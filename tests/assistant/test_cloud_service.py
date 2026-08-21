import unittest

from src.assistant.cloud_service import CloudRouteBridge
from src.assistant.contracts import (
    AssistantError,
    AssistantGenerationError,
    AssistantRequest,
    AssistantResponse,
)
from src.assistant.service import AssistantService
from src.context.contracts import ContextBlock, ContextMessage
from src.llm.approval import ApprovalGate, ApprovalRequiredError, CloudPreview
from src.llm.providers.cloud import FakeCloudProvider
from src.llm.providers.fake import FakeProvider
from src.llm.registry import ProviderRegistry
from src.llm.routing.static import StaticRouter


def _request(privacy: str = "cloud_allowed", allow_cloud: bool = True, request_id: str = "r1") -> AssistantRequest:
    return AssistantRequest(
        text="hi",
        conversation_id="c1",
        channel="web",
        privacy=privacy,  # type: ignore[arg-type]
        allow_cloud=allow_cloud,
        request_id=request_id,
    )


def _registry_with_cloud(fail: bool = False) -> ProviderRegistry:
    reg = ProviderRegistry()
    reg.register("ollama", FakeProvider(response="local reply"), local=True)
    reg.register("cloud", FakeCloudProvider(fail=fail), local=False)
    return reg


def _local_service(reg: ProviderRegistry) -> AssistantService:
    router = StaticRouter(reg, default_provider_id="ollama")
    return AssistantService(reg, router)


def _blocks():
    return [
        ContextBlock(source="calendar", content="calendar secret", sensitivity="personal", local_only=True),
        ContextBlock(source="screen", content="screen secret", sensitivity="secret", local_only=True),
        ContextBlock(source="tasks", content="task list secret", sensitivity="personal", local_only=True),
        ContextBlock(source="public_note", content="public info", sensitivity="public", local_only=False),
    ]


class CloudRouteBridgeTest(unittest.TestCase):
    def test_requires_cloud_privacy(self):
        reg = _registry_with_cloud()
        bridge = CloudRouteBridge(reg, "cloud", approval=ApprovalGate(), local_service=_local_service(reg))
        with self.assertRaises(AssistantError):
            bridge.send(
                _request(privacy="local_only"),
                _blocks(),
            )

    def test_requires_approval(self):
        reg = _registry_with_cloud()
        bridge = CloudRouteBridge(reg, "cloud", approval=ApprovalGate(), local_service=_local_service(reg))
        with self.assertRaises(ApprovalRequiredError):
            bridge.send(_request(), _blocks())

    def test_send_success_anonymized(self):
        reg = _registry_with_cloud()
        gate = ApprovalGate()
        gate.approve("r1")
        bridge = CloudRouteBridge(reg, "cloud", approval=gate, local_service=_local_service(reg))
        resp = bridge.send(_request(), _blocks())
        self.assertIsInstance(resp, AssistantResponse)
        self.assertFalse(resp.route.local)
        self.assertEqual(resp.text, "cloud response")

        sent = reg.get("cloud").provider.sent_payloads[0]
        joined = str(sent)
        self.assertNotIn("calendar", joined)
        self.assertNotIn("screen", joined)
        self.assertNotIn("task list", joined)
        self.assertIn("public info", joined)

    def test_fallback_to_local_on_cloud_failure(self):
        reg = _registry_with_cloud(fail=True)
        gate = ApprovalGate()
        gate.approve("r1")
        bridge = CloudRouteBridge(reg, "cloud", approval=gate, local_service=_local_service(reg))
        resp = bridge.send(_request(), _blocks())
        self.assertTrue(resp.route.local)
        self.assertEqual(resp.text, "local reply")

    def test_no_fallback_without_local_service(self):
        reg = _registry_with_cloud(fail=True)
        gate = ApprovalGate()
        gate.approve("r1")
        bridge = CloudRouteBridge(reg, "cloud", approval=gate, local_service=None)
        with self.assertRaises(AssistantGenerationError):
            bridge.send(_request(), _blocks())

    def test_disabled_by_default_no_cloud_registered(self):
        reg = ProviderRegistry()
        reg.register("ollama", FakeProvider(), local=True)
        self.assertNotIn("cloud", reg)

    def test_preview_returns_anonymized(self):
        """preview() returns CloudPreview with only public content."""
        reg = _registry_with_cloud()
        gate = ApprovalGate()
        bridge = CloudRouteBridge(reg, "cloud", approval=gate, local_service=_local_service(reg))
        pv = bridge.preview(_request(), _blocks())
        self.assertIsInstance(pv, CloudPreview)
        self.assertEqual(pv.request_id, "r1")
        joined = str(pv.messages)
        self.assertNotIn("calendar", joined)
        self.assertNotIn("screen", joined)
        self.assertNotIn("task list", joined)
        self.assertIn("public info", joined)

    def test_entry_local_fallback(self):
        """entry.local=True should fallback to local, not raise."""
        reg = ProviderRegistry()
        reg.register("ollama", FakeProvider(response="local reply"), local=True)
        reg.register("cloud", FakeCloudProvider(), local=True)  # local=True!
        gate = ApprovalGate()
        gate.approve("r1")
        bridge = CloudRouteBridge(reg, "cloud", approval=gate, local_service=_local_service(reg))
        resp = bridge.send(_request(), _blocks())
        self.assertTrue(resp.route.local)
        self.assertEqual(resp.text, "local reply")

    def test_successful_send_revokes_approval(self):
        """After successful cloud send, approval is revoked."""
        reg = _registry_with_cloud()
        gate = ApprovalGate()
        gate.approve("r1")
        bridge = CloudRouteBridge(reg, "cloud", approval=gate, local_service=_local_service(reg))
        bridge.send(_request(), _blocks())
        self.assertFalse(gate.is_approved("r1"))

    def test_request_id_none_fallback(self):
        """request_id=None should fallback to local, not raise."""
        reg = _registry_with_cloud()
        gate = ApprovalGate()
        bridge = CloudRouteBridge(reg, "cloud", approval=gate, local_service=_local_service(reg))
        req = _request(request_id=None)
        resp = bridge.send(req, _blocks())
        self.assertTrue(resp.route.local)
        self.assertEqual(resp.text, "local reply")

    def test_personal_history_not_in_cloud_payload(self):
        """History as a ContextBlock in blocks should be filtered out for cloud."""
        reg = _registry_with_cloud()
        gate = ApprovalGate()
        gate.approve("r1")
        bridge = CloudRouteBridge(reg, "cloud", approval=gate, local_service=_local_service(reg))
        history_block = ContextBlock(
            source="history",
            content=tuple([ContextMessage(role="user", content="secret chat")]),
            sensitivity="personal",
            local_only=True,
        )
        blocks = _blocks() + [history_block]
        resp = bridge.send(_request(), blocks)
        self.assertFalse(resp.route.local)
        sent = reg.get("cloud").provider.sent_payloads[0]
        self.assertNotIn("secret chat", str(sent))
        self.assertIn("public info", str(sent))

    def test_base_system_in_local_fallback_messages(self):
        """CloudRouteBridge with base_system produces local fallback messages including it."""
        reg = _registry_with_cloud(fail=True)
        gate = ApprovalGate()
        gate.approve("r1")
        # Use a FakeProvider as local_service so we can inspect messages
        local_provider = FakeProvider(response="local reply")
        local_reg = ProviderRegistry()
        local_reg.register("ollama", local_provider, local=True)
        local_router = StaticRouter(local_reg, default_provider_id="ollama")
        local_svc = AssistantService(local_reg, local_router)

        bridge = CloudRouteBridge(
            reg, "cloud", approval=gate,
            local_service=local_svc, base_system="SYS",
        )
        resp = bridge.send(_request(), _blocks())
        self.assertTrue(resp.route.local)
        # The local_provider should have received messages with SYS in system content
        self.assertTrue(len(local_provider.calls) > 0)
        messages = local_provider.calls[0]["messages"]
        system_msgs = [m for m in messages if m.get("role") == "system"]
        self.assertTrue(len(system_msgs) > 0)
        self.assertIn("SYS", system_msgs[0]["content"])

    def test_base_system_in_cloud_payload(self):
        """CloudRouteBridge with base_system includes it in cloud payload."""
        reg = _registry_with_cloud()
        gate = ApprovalGate()
        gate.approve("r1")
        bridge = CloudRouteBridge(
            reg, "cloud", approval=gate,
            local_service=_local_service(reg), base_system="SYS",
        )
        resp = bridge.send(_request(), _blocks())
        self.assertFalse(resp.route.local)
        sent = reg.get("cloud").provider.sent_payloads[0]
        joined = str(sent)
        self.assertIn("SYS", joined)


if __name__ == "__main__":
    unittest.main()
