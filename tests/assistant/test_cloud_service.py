import unittest

from src.assistant.cloud_service import CloudRouteBridge
from src.assistant.contracts import (
    AssistantError,
    AssistantGenerationError,
    AssistantRequest,
    AssistantResponse,
)
from src.assistant.service import AssistantService
from src.context.contracts import ContextBlock
from src.llm.approval import ApprovalGate, ApprovalRequiredError
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
                local_messages=[{"role": "user", "content": "hi"}],
            )

    def test_requires_approval(self):
        reg = _registry_with_cloud()
        bridge = CloudRouteBridge(reg, "cloud", approval=ApprovalGate(), local_service=_local_service(reg))
        with self.assertRaises(ApprovalRequiredError):
            bridge.send(_request(), _blocks(), local_messages=[{"role": "user", "content": "hi"}])

    def test_send_success_anonymized(self):
        reg = _registry_with_cloud()
        gate = ApprovalGate()
        gate.approve("r1")
        bridge = CloudRouteBridge(reg, "cloud", approval=gate, local_service=_local_service(reg))
        resp = bridge.send(_request(), _blocks(), local_messages=[{"role": "user", "content": "hi"}])
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
        resp = bridge.send(_request(), _blocks(), local_messages=[{"role": "user", "content": "hi"}])
        self.assertTrue(resp.route.local)
        self.assertEqual(resp.text, "local reply")

    def test_no_fallback_without_local_service(self):
        reg = _registry_with_cloud(fail=True)
        gate = ApprovalGate()
        gate.approve("r1")
        bridge = CloudRouteBridge(reg, "cloud", approval=gate, local_service=None)
        with self.assertRaises(AssistantGenerationError):
            bridge.send(_request(), _blocks(), local_messages=[{"role": "user", "content": "hi"}])

    def test_disabled_by_default_no_cloud_registered(self):
        reg = ProviderRegistry()
        reg.register("ollama", FakeProvider(), local=True)
        self.assertNotIn("cloud", reg)


if __name__ == "__main__":
    unittest.main()
