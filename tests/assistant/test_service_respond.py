import unittest

from src.assistant.contracts import (
    AssistantRequest,
    AssistantResponse,
)
from src.assistant.service import AssistantService
from src.context.contracts import ContextBlock
from src.llm.approval import ApprovalGate, CloudPreview
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


def _blocks():
    return [
        ContextBlock(source="calendar", content="calendar secret", sensitivity="personal", local_only=True),
        ContextBlock(source="public_note", content="public info", sensitivity="public", local_only=False),
    ]


class AssistantServiceRespondStreamTest(unittest.TestCase):
    """respond_stream は StreamResult を返し、反復後に .response で route/stats を取れる。"""

    def test_respond_stream_returns_stream_result_with_local_provider(self):
        from src.assistant.service import StreamResult

        reg = ProviderRegistry()
        provider = FakeProvider(response="stream reply")
        reg.register("ollama", provider, local=True)
        router = StaticRouter(reg, default_provider_id="ollama")
        service = AssistantService(reg, router)

        from src.context.providers.history import HistoryContextProvider
        history_block = HistoryContextProvider.collect([
            {"role": "user", "content": "hello"},
        ])
        blocks = [b for b in [_block_for_stream()] + ([history_block] if history_block else []) if b]

        stream = service.respond_stream(
            _request(privacy="local_only", allow_cloud=False),
            blocks,
            base_system="SYS",
        )
        self.assertIsInstance(stream, StreamResult)
        tokens = list(stream)
        self.assertEqual("".join(tokens), "stream reply")
        # 反復後に .response で route/stats にアクセスできる
        self.assertTrue(stream.response.route.local)
        self.assertEqual(stream.response.text, "stream reply")

    def test_respond_stream_base_system_preserved(self):
        """respond_stream も base_system を system message へ反映する。"""
        reg = ProviderRegistry()
        provider = FakeProvider(response="ok")
        reg.register("ollama", provider, local=True)
        router = StaticRouter(reg, default_provider_id="ollama")
        service = AssistantService(reg, router)

        from src.context.providers.history import HistoryContextProvider
        history_block = HistoryContextProvider.collect([
            {"role": "user", "content": "hello"},
        ])
        blocks = [b for b in [_block_for_stream()] + ([history_block] if history_block else []) if b]

        stream = service.respond_stream(
            _request(privacy="local_only", allow_cloud=False),
            blocks,
            base_system="You are a helpful assistant.",
        )
        list(stream)
        messages = provider.calls[0]["messages"]
        system_msgs = [m for m in messages if m.get("role") == "system"]
        self.assertTrue(len(system_msgs) > 0)
        self.assertIn("You are a helpful assistant.", system_msgs[0]["content"])

    def test_respond_stream_ignores_cloud_bridge(self):
        """respond_stream は cloud_bridge があってもローカルストリームへ（cloud は非ストリーム）。"""
        from src.assistant.cloud_service import CloudRouteBridge
        from src.assistant.service import StreamResult

        reg = ProviderRegistry()
        provider = FakeProvider(response="local stream")
        reg.register("ollama", provider, local=True)
        cloud = FakeCloudProvider()
        reg.register("cloud", cloud, local=False)
        router = StaticRouter(reg, default_provider_id="ollama")
        service = AssistantService(reg, router)

        gate = ApprovalGate()
        gate.approve("r1")
        bridge = CloudRouteBridge(reg, "cloud", approval=gate, local_service=service)
        service.set_cloud_bridge(bridge)

        stream = service.respond_stream(_request(), [_block_for_stream()])
        self.assertIsInstance(stream, StreamResult)
        tokens = list(stream)
        self.assertEqual("".join(tokens), "local stream")
        # cloud provider は使われない（ストリームはローカルへ fallback）
        self.assertEqual(len(cloud.sent_payloads), 0)
        self.assertTrue(stream.response.route.local)


def _block_for_stream() -> ContextBlock:
    return ContextBlock(
        source="public_note", content="public info",
        sensitivity="public", local_only=False,
    )


class AssistantServiceRespondTest(unittest.TestCase):
    def test_respond_with_cloud_bridge(self):
        """respond with cloud_bridge + cloud_allowed + allow_cloud delegates to bridge."""
        from src.assistant.cloud_service import CloudRouteBridge

        reg = ProviderRegistry()
        reg.register("ollama", FakeProvider(response="local reply"), local=True)
        reg.register("cloud", FakeCloudProvider(), local=False)
        router = StaticRouter(reg, default_provider_id="ollama")
        service = AssistantService(reg, router)

        gate = ApprovalGate()
        gate.approve("r1")
        bridge = CloudRouteBridge(reg, "cloud", approval=gate, local_service=service)
        service.set_cloud_bridge(bridge)

        resp, preview = service.respond(_request(), _blocks())
        self.assertIsInstance(resp, AssistantResponse)
        self.assertFalse(resp.route.local)
        self.assertEqual(resp.text, "cloud response")
        self.assertIsNotNone(preview)
        self.assertIsInstance(preview, CloudPreview)

    def test_respond_without_cloud_bridge(self):
        """respond without cloud_bridge goes local."""
        reg = ProviderRegistry()
        reg.register("ollama", FakeProvider(response="local reply"), local=True)
        router = StaticRouter(reg, default_provider_id="ollama")
        service = AssistantService(reg, router)

        resp, preview = service.respond(_request(), _blocks())
        self.assertIsInstance(resp, AssistantResponse)
        self.assertTrue(resp.route.local)
        self.assertEqual(resp.text, "local reply")
        self.assertIsNone(preview)

    def test_respond_with_cloud_bridge_but_local_privacy(self):
        """respond with cloud_bridge but privacy=local_only goes local."""
        from src.assistant.cloud_service import CloudRouteBridge

        reg = ProviderRegistry()
        reg.register("ollama", FakeProvider(response="local reply"), local=True)
        reg.register("cloud", FakeCloudProvider(), local=False)
        router = StaticRouter(reg, default_provider_id="ollama")
        service = AssistantService(reg, router)

        gate = ApprovalGate()
        bridge = CloudRouteBridge(reg, "cloud", approval=gate, local_service=service)
        service.set_cloud_bridge(bridge)

        resp, preview = service.respond(_request(privacy="local_only", allow_cloud=False), _blocks())
        self.assertTrue(resp.route.local)
        self.assertEqual(resp.text, "local reply")
        self.assertIsNone(preview)

    def test_set_cloud_bridge(self):
        """set_cloud_bridge sets the bridge."""
        from src.assistant.cloud_service import CloudRouteBridge

        reg = ProviderRegistry()
        reg.register("ollama", FakeProvider(), local=True)
        router = StaticRouter(reg, default_provider_id="ollama")
        service = AssistantService(reg, router)
        self.assertIsNone(service._cloud_bridge)

        gate = ApprovalGate()
        bridge = CloudRouteBridge(reg, "cloud", approval=gate)
        service.set_cloud_bridge(bridge)
        self.assertIs(service._cloud_bridge, bridge)

    def test_respond_base_system_preserved_in_messages(self):
        """respond with base_system includes it in the system message."""
        from src.context.providers.history import HistoryContextProvider

        reg = ProviderRegistry()
        provider = FakeProvider(response="reply")
        reg.register("ollama", provider, local=True)
        router = StaticRouter(reg, default_provider_id="ollama")
        service = AssistantService(reg, router)

        history_block = HistoryContextProvider.collect([
            {"role": "user", "content": "hello"},
        ])
        blocks = list(_blocks()) + ([history_block] if history_block is not None else [])

        service.respond(
            _request(privacy="local_only", allow_cloud=False),
            blocks,
            base_system="You are a helpful assistant.",
        )

        self.assertTrue(len(provider.calls) > 0)
        messages = provider.calls[0]["messages"]
        system_msgs = [m for m in messages if m.get("role") == "system"]
        self.assertTrue(len(system_msgs) > 0)
        self.assertIn("You are a helpful assistant.", system_msgs[0]["content"])

    def test_respond_without_base_system(self):
        """respond without base_system (default '') omits system message or has empty content."""
        from src.context.providers.history import HistoryContextProvider

        reg = ProviderRegistry()
        provider = FakeProvider(response="reply")
        reg.register("ollama", provider, local=True)
        router = StaticRouter(reg, default_provider_id="ollama")
        service = AssistantService(reg, router)

        history_block = HistoryContextProvider.collect([
            {"role": "user", "content": "hello"},
        ])
        blocks = list(_blocks()) + ([history_block] if history_block is not None else [])

        service.respond(
            _request(privacy="local_only", allow_cloud=False),
            blocks,
        )

        self.assertTrue(len(provider.calls) > 0)
        messages = provider.calls[0]["messages"]
        system_msgs = [m for m in messages if m.get("role") == "system"]
        # With base_system="" and no public str blocks, system message should be absent
        # (ContextBuilder only adds system message when system_content is truthy)
        for m in system_msgs:
            self.assertNotIn("You are a helpful assistant.", m["content"])

    def test_respond_cloud_bridge_base_system_in_preview(self):
        """respond with cloud_bridge + base_system -> preview includes base_system."""
        from src.assistant.cloud_service import CloudRouteBridge

        reg = ProviderRegistry()
        reg.register("ollama", FakeProvider(response="local reply"), local=True)
        reg.register("cloud", FakeCloudProvider(), local=False)
        router = StaticRouter(reg, default_provider_id="ollama")
        service = AssistantService(reg, router)

        gate = ApprovalGate()
        gate.approve("r1")
        bridge = CloudRouteBridge(
            reg, "cloud", approval=gate,
            local_service=service, base_system="SYS_PROMPT",
        )
        service.set_cloud_bridge(bridge)

        resp, preview = service.respond(_request(), _blocks())
        self.assertIsNotNone(preview)
        joined = str(preview.messages)
        self.assertIn("SYS_PROMPT", joined)


if __name__ == "__main__":
    unittest.main()
