from __future__ import annotations

import asyncio
import json
import threading
import unittest
from dataclasses import dataclass
from unittest.mock import patch

from src.assistant.contracts import AssistantRequest
from src.assistant.factory import build_local_service
from src.assistant.stream_queue import QueueStream, stream_to_queue
from src.context.contracts import ContextBlock
from src.llm.providers.fake import FakeProvider
from src.llm.routing.contracts import NoRouteError
from src.web import server


@dataclass
class _FactoryConfig:
    ollama_base_url: str = "http://unused.invalid"
    model: str = "config-model"
    temperature: float = 0.2
    top_p: float = 0.8
    top_k: int = 20
    repeat_penalty: float = 1.2
    num_ctx: int = 4096
    num_predict: int | None = 128


class _FailingProvider(FakeProvider):
    def generate_stream(self, messages, **kwargs):
        raise RuntimeError("planned provider failure")


class _EndlessProvider(FakeProvider):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()

    def generate_stream(self, messages, **kwargs):
        self.started.set()
        while True:
            yield "token"


class _UnexpectedService:
    def generate(self, *args, **kwargs):
        raise AssertionError("task extraction must not use AssistantService")


class WebAssistantStreamTest(unittest.TestCase):
    def setUp(self) -> None:
        self.original_config = server.config
        self.original_llm = server.llm
        self.original_assistant_service = server.assistant_service
        self.original_provider_registry = server.provider_registry
        self.original_task_store = server.task_store

    def tearDown(self) -> None:
        server.config = self.original_config
        server.llm = self.original_llm
        server.assistant_service = self.original_assistant_service
        server.provider_registry = self.original_provider_registry
        server.task_store = self.original_task_store

    @staticmethod
    def _request() -> AssistantRequest:
        return AssistantRequest(
            text="質問",
            conversation_id="web-test",
            channel="web",
            privacy="local_only",
        )

    @staticmethod
    def _consume(queue_stream: QueueStream) -> list[object]:
        items: list[object] = []
        while True:
            item = queue_stream.queue.get(timeout=1.0)
            items.append(item)
            if item is None:
                return items

    def test_service_stream_keeps_token_order_and_sentinel_shape(self) -> None:
        provider = FakeProvider(stream_chunks=("first", "second", "third"))
        service, _ = build_local_service(_FactoryConfig(), provider=provider)

        queue_stream = stream_to_queue(
            service.generate_stream(
                self._request(), [{"role": "user", "content": "質問"}]
            )
        )

        self.assertEqual(
            self._consume(queue_stream), ["first", "second", "third", None]
        )
        self.assertTrue(queue_stream.join(timeout=1.0))

    def test_start_assistant_stream_routes_and_starts_queue_worker(self) -> None:
        provider = FakeProvider(stream_chunks=("first", "second"))
        server.assistant_service, _ = build_local_service(
            _FactoryConfig(), provider=provider
        )
        blocks = (
            ContextBlock(
                source="history",
                content="user says 質問",
                sensitivity="personal",
                local_only=True,
            ),
        )

        queue_stream = server._start_assistant_stream(
            self._request(), blocks, base_system="test-system"
        )

        self.assertIsInstance(queue_stream, QueueStream)
        self.assertEqual(self._consume(queue_stream), ["first", "second", None])
        self.assertTrue(queue_stream.join(timeout=1.0))
        self.assertEqual(provider.calls[0]["kind"], "generate_stream")

    def test_start_assistant_stream_propagates_route_error(self) -> None:
        provider = FakeProvider(available=False)
        server.assistant_service, _ = build_local_service(
            _FactoryConfig(), provider=provider
        )
        blocks = (
            ContextBlock(
                source="history",
                content="user says 質問",
                sensitivity="personal",
                local_only=True,
            ),
        )

        with self.assertRaises(NoRouteError):
            server._start_assistant_stream(
                self._request(), blocks, base_system="test-system"
            )

    def test_provider_exception_is_followed_by_sentinel(self) -> None:
        service, _ = build_local_service(
            _FactoryConfig(), provider=_FailingProvider()
        )
        queue_stream = stream_to_queue(
            service.generate_stream(
                self._request(), [{"role": "user", "content": "質問"}]
            )
        )

        items = self._consume(queue_stream)

        self.assertIsInstance(items[0], RuntimeError)
        self.assertEqual(str(items[0]), "planned provider failure")
        self.assertIsNone(items[1])
        self.assertTrue(queue_stream.join(timeout=1.0))

    def test_cancel_stops_stream_worker(self) -> None:
        provider = _EndlessProvider()
        service, _ = build_local_service(_FactoryConfig(), provider=provider)
        queue_stream = stream_to_queue(
            service.generate_stream(
                self._request(), [{"role": "user", "content": "質問"}]
            )
        )
        self.assertTrue(provider.started.wait(timeout=1.0))

        queue_stream.cancel()

        self.assertTrue(queue_stream.join(timeout=1.0))
        self.assertFalse(queue_stream.is_running)

    def test_status_uses_registry_availability_and_model(self) -> None:
        server.config = _FactoryConfig()
        server.provider_registry = None

        uninitialized = asyncio.run(server.status())

        self.assertIs(uninitialized["ollama"], False)
        self.assertEqual(uninitialized["model"], "config-model")
        self.assertIsInstance(uninitialized["ollama"], bool)
        self.assertIsInstance(uninitialized["model"], str)

        provider = FakeProvider(model="registry-model")
        service, registry = build_local_service(server.config, provider=provider)
        server.assistant_service = service
        server.provider_registry = registry
        server.llm = provider

        initialized = asyncio.run(server.status())

        self.assertIs(initialized["ollama"], True)
        self.assertEqual(initialized["model"], "registry-model")
        self.assertIsInstance(initialized["ollama"], bool)
        self.assertIsInstance(initialized["model"], str)

    def test_start_assistant_stream_uses_respond_stream_with_blocks_and_base_system(
        self,
    ) -> None:
        """_start_assistant_stream は assistant_service.respond_stream を
        blocks + base_system を渡して呼ぶこと。"""
        provider = FakeProvider(stream_chunks=("ok",))
        service, _ = build_local_service(_FactoryConfig(), provider=provider)
        server.assistant_service = service
        blocks = (
            ContextBlock(
                source="history",
                content="user says 質問",
                sensitivity="personal",
                local_only=True,
            ),
        )

        with patch.object(service, "respond_stream", wraps=service.respond_stream) as mock_rs:
            queue_stream = server._start_assistant_stream(
                self._request(), blocks, base_system="my-system"
            )
            self._consume(queue_stream)

        mock_rs.assert_called_once()
        call_args = mock_rs.call_args
        # positional: request, blocks
        self.assertIsInstance(call_args.args[0], AssistantRequest)
        self.assertEqual(call_args.args[1], blocks)
        # keyword: base_system
        self.assertEqual(call_args.kwargs["base_system"], "my-system")

    def test_task_extraction_keeps_direct_provider_options(self) -> None:
        provider = FakeProvider(response=json.dumps({"tasks": []}))
        server.config = _FactoryConfig()
        server.llm = provider
        server.assistant_service = _UnexpectedService()
        server.task_store = object()

        with patch.dict(
            "os.environ",
            {
                "TASKS_CHAT_EXTRACTION_ENABLED": "true",
                "TASKS_CHAT_EXTRACTION_TIMEOUT_SECONDS": "17",
            },
        ):
            self.assertEqual(server._extract_task_candidates("資料を提出する"), [])

        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(provider.calls[0]["kind"], "generate")
        self.assertEqual(provider.calls[0]["options"]["temperature"], 0.0)
        self.assertEqual(provider.calls[0]["options"]["timeout"], 17.0)


if __name__ == "__main__":
    unittest.main()
