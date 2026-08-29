from __future__ import annotations

import asyncio
import json
import queue
import tempfile
import threading
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from unittest.mock import patch

from src.assistant.contracts import AssistantRequest
from src.assistant.factory import build_local_service
from src.assistant.stream_queue import QueueStream, stream_to_queue
from src.context.contracts import ContextBlock
from src.llm.providers.fake import FakeProvider
from src.llm.registry import ProviderRegistry
from src.llm.routing.contracts import NoRouteError
from src.perception import SensorPolicy
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
        self.original_primary_provider_id = server.primary_provider_id
        self.original_primary_provider_kind = server.primary_provider_kind
        self.original_primary_provider_base_url = server.primary_provider_base_url

    def tearDown(self) -> None:
        server.config = self.original_config
        server.llm = self.original_llm
        server.assistant_service = self.original_assistant_service
        server.provider_registry = self.original_provider_registry
        server.task_store = self.original_task_store
        server.primary_provider_id = self.original_primary_provider_id
        server.primary_provider_kind = self.original_primary_provider_kind
        server.primary_provider_base_url = self.original_primary_provider_base_url

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

    def test_status_omits_provider_model_metadata(self) -> None:
        server.config = _FactoryConfig()
        server.provider_registry = None

        uninitialized = asyncio.run(server.status())

        self.assertIs(uninitialized["ollama"], False)
        self.assertNotIn("model", uninitialized)
        self.assertIsInstance(uninitialized["ollama"], bool)
        # base_url 未解決のときは到達可能性 unconfigured としてプローブしない
        self.assertEqual(uninitialized["provider_reachability"], "unconfigured")

        provider = FakeProvider(model="registry-model")
        service, registry = build_local_service(server.config, provider=provider)
        server.assistant_service = service
        server.provider_registry = registry
        server.llm = provider
        server.primary_provider_kind = "ollama"
        server.primary_provider_base_url = server.config.ollama_base_url

        with patch("src.web.server.HealthChecker") as mock_cls:
            mock_cls.return_value.check_all.return_value = {
                "status": "ok",
                "checks": {"ollama": {"status": "ok"}},
            }
            initialized = asyncio.run(server.status())

        self.assertIs(initialized["ollama"], True)
        self.assertNotIn("model", initialized)
        self.assertIsInstance(initialized["ollama"], bool)
        self.assertEqual(initialized["provider_reachability"], "ok")

    def test_status_resolves_through_tracked_primary_provider_id(self) -> None:
        """追跡中 primary_provider_id を優先し、ハードコードされた "ollama" に依存しない。"""
        server.config = _FactoryConfig()
        server.primary_provider_id = "local-openai"
        server.primary_provider_kind = "openai_compatible"
        server.primary_provider_base_url = "http://localhost:8080/v1"
        registry = ProviderRegistry()
        registry.register(
            "ollama", FakeProvider(model="ollama-model", available=False), local=True
        )
        registry.register("local-openai", FakeProvider(model="local-model"), local=True)
        server.provider_registry = registry

        with patch("src.web.server.HealthChecker") as mock_cls:
            mock_cls.return_value.check_all.return_value = {
                "status": "ok",
                "checks": {"local_provider": {"status": "ok"}},
            }
            body = asyncio.run(server.status())

        self.assertEqual(body["provider_id"], "local-openai")
        self.assertEqual(body["provider_kind"], "openai_compatible")
        self.assertNotIn("model", body)
        self.assertEqual(body["provider_reachability"], "ok")
        self.assertTrue(body["ollama"])  # 後方互換エイリアス = 選択中Providerの到達性
        self.assertTrue(body["local_provider"])

    def test_status_resolves_chat_config_resolved_provider_id(self) -> None:
        """ChatConfig の resolved provider_id (明示 local_provider_id) を通じて解決する。"""
        from src.chat.config import ChatConfig

        cfg = ChatConfig(
            model="cfg-model",
            local_provider_kind="openai_compatible",
            local_base_url="http://localhost:8080/v1",
            local_provider_id="custom-openai",
        )
        server.config = cfg
        service, registry = build_local_service(cfg)
        server.assistant_service = service
        server.provider_registry = registry
        server.primary_provider_id = cfg.resolved_local_provider_id()
        server.primary_provider_kind = "openai_compatible"
        server.primary_provider_base_url = cfg.resolved_local_base_url()

        with patch("src.web.server.HealthChecker") as mock_cls:
            mock_cls.return_value.check_all.return_value = {
                "status": "ok",
                "checks": {"local_provider": {"status": "ok"}},
            }
            body = asyncio.run(server.status())

        self.assertEqual(body["provider_id"], "custom-openai")
        self.assertEqual(body["provider_kind"], "openai_compatible")
        self.assertNotIn("model", body)
        self.assertEqual(body["provider_reachability"], "ok")
        self.assertTrue(body["ollama"])
        self.assertTrue(body["local_provider"])

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


class _Raise:
    """_ScriptedQueue に例外を上げさせる script 要素。"""

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc


class _ScriptedQueue:
    """script に決め打ちした順で値を返すか例外を上げる疑似Queue。"""

    def __init__(self, script: list[object] = ()) -> None:
        self._script = list(script)

    def get(self, block: bool = True, timeout: float | None = None) -> object:
        if not self._script:
            raise queue.Empty
        item = self._script.pop(0)
        if isinstance(item, _Raise):
            raise item.exc
        return item


class _FakeQueueStream:
    """websocket_chat が期待する queue + cancel を持った疑似 QueueStream。"""

    def __init__(self, script: list[object] = ()) -> None:
        self.queue = _ScriptedQueue(script)
        self.cancel_called = False

    def cancel(self) -> None:
        self.cancel_called = True


class _BlockingQueue:
    """cancel() で解放されるまで get をブロックする疑似Queue。"""

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def get(self, block: bool = True, timeout: float | None = None) -> object:
        self.entered.set()
        self.release.wait(timeout=timeout)
        if not self.release.is_set():
            raise queue.Empty
        return None


class _CancellableQueueStream:
    """cancel() がブロック中の consumer を解放する疑似 QueueStream。"""

    def __init__(self, block_queue: _BlockingQueue) -> None:
        self.queue = block_queue
        self.cancel_called = False

    def cancel(self) -> None:
        self.cancel_called = True
        self.queue.release.set()


class _FakeIdleManager:
    def __init__(self) -> None:
        self.starts = 0
        self.ends = 0

    def notify_inference_start(self, wait_for_gpu: bool = False) -> None:
        self.starts += 1

    def notify_inference_end(self) -> None:
        self.ends += 1


class _FakeWebSocket:
    """accept / receive_text / send_json を持ったオフラインWebSocket。"""

    def __init__(self, incoming: list[str] = ()) -> None:
        self.incoming = list(incoming)
        self.sent: list[dict] = []
        self.accepted = False

    async def accept(self) -> None:
        self.accepted = True

    async def receive_text(self) -> str:
        if not self.incoming:
            raise server.WebSocketDisconnect()
        return self.incoming.pop(0)

    async def send_json(self, payload: dict) -> None:
        self.sent.append(payload)


class WebSocketChatStreamTest(unittest.IsolatedAsyncioTestCase):
    """websocket_chat のキューストリーミング失敗セマンティクス (オフライン)。"""

    _MESSAGE = json.dumps({
        "type": "message",
        "text": "質問",
        "session_id": "web-test",
        "tts": False,
    })

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self._saved = {
            "config": server.config,
            "assistant_service": server.assistant_service,
            "idle_manager": server.idle_manager,
            "task_store": server.task_store,
            "llm": server.llm,
            "stt": server.stt,
            "tts": server.tts,
            "rag": server.rag,
            "vision": server.vision,
            "screen": server.screen,
            "monitor": server.monitor,
            "profile": server.profile,
            "preloader": server.preloader,
            "web_search": server.web_search,
            "growth_tracker": server.growth_tracker,
            "calendar_client": server.calendar_client,
            "sessions": server.sessions,
        }
        server.config = SimpleNamespace(
            system_prompt="test-system",
            max_history_turns=20,
            history_dir=str(Path(self.tmp.name) / "history"),
            emotion_tag_enabled=False,
        )
        server.assistant_service = None
        server.idle_manager = _FakeIdleManager()
        server.task_store = None
        server.llm = None
        server.stt = None
        server.tts = None
        server.rag = None
        server.vision = None
        server.screen = None
        server.monitor = None
        server.profile = None
        server.preloader = None
        server.web_search = None
        server.growth_tracker = None
        server.calendar_client = None
        server.sessions = {}

    def tearDown(self) -> None:
        for key, value in self._saved.items():
            setattr(server, key, value)
        self.tmp.cleanup()

    async def _run_message(
        self, *, stream: _FakeQueueStream | None = None, start_error: BaseException | None = None
    ) -> _FakeWebSocket:
        ws = _FakeWebSocket([self._MESSAGE])
        if start_error is not None:
            patcher = patch.object(
                server, "_start_assistant_stream", side_effect=start_error
            )
        else:
            patcher = patch.object(
                server, "_start_assistant_stream", return_value=stream
            )
        with patcher:
            await server.websocket_chat(ws)
        return ws

    async def test_start_failure_sends_neutral_error_and_rolls_back(self) -> None:
        ws = await self._run_message(start_error=NoRouteError())

        self.assertEqual(ws.sent[-1]["type"], "error")
        self.assertEqual(ws.sent[-1]["message"], server._STREAM_ERROR_MESSAGE)
        self.assertNotIn("done", [item["type"] for item in ws.sent])
        self.assertEqual(server.sessions["web-test"].messages, [])
        self.assertEqual(server.idle_manager.ends, 1)

    async def test_immediate_timeout_sends_error_and_no_done(self) -> None:
        stream = _FakeQueueStream([_Raise(queue.Empty())])
        ws = await self._run_message(stream=stream)

        self.assertEqual(ws.sent[-1]["type"], "error")
        self.assertEqual(ws.sent[-1]["message"], server._STREAM_TIMEOUT_MESSAGE)
        self.assertNotIn("done", [item["type"] for item in ws.sent])
        self.assertTrue(stream.cancel_called)
        self.assertEqual(server.sessions["web-test"].messages, [])
        self.assertEqual(server.idle_manager.ends, 1)

    async def test_arbitrary_get_exception_is_not_swallowed(self) -> None:
        stream = _FakeQueueStream([_Raise(RuntimeError("boom"))])
        ws = await self._run_message(stream=stream)

        self.assertEqual(ws.sent[-1]["type"], "error")
        self.assertEqual(ws.sent[-1]["message"], server._STREAM_ERROR_MESSAGE)
        self.assertNotIn("boom", ws.sent[-1]["message"])
        self.assertNotIn("done", [item["type"] for item in ws.sent])
        self.assertTrue(stream.cancel_called)
        self.assertEqual(server.sessions["web-test"].messages, [])
        self.assertEqual(server.idle_manager.ends, 1)

    async def test_empty_sentinel_treated_as_error(self) -> None:
        stream = _FakeQueueStream([None])
        ws = await self._run_message(stream=stream)

        self.assertEqual(
            ws.sent, [{"type": "error", "message": server._STREAM_EMPTY_MESSAGE}]
        )
        self.assertTrue(stream.cancel_called)
        self.assertEqual(server.sessions["web-test"].messages, [])
        self.assertEqual(server.idle_manager.ends, 1)

    async def test_provider_exception_sentinel_sends_error_no_done(self) -> None:
        stream = _FakeQueueStream(["hello", RuntimeError("provider exploded"), None])
        ws = await self._run_message(stream=stream)

        types = [item["type"] for item in ws.sent]
        self.assertEqual(types, ["token", "error"])
        self.assertEqual(ws.sent[0]["content"], "hello")
        self.assertNotIn("done", types)
        self.assertNotIn("provider exploded", ws.sent[-1]["message"])
        self.assertTrue(stream.cancel_called)
        self.assertEqual(server.sessions["web-test"].messages, [])
        self.assertEqual(server.idle_manager.ends, 1)

    async def test_partial_then_timeout_keeps_tokens_then_error(self) -> None:
        stream = _FakeQueueStream(["first", "second", _Raise(queue.Empty())])
        ws = await self._run_message(stream=stream)

        types = [item["type"] for item in ws.sent]
        self.assertEqual(types, ["token", "token", "error"])
        self.assertEqual(
            [item["content"] for item in ws.sent if item["type"] == "token"],
            ["first", "second"],
        )
        self.assertEqual(ws.sent[-1]["message"], server._STREAM_TIMEOUT_MESSAGE)
        self.assertNotIn("done", types)
        self.assertTrue(stream.cancel_called)
        self.assertEqual(server.sessions["web-test"].messages, [])
        self.assertEqual(server.idle_manager.ends, 1)

    async def test_partial_then_provider_error_keeps_tokens_then_error(self) -> None:
        stream = _FakeQueueStream(["first", RuntimeError("boom"), None])
        ws = await self._run_message(stream=stream)

        types = [item["type"] for item in ws.sent]
        self.assertEqual(types, ["token", "error"])
        self.assertEqual(ws.sent[0]["content"], "first")
        self.assertEqual(ws.sent[-1]["message"], server._STREAM_ERROR_MESSAGE)
        self.assertNotIn("done", types)
        self.assertTrue(stream.cancel_called)
        self.assertEqual(server.sessions["web-test"].messages, [])
        self.assertEqual(server.idle_manager.ends, 1)

    async def test_normal_completion_commits_and_sends_done(self) -> None:
        stream = _FakeQueueStream(["first", "second", None])
        ws = await self._run_message(stream=stream)

        types = [item["type"] for item in ws.sent]
        self.assertEqual(types, ["token", "token", "done"])
        self.assertEqual(ws.sent[-1]["full_text"], "firstsecond")
        self.assertTrue(stream.cancel_called)
        session = server.sessions["web-test"]
        self.assertEqual([m["role"] for m in session.messages], ["user", "assistant"])
        self.assertEqual(session.messages[-1]["content"], "firstsecond")
        self.assertEqual(server.idle_manager.ends, 1)
        history = Path(self.tmp.name) / "history"
        self.assertTrue(list(history.glob("session_*.json")))

    async def test_cancellation_cancels_stream_and_ends_inference(self) -> None:
        blocking = _BlockingQueue()
        stream = _CancellableQueueStream(blocking)
        ws = _FakeWebSocket([self._MESSAGE])

        with patch.object(server, "_start_assistant_stream", return_value=stream):
            task = asyncio.create_task(server.websocket_chat(ws))
            for _ in range(200):
                if blocking.entered.is_set():
                    break
                await asyncio.sleep(0.01)
            self.assertTrue(blocking.entered.is_set())
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        self.assertTrue(stream.cancel_called)
        self.assertEqual(server.idle_manager.ends, 1)
        self.assertNotIn("done", [item["type"] for item in ws.sent])
        self.assertNotIn("error", [item["type"] for item in ws.sent])
        # アシスタント応答はコミットされない (ユーザー発言は残る)
        self.assertEqual(
            [m["role"] for m in server.sessions["web-test"].messages], ["user"]
        )


class _FakeVision:
    def __init__(self) -> None:
        self.started = False

    def start(self) -> bool:
        self.started = True
        return True

    def get_status(self) -> dict:
        return {"emotion_detection": False}


class _FakeMonitor:
    def __init__(self) -> None:
        self.started = False

    def start(self) -> bool:
        self.started = True
        return True


class SensorInitPolicyTest(unittest.TestCase):
    """SensorPolicy 既定オフ時の vision/monitor 構築・start ゲート (オフライン)。"""

    def test_vision_disabled_does_not_construct_or_start(self) -> None:
        with mock.patch(
            "src.web.server.VisionContext",
            side_effect=AssertionError("must not construct vision"),
        ):
            self.assertIsNone(server._init_vision_from_policy(SensorPolicy(camera=False)))

    def test_vision_policy_none_fails_closed(self) -> None:
        with mock.patch(
            "src.web.server.VisionContext",
            side_effect=AssertionError("must not construct vision"),
        ):
            self.assertIsNone(server._init_vision_from_policy(None))

    def test_vision_enabled_constructs_and_starts_fake(self) -> None:
        fake = _FakeVision()
        with mock.patch("src.web.server.VisionContext", return_value=fake), (
            mock.patch.object(server.time, "sleep")
        ):
            vision = server._init_vision_from_policy(SensorPolicy(camera=True))
        self.assertIs(vision, fake)
        self.assertTrue(fake.started)

    def test_vision_start_failure_returns_none(self) -> None:
        class _Unstartable:
            def start(self) -> bool:
                return False

        with mock.patch("src.web.server.VisionContext", return_value=_Unstartable()), (
            mock.patch.object(server.time, "sleep")
        ):
            self.assertIsNone(server._init_vision_from_policy(SensorPolicy(camera=True)))

    def test_monitor_disabled_does_not_construct_or_start(self) -> None:
        with mock.patch(
            "src.web.server.MonitorContext",
            side_effect=AssertionError("must not construct monitor"),
        ):
            self.assertIsNone(server._init_monitor_from_policy(SensorPolicy(monitor=False)))

    def test_monitor_policy_none_fails_closed(self) -> None:
        with mock.patch(
            "src.web.server.MonitorContext",
            side_effect=AssertionError("must not construct monitor"),
        ):
            self.assertIsNone(server._init_monitor_from_policy(None))

    def test_monitor_enabled_constructs_and_starts_fake(self) -> None:
        fake = _FakeMonitor()
        with mock.patch("src.web.server.MonitorContext", return_value=fake):
            monitor = server._init_monitor_from_policy(SensorPolicy(monitor=True))
        self.assertIs(monitor, fake)
        self.assertTrue(fake.started)

    def test_monitor_start_failure_returns_none(self) -> None:
        class _Unstartable:
            def start(self) -> bool:
                return False

        with mock.patch("src.web.server.MonitorContext", return_value=_Unstartable()):
            self.assertIsNone(server._init_monitor_from_policy(SensorPolicy(monitor=True)))


if __name__ == "__main__":
    unittest.main()
