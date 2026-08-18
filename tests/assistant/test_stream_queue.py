import queue
import threading
import time
import unittest
from unittest.mock import patch

from src.assistant import (
    AssistantError,
    AssistantRequest,
    AssistantService,
    QueueStream,
    stream_to_queue,
)
from src.llm.providers.fake import FakeProvider
from src.llm.registry import ProviderRegistry
from src.llm.routing.static import StaticRouter


class CloseRecordingStream:
    def __init__(self) -> None:
        self.closed = False
        self.second_token_requested = threading.Event()
        self._count = 0

    def __iter__(self):
        return self

    def __next__(self) -> str:
        self._count += 1
        if self._count >= 2:
            self.second_token_requested.set()
        return str(self._count)

    def close(self) -> None:
        self.closed = True


class BlockingAfterFirstProvider(FakeProvider):
    def __init__(self) -> None:
        super().__init__()
        self.blocked = threading.Event()
        self.release = threading.Event()
        self.third_token_requested = threading.Event()

    def generate_stream(self, messages, **kwargs):
        yield "first"
        self.blocked.set()
        self.release.wait()
        yield "second"
        self.third_token_requested.set()
        yield "third"


class QueueStreamTest(unittest.TestCase):
    @staticmethod
    def consume(stream: QueueStream) -> list[object]:
        items: list[object] = []
        while True:
            item = stream.queue.get(timeout=1.0)
            items.append(item)
            if item is None:
                return items

    def test_tokens_keep_order_and_end_with_one_sentinel(self) -> None:
        stream = stream_to_queue(("a", "b", "c"))

        self.assertEqual(self.consume(stream), ["a", "b", "c", None])
        self.assertTrue(stream.join(timeout=1.0))
        with self.assertRaises(queue.Empty):
            stream.queue.get_nowait()

    def test_exception_object_precedes_sentinel(self) -> None:
        failure = RuntimeError("planned failure")

        def failing_stream():
            yield "first"
            raise failure

        stream = stream_to_queue(failing_stream())
        items = self.consume(stream)

        self.assertEqual(items[0], "first")
        self.assertIs(items[1], failure)
        self.assertIsNone(items[2])
        self.assertTrue(stream.join(timeout=1.0))

    def test_full_queue_applies_backpressure_without_losing_tokens(self) -> None:
        third_token_requested = threading.Event()

        def source():
            yield "a"
            yield "b"
            third_token_requested.set()
            yield "c"

        stream = stream_to_queue(source(), maxsize=2)
        self.assertTrue(third_token_requested.wait(timeout=1.0))
        self.assertEqual(stream.queue.qsize(), 2)
        self.assertTrue(stream.is_running)

        self.assertEqual(stream.queue.get(timeout=1.0), "a")
        self.assertEqual(self.consume(stream), ["b", "c", None])
        self.assertTrue(stream.join(timeout=1.0))

    def test_cancel_ends_worker_while_queue_is_full(self) -> None:
        source = CloseRecordingStream()
        stream = stream_to_queue(source, maxsize=1)
        self.assertTrue(source.second_token_requested.wait(timeout=1.0))
        self.assertEqual(stream.queue.qsize(), 1)

        stream.cancel()

        self.assertTrue(stream.join(timeout=1.0))
        self.assertFalse(stream.is_running)
        self.assertEqual(self.consume(stream), [None])

    def test_cancel_closes_source(self) -> None:
        source = CloseRecordingStream()
        stream = stream_to_queue(source, maxsize=1)
        self.assertTrue(source.second_token_requested.wait(timeout=1.0))

        stream.cancel()

        self.assertTrue(stream.join(timeout=1.0))
        self.assertTrue(source.closed)

    def test_cancel_is_idempotent(self) -> None:
        source = CloseRecordingStream()
        stream = stream_to_queue(source, maxsize=1)
        self.assertTrue(source.second_token_requested.wait(timeout=1.0))

        stream.cancel()
        stream.cancel()

        self.assertTrue(stream.join(timeout=1.0))
        self.assertTrue(source.closed)

    def test_assistant_stream_result_exposes_response_after_consumption(self) -> None:
        provider = FakeProvider(
            stream_chunks=("hel", "lo"),
            model="stream-model",
            stats={"eval_count": 2},
        )
        registry = ProviderRegistry()
        registry.register("local", provider, local=True)
        router = StaticRouter(registry, default_provider_id="local")
        service = AssistantService(registry, router)
        request = AssistantRequest(
            text="hello", conversation_id="conversation", channel="web"
        )
        result = service.generate_stream(
            request, [{"role": "user", "content": "hello"}]
        )

        stream = stream_to_queue(result)

        self.assertIs(stream.source, result)
        self.assertEqual(self.consume(stream), ["hel", "lo", None])
        self.assertTrue(stream.join(timeout=1.0))
        self.assertEqual(stream.source.response.text, "hello")
        self.assertEqual(stream.source.response.route.model, "stream-model")

    def test_cancel_closes_stream_result_and_stops_further_source_iteration(self) -> None:
        provider = BlockingAfterFirstProvider()
        registry = ProviderRegistry()
        registry.register("local", provider, local=True)
        service = AssistantService(
            registry, StaticRouter(registry, default_provider_id="local")
        )
        result = service.generate_stream(
            AssistantRequest(text="hello", conversation_id="c", channel="web"),
            [{"role": "user", "content": "hello"}],
        )
        stream = stream_to_queue(result)
        self.assertTrue(provider.blocked.wait(timeout=1.0))

        with patch.object(result, "close", wraps=result.close) as close:
            stream.cancel()
            close.assert_called_once_with()
        provider.release.set()

        self.assertTrue(stream.join(timeout=1.0))
        self.assertFalse(provider.third_token_requested.is_set())
        self.assertEqual(self.consume(stream), ["first", None])
        with self.assertRaises(AssistantError):
            _ = result.response

    def test_cancel_returns_quickly_while_source_read_is_blocked(self) -> None:
        blocked = threading.Event()
        release = threading.Event()

        def source():
            blocked.set()
            release.wait()
            yield "late"

        stream = stream_to_queue(source())
        self.assertTrue(blocked.wait(timeout=1.0))

        started_at = time.monotonic()
        stream.cancel()
        elapsed = time.monotonic() - started_at

        self.assertLess(elapsed, 0.5)
        self.assertFalse(stream.join(timeout=0.05))
        release.set()
        self.assertTrue(stream.join(timeout=1.0))

    def test_stream_result_close_is_safe_before_during_and_repeatedly(self) -> None:
        provider = FakeProvider(stream_chunks=("first", "second"))
        registry = ProviderRegistry()
        registry.register("local", provider, local=True)
        service = AssistantService(
            registry, StaticRouter(registry, default_provider_id="local")
        )
        request = AssistantRequest(text="hello", conversation_id="c", channel="web")
        messages = [{"role": "user", "content": "hello"}]

        unopened = service.generate_stream(request, messages)
        unopened.close()
        unopened.close()
        with self.assertRaises(AssistantError):
            iter(unopened)
        with self.assertRaises(AssistantError):
            _ = unopened.response

        active = service.generate_stream(request, messages)
        iterator = iter(active)
        self.assertEqual(next(iterator), "first")
        active.close()
        active.close()
        with self.assertRaises(StopIteration):
            next(iterator)
        with self.assertRaises(AssistantError):
            _ = active.response


if __name__ == "__main__":
    unittest.main()
