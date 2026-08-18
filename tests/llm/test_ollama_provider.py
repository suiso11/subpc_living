from __future__ import annotations

import unittest
from typing import Any

import httpx

from src.chat.client import OllamaResponseError
from src.llm.contracts import GenerationOptions
from src.llm.errors import ProviderRequestError, ProviderTimeoutError
from src.llm.provider import LLMProvider
from src.llm.providers.ollama import OllamaProvider


class StubOllamaClient:
    def __init__(self) -> None:
        self.model = "stub-model"
        self.available = True
        self.calls: list[dict[str, Any]] = []
        self._last_stats = {"eval_count": 3}
        self.closed = False
        self.generate_error: Exception | None = None
        self.stream_error: Exception | None = None

    def is_available(self) -> bool:
        return self.available

    def list_models(self) -> list[str]:
        return [self.model]

    def has_model(self, model: str | None = None) -> bool:
        return (model or self.model) == self.model

    def generate(self, messages: list[dict], **kwargs: Any) -> str:
        self.calls.append({"kind": "generate", "messages": messages, "kwargs": kwargs})
        if self.generate_error is not None:
            raise self.generate_error
        return "generated"

    def generate_stream(self, messages: list[dict], **kwargs: Any):
        self.calls.append(
            {"kind": "generate_stream", "messages": messages, "kwargs": kwargs}
        )
        yield "first"
        if self.stream_error is not None:
            raise self.stream_error
        yield "second"

    @property
    def last_stats(self) -> dict[str, Any]:
        return self._last_stats

    def close(self) -> None:
        self.closed = True


class GenerationOptionsTest(unittest.TestCase):
    def test_generate_and_stream_kwargs_match_existing_signatures(self) -> None:
        options = GenerationOptions(temperature=0.2, num_predict=64, timeout=4.5)

        self.assertEqual(options.as_generate_kwargs()["timeout"], 4.5)
        self.assertEqual(options.as_generate_kwargs()["num_predict"], 64)
        self.assertNotIn("timeout", options.as_stream_kwargs())
        self.assertEqual(options.as_stream_kwargs()["temperature"], 0.2)


class OllamaProviderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = StubOllamaClient()
        self.provider = OllamaProvider(
            provider_id="local-strong", client=self.client  # type: ignore[arg-type]
        )

    def test_adapter_satisfies_provider_and_delegates_options(self) -> None:
        result = self.provider.generate(
            [{"role": "user", "content": "hello"}],
            temperature=0.3,
            top_p=0.8,
            top_k=12,
            repeat_penalty=1.2,
            num_ctx=4096,
            num_predict=32,
            timeout=3.0,
        )

        self.assertIsInstance(self.provider, LLMProvider)
        self.assertEqual(result, "generated")
        self.assertEqual(self.client.calls[0]["kwargs"]["temperature"], 0.3)
        self.assertEqual(self.client.calls[0]["kwargs"]["timeout"], 3.0)
        self.assertEqual(self.provider.last_stats, {"eval_count": 3})

    def test_model_catalog_and_close_are_delegated(self) -> None:
        self.assertTrue(self.provider.is_available())
        self.assertEqual(self.provider.list_models(), ["stub-model"])
        self.assertTrue(self.provider.has_model())

        self.provider.model = "changed"
        self.assertEqual(self.client.model, "changed")
        self.provider.close()
        self.assertTrue(self.client.closed)

    def test_generate_timeout_is_normalized_with_cause(self) -> None:
        self.client.generate_error = httpx.TimeoutException("slow")

        with self.assertRaises(ProviderTimeoutError) as raised:
            self.provider.generate([{"role": "user", "content": "hello"}])

        self.assertEqual(raised.exception.provider_id, "local-strong")
        self.assertEqual(raised.exception.operation, "generate")
        self.assertIsInstance(raised.exception.__cause__, httpx.TimeoutException)

    def test_generate_request_error_is_normalized(self) -> None:
        self.client.generate_error = httpx.ConnectError("offline")

        with self.assertRaises(ProviderRequestError) as raised:
            self.provider.generate([{"role": "user", "content": "hello"}])

        self.assertEqual(raised.exception.operation, "generate")
        self.assertIsInstance(raised.exception.__cause__, httpx.ConnectError)

    def test_generate_response_error_is_normalized_with_cause(self) -> None:
        self.client.generate_error = OllamaResponseError("bad payload")

        with self.assertRaises(ProviderRequestError) as raised:
            self.provider.generate([{"role": "user", "content": "hello"}])

        self.assertEqual(raised.exception.provider_id, "local-strong")
        self.assertEqual(raised.exception.operation, "generate")
        self.assertIsInstance(raised.exception.__cause__, OllamaResponseError)

    def test_stream_error_is_normalized_during_iteration(self) -> None:
        self.client.stream_error = httpx.ConnectError("stream stopped")
        stream = self.provider.generate_stream(
            [{"role": "user", "content": "hello"}]
        )

        self.assertEqual(next(stream), "first")
        with self.assertRaises(ProviderRequestError) as raised:
            next(stream)

        self.assertEqual(raised.exception.operation, "generate_stream")
        self.assertIsInstance(raised.exception.__cause__, httpx.ConnectError)

    def test_stream_response_error_is_normalized_during_iteration(self) -> None:
        self.client.stream_error = OllamaResponseError("bad chunk")
        stream = self.provider.generate_stream(
            [{"role": "user", "content": "hello"}]
        )

        self.assertEqual(next(stream), "first")
        with self.assertRaises(ProviderRequestError) as raised:
            next(stream)

        self.assertEqual(raised.exception.provider_id, "local-strong")
        self.assertEqual(raised.exception.operation, "generate_stream")
        self.assertIsInstance(raised.exception.__cause__, OllamaResponseError)

    def test_arbitrary_programming_error_is_not_reclassified(self) -> None:
        self.client.generate_error = ValueError("bad test setup")

        with self.assertRaisesRegex(ValueError, "bad test setup"):
            self.provider.generate([{"role": "user", "content": "hello"}])

    def test_generate_arbitrary_type_error_is_not_reclassified(self) -> None:
        self.client.generate_error = TypeError("bad test setup")

        with self.assertRaises(TypeError):
            self.provider.generate([{"role": "user", "content": "hello"}])

    def test_stream_arbitrary_type_error_is_not_reclassified(self) -> None:
        self.client.stream_error = TypeError("bad test setup")
        stream = self.provider.generate_stream(
            [{"role": "user", "content": "hello"}]
        )

        self.assertEqual(next(stream), "first")
        with self.assertRaises(TypeError):
            next(stream)


if __name__ == "__main__":
    unittest.main()
