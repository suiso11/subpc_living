import json
import unittest

import httpx

from src.llm.errors import ProviderRequestError, ProviderTimeoutError
from src.llm.providers.cloud_http import OpenAICompatibleProvider


class _MockTransport(httpx.BaseTransport):
    """Simple mock transport that returns pre-configured responses."""

    def __init__(
        self,
        status_code: int = 200,
        json_data: dict | None = None,
        text: str = "",
        raise_exc: Exception | None = None,
    ) -> None:
        self.status_code = status_code
        self.json_data = json_data
        self.text = text
        self.raise_exc = raise_exc
        self.requests: list[httpx.Request] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.raise_exc is not None:
            raise self.raise_exc
        if self.json_data is not None:
            return httpx.Response(
                self.status_code,
                json=self.json_data,
                request=request,
            )
        return httpx.Response(
            self.status_code,
            text=self.text,
            request=request,
        )


class OpenAICompatibleProviderGenerateTest(unittest.TestCase):
    def test_happy_path(self):
        transport = _MockTransport(
            json_data={
                "choices": [{"message": {"content": "hello world"}}],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 3,
                    "total_tokens": 8,
                },
            }
        )
        client = httpx.Client(transport=transport, base_url="https://api.example.com")
        p = OpenAICompatibleProvider(
            model="gpt-4", api_key="test-key", client=client
        )
        result = p.generate(
            [{"role": "user", "content": "hi"}],
            temperature=0.5,
            top_p=0.8,
            num_predict=100,
        )
        self.assertEqual(result, "hello world")
        self.assertEqual(len(transport.requests), 1)
        req = transport.requests[0]
        self.assertEqual(str(req.url), "https://api.example.com/chat/completions")
        self.assertEqual(req.headers["authorization"], "Bearer test-key")
        body = json.loads(req.content)
        self.assertEqual(body["model"], "gpt-4")
        self.assertEqual(body["messages"], [{"role": "user", "content": "hi"}])
        self.assertEqual(body["temperature"], 0.5)
        self.assertEqual(body["top_p"], 0.8)
        self.assertEqual(body["max_tokens"], 100)

    def test_max_tokens_omitted_when_none(self):
        transport = _MockTransport(
            json_data={"choices": [{"message": {"content": "ok"}}]}
        )
        client = httpx.Client(transport=transport, base_url="https://api.example.com")
        p = OpenAICompatibleProvider(
            model="m", api_key="k", client=client
        )
        p.generate([{"role": "user", "content": "hi"}])
        body = json.loads(transport.requests[0].content)
        self.assertNotIn("max_tokens", body)

    def test_max_tokens_omitted_when_zero(self):
        transport = _MockTransport(
            json_data={"choices": [{"message": {"content": "ok"}}]}
        )
        client = httpx.Client(transport=transport, base_url="https://api.example.com")
        p = OpenAICompatibleProvider(
            model="m", api_key="k", client=client
        )
        p.generate([{"role": "user", "content": "hi"}], num_predict=0)
        body = json.loads(transport.requests[0].content)
        self.assertNotIn("max_tokens", body)

    def test_usage_stats_populated(self):
        transport = _MockTransport(
            json_data={
                "choices": [{"message": {"content": "x"}}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                },
            }
        )
        client = httpx.Client(transport=transport, base_url="https://api.example.com")
        p = OpenAICompatibleProvider(
            model="m", api_key="k", client=client
        )
        p.generate([{"role": "user", "content": "hi"}])
        self.assertEqual(
            p.last_stats,
            {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        )

    def test_usage_stats_empty_when_no_usage(self):
        transport = _MockTransport(
            json_data={"choices": [{"message": {"content": "x"}}]}
        )
        client = httpx.Client(transport=transport, base_url="https://api.example.com")
        p = OpenAICompatibleProvider(
            model="m", api_key="k", client=client
        )
        p.generate([{"role": "user", "content": "hi"}])
        self.assertEqual(p.last_stats, {})

    def test_non_2xx_raises_provider_request_error(self):
        transport = _MockTransport(status_code=500, text="Internal Server Error")
        client = httpx.Client(transport=transport, base_url="https://api.example.com")
        p = OpenAICompatibleProvider(
            model="m", api_key="super-secret-key-12345", client=client
        )
        with self.assertRaises(ProviderRequestError) as ctx:
            p.generate([{"role": "user", "content": "hi"}])
        self.assertIn("500", str(ctx.exception))
        self.assertNotIn("super-secret-key-12345", str(ctx.exception))

    def test_invalid_json_raises_provider_request_error(self):
        transport = _MockTransport(status_code=200, text="not json at all")
        client = httpx.Client(transport=transport, base_url="https://api.example.com")
        p = OpenAICompatibleProvider(
            model="m", api_key="secretkey99", client=client
        )
        with self.assertRaises(ProviderRequestError):
            p.generate([{"role": "user", "content": "hi"}])

    def test_malformed_choices_raises_provider_request_error(self):
        transport = _MockTransport(
            json_data={"choices": []}
        )
        client = httpx.Client(transport=transport, base_url="https://api.example.com")
        p = OpenAICompatibleProvider(
            model="m", api_key="k", client=client
        )
        with self.assertRaises(ProviderRequestError):
            p.generate([{"role": "user", "content": "hi"}])

    def test_timeout_maps_to_provider_timeout_error(self):
        transport = _MockTransport(raise_exc=httpx.ConnectTimeout("connection timed out"))
        client = httpx.Client(transport=transport, base_url="https://api.example.com")
        p = OpenAICompatibleProvider(
            model="m", api_key="mysecretkey", client=client
        )
        with self.assertRaises(ProviderTimeoutError) as ctx:
            p.generate([{"role": "user", "content": "hi"}])
        self.assertIn("generate", ctx.exception.operation)
        self.assertNotIn("mysecretkey", str(ctx.exception))

    def test_http_error_maps_to_provider_request_error(self):
        transport = _MockTransport(raise_exc=httpx.ConnectError("refused"))
        client = httpx.Client(transport=transport, base_url="https://api.example.com")
        p = OpenAICompatibleProvider(
            model="m", api_key="mysecretkey", client=client
        )
        with self.assertRaises(ProviderRequestError) as ctx:
            p.generate([{"role": "user", "content": "hi"}])
        self.assertIn("generate", ctx.exception.operation)
        self.assertNotIn("mysecretkey", str(ctx.exception))


class OpenAICompatibleProviderStreamTest(unittest.TestCase):
    def test_happy_path(self):
        sse = (
            'data: {"choices":[{"delta":{"content":"Hello"}}]}\n'
            'data: {"choices":[{"delta":{"content":" world"}}]}\n'
            "data: [DONE]\n"
        )
        transport = _MockTransport(text=sse)
        client = httpx.Client(transport=transport, base_url="https://api.example.com")
        p = OpenAICompatibleProvider(
            model="m", api_key="k", client=client
        )
        chunks = list(p.generate_stream([{"role": "user", "content": "hi"}]))
        self.assertEqual(chunks, ["Hello", " world"])

    def test_skips_chunks_without_content(self):
        sse = (
            'data: {"choices":[{"delta":{"role":"assistant"}}]}\n'
            'data: {"choices":[{"delta":{"content":"hi"}}]}\n'
            "data: [DONE]\n"
        )
        transport = _MockTransport(text=sse)
        client = httpx.Client(transport=transport, base_url="https://api.example.com")
        p = OpenAICompatibleProvider(
            model="m", api_key="k", client=client
        )
        chunks = list(p.generate_stream([{"role": "user", "content": "hi"}]))
        self.assertEqual(chunks, ["hi"])

    def test_non_2xx_raises_provider_request_error(self):
        transport = _MockTransport(status_code=401, text="Unauthorized")
        client = httpx.Client(transport=transport, base_url="https://api.example.com")
        p = OpenAICompatibleProvider(
            model="m", api_key="secret123", client=client
        )
        with self.assertRaises(ProviderRequestError) as ctx:
            list(p.generate_stream([{"role": "user", "content": "hi"}]))
        self.assertIn("401", str(ctx.exception))
        self.assertNotIn("secret123", str(ctx.exception))

    def test_timeout_maps_to_provider_timeout_error(self):
        transport = _MockTransport(raise_exc=httpx.ConnectTimeout("timed out"))
        client = httpx.Client(transport=transport, base_url="https://api.example.com")
        p = OpenAICompatibleProvider(
            model="m", api_key="k", client=client
        )
        with self.assertRaises(ProviderTimeoutError):
            list(p.generate_stream([{"role": "user", "content": "hi"}]))

    def test_stream_headers_and_body(self):
        sse = 'data: {"choices":[{"delta":{"content":"ok"}}]}\ndata: [DONE]\n'
        transport = _MockTransport(text=sse)
        client = httpx.Client(transport=transport, base_url="https://api.example.com")
        p = OpenAICompatibleProvider(
            model="m", api_key="tok", client=client
        )
        list(p.generate_stream(
            [{"role": "user", "content": "q"}],
            temperature=0.3,
            top_p=0.5,
            num_predict=50,
        ))
        req = transport.requests[0]
        self.assertEqual(req.headers["authorization"], "Bearer tok")
        body = json.loads(req.content)
        self.assertTrue(body["stream"])
        self.assertEqual(body["temperature"], 0.3)
        self.assertEqual(body["max_tokens"], 50)


class OpenAICompatibleProviderLifecycleTest(unittest.TestCase):
    def test_is_available_before_close(self):
        p = OpenAICompatibleProvider(model="m", api_key="k")
        self.assertTrue(p.is_available())

    def test_is_available_after_close(self):
        p = OpenAICompatibleProvider(model="m", api_key="k")
        p.close()
        self.assertFalse(p.is_available())

    def test_last_stats_returns_copy(self):
        transport = _MockTransport(
            json_data={
                "choices": [{"message": {"content": "x"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            }
        )
        client = httpx.Client(transport=transport, base_url="https://api.example.com")
        p = OpenAICompatibleProvider(model="m", api_key="k", client=client)
        p.generate([{"role": "user", "content": "hi"}])
        stats1 = p.last_stats
        stats2 = p.last_stats
        self.assertEqual(stats1, stats2)
        self.assertIsNot(stats1, stats2)

    def test_exception_messages_never_contain_api_key(self):
        api_key = "super-secret-api-key-xyz"
        # Test various error paths
        for exc, op in [
            (httpx.ConnectTimeout("timeout"), "generate"),
            (httpx.ConnectError("refused"), "generate"),
            (httpx.ConnectTimeout("timeout"), "generate_stream"),
            (httpx.ConnectError("refused"), "generate_stream"),
        ]:
            transport = _MockTransport(raise_exc=exc)
            client = httpx.Client(transport=transport, base_url="https://api.example.com")
            p = OpenAICompatibleProvider(model="m", api_key=api_key, client=client)
            try:
                if op == "generate":
                    p.generate([{"role": "user", "content": "hi"}])
                else:
                    list(p.generate_stream([{"role": "user", "content": "hi"}]))
            except Exception as e:
                self.assertNotIn(api_key, str(e))

    def test_non_2xx_error_never_contains_api_key(self):
        api_key = "secret-api-key-42"
        transport = _MockTransport(status_code=403, text="Forbidden")
        client = httpx.Client(transport=transport, base_url="https://api.example.com")
        p = OpenAICompatibleProvider(model="m", api_key=api_key, client=client)
        with self.assertRaises(ProviderRequestError) as ctx:
            p.generate([{"role": "user", "content": "hi"}])
        self.assertNotIn(api_key, str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
