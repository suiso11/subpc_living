import json
import unittest

import httpx

from src.llm.errors import ProviderRequestError, ProviderTimeoutError
from src.llm.provider import LLMProvider
from src.llm.providers.local_openai import LocalOpenAICompatibleProvider


class _MockTransport(httpx.BaseTransport):
    """Mock transport that returns pre-configured chat and /models responses."""

    def __init__(
        self,
        *,
        status_code: int = 200,
        json_data: dict | None = None,
        text: str = "",
        raise_exc: Exception | None = None,
        models_status: int = 200,
        models_json: dict | None = None,
        models_raise_exc: Exception | None = None,
    ) -> None:
        self.status_code = status_code
        self.json_data = json_data
        self.text = text
        self.raise_exc = raise_exc
        self.models_status = models_status
        self.models_json = models_json
        self.models_raise_exc = models_raise_exc
        self.requests: list[httpx.Request] = []

    def _base_response(self, request: httpx.Request) -> httpx.Response:
        if self.json_data is not None:
            return httpx.Response(self.status_code, json=self.json_data, request=request)
        return httpx.Response(self.status_code, text=self.text, request=request)

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path.endswith("/models"):
            if self.models_raise_exc is not None:
                raise self.models_raise_exc
            if self.models_json is not None:
                return httpx.Response(
                    self.models_status, json=self.models_json, request=request
                )
            return httpx.Response(self.models_status, text="", request=request)
        if self.raise_exc is not None:
            raise self.raise_exc
        return self._base_response(request)


MODELS = {"data": [{"id": "local-model"}, {"id": "local-model-2"}]}


def _client(transport: _MockTransport) -> httpx.Client:
    return httpx.Client(transport=transport, base_url="http://localhost:8080/v1")


def _client_at(transport: _MockTransport, base_url: str) -> httpx.Client:
    return httpx.Client(transport=transport, base_url=base_url)


class LocalOpenAICompatibleProviderHeadersTest(unittest.TestCase):
    def test_no_authorization_header_without_key(self):
        transport = _MockTransport(
            json_data={"choices": [{"message": {"content": "hi"}}]}
        )
        p = LocalOpenAICompatibleProvider(model="m", client=_client(transport))
        p.generate([{"role": "user", "content": "hello"}])
        req = transport.requests[0]
        self.assertNotIn("authorization", req.headers)

    def test_no_authorization_header_with_blank_key(self):
        transport = _MockTransport(
            json_data={"choices": [{"message": {"content": "hi"}}]}
        )
        p = LocalOpenAICompatibleProvider(
            model="m", api_key="   ", client=_client(transport)
        )
        p.generate([{"role": "user", "content": "hello"}])
        req = transport.requests[0]
        self.assertNotIn("authorization", req.headers)

    def test_bearer_header_with_configured_key(self):
        transport = _MockTransport(
            json_data={"choices": [{"message": {"content": "hi"}}]}
        )
        p = LocalOpenAICompatibleProvider(
            model="m", api_key="secret-key", client=_client(transport)
        )
        p.generate([{"role": "user", "content": "hello"}])
        req = transport.requests[0]
        self.assertEqual(req.headers["authorization"], "Bearer secret-key")

    def test_stream_headers_omit_authorization_without_key(self):
        sse = 'data: {"choices":[{"delta":{"content":"ok"}}]}\ndata: [DONE]\n'
        transport = _MockTransport(text=sse)
        p = LocalOpenAICompatibleProvider(model="m", client=_client(transport))
        list(p.generate_stream([{"role": "user", "content": "hello"}]))
        req = transport.requests[0]
        self.assertNotIn("authorization", req.headers)


class LocalOpenAICompatibleProviderGenerateTest(unittest.TestCase):
    def test_happy_path_payload_and_response(self):
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
        p = LocalOpenAICompatibleProvider(model="local-model", client=_client(transport))
        result = p.generate(
            [{"role": "user", "content": "hi"}],
            temperature=0.5,
            top_p=0.8,
            num_predict=100,
        )
        self.assertEqual(result, "hello world")
        self.assertEqual(len(transport.requests), 1)
        req = transport.requests[0]
        self.assertEqual(str(req.url), "http://localhost:8080/v1/chat/completions")
        body = json.loads(req.content)
        self.assertEqual(body["model"], "local-model")
        self.assertEqual(body["messages"], [{"role": "user", "content": "hi"}])
        self.assertEqual(body["temperature"], 0.5)
        self.assertEqual(body["top_p"], 0.8)
        self.assertEqual(body["max_tokens"], 100)

    def test_max_tokens_omitted_when_none(self):
        transport = _MockTransport(
            json_data={"choices": [{"message": {"content": "ok"}}]}
        )
        p = LocalOpenAICompatibleProvider(model="m", client=_client(transport))
        p.generate([{"role": "user", "content": "hi"}])
        body = json.loads(transport.requests[0].content)
        self.assertNotIn("max_tokens", body)

    def test_max_tokens_omitted_when_zero(self):
        transport = _MockTransport(
            json_data={"choices": [{"message": {"content": "ok"}}]}
        )
        p = LocalOpenAICompatibleProvider(model="m", client=_client(transport))
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
        p = LocalOpenAICompatibleProvider(model="m", client=_client(transport))
        p.generate([{"role": "user", "content": "hi"}])
        self.assertEqual(
            p.last_stats,
            {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        )

    def test_usage_stats_empty_when_no_usage(self):
        transport = _MockTransport(
            json_data={"choices": [{"message": {"content": "x"}}]}
        )
        p = LocalOpenAICompatibleProvider(model="m", client=_client(transport))
        p.generate([{"role": "user", "content": "hi"}])
        self.assertEqual(p.last_stats, {})

    def test_non_2xx_raises_provider_request_error(self):
        transport = _MockTransport(status_code=500, text="Internal Server Error")
        p = LocalOpenAICompatibleProvider(
            model="m", api_key="super-secret-key-12345", client=_client(transport)
        )
        with self.assertRaises(ProviderRequestError) as ctx:
            p.generate([{"role": "user", "content": "hi"}])
        self.assertIn("500", str(ctx.exception))
        self.assertNotIn("super-secret-key-12345", str(ctx.exception))

    def test_invalid_json_raises_provider_request_error(self):
        transport = _MockTransport(status_code=200, text="not json at all")
        p = LocalOpenAICompatibleProvider(
            model="m", api_key="secretkey99", client=_client(transport)
        )
        with self.assertRaises(ProviderRequestError):
            p.generate([{"role": "user", "content": "hi"}])

    def test_malformed_choices_raises_provider_request_error(self):
        transport = _MockTransport(json_data={"choices": []})
        p = LocalOpenAICompatibleProvider(model="m", client=_client(transport))
        with self.assertRaises(ProviderRequestError):
            p.generate([{"role": "user", "content": "hi"}])

    def test_missing_choices_raises_provider_request_error(self):
        transport = _MockTransport(json_data={"unexpected": True})
        p = LocalOpenAICompatibleProvider(model="m", client=_client(transport))
        with self.assertRaises(ProviderRequestError):
            p.generate([{"role": "user", "content": "hi"}])

    def test_non_string_content_raises_provider_request_error(self):
        transport = _MockTransport(
            json_data={"choices": [{"message": {"content": {"text": "hi"}}}]}
        )
        p = LocalOpenAICompatibleProvider(model="m", client=_client(transport))
        with self.assertRaises(ProviderRequestError) as ctx:
            p.generate([{"role": "user", "content": "hi"}])
        self.assertIn("generate", ctx.exception.operation)

    def test_timeout_maps_to_provider_timeout_error(self):
        transport = _MockTransport(
            raise_exc=httpx.ConnectTimeout("connection timed out")
        )
        p = LocalOpenAICompatibleProvider(
            model="m", api_key="mysecretkey", client=_client(transport)
        )
        with self.assertRaises(ProviderTimeoutError) as ctx:
            p.generate([{"role": "user", "content": "hi"}])
        self.assertIn("generate", ctx.exception.operation)
        self.assertNotIn("mysecretkey", str(ctx.exception))

    def test_http_error_maps_to_provider_request_error(self):
        transport = _MockTransport(raise_exc=httpx.ConnectError("refused"))
        p = LocalOpenAICompatibleProvider(
            model="m", api_key="mysecretkey", client=_client(transport)
        )
        with self.assertRaises(ProviderRequestError) as ctx:
            p.generate([{"role": "user", "content": "hi"}])
        self.assertIn("generate", ctx.exception.operation)
        self.assertNotIn("mysecretkey", str(ctx.exception))


class LocalOpenAICompatibleProviderStreamTest(unittest.TestCase):
    def test_happy_path(self):
        sse = (
            'data: {"choices":[{"delta":{"content":"Hello"}}]}\n'
            'data: {"choices":[{"delta":{"content":" world"}}]}\n'
            "data: [DONE]\n"
        )
        transport = _MockTransport(text=sse)
        p = LocalOpenAICompatibleProvider(model="m", client=_client(transport))
        chunks = list(p.generate_stream([{"role": "user", "content": "hi"}]))
        self.assertEqual(chunks, ["Hello", " world"])

    def test_skips_chunks_without_content(self):
        sse = (
            'data: {"choices":[{"delta":{"role":"assistant"}}]}\n'
            'data: {"choices":[{"delta":{"content":"hi"}}]}\n'
            "data: [DONE]\n"
        )
        transport = _MockTransport(text=sse)
        p = LocalOpenAICompatibleProvider(model="m", client=_client(transport))
        chunks = list(p.generate_stream([{"role": "user", "content": "hi"}]))
        self.assertEqual(chunks, ["hi"])

    def test_skips_non_data_lines_and_metadata_only_chunks(self):
        sse = (
            "event: ping\n"
            "id: 1\n"
            ': keepalive\n'
            'data: {"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}\n'
            'data: {"choices":[]}\n'
            'data: {"choices":[{"delta":{}}]}\n'
            'data: {"choices":[{"delta":{"role":"assistant"}}]}\n'
            'data: {"choices":[{"delta":{"content":"a"}}]}\n'
            "data: [DONE]\n"
        )
        transport = _MockTransport(text=sse)
        p = LocalOpenAICompatibleProvider(model="m", client=_client(transport))
        chunks = list(p.generate_stream([{"role": "user", "content": "hi"}]))
        self.assertEqual(chunks, ["a"])

    def test_malformed_json_chunk_raises(self):
        sse = (
            'data: {"choices":[{"delta":{"content":"a"}}]}\n'
            "data: not json at all\n"
            "data: [DONE]\n"
        )
        transport = _MockTransport(text=sse)
        p = LocalOpenAICompatibleProvider(model="m", client=_client(transport))
        chunks = []
        gen = p.generate_stream([{"role": "user", "content": "hi"}])
        with self.assertRaises(ProviderRequestError) as ctx:
            for chunk in gen:
                chunks.append(chunk)
        self.assertEqual(chunks, ["a"])
        self.assertEqual(ctx.exception.operation, "generate_stream")

    def test_non_object_json_chunk_raises(self):
        sse = "data: [1, 2]\n" + "data: [DONE]\n"
        transport = _MockTransport(text=sse)
        p = LocalOpenAICompatibleProvider(model="m", client=_client(transport))
        with self.assertRaises(ProviderRequestError):
            list(p.generate_stream([{"role": "user", "content": "hi"}]))

    def test_error_payload_raises(self):
        sse = (
            'data: {"choices":[{"delta":{"content":"a"}}]}\n'
            'data: {"error":{"message":"server exploded","type":"internal_error"}}\n'
            "data: [DONE]\n"
        )
        transport = _MockTransport(text=sse)
        p = LocalOpenAICompatibleProvider(
            model="m", api_key="secret-key-xyz", client=_client(transport)
        )
        chunks = []
        gen = p.generate_stream([{"role": "user", "content": "hi"}])
        with self.assertRaises(ProviderRequestError) as ctx:
            for chunk in gen:
                chunks.append(chunk)
        self.assertEqual(chunks, ["a"])
        self.assertNotIn("secret-key-xyz", str(ctx.exception))

    def test_non_string_content_raises(self):
        for bad in ('{"text":"bad"}', "123", "true", '["x"]'):
            sse = (
                f'data: {{"choices":[{{"delta":{{"content":{bad}}}}}]}}\n'
                "data: [DONE]\n"
            )
            transport = _MockTransport(text=sse)
            p = LocalOpenAICompatibleProvider(model="m", client=_client(transport))
            with self.subTest(bad=bad):
                with self.assertRaises(ProviderRequestError) as ctx:
                    list(p.generate_stream([{"role": "user", "content": "hi"}]))
                self.assertEqual(ctx.exception.operation, "generate_stream")

    def test_null_content_is_ignorable(self):
        sse = (
            'data: {"choices":[{"delta":{"content":null}}]}\n'
            'data: {"choices":[{"delta":{"content":"ok"}}]}\n'
            "data: [DONE]\n"
        )
        transport = _MockTransport(text=sse)
        p = LocalOpenAICompatibleProvider(model="m", client=_client(transport))
        self.assertEqual(
            list(p.generate_stream([{"role": "user", "content": "hi"}])), ["ok"]
        )

    def test_invalid_delta_raises(self):
        sse = 'data: {"choices":[{"delta":"not-a-dict"}]}\n' + "data: [DONE]\n"
        transport = _MockTransport(text=sse)
        p = LocalOpenAICompatibleProvider(model="m", client=_client(transport))
        with self.assertRaises(ProviderRequestError):
            list(p.generate_stream([{"role": "user", "content": "hi"}]))

    def test_invalid_choice_raises(self):
        sse = 'data: {"choices":["not-a-dict"]}\n' + "data: [DONE]\n"
        transport = _MockTransport(text=sse)
        p = LocalOpenAICompatibleProvider(model="m", client=_client(transport))
        with self.assertRaises(ProviderRequestError):
            list(p.generate_stream([{"role": "user", "content": "hi"}]))

    def test_empty_stream_raises_without_done(self):
        transport = _MockTransport(text="")
        p = LocalOpenAICompatibleProvider(model="m", client=_client(transport))
        with self.assertRaises(ProviderRequestError) as ctx:
            list(p.generate_stream([{"role": "user", "content": "hi"}]))
        self.assertEqual(ctx.exception.operation, "generate_stream")

    def test_eof_without_done_after_content_raises(self):
        sse = 'data: {"choices":[{"delta":{"content":"partial"}}]}\n'
        transport = _MockTransport(text=sse)
        p = LocalOpenAICompatibleProvider(model="m", client=_client(transport))
        chunks = []
        gen = p.generate_stream([{"role": "user", "content": "hi"}])
        with self.assertRaises(ProviderRequestError):
            for chunk in gen:
                chunks.append(chunk)
        self.assertEqual(chunks, ["partial"])

    def test_done_without_any_content_fails(self):
        for sse in (
            "data: [DONE]\n",
            'data: {"usage":{"total_tokens":3}}\n' + "data: [DONE]\n",
            'data: {"choices":[{"delta":{"role":"assistant"}}]}\n' + "data: [DONE]\n",
        ):
            transport = _MockTransport(text=sse)
            p = LocalOpenAICompatibleProvider(model="m", client=_client(transport))
            with self.subTest(sse=sse):
                with self.assertRaises(ProviderRequestError) as ctx:
                    list(p.generate_stream([{"role": "user", "content": "hi"}]))
                self.assertEqual(ctx.exception.operation, "generate_stream")

    def test_stream_headers_and_body(self):
        sse = 'data: {"choices":[{"delta":{"content":"ok"}}]}\ndata: [DONE]\n'
        transport = _MockTransport(text=sse)
        p = LocalOpenAICompatibleProvider(
            model="m", api_key="tok", client=_client(transport)
        )
        list(
            p.generate_stream(
                [{"role": "user", "content": "q"}],
                temperature=0.3,
                top_p=0.5,
                num_predict=50,
            )
        )
        req = transport.requests[0]
        self.assertEqual(req.headers["authorization"], "Bearer tok")
        body = json.loads(req.content)
        self.assertTrue(body["stream"])
        self.assertEqual(body["temperature"], 0.3)
        self.assertEqual(body["max_tokens"], 50)

    def test_non_2xx_raises_provider_request_error(self):
        transport = _MockTransport(status_code=401, text="Unauthorized")
        p = LocalOpenAICompatibleProvider(
            model="m", api_key="secret123", client=_client(transport)
        )
        with self.assertRaises(ProviderRequestError) as ctx:
            list(p.generate_stream([{"role": "user", "content": "hi"}]))
        self.assertIn("401", str(ctx.exception))
        self.assertNotIn("secret123", str(ctx.exception))

    def test_timeout_maps_to_provider_timeout_error(self):
        transport = _MockTransport(raise_exc=httpx.ConnectTimeout("timed out"))
        p = LocalOpenAICompatibleProvider(model="m", client=_client(transport))
        with self.assertRaises(ProviderTimeoutError):
            list(p.generate_stream([{"role": "user", "content": "hi"}]))

    def test_http_error_maps_to_provider_request_error(self):
        transport = _MockTransport(raise_exc=httpx.ConnectError("refused"))
        p = LocalOpenAICompatibleProvider(model="m", client=_client(transport))
        with self.assertRaises(ProviderRequestError):
            list(p.generate_stream([{"role": "user", "content": "hi"}]))


class LocalOpenAICompatibleProviderModelsTest(unittest.TestCase):
    def test_list_models_from_data_ids(self):
        transport = _MockTransport(models_json=MODELS)
        p = LocalOpenAICompatibleProvider(model="local-model", client=_client(transport))
        self.assertEqual(p.list_models(), ["local-model", "local-model-2"])
        req = transport.requests[0]
        self.assertEqual(str(req.url), "http://localhost:8080/v1/models")

    def test_has_model_exact_match(self):
        transport = _MockTransport(models_json=MODELS)
        p = LocalOpenAICompatibleProvider(model="local-model", client=_client(transport))
        self.assertTrue(p.has_model("local-model"))
        self.assertTrue(p.has_model("local-model-2"))
        self.assertFalse(p.has_model("local-model-3"))

    def test_has_model_is_case_sensitive(self):
        transport = _MockTransport(models_json=MODELS)
        p = LocalOpenAICompatibleProvider(model="local-model", client=_client(transport))
        self.assertFalse(p.has_model("Local-Model"))

    def test_has_model_defaults_to_selected_model(self):
        transport = _MockTransport(models_json=MODELS)
        p = LocalOpenAICompatibleProvider(model="local-model-2", client=_client(transport))
        self.assertTrue(p.has_model())
        p2 = LocalOpenAICompatibleProvider(model="missing", client=_client(transport))
        self.assertFalse(p2.has_model())

    def test_is_available_true_when_not_closed(self):
        transport = _MockTransport()
        p = LocalOpenAICompatibleProvider(model="m", client=_client(transport))
        self.assertTrue(p.is_available())
        self.assertEqual(transport.requests, [])

    def test_is_available_false_after_close(self):
        transport = _MockTransport()
        p = LocalOpenAICompatibleProvider(model="m", client=_client(transport))
        p.close()
        self.assertFalse(p.is_available())
        self.assertEqual(transport.requests, [])

    def test_is_available_never_probes_models(self):
        transport = _MockTransport(models_raise_exc=httpx.ConnectError("refused"))
        p = LocalOpenAICompatibleProvider(model="m", client=_client(transport))
        self.assertTrue(p.is_available())
        self.assertEqual(transport.requests, [])

    def test_list_models_empty_on_non_2xx(self):
        transport = _MockTransport(models_status=500, models_json={"data": []})
        p = LocalOpenAICompatibleProvider(model="m", client=_client(transport))
        self.assertEqual(p.list_models(), [])

    def test_list_models_empty_on_connection_error(self):
        transport = _MockTransport(models_raise_exc=httpx.ConnectError("refused"))
        p = LocalOpenAICompatibleProvider(model="m", client=_client(transport))
        self.assertEqual(p.list_models(), [])

    def test_list_models_empty_on_timeout(self):
        transport = _MockTransport(models_raise_exc=httpx.ConnectTimeout("slow"))
        p = LocalOpenAICompatibleProvider(model="m", client=_client(transport))
        self.assertEqual(p.list_models(), [])

    def test_list_models_empty_on_invalid_shape(self):
        transport = _MockTransport(models_json={"unexpected": True})
        p = LocalOpenAICompatibleProvider(model="m", client=_client(transport))
        self.assertEqual(p.list_models(), [])

    def test_list_models_empty_on_invalid_json(self):
        transport = _MockTransport(
            models_status=200, models_json=None, text="not json at all"
        )
        p = LocalOpenAICompatibleProvider(model="m", client=_client(transport))
        self.assertEqual(p.list_models(), [])

    def test_list_models_empty_after_close(self):
        transport = _MockTransport(models_json=MODELS)
        p = LocalOpenAICompatibleProvider(model="m", client=_client(transport))
        p.close()
        self.assertEqual(p.list_models(), [])
        self.assertEqual(transport.requests, [])

    def test_has_model_false_when_models_unavailable(self):
        transport = _MockTransport(models_raise_exc=httpx.ConnectError("refused"))
        p = LocalOpenAICompatibleProvider(model="local-model", client=_client(transport))
        self.assertFalse(p.has_model("local-model"))

    def test_list_models_skips_non_string_ids(self):
        transport = _MockTransport(
            models_json={"data": [{"id": "real"}, {"id": 42}, {"id": ""}, "not-a-dict"]}
        )
        p = LocalOpenAICompatibleProvider(model="m", client=_client(transport))
        self.assertEqual(p.list_models(), ["real"])


class LocalOpenAICompatibleProviderLifecycleTest(unittest.TestCase):
    def test_satisfies_llm_provider_protocol(self):
        p = LocalOpenAICompatibleProvider(model="m")
        self.assertIsInstance(p, LLMProvider)

    def test_is_available_before_close(self):
        transport = _MockTransport(models_json=MODELS)
        p = LocalOpenAICompatibleProvider(model="m", client=_client(transport))
        self.assertTrue(p.is_available())
        self.assertEqual(transport.requests, [])

    def test_is_available_false_after_close(self):
        transport = _MockTransport(models_json=MODELS)
        p = LocalOpenAICompatibleProvider(model="m", client=_client(transport))
        p.close()
        self.assertFalse(p.is_available())

    def test_close_is_idempotent(self):
        transport = _MockTransport(models_json=MODELS)
        p = LocalOpenAICompatibleProvider(model="m", client=_client(transport))
        p.close()
        p.close()
        self.assertFalse(p.is_available())

    def test_close_leaves_injected_client_usable(self):
        transport = _MockTransport(models_json=MODELS)
        client = _client(transport)
        p = LocalOpenAICompatibleProvider(model="m", client=client)
        p.close()
        self.assertFalse(p.is_available())
        self.assertFalse(client.is_closed)
        resp = client.get("/models")
        self.assertEqual(resp.status_code, 200)

    def test_close_closes_owned_client(self):
        p = LocalOpenAICompatibleProvider(model="m")
        self.assertTrue(p._owns_client)
        p.close()
        self.assertTrue(p._client.is_closed)
        p.close()
        self.assertFalse(p.is_available())

    def test_owned_client_never_closed_by_reuse(self):
        transport = _MockTransport(json_data={"choices": [{"message": {"content": "x"}}]})
        client = _client(transport)
        p = LocalOpenAICompatibleProvider(model="m", client=client)
        p.close()
        p2 = LocalOpenAICompatibleProvider(model="m", client=client)
        self.assertEqual(p2.generate([{"role": "user", "content": "hi"}]), "x")

    def test_last_stats_returns_copy(self):
        transport = _MockTransport(
            json_data={
                "choices": [{"message": {"content": "x"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            }
        )
        p = LocalOpenAICompatibleProvider(model="m", client=_client(transport))
        p.generate([{"role": "user", "content": "hi"}])
        stats1 = p.last_stats
        stats2 = p.last_stats
        self.assertEqual(stats1, stats2)
        self.assertIsNot(stats1, stats2)

    def test_exception_messages_never_contain_api_key(self):
        api_key = "super-secret-api-key-xyz"
        for exc, op in [
            (httpx.ConnectTimeout("timeout"), "generate"),
            (httpx.ConnectError("refused"), "generate"),
            (httpx.ConnectTimeout("timeout"), "generate_stream"),
            (httpx.ConnectError("refused"), "generate_stream"),
        ]:
            transport = _MockTransport(raise_exc=exc)
            p = LocalOpenAICompatibleProvider(
                model="m", api_key=api_key, client=_client(transport)
            )
            try:
                if op == "generate":
                    p.generate([{"role": "user", "content": "hi"}])
                else:
                    list(p.generate_stream([{"role": "user", "content": "hi"}]))
            except Exception as e:
                self.assertNotIn(api_key, str(e))


class LocalOpenAICompatibleProviderEndpointValidationTest(unittest.TestCase):
    """Constructor enforces the shared loopback-only base URL policy."""

    def test_default_base_url_is_loopback_openai_convention(self):
        p = LocalOpenAICompatibleProvider(model="m")
        self.assertEqual(
            str(p._client.base_url), "http://localhost:8080/v1/"
        )
        p.close()

    def test_accepts_loopback_base_urls(self):
        for url in (
            "http://localhost:8080/v1",
            "https://localhost:8443/v1",
            "http://127.0.0.1:8080/v1",
            "http://[::1]:8080/v1",
        ):
            with self.subTest(url=url):
                p = LocalOpenAICompatibleProvider(model="m", base_url=url)
                p.close()

    def test_rejects_non_loopback_base_urls(self):
        for url in (
            "http://192.168.1.5:8080/v1",
            "http://8.8.8.8/v1",
            "ftp://localhost:8080/v1",
            "http://user:pass@localhost:8080/v1",
            "http://my-laptop:8080/v1",
            "localhost:8080/v1",
        ):
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    LocalOpenAICompatibleProvider(model="m", base_url=url)

    def test_rejects_non_loopback_even_with_injected_client(self):
        client = _client(_MockTransport())
        with self.assertRaises(ValueError):
            LocalOpenAICompatibleProvider(
                model="m", base_url="http://192.168.1.5:8080/v1", client=client
            )
        self.assertFalse(client.is_closed)
        client.close()

    def test_rejects_query_or_fragment_base_urls(self):
        for url in (
            "http://localhost:8080/v1?key=value",
            "http://127.0.0.1:8080/v1?apikey=sk-secret-123",
            "http://[::1]:8080/v1#frag",
        ):
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    LocalOpenAICompatibleProvider(model="m", base_url=url)

    def test_rejects_bare_delimiter_base_urls(self):
        for url in (
            "http://localhost:8080/v1?",
            "http://localhost:8080/v1#",
            "http://localhost:8080/v1?#",
            "http://127.0.0.1:8080/v1?",
            "http://[::1]:8080/v1#",
        ):
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    LocalOpenAICompatibleProvider(model="m", base_url=url)

    def test_accepts_encoded_delimiters_in_base_url_path(self):
        for url in (
            "http://localhost:8080/v1%3F",
            "http://localhost:8080/v1%23",
            "http://localhost:8080/v1%3Fpath%23",
        ):
            with self.subTest(url=url):
                p = LocalOpenAICompatibleProvider(model="m", base_url=url)
                p.close()

    def test_validation_errors_never_echo_canary_url_parts(self):
        canary = "canary-userinfo-77"
        for url in (
            f"http://user:{canary}@localhost:8080/v1",
            f"http://{canary}:8080/v1",
            f"http://localhost:8080/v1?key={canary}",
            f"http://localhost:8080/v1?",
            f"http://localhost:8080/v1#{canary}",
            "http://[::1",
        ):
            with self.subTest(url=url):
                with self.assertRaises(ValueError) as ctx:
                    LocalOpenAICompatibleProvider(model="m", base_url=url)
                message = str(ctx.exception)
                self.assertNotIn(canary, message)
                self.assertNotIn(url, message)
                self.assertNotIn("http://", message)
                self.assertNotIn("IPv6", message)

    def test_query_rejection_error_does_not_echo_query_value(self):
        secret = "sk-query-secret-999"
        with self.assertRaises(ValueError) as ctx:
            LocalOpenAICompatibleProvider(
                model="m", base_url=f"http://localhost:8080/v1?key={secret}"
            )
        self.assertNotIn(secret, str(ctx.exception))
        self.assertNotIn("sk-", str(ctx.exception))

    def test_default_url_works_with_matching_injected_client(self):
        transport = _MockTransport(json_data={"choices": [{"message": {"content": "x"}}]})
        p = LocalOpenAICompatibleProvider(model="m", client=_client(transport))
        p.generate([{"role": "user", "content": "hi"}])
        self.assertEqual(str(transport.requests[0].url), "http://localhost:8080/v1/chat/completions")


class LocalOpenAICompatibleProviderDiscoveryHeadersTest(unittest.TestCase):
    """GET /models discovery sends the same optional auth as generation."""

    def test_keyless_discovery_sends_no_authorization(self):
        transport = _MockTransport(models_json=MODELS)
        p = LocalOpenAICompatibleProvider(model="m", client=_client(transport))
        self.assertEqual(p.list_models(), ["local-model", "local-model-2"])
        req = transport.requests[0]
        self.assertNotIn("authorization", req.headers)

    def test_blank_key_discovery_sends_no_authorization(self):
        transport = _MockTransport(models_json=MODELS)
        p = LocalOpenAICompatibleProvider(
            model="m", api_key="   ", client=_client(transport)
        )
        self.assertEqual(p.list_models(), ["local-model", "local-model-2"])
        req = transport.requests[0]
        self.assertNotIn("authorization", req.headers)

    def test_authenticated_discovery_sends_bearer_header(self):
        transport = _MockTransport(models_json=MODELS)
        p = LocalOpenAICompatibleProvider(
            model="m", api_key="discovery-secret", client=_client(transport)
        )
        self.assertEqual(p.list_models(), ["local-model", "local-model-2"])
        req = transport.requests[0]
        self.assertEqual(req.headers["authorization"], "Bearer discovery-secret")

    def test_discovery_uses_same_header_policy_as_generation(self):
        transport = _MockTransport(
            models_json=MODELS,
            json_data={"choices": [{"message": {"content": "hi"}}]},
        )
        p = LocalOpenAICompatibleProvider(
            model="m", api_key="shared-secret", client=_client(transport)
        )
        p.list_models()
        discovery_req = transport.requests[0]
        p.generate([{"role": "user", "content": "hi"}])
        generation_req = transport.requests[1]
        self.assertEqual(
            discovery_req.headers.get("authorization"),
            generation_req.headers.get("authorization"),
        )


class LocalOpenAICompatibleProviderClientBoundaryTest(unittest.TestCase):
    """An injected client must itself honor the loopback trust boundary."""

    def test_remote_injected_client_is_rejected(self):
        for base_url in (
            "http://8.8.8.8/v1",
            "https://example.com/v1",
            "http://192.168.1.5:8080/v1",
        ):
            with self.subTest(base_url=base_url):
                client = _client_at(_MockTransport(), base_url)
                with self.assertRaises(ValueError):
                    LocalOpenAICompatibleProvider(model="m", client=client)
                self.assertFalse(client.is_closed)
                client.close()

    def test_mismatched_loopback_injected_client_is_rejected(self):
        client = _client_at(_MockTransport(), "http://localhost:9999/v1")
        with self.assertRaises(ValueError) as ctx:
            LocalOpenAICompatibleProvider(
                model="m", base_url="http://localhost:8080/v1", client=client
            )
        self.assertIn("does not match", str(ctx.exception))
        self.assertFalse(client.is_closed)
        client.close()

    def test_mismatch_error_does_not_echo_urls(self):
        canary = "canary-client-url-42"
        client = _client_at(_MockTransport(), f"http://localhost:9999/{canary}/v1")
        with self.assertRaises(ValueError) as ctx:
            LocalOpenAICompatibleProvider(
                model="m", base_url="http://localhost:8080/v1", client=client
            )
        message = str(ctx.exception)
        self.assertNotIn(canary, message)
        self.assertNotIn("localhost:9999", message)
        self.assertNotIn("localhost:8080", message)
        self.assertNotIn("http://", message)
        client.close()

    def test_mismatch_rejected_even_with_default_base_url(self):
        client = _client_at(_MockTransport(), "http://127.0.0.1:8080/v1")
        with self.assertRaises(ValueError):
            LocalOpenAICompatibleProvider(model="m", client=client)
        client.close()

    def test_trailing_slash_variants_are_equivalent(self):
        transport = _MockTransport(models_json=MODELS)
        client = _client_at(transport, "http://localhost:8080/v1/")
        p = LocalOpenAICompatibleProvider(
            model="m", base_url="http://localhost:8080/v1", client=client
        )
        self.assertEqual(p.list_models(), ["local-model", "local-model-2"])
        self.assertEqual(str(transport.requests[0].url), "http://localhost:8080/v1/models")
        p.close()
        self.assertFalse(client.is_closed)
        client.close()

    def test_rejection_leaves_injected_client_usable(self):
        client = _client_at(_MockTransport(), "http://localhost:9090/v1")
        with self.assertRaises(ValueError):
            LocalOpenAICompatibleProvider(model="m", client=client)
        self.assertFalse(client.is_closed)
        resp = client.get("/models")
        self.assertEqual(resp.status_code, 200)
        client.close()

    def test_no_key_leakage_in_repr_and_boundary_errors(self):
        api_key = "super-secret-key-for-repr"
        transport = _MockTransport(models_json=MODELS)
        p = LocalOpenAICompatibleProvider(
            model="m", api_key=api_key, client=_client(transport)
        )
        self.assertNotIn(api_key, repr(p))
        self.assertNotIn(api_key, str(p))
        p.close()

        mismatched = _client_at(_MockTransport(), "http://localhost:9999/v1")
        with self.assertRaises(ValueError) as ctx:
            LocalOpenAICompatibleProvider(
                model="m",
                base_url="http://localhost:8080/v1",
                api_key=api_key,
                client=mismatched,
            )
        self.assertNotIn(api_key, str(ctx.exception))
        mismatched.close()


if __name__ == "__main__":
    unittest.main()