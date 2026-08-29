import unittest

from src.assistant.contracts import AssistantError, AssistantRequest
from src.assistant.service import AssistantService
from src.llm.errors import ProviderRequestError
from src.llm.providers.fake import FakeProvider
from src.llm.registry import ProviderRegistry
from src.llm.routing.static import StaticRouter


def _request(request_id: str = "r1") -> AssistantRequest:
    return AssistantRequest(
        text="hi",
        conversation_id="c1",
        channel="web",
        privacy="local_only",  # type: ignore[arg-type]
        allow_cloud=False,
        request_id=request_id,
    )


def _messages():
    return [{"role": "user", "content": "hi"}]


class CloseRaisingIterable:
    """chunk を返した後に生成例外を送出し、close() で別の例外を送出するIterable。"""

    def __init__(self, chunks, gen_exc, close_exc) -> None:
        self.chunks = chunks
        self.gen_exc = gen_exc
        self.close_exc = close_exc

    def __iter__(self):
        for chunk in self.chunks:
            yield chunk
        if self.gen_exc is not None:
            raise self.gen_exc

    def close(self) -> None:
        if self.close_exc is not None:
            raise self.close_exc


class CloseRaisingProvider(FakeProvider):
    def __init__(self, gen_exc, close_exc) -> None:
        super().__init__()
        self.gen_exc = gen_exc
        self.close_exc = close_exc

    def generate_stream(self, messages, **options):
        return CloseRaisingIterable(("x", "y"), self.gen_exc, self.close_exc)


def _make_stream(provider, request_id: str = "r1"):
    reg = ProviderRegistry()
    reg.register("ollama", provider, local=True)
    router = StaticRouter(reg, default_provider_id="ollama")
    service = AssistantService(reg, router)
    return service.generate_stream(_request(request_id), _messages())


class StreamResultCloseMaskingTest(unittest.TestCase):
    """StreamResult の iteration/close が下層 close エラーで元例外を隠さないこと。"""

    def test_generation_error_not_masked_by_underlying_close_error(self) -> None:
        gen_exc = ProviderRequestError("fake", "generate_stream", "generation failed")
        close_exc = RuntimeError("underlying generator close failed")
        stream = _make_stream(CloseRaisingProvider(gen_exc, close_exc))

        with self.assertRaises(ProviderRequestError) as ctx:
            list(stream)

        self.assertIs(ctx.exception, gen_exc)

    def test_explicit_close_with_no_prior_error_is_safe(self) -> None:
        close_exc = RuntimeError("underlying generator close failed")
        stream = _make_stream(CloseRaisingProvider(None, close_exc))

        it = iter(stream)
        self.assertEqual(next(it), "x")

        stream.close()  # must not raise despite underlying close error

        self.assertTrue(stream._closed)
        with self.assertRaises(AssistantError):
            stream.response

    def test_normal_completion_close_error_propagates(self) -> None:
        close_exc = RuntimeError("underlying generator close failed")
        stream = _make_stream(CloseRaisingProvider(None, close_exc))

        with self.assertRaises(RuntimeError) as ctx:
            list(stream)

        self.assertIs(ctx.exception, close_exc)


if __name__ == "__main__":
    unittest.main()