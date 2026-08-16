"""ネットワークや実モデルを使わない決定的なテスト用Provider。"""

from collections.abc import Generator, Iterable
from typing import Any


class FakeProvider:
    """固定応答を返し、呼び出し内容を検査できるProvider。"""

    def __init__(
        self,
        response: str = "fake response",
        *,
        stream_chunks: Iterable[str] | None = None,
        model: str = "fake",
        available: bool = True,
        stats: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.response = response
        self.stream_chunks = tuple(stream_chunks) if stream_chunks is not None else (response,)
        self.available = available
        self.calls: list[dict[str, Any]] = []
        self.closed = False
        self._last_stats = dict(stats or {})

    def is_available(self) -> bool:
        return self.available and not self.closed

    @staticmethod
    def _options(
        *,
        temperature: float,
        top_p: float,
        top_k: int,
        repeat_penalty: float,
        num_ctx: int,
        num_predict: int | None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        options: dict[str, Any] = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repeat_penalty": repeat_penalty,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        }
        if timeout is not None:
            options["timeout"] = timeout
        return options

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        num_ctx: int = 8192,
        num_predict: int | None = None,
        timeout: float | None = None,
    ) -> str:
        self.calls.append(
            {
                "kind": "generate",
                "messages": [dict(message) for message in messages],
                "options": self._options(
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repeat_penalty=repeat_penalty,
                    num_ctx=num_ctx,
                    num_predict=num_predict,
                    timeout=timeout,
                ),
            }
        )
        return self.response

    def generate_stream(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        num_ctx: int = 8192,
        num_predict: int | None = None,
    ) -> Generator[str, None, None]:
        self.calls.append(
            {
                "kind": "generate_stream",
                "messages": [dict(message) for message in messages],
                "options": self._options(
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repeat_penalty=repeat_penalty,
                    num_ctx=num_ctx,
                    num_predict=num_predict,
                ),
            }
        )
        yield from self.stream_chunks

    @property
    def last_stats(self) -> dict[str, Any]:
        return dict(self._last_stats)

    def close(self) -> None:
        self.closed = True
