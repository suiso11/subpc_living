"""LLM実装を既存の呼び出し元から分離する共通インターフェース。"""

from collections.abc import Generator
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    """現在の同期チャット経路が必要とする最小Provider契約。

    既存の ``OllamaClient`` と構造的に互換になるよう、移行前のメソッド名と
    生成オプションを維持する。ルーティングや非同期化はこの境界の外で扱う。
    """

    model: str

    def is_available(self) -> bool:
        """プロバイダーが応答可能か返す。"""
        ...

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
        """非ストリーミングで応答を生成する。"""
        ...

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
        """生成したテキストを順番に返す。"""
        ...

    @property
    def last_stats(self) -> dict[str, Any]:
        """直近生成の統計値を返す。"""
        ...

    def close(self) -> None:
        """プロバイダーが保持するリソースを解放する。"""
        ...
