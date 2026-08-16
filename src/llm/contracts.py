"""LLMプロバイダー間で共有する最小データ契約。"""

from dataclasses import dataclass
from typing import Any, Literal, TypedDict


ChatRole = Literal["system", "user", "assistant", "tool"]


class ChatMessage(TypedDict):
    """テキスト生成へ渡す1件のメッセージ。"""

    role: ChatRole
    content: str


class GenerationStats(TypedDict, total=False):
    """生成後に取得できるプロバイダー共通の統計値。"""

    total_duration: int
    eval_count: int
    eval_duration: int


@dataclass(frozen=True)
class GenerationOptions:
    """既存Ollama経路と同じ既定値を持つ生成設定。"""

    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    num_ctx: int = 8192
    num_predict: int | None = None
    timeout: float | None = None

    def as_generate_kwargs(self) -> dict[str, Any]:
        """非ストリーミングProvider呼び出し用のキーワード引数を返す。"""
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repeat_penalty": self.repeat_penalty,
            "num_ctx": self.num_ctx,
            "num_predict": self.num_predict,
            "timeout": self.timeout,
        }

    def as_stream_kwargs(self) -> dict[str, Any]:
        """現行stream契約で利用可能なキーワード引数を返す。"""
        kwargs = self.as_generate_kwargs()
        kwargs.pop("timeout")
        return kwargs
