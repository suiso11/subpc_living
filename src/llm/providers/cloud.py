"""クラウドProviderの境界とテスト用Fake。

実際のHTTPクラウドProviderは今回実装しない。Phase Kは「無効が既定」の
設定ゲートと承認経路を完成させることが目的であり、実送信は将来の
CloudConfig有効時へ swap する。ここではネットワーク・実キーを使わない
``FakeCloudProvider`` を非Local Providerのテスト双重として提供する。
"""

from collections.abc import Generator
from typing import Any

from src.llm.errors import ProviderRequestError
from src.llm.provider import LLMProvider


class FakeCloudProvider:
    """ネットワーク・実モデルを使わない非LocalクラウドProviderのテスト双重。

    ``local=False`` は登録時に ``ProviderRegistry.register(..., local=False)`` で
    宣言する。この実体は ``LLMProvider`` に構造的に適合する。
    """

    def __init__(
        self,
        model: str = "cloud-model",
        *,
        fail: bool = False,
        stats: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self._fail = fail
        self._closed = False
        self._sent_payloads: list[list[dict[str, Any]]] = []
        self._last_stats = dict(stats or {})

    def is_available(self) -> bool:
        return not self._closed

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
        self._sent_payloads.append([dict(message) for message in messages])
        if self._fail:
            raise ProviderRequestError(self.model, "generate", "cloud provider unavailable")
        return "cloud response"

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
        self._sent_payloads.append([dict(message) for message in messages])
        if self._fail:
            raise ProviderRequestError(
                self.model, "generate_stream", "cloud provider unavailable"
            )
        yield "cloud "
        yield "response"

    @property
    def sent_payloads(self) -> list[list[dict[str, Any]]]:
        return [list(payload) for payload in self._sent_payloads]

    @property
    def last_stats(self) -> dict[str, Any]:
        return dict(self._last_stats)

    def close(self) -> None:
        self._closed = True
