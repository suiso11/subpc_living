"""既存OllamaClientを共通Provider契約へ適合させるAdapter。"""

from collections.abc import Generator
from typing import Any

import httpx

from src.chat.client import OllamaClient, OllamaResponseError
from src.llm.errors import ProviderRequestError, ProviderTimeoutError


class OllamaProvider:
    """OllamaClientへ委譲し、通信例外と応答解析例外 (OllamaResponseError) を共通形式へ変換する。"""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str | None = None,
        *,
        provider_id: str = "ollama",
        client: OllamaClient | None = None,
    ) -> None:
        self.provider_id = provider_id
        if client is not None:
            self._client = client
        elif model is None:
            self._client = OllamaClient(base_url=base_url)
        else:
            self._client = OllamaClient(base_url=base_url, model=model)

    @property
    def model(self) -> str:
        return self._client.model

    @model.setter
    def model(self, value: str) -> None:
        self._client.model = value

    def is_available(self) -> bool:
        try:
            return self._client.is_available()
        except httpx.HTTPError:
            return False

    def list_models(self) -> list[str]:
        """Ollama固有のモデル一覧機能を移行期間中も公開する。"""
        return self._client.list_models()

    def has_model(self, model: str | None = None) -> bool:
        """Ollama固有のモデル存在確認を移行期間中も公開する。"""
        return self._client.has_model(model)

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
        try:
            return self._client.generate(
                messages,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repeat_penalty,
                num_ctx=num_ctx,
                num_predict=num_predict,
                timeout=timeout,
            )
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                self.provider_id, "generate", f"request timed out: {exc!r}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderRequestError(
                self.provider_id, "generate", f"request failed: {exc!r}"
            ) from exc
        except OllamaResponseError as exc:
            raise ProviderRequestError(
                self.provider_id, "generate", f"invalid response: {exc}"
            ) from exc

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
        try:
            yield from self._client.generate_stream(
                messages,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repeat_penalty,
                num_ctx=num_ctx,
                num_predict=num_predict,
            )
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                self.provider_id, "generate_stream", f"request timed out: {exc!r}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderRequestError(
                self.provider_id, "generate_stream", f"request failed: {exc!r}"
            ) from exc
        except OllamaResponseError as exc:
            raise ProviderRequestError(
                self.provider_id, "generate_stream", f"invalid response: {exc}"
            ) from exc

    @property
    def last_stats(self) -> dict[str, Any]:
        return dict(self._client.last_stats)

    def close(self) -> None:
        self._client.close()
