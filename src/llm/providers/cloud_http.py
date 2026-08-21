"""Phase K: OpenAI-compatible HTTP cloud provider (real swap target).

This is the real-cloud swap target for Phase K; only public/anonymized context
reaches it via CloudRouteBridge; key comes from env via
CloudConfig.resolve_api_key(), never hardcoded.

Structurally implements the same interface as FakeCloudProvider/OllamaProvider.
"""

from __future__ import annotations

import json
from collections.abc import Generator
from typing import Any

import httpx

from src.llm.errors import ProviderRequestError, ProviderTimeoutError


class OpenAICompatibleProvider:
    """Real cloud provider using the OpenAI-compatible /chat/completions API.

    Parameters ``top_k``, ``repeat_penalty``, and ``num_ctx`` have no OpenAI
    equivalent and are silently ignored (documented in each method docstring).
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        provider_id: str = "cloud",
        timeout: float = 60.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self.provider_id = provider_id
        self._api_key = api_key
        self._closed = False
        self._last_stats: dict[str, Any] = {}
        if client is not None:
            self._client = client
            self._owns_client = False
        else:
            self._client = httpx.Client(
                base_url=base_url.rstrip("/"), timeout=timeout
            )
            self._owns_client = True

    def is_available(self) -> bool:
        """Return ``not self._closed``.

        Does NOT probe the network to avoid burning quota; availability is
        determined solely by whether this provider instance is still open.
        """
        return not self._closed

    def _build_payload(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float,
        top_p: float,
        num_predict: int | None,
        stream: bool = False,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [dict(m) for m in messages],
            "temperature": temperature,
            "top_p": top_p,
        }
        if num_predict is not None and num_predict > 0:
            body["max_tokens"] = num_predict
        if stream:
            body["stream"] = True
        return body

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

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
        """Generate a completion (non-streaming).

        Parameters ``top_k``, ``repeat_penalty``, ``num_ctx`` are accepted for
        interface compatibility but have no OpenAI equivalent and are silently
        ignored.
        """
        body = self._build_payload(
            messages, temperature=temperature, top_p=top_p, num_predict=num_predict
        )
        kwargs: dict[str, Any] = {"headers": self._headers(), "json": body}
        if timeout is not None:
            kwargs["timeout"] = timeout
        try:
            resp = self._client.post("/chat/completions", **kwargs)
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                self.provider_id, "generate", f"request timed out: {exc!r}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderRequestError(
                self.provider_id, "generate", f"request failed: {exc!r}"
            ) from exc

        if resp.status_code < 200 or resp.status_code >= 300:
            raise ProviderRequestError(
                self.provider_id,
                "generate",
                f"HTTP {resp.status_code}",
            )
        try:
            data = resp.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise ProviderRequestError(
                self.provider_id, "generate", f"invalid JSON response: {exc!r}"
            ) from exc

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderRequestError(
                self.provider_id, "generate", f"unexpected response structure: {exc!r}"
            ) from exc

        usage = data.get("usage")
        if isinstance(usage, dict):
            self._last_stats = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }
        else:
            self._last_stats = {}

        return content

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
        """Generate a streaming completion (SSE).

        Parameters ``top_k``, ``repeat_penalty``, ``num_ctx`` are accepted for
        interface compatibility but have no OpenAI equivalent and are silently
        ignored.
        """
        body = self._build_payload(
            messages,
            temperature=temperature,
            top_p=top_p,
            num_predict=num_predict,
            stream=True,
        )
        # client.stream + iter_lines で SSE を逐次消流する (全体バッファしない)。
        try:
            with self._client.stream(
                "POST",
                "/chat/completions",
                headers=self._headers(),
                json=body,
            ) as resp:
                if resp.status_code < 200 or resp.status_code >= 300:
                    raise ProviderRequestError(
                        self.provider_id,
                        "generate_stream",
                        f"HTTP {resp.status_code}",
                    )
                yield from self._iter_sse_content(resp)
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                self.provider_id, "generate_stream", f"request timed out: {exc!r}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderRequestError(
                self.provider_id, "generate_stream", f"request failed: {exc!r}"
            ) from exc

    def _iter_sse_content(self, resp: httpx.Response) -> Generator[str, None, None]:
        """SSE 行を逐次解析し、delta.content の断片だけを yield する。"""
        for line in resp.iter_lines():
            line = line.strip()
            if not line:
                continue
            if not line.startswith("data: "):
                continue
            payload = line[len("data: "):]
            if payload == "[DONE]":
                return
            try:
                chunk = json.loads(payload)
            except (json.JSONDecodeError, ValueError):
                continue
            try:
                delta = chunk["choices"][0]["delta"]
                content = delta.get("content")
                if content:
                    yield content
            except (KeyError, IndexError, TypeError):
                continue

    @property
    def last_stats(self) -> dict[str, Any]:
        """Return a copy of the stats from the last non-streaming generation."""
        return dict(self._last_stats)

    def close(self) -> None:
        """Close the underlying HTTP client and mark this provider as closed."""
        self._client.close()
        self._closed = True
