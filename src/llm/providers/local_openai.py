"""Keyless-capable OpenAI-compatible provider for local /v1 API servers.

Targets llama.cpp, LM Studio, and vLLM style servers exposing the OpenAI
``/v1`` surface: ``POST /chat/completions`` and ``GET /models``. No cloud
approval/redaction behavior; the server is assumed to be a local inference
daemon reached only over loopback. The destination is enforced to
``localhost`` / ``127.0.0.1`` / ``::1`` (or another ``is_loopback`` IP) by
:func:`~src.llm.local_endpoint.validate_loopback_openai_base_url`; remote
trusted nodes are not yet supported and are deferred.
"""

from __future__ import annotations

import json
from collections.abc import Generator
from typing import Any

import httpx

from src.llm.errors import ProviderRequestError, ProviderTimeoutError
from src.llm.local_endpoint import validate_loopback_openai_base_url


class LocalOpenAICompatibleProvider:
    """Synchronous OpenAI-compatible provider for a local /v1 API server.

    ``api_key`` is optional. When it is absent or blank, no ``Authorization``
    header is sent at all; a ``Bearer`` header is sent only when a non-blank
    key is explicitly configured. The same header policy applies to ``GET
    /models`` discovery and to generation/streaming.

    A caller-injected ``client`` must itself target a loopback URL whose
    normalized effective ``base_url`` matches this provider's ``base_url``;
    otherwise the trust boundary could be bypassed by a remote or mismatched
    client. Ownership semantics are unchanged: an injected client is left
    usable by the caller and is never closed by this provider.

    Parameters ``top_k``, ``repeat_penalty``, and ``num_ctx`` are accepted for
    interface compatibility but have no OpenAI equivalent and are silently
    ignored (documented in each method docstring).
    """

    def __init__(
        self,
        *,
        model: str,
        base_url: str = "http://localhost:8080/v1",
        provider_id: str = "local-openai",
        api_key: str | None = None,
        timeout: float = 60.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self.provider_id = provider_id
        self._api_key = (api_key or "").strip()
        self._closed = False
        self._last_stats: dict[str, Any] = {}
        validate_loopback_openai_base_url(base_url)
        normalized_base_url = base_url.rstrip("/")
        if client is not None:
            client_base_url = str(client.base_url)
            validate_loopback_openai_base_url(client_base_url)
            if client_base_url.rstrip("/") != normalized_base_url:
                raise ValueError(
                    "injected client base_url does not match provider base_url"
                )
            self._client = client
            self._owns_client = False
        else:
            self._client = httpx.Client(
                base_url=normalized_base_url, timeout=timeout
            )
            self._owns_client = True

    def is_available(self) -> bool:
        """Report lifecycle availability without any network probe.

        Returns ``True`` as long as the provider is not closed. Actual
        reachability is established lazily by ``generate``/``generate_stream``,
        so chat-only servers that do not implement ``GET /models`` remain
        usable and calls do not incur a duplicate ``/models`` probe.
        """
        return not self._closed

    def list_models(self) -> list[str]:
        """Best-effort discovery of ``GET /models`` ids (``data[].id``).

        Optional explicit discovery: returns ``[]`` (never raises) when the
        provider is closed, the request times out, transport fails, the server
        responds non-2xx, or the payload is not valid JSON with a ``data`` list.
        """
        data = self._fetch_models()
        data_list = data.get("data")
        if not isinstance(data_list, list):
            return []
        models: list[str] = []
        for item in data_list:
            if not isinstance(item, dict):
                continue
            model_id = item.get("id")
            if isinstance(model_id, str) and model_id:
                models.append(model_id)
        return models

    def has_model(self, model: str | None = None) -> bool:
        """Return whether the exact selected model id is in the /models list."""
        target = model or self.model
        return target in self.list_models()

    def _fetch_models(self) -> dict[str, Any]:
        """Best-effort ``GET /models``; returns ``{}`` on any failure."""
        if self._closed:
            return {}
        try:
            resp = self._client.get("/models", headers=self._headers())
        except (httpx.TimeoutException, httpx.HTTPError):
            return {}
        if resp.status_code < 200 or resp.status_code >= 300:
            return {}
        try:
            data = resp.json()
        except (json.JSONDecodeError, ValueError):
            return {}
        if not isinstance(data, dict):
            return {}
        return data

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

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
        if not isinstance(content, str):
            raise ProviderRequestError(
                self.provider_id,
                "generate",
                f"unexpected response structure: content is {type(content).__name__}",
            )

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
        """Parse SSE lines and yield ``delta.content`` string fragments.

        Protocol violations are fatal and raise
        :class:`~src.llm.errors.ProviderRequestError`: malformed JSON data,
        non-object chunks, server ``error`` payloads, invalid (non-string)
        ``content`` values, EOF without ``[DONE]``, and a ``[DONE]`` stream
        that produced no text content. Usage-only / metadata-only chunks
        (no ``content``) are ignored. When content has already been yielded,
        a later error still propagates so partial text is not silently
        presented as a complete answer.
        """
        saw_content = False
        saw_done = False
        for line in resp.iter_lines():
            line = line.strip()
            if not line:
                continue
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                saw_done = True
                break
            try:
                chunk = json.loads(payload)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ProviderRequestError(
                    self.provider_id,
                    "generate_stream",
                    f"malformed SSE data chunk: {exc!r}",
                ) from exc
            if not isinstance(chunk, dict):
                raise ProviderRequestError(
                    self.provider_id,
                    "generate_stream",
                    f"SSE data chunk is not a JSON object: {type(chunk).__name__}",
                )
            error = chunk.get("error")
            if error is not None:
                raise ProviderRequestError(
                    self.provider_id,
                    "generate_stream",
                    f"server error in SSE stream: {error!r}",
                )
            choices = chunk.get("choices")
            if not isinstance(choices, list) or not choices:
                continue
            choice = choices[0]
            if not isinstance(choice, dict):
                raise ProviderRequestError(
                    self.provider_id,
                    "generate_stream",
                    f"invalid SSE choices[0]: {type(choice).__name__}",
                )
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                raise ProviderRequestError(
                    self.provider_id,
                    "generate_stream",
                    f"invalid SSE delta: {type(delta).__name__}",
                )
            content = delta.get("content")
            if content is None:
                continue
            if not isinstance(content, str):
                raise ProviderRequestError(
                    self.provider_id,
                    "generate_stream",
                    f"invalid SSE content type: {type(content).__name__}",
                )
            if content:
                saw_content = True
                yield content
        if not saw_done:
            raise ProviderRequestError(
                self.provider_id,
                "generate_stream",
                "SSE stream ended without [DONE]",
            )
        if not saw_content:
            raise ProviderRequestError(
                self.provider_id,
                "generate_stream",
                "SSE stream completed with [DONE] but produced no text content",
            )

    @property
    def last_stats(self) -> dict[str, Any]:
        """Return a copy of the stats from the last non-streaming generation."""
        return dict(self._last_stats)

    def close(self) -> None:
        """Mark the provider closed and release owned resources.

        The underlying HTTP client is closed only when the provider created it.
        A caller-injected client is left usable by the caller. Closing is
        idempotent.
        """
        if not self._closed:
            if self._owns_client:
                self._client.close()
            self._closed = True