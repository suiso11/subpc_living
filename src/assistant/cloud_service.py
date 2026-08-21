"""クラウド送信の明示経路（Phase K）。

経路選択は Router ではなく、この Bridge が担当する。前提:
- ``AssistantRequest.privacy == "cloud_allowed"`` かつ ``allow_cloud`` のみ許可
- 1リクエスト単位の明示承認 (``ApprovalGate``) が必須
- ``CloudPayloadBuilder`` で personal/secret を除外（匿名化）して送信
- クラウド失敗時はローカル ``AssistantService`` へ fallback
構築済みmessagesをそのままクラウドへ渡す経路は存在しない（既存保証を維持）。
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

from src.assistant.contracts import (
    AssistantError,
    AssistantGenerationError,
    AssistantRequest,
    AssistantResponse,
)
from src.assistant.service import AssistantService
from src.context.contracts import ContextBlock
from src.llm.approval import ApprovalGate, CloudPayloadBuilder, CloudPreview
from src.llm.contracts import ChatMessage, GenerationOptions
from src.llm.errors import LLMProviderError
from src.llm.registry import ProviderRegistry
from src.llm.routing.contracts import RouteDecision


class CloudRouteBridge:
    """承認済み・匿名化済みのクラウド送信とローカルFallback。"""

    def __init__(
        self,
        registry: ProviderRegistry,
        cloud_provider_id: str,
        *,
        approval: ApprovalGate,
        local_service: AssistantService | None = None,
        builder: CloudPayloadBuilder | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._registry = registry
        self._cloud_provider_id = cloud_provider_id
        self._approval = approval
        self._local_service = local_service
        self._builder = builder or CloudPayloadBuilder()
        self._clock = clock
        self._last_preview: CloudPreview | None = None

    def send(
        self,
        request: AssistantRequest,
        blocks: Sequence[ContextBlock],
        *,
        local_messages: Sequence[ChatMessage],
        history: Sequence[ChatMessage] = (),
        options: GenerationOptions | None = None,
    ) -> AssistantResponse:
        if not (request.privacy == "cloud_allowed" and request.allow_cloud):
            raise AssistantError(
                "cloud route requires privacy=cloud_allowed and allow_cloud"
            )
        request_id = request.request_id
        if not request_id:
            raise AssistantError("cloud send requires explicit request_id for approval")
        self._approval.require(request_id)

        messages = self._builder.build(blocks, history=history)
        self._last_preview = self._approval.preview(request_id, messages)

        try:
            entry = self._registry.get(self._cloud_provider_id)
        except Exception as exc:
            return self._fallback(
                request,
                local_messages,
                attempts=[(self._cloud_provider_id, f"{type(exc).__name__}: {exc}")],
            )

        if entry.local:
            raise AssistantError("cloud provider must be non-local")

        started_at = self._clock()
        try:
            text = entry.provider.generate(
                [dict(message) for message in messages],
                **(options or GenerationOptions()).as_generate_kwargs(),
            )
        except LLMProviderError as exc:
            return self._fallback(
                request,
                local_messages,
                attempts=[
                    (self._cloud_provider_id, f"{type(exc).__name__}: {exc}")
                ],
            )

        latency_ms = int(max(0.0, self._clock() - started_at) * 1000)
        return AssistantResponse(
            text=text,
            route=RouteDecision(
                provider_id=entry.provider_id,
                model=entry.provider.model,
                local=False,
                reason="cloud approved + anonymized",
            ),
            latency_ms=latency_ms,
            stats=dict(entry.provider.last_stats),
        )

    def _fallback(
        self,
        request: AssistantRequest,
        local_messages: Sequence[ChatMessage],
        *,
        attempts: list[tuple[str, str]],
    ) -> AssistantResponse:
        if self._local_service is None:
            raise AssistantGenerationError(tuple(attempts))
        return self._local_service.generate(request, local_messages)

    @property
    def last_preview(self) -> CloudPreview | None:
        return self._last_preview
