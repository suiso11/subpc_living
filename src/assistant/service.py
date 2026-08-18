"""経路選択とProvider fallbackを統合するAssistantサービス。"""

from collections.abc import Callable, Iterable, Iterator, Sequence
import time
from typing import NoReturn

from src.assistant.contracts import (
    AssistantError,
    AssistantGenerationError,
    AssistantRequest,
    AssistantResponse,
)
from src.llm.contracts import ChatMessage, GenerationOptions
from src.llm.errors import LLMProviderError
from src.llm.registry import ProviderEntry, ProviderRegistry, UnknownProviderError
from src.llm.routing.contracts import ModelRouter, RouteDecision


def _candidate_ids(decision: RouteDecision) -> tuple[str, ...]:
    candidates: list[str] = []
    for provider_id in (decision.provider_id, *decision.fallback_provider_ids):
        if provider_id not in candidates:
            candidates.append(provider_id)
    return tuple(candidates)


def _actual_route(
    decision: RouteDecision,
    entry: ProviderEntry,
    candidates: tuple[str, ...],
    attempts: list[tuple[str, str]],
) -> RouteDecision:
    reason = decision.reason
    if entry.provider_id != decision.provider_id:
        failures = ", ".join(
            f"{provider_id}={failure}" for provider_id, failure in attempts
        )
        reason = f"{reason} | fallback after {failures}"
    return RouteDecision(
        provider_id=entry.provider_id,
        model=entry.provider.model,
        local=entry.local,
        reason=reason,
        fallback_provider_ids=tuple(
            provider_id
            for provider_id in candidates
            if provider_id != entry.provider_id
        ),
    )


def _raise_generation_error(
    attempts: list[tuple[str, str]], last_error: Exception | None
) -> NoReturn:
    error = AssistantGenerationError(tuple(attempts))
    if last_error is None:
        raise error
    raise error from last_error


class AssistantService:
    """履歴を持たず、選択済み経路から同期応答を生成する。

    ``requested_model`` はPhase Cでは適用せず、各Providerが固定で持つmodelを利用する。
    構築済みmessagesを安全に扱うため、localでないProviderは常に拒否する。
    """

    def __init__(
        self,
        registry: ProviderRegistry,
        router: ModelRouter,
        *,
        options: GenerationOptions | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._registry = registry
        self._router = router
        self._options = options or GenerationOptions()
        self._clock = clock

    def generate(
        self, request: AssistantRequest, messages: Sequence[ChatMessage]
    ) -> AssistantResponse:
        """経路順にProviderを試し、最初に成功した同期応答を返す。"""
        decision = self._router.route(request)
        candidates = _candidate_ids(decision)
        attempts: list[tuple[str, str]] = []
        last_error: Exception | None = None

        for provider_id in candidates:
            try:
                entry = self._registry.get(provider_id)
            except UnknownProviderError as exc:
                attempts.append((provider_id, f"{type(exc).__name__}: {exc}"))
                last_error = exc
                continue

            if not entry.local:
                attempts.append(
                    (provider_id, "non-local provider rejected for prebuilt messages")
                )
                continue
            if not entry.provider.is_available():
                attempts.append((provider_id, "unavailable"))
                continue

            started_at = self._clock()
            try:
                text = entry.provider.generate(
                    [dict(message) for message in messages],
                    **self._options.as_generate_kwargs(),
                )
            except LLMProviderError as exc:
                attempts.append((provider_id, f"{type(exc).__name__}: {exc}"))
                last_error = exc
                continue

            latency_ms = int(max(0.0, self._clock() - started_at) * 1000)
            return AssistantResponse(
                text=text,
                route=_actual_route(decision, entry, candidates, attempts),
                latency_ms=latency_ms,
                stats=dict(entry.provider.last_stats),
            )

        _raise_generation_error(attempts, last_error)

    def generate_stream(
        self, request: AssistantRequest, messages: Sequence[ChatMessage]
    ) -> "StreamResult":
        """経路を即時決定し、Provider呼び出しを反復時に始めるstreamを返す。

        最初のtokenを返す前の ``LLMProviderError`` だけfallbackする。tokenを1件以上
        返した後の同例外は、部分出力の重複を避けるためそのまま呼び出し元へ伝播する。
        """
        decision = self._router.route(request)
        return StreamResult(
            registry=self._registry,
            decision=decision,
            messages=[dict(message) for message in messages],
            options=self._options,
            clock=self._clock,
        )


class StreamResult(Iterable[str]):
    """一度だけ反復でき、完了後に統合応答を公開する遅延stream。"""

    def __init__(
        self,
        *,
        registry: ProviderRegistry,
        decision: RouteDecision,
        messages: Sequence[ChatMessage],
        options: GenerationOptions,
        clock: Callable[[], float],
    ) -> None:
        self._registry = registry
        self._decision = decision
        self._messages = [dict(message) for message in messages]
        self._options = options
        self._clock = clock
        self._started = False
        self._response: AssistantResponse | None = None
        self._parts: list[str] = []

    def __iter__(self) -> Iterator[str]:
        """候補Providerからtokenを順に返し、正常終了時に応答を確定する。"""
        if self._started:
            raise AssistantError("stream already consumed")
        self._started = True

        candidates = _candidate_ids(self._decision)
        attempts: list[tuple[str, str]] = []
        last_error: Exception | None = None
        started_at: float | None = None

        for provider_id in candidates:
            try:
                entry = self._registry.get(provider_id)
            except UnknownProviderError as exc:
                attempts.append((provider_id, f"{type(exc).__name__}: {exc}"))
                last_error = exc
                continue

            if not entry.local:
                attempts.append(
                    (provider_id, "non-local provider rejected for prebuilt messages")
                )
                continue
            if not entry.provider.is_available():
                attempts.append((provider_id, "unavailable"))
                continue

            if started_at is None:
                started_at = self._clock()
            emitted = False
            chunks: Iterable[str] | None = None
            try:
                chunks = entry.provider.generate_stream(
                    [dict(message) for message in self._messages],
                    **self._options.as_stream_kwargs(),
                )
                for chunk in chunks:
                    emitted = True
                    self._parts.append(chunk)
                    yield chunk
            except LLMProviderError as exc:
                if emitted:
                    raise
                attempts.append((provider_id, f"{type(exc).__name__}: {exc}"))
                last_error = exc
                continue
            finally:
                close = getattr(chunks, "close", None)
                if callable(close):
                    close()

            latency_ms = int(max(0.0, self._clock() - started_at) * 1000)
            self._response = AssistantResponse(
                text="".join(self._parts),
                route=_actual_route(self._decision, entry, candidates, attempts),
                latency_ms=latency_ms,
                stats=dict(entry.provider.last_stats),
            )
            return

        _raise_generation_error(attempts, last_error)

    @property
    def response(self) -> AssistantResponse:
        """完了済みstreamの統合応答を返す。"""
        if self._response is None:
            raise AssistantError("stream not finished")
        return self._response

    @property
    def text(self) -> str:
        """完了済みstreamの結合本文を返す。"""
        return self.response.text
