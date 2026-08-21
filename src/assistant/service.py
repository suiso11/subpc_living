"""経路選択とProvider fallbackを統合するAssistantサービス。"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
import threading
import time
from typing import TYPE_CHECKING
from uuid import uuid4

from src.assistant.run_logger import RunLogger
from src.assistant.contracts import (
    AssistantError,
    AssistantGenerationError,
    AssistantRequest,
    AssistantResponse,
)
from src.context.contracts import ContextBlock
from src.llm.contracts import ChatMessage, GenerationOptions
from src.llm.errors import LLMProviderError
from src.llm.registry import ProviderEntry, ProviderRegistry, UnknownProviderError
from src.llm.routing.contracts import ModelRouter, RouteDecision

if TYPE_CHECKING:
    from src.assistant.cloud_service import CloudRouteBridge
    from src.llm.approval import CloudPreview


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


def _generation_error(
    attempts: list[tuple[str, str]], last_error: Exception | None
) -> AssistantGenerationError:
    error = AssistantGenerationError(tuple(attempts))
    if last_error is not None:
        error.__cause__ = last_error
    return error


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
        run_logger: RunLogger | None = None,
        request_id_factory: Callable[[], str] = lambda: str(uuid4()),
    ) -> None:
        self._registry = registry
        self._router = router
        self._options = options or GenerationOptions()
        self._clock = clock
        self._run_logger = run_logger
        self._request_id_factory = request_id_factory
        self._cloud_bridge: CloudRouteBridge | None = None

    def _request_id(self, request: AssistantRequest) -> str:
        return request.request_id or self._request_id_factory()

    def set_cloud_bridge(self, bridge: "CloudRouteBridge") -> None:
        """クラウドブリッジを設定する。"""
        self._cloud_bridge = bridge

    def respond(
        self,
        request: AssistantRequest,
        blocks: Sequence[ContextBlock],
        *,
        base_system: str = "",
        options: GenerationOptions | None = None,
    ) -> tuple[AssistantResponse, CloudPreview | None]:
        """blocks からメッセージを構築して生成する。

        クラウドブリッジが設定済みで privacy=cloud_allowed + allow_cloud の場合は
        Bridge へ委譲する。それ以外はローカルで生成する。
        """
        if (
            self._cloud_bridge is not None
            and request.privacy == "cloud_allowed"
            and request.allow_cloud
        ):
            preview = self._cloud_bridge.preview(request, blocks)
            response = self._cloud_bridge.send(request, blocks, options=options)
            return response, preview

        from src.context.builder import ContextBuilder

        builder = ContextBuilder(base_system)
        messages = builder.build_messages(
            blocks,
            privacy=request.privacy,
            target_local=True,
        )
        response = self.generate(request, messages)
        return response, None

    def _record_route(
        self, request_id: str, request: AssistantRequest, decision: RouteDecision
    ) -> None:
        if self._run_logger is None:
            return
        try:
            self._run_logger.record_route(
                request_id,
                channel=request.channel,
                profile=request.profile,
                decision=decision,
            )
        except Exception:
            pass

    def _record_run(
        self,
        request_id: str,
        request: AssistantRequest,
        route: RouteDecision | None,
        latency_ms: int,
        success: bool,
        error: str | None,
    ) -> None:
        if self._run_logger is None:
            return
        try:
            self._run_logger.record_run(
                request_id,
                channel=request.channel,
                profile=request.profile,
                route=route,
                latency_ms=latency_ms,
                success=success,
                error=error,
            )
        except Exception:
            pass

    def generate(
        self, request: AssistantRequest, messages: Sequence[ChatMessage]
    ) -> AssistantResponse:
        """経路順にProviderを試し、最初に成功した同期応答を返す。"""
        request_id = self._request_id(request)
        try:
            decision = self._router.route(request)
        except Exception as exc:
            self._record_run(request_id, request, None, 0, False, type(exc).__name__)
            raise
        self._record_route(request_id, request, decision)
        candidates = _candidate_ids(decision)
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
            route = _actual_route(decision, entry, candidates, attempts)
            self._record_run(request_id, request, route, latency_ms, True, None)
            return AssistantResponse(
                text=text,
                route=route,
                latency_ms=latency_ms,
                stats=dict(entry.provider.last_stats),
            )

        error = _generation_error(attempts, last_error)
        latency_ms = (
            int(max(0.0, self._clock() - started_at) * 1000)
            if started_at is not None
            else 0
        )
        self._record_run(request_id, request, None, latency_ms, False, type(error).__name__)
        raise error

    def generate_stream(
        self, request: AssistantRequest, messages: Sequence[ChatMessage]
    ) -> "StreamResult":
        """経路を即時決定し、Provider呼び出しを反復時に始めるstreamを返す。

        最初のtokenを返す前の ``LLMProviderError`` だけfallbackする。tokenを1件以上
        返した後の同例外は、部分出力の重複を避けるためそのまま呼び出し元へ伝播する。
        """
        request_id = self._request_id(request)
        try:
            decision = self._router.route(request)
        except Exception as exc:
            self._record_run(request_id, request, None, 0, False, type(exc).__name__)
            raise
        self._record_route(request_id, request, decision)
        return StreamResult(
            registry=self._registry,
            decision=decision,
            messages=[dict(message) for message in messages],
            options=self._options,
            clock=self._clock,
            run_logger=self._run_logger,
            request_id=request_id,
            channel=request.channel,
            profile=request.profile,
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
        run_logger: RunLogger | None = None,
        request_id: str = "",
        channel: str = "",
        profile: str = "",
    ) -> None:
        self._registry = registry
        self._decision = decision
        self._messages = [dict(message) for message in messages]
        self._options = options
        self._clock = clock
        self._run_logger = run_logger
        self._request_id = request_id
        self._channel = channel
        self._profile = profile
        self._started = False
        self._closed = False
        self._state_lock = threading.Lock()
        self._run_recorded = False
        self._iterator: Iterator[str] | None = None
        self._response: AssistantResponse | None = None
        self._parts: list[str] = []
        self._started_at: float | None = None

    def _record_run(
        self,
        route: RouteDecision | None,
        latency_ms: int,
        success: bool,
        error: str | None,
    ) -> None:
        if self._run_logger is None:
            return
        with self._state_lock:
            if self._run_recorded:
                return
            self._run_recorded = True
        try:
            self._run_logger.record_run(
                self._request_id,
                channel=self._channel,
                profile=self._profile,
                route=route,
                latency_ms=latency_ms,
                success=success,
                error=error,
            )
        except Exception:
            pass

    def __iter__(self) -> Iterator[str]:
        """候補Providerからtokenを順に返し、正常終了時に応答を確定する。"""
        with self._state_lock:
            if self._started:
                raise AssistantError("stream already consumed")
            self._started = True
            self._iterator = self._iterate()
            return self._iterator

    def _iterate(self) -> Iterator[str]:
        """選択済み候補を反復する内部generator。"""

        candidates = _candidate_ids(self._decision)
        attempts: list[tuple[str, str]] = []
        last_error: Exception | None = None

        def latency_ms() -> int:
            if self._started_at is None:
                return 0
            return int(max(0.0, self._clock() - self._started_at) * 1000)

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

            if self._started_at is None:
                self._started_at = self._clock()
            emitted = False
            chunks: Iterable[str] | None = None
            try:
                chunks = entry.provider.generate_stream(
                    [dict(message) for message in self._messages],
                    **self._options.as_stream_kwargs(),
                )
                for chunk in chunks:
                    with self._state_lock:
                        if self._closed:
                            return
                    emitted = True
                    self._parts.append(chunk)
                    yield chunk
            except LLMProviderError as exc:
                if emitted:
                    self._record_run(
                        None,
                        latency_ms(),
                        False,
                        type(exc).__name__,
                    )
                    raise
                attempts.append((provider_id, f"{type(exc).__name__}: {exc}"))
                last_error = exc
                continue
            finally:
                close = getattr(chunks, "close", None)
                if callable(close):
                    close()

            latency = latency_ms()
            with self._state_lock:
                if self._closed:
                    return
                self._response = AssistantResponse(
                    text="".join(self._parts),
                    route=_actual_route(
                        self._decision, entry, candidates, attempts
                    ),
                    latency_ms=latency,
                    stats=dict(entry.provider.last_stats),
                )
            self._record_run(self._response.route, latency, True, None)
            return

        error = _generation_error(attempts, last_error)
        self._record_run(None, latency_ms(), False, type(error).__name__)
        raise error

    def close(self) -> None:
        """内部generatorを閉じ、以後の反復を停止する。"""
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            self._started = True
            iterator = self._iterator

        if iterator is not None:
            try:
                iterator.close()
            except ValueError:
                # 別threadで実行中のgeneratorは閉じられないため停止フラグに委ねる。
                pass

        if self._response is None:
            latency_ms = 0
            if self._started_at is not None:
                latency_ms = int(max(0.0, self._clock() - self._started_at) * 1000)
            self._record_run(None, latency_ms, False, "cancelled")

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
