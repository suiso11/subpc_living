"""スレッドセーフでオプトインのアクティビティ収集ランタイム。

ActivitySource -> ActivityEventCollector -> StateAggregator を一定間隔で回し、
最新の CompanionState と status (エラー種別名と失敗回数だけ) を公開する。
ActivitySample や PerceptionEvent はサイクルごとに破棄し、生データは保持しない。
OS 収集は source の注入でのみ有効になる (オプトイン)。
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.companion.contracts import CompanionState
from src.companion.state import StateAggregator
from src.perception.activity import ActivityEventCollector

if TYPE_CHECKING:
    from src.companion.contracts import PerceptionEvent
    from src.perception.activity import ActivitySample
    from src.perception.sources import ActivitySource

StateCallback = Callable[[CompanionState], None]


@dataclass(frozen=True)
class ActivityRuntimeStatus:
    """収集状態の要約。生データ・サンプル・イベントは含まない。"""

    running: bool
    last_state: CompanionState | None = None
    last_update_at: float | None = None
    failure_count: int = 0
    consecutive_failures: int = 0
    last_error_type: str | None = None


class ActivityRuntime:
    """ActivitySource を定期的に回して最新状態だけを公開する収集ランタイム。

    - コンストラクタは収集を開始しない。start() までスレッドは動かない。
    - start() / stop() は冪等。stop() は注入された threading.Event で即時解除する。
    - collect_once() は決定論的テスト用に 1 サイクルだけ実行する。
    - source の失敗はループを殺さず、メッセージやサンプル内容を記録しない。
      status はエラー種別名と失敗回数だけを公開する。
    - サイクルごとに ActivitySample / PerceptionEvent は保持せず、
      CompanionState と集計カウンタ・タイムスタンプだけを残す。
    - callback は CompanionState のみを受け取り、失敗しても収集を止めない。
    """

    def __init__(
        self,
        source: ActivitySource,
        *,
        poll_interval: float = 5.0,
        idle_threshold: float = 300,
        away_threshold: float = 1800,
        min_confidence: float = 0.5,
        source_name: str = "pc_activity",
        stop_event: threading.Event | None = None,
        callback: StateCallback | None = None,
    ) -> None:
        if (
            not isinstance(poll_interval, (int, float))
            or isinstance(poll_interval, bool)
            or poll_interval <= 0
        ):
            raise ValueError(
                f"poll_interval must be a positive number, got {poll_interval!r}"
            )
        if callback is not None and not callable(callback):
            raise TypeError(
                f"callback must be callable or None, got {type(callback).__name__}"
            )
        self._source = source
        self._collector = ActivityEventCollector(
            idle_threshold=idle_threshold,
            away_threshold=away_threshold,
            source=source_name,
        )
        self._aggregator = StateAggregator(min_confidence=min_confidence)
        self._poll_interval = poll_interval
        self._stop_event = stop_event if stop_event is not None else threading.Event()
        self._callback = callback
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._latest: CompanionState | None = None
        self._last_update_at: float | None = None
        self._total_failures = 0
        self._consecutive_failures = 0
        self._last_error_type: str | None = None

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    @property
    def state(self) -> CompanionState | None:
        with self._lock:
            return self._latest

    @property
    def status(self) -> ActivityRuntimeStatus:
        with self._lock:
            return ActivityRuntimeStatus(
                running=self._thread is not None and self._thread.is_alive(),
                last_state=self._latest,
                last_update_at=self._last_update_at,
                failure_count=self._total_failures,
                consecutive_failures=self._consecutive_failures,
                last_error_type=self._last_error_type,
            )

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            thread = threading.Thread(
                target=self._run,
                name="activity-runtime",
                daemon=True,
            )
            self._thread = thread
        thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        with self._lock:
            thread = self._thread
            self._thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)

    def collect_once(self) -> CompanionState:
        """1 サイクルだけ収集し、現在の CompanionState を返す。

        source が失敗しても例外を上げず、直前の状態を返して status に記録する。
        """
        with self._lock:
            state = self._collect_locked()
        self._notify(state)
        return state

    def _run(self) -> None:
        while True:
            if self._stop_event.is_set():
                return
            try:
                self.collect_once()
            except Exception:
                pass
            if self._stop_event.wait(self._poll_interval):
                return

    def _collect_locked(self) -> CompanionState:
        try:
            sample: ActivitySample = self._source.sample()
            event: PerceptionEvent | None = self._collector.update(sample)
        except Exception as exc:
            self._total_failures += 1
            self._consecutive_failures += 1
            self._last_error_type = type(exc).__name__
            self._latest = self._aggregator.state
            return self._latest
        if event is not None:
            self._aggregator.apply(event)
        self._consecutive_failures = 0
        state = self._aggregator.state
        self._latest = state
        self._last_update_at = state.updated_at
        return state

    def _notify(self, state: CompanionState) -> None:
        if self._callback is None:
            return
        try:
            self._callback(state)
        except Exception:
            pass
