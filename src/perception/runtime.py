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
    """収集状態の要約。生データ・サンプル・イベントは含まない。

    stop_failed / stop_failed_at は停止のタイムアウトという固定の失敗状態を表す。
    例外オブジェクトやその内容は公開しない (エラー種別名とカウンタのみ)。
    """

    running: bool
    last_state: CompanionState | None = None
    last_update_at: float | None = None
    failure_count: int = 0
    consecutive_failures: int = 0
    last_error_type: str | None = None
    stop_failed: bool = False
    stop_failed_at: float | None = None


class ActivityRuntime:
    """ActivitySource を定期的に回して最新状態だけを公開する収集ランタイム。

    - コンストラクタは収集を開始しない。start() までスレッドは動かない。
    - start() / stop() は冪等。start() は起動できた場合のみ True を返し、
      生成/start 例外や即死は False を返す。stop() は注入された threading.Event で
      即時解除する。
    - stop() はスレッド所有権を終了確認まで保持する。join がタイムアウトしても
      停止済みとは報告せず、status.stop_failed で固定の失敗状態を公開する。
      生存スレッドがある間は start() が再起動を拒否し、複製コレクタを防ぐ。
      前スレッドの終了を確認できた場合のみリソースを回収して再起動を許可する。
    - collect_once() は決定論的テスト用に 1 サイクルだけ実行する。
       source 呼び出し中はロックを保持しないため、ブロックする source でも
       stop() が安全に join できる。
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
        self._stop_failed = False
        self._stop_failed_at: float | None = None

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
                stop_failed=self._stop_failed,
                stop_failed_at=self._stop_failed_at,
            )

    def start(self) -> bool:
        """収集スレッドを起動し、起動できたかどうかを真偽で返す。

        - 生存中のスレッドがあれば何もしない (冪等、False)。
        - 前回 stop がタイムアウトしてスレッドが生存中の間は再起動を拒否し、
          複製コレクタを作らない (False)。
        - 前スレッドが終了を確認できた (is_alive() が False になった) 場合のみ
          所有権と停止失敗状態を回収して新しいスレッドを起動する。
        - スレッド生成 (factory) や start の例外は捕捉して漏らさない。start が
          例外を投げてもスレッドが生存した場合 (部分起動) は True を返し、生存
          スレッドの所有権を保持して二重起動を防ぐ。生存していなければ所有権を
          回収して False を返す。
        - start 直後にスレッドが生存していない (即死) 場合は成功とせず False を返し、
          所有権を回収する。
        - False を返すときは生存スレッドを保持しない (False は常に非生存を意味する)。
        """
        with self._lock:
            thread = self._thread
            if thread is not None and thread.is_alive():
                return False
            if thread is not None:
                # 終了を確認: 前スレッドのリソースを回収して再起動を許可する。
                self._thread = None
                self._stop_failed = False
                self._stop_failed_at = None
            self._stop_event.clear()
            try:
                new_thread = threading.Thread(
                    target=self._run,
                    name="activity-runtime",
                    daemon=True,
                )
            except Exception:
                return False
            self._thread = new_thread
        try:
            new_thread.start()
        except Exception:
            with self._lock:
                if self._is_live(new_thread):
                    return True
                self._thread = None
                return False
        with self._lock:
            if new_thread.is_alive():
                return True
            self._thread = None
            return False

    def stop(self, timeout: float = 5.0) -> None:
        """収集を停止する。冪等。

        - スレッドの所有権は終了を確認するまで保持する。join 後に is_alive() が
          False になるまで _thread は解放しない (生存スレッドを停止済みと報告しない)。
        - join がタイムアウトした場合、生存スレッドを残したまま固定の失敗状態
          (stop_failed / stop_failed_at) を公開する。この間 start() は再起動を
          拒否し、複製コレクタを防ぐ。例外の内容は公開しない。
        - 生存スレッドが後から自ら終了した場合は、次の stop() / start() が
          所有権を回収して通常運用へ戻れる。
        - start に失敗して未起動のまま保持されたワーカーは join せず、
          所有権だけ回収する。
        """
        self._stop_event.set()
        with self._lock:
            thread = self._thread
        if thread is None:
            return
        if thread.is_alive():
            thread.join(timeout=timeout)
            with self._lock:
                if thread.is_alive():
                    self._stop_failed = True
                    self._stop_failed_at = time.monotonic()
                    return
        with self._lock:
            if self._thread is thread:
                self._thread = None
            self._stop_failed = False
            self._stop_failed_at = None

    def collect_once(self) -> CompanionState:
        """1 サイクルだけ収集し、現在の CompanionState を返す。

        source が失敗しても例外を上げず、直前の状態を返して status に記録する。
        source 呼び出し中はロックを保持しないため、ブロックする source でも
        stop() が安全にスレッドを join できる。
        """
        try:
            sample_data: ActivitySample = self._source.sample()
            event: PerceptionEvent | None = self._collector.update(sample_data)
        except Exception as exc:
            with self._lock:
                self._total_failures += 1
                self._consecutive_failures += 1
                self._last_error_type = type(exc).__name__
                state = self._aggregator.state
                self._latest = state
        else:
            with self._lock:
                if event is not None:
                    self._aggregator.apply(event)
                state = self._aggregator.state
                self._latest = state
                self._last_update_at = state.updated_at
                self._consecutive_failures = 0
        self._notify(state)
        return state

    @staticmethod
    def _is_live(thread: threading.Thread) -> bool:
        try:
            return thread.is_alive()
        except Exception:
            return False

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

    def _notify(self, state: CompanionState) -> None:
        if self._callback is None:
            return
        try:
            self._callback(state)
        except Exception:
            pass
