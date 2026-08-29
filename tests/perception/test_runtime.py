from __future__ import annotations

import threading
import time
import unittest
from collections import deque
from unittest import mock

from src.companion.contracts import CompanionState, PerceptionEvent
from src.perception import ActivityRuntime, ActivityRuntimeStatus
from src.perception.activity import ActivitySample
from src.perception.runtime import ActivityRuntime as ModActivityRuntime


class _FakeSource:
    """キュー駆動の ActivitySource。sample か例外を順に返す。"""

    def __init__(self) -> None:
        self._queue: deque = deque()
        self.calls = 0

    def push(self, sample: ActivitySample) -> None:
        self._queue.append(sample)

    def push_error(self, exc: BaseException) -> None:
        self._queue.append(exc)

    def sample(self) -> ActivitySample:
        self.calls += 1
        item = self._queue.popleft()
        if isinstance(item, BaseException):
            raise item
        return item


class _FlagSource:
    """failing フラグを切り替えられる ActivitySource。スレッドテスト用。"""

    def __init__(self) -> None:
        self.failing = False
        self.calls = 0

    def sample(self) -> ActivitySample:
        self.calls += 1
        if self.failing:
            raise OSError("transient source outage")
        return ActivitySample(
            timestamp=float(self.calls),
            idle_seconds=0.0,
            app_category="work",
        )


class _BlockingSource:
    """sample() が release イベントで解放されるまでブロックする ActivitySource。

    停止タイムアウトの決定論的テスト用。解放されなければスレッドは止まらない。
    """

    def __init__(self) -> None:
        self.release = threading.Event()
        self.calls = 0

    def sample(self) -> ActivitySample:
        self.calls += 1
        if not self.release.wait(30.0):
            raise TimeoutError("block never released")
        return ActivitySample(
            timestamp=float(self.calls),
            idle_seconds=0.0,
            app_category="work",
        )


def sample(
    timestamp: float,
    idle_seconds: float,
    app_category: str = "work",
) -> ActivitySample:
    return ActivitySample(
        timestamp=timestamp,
        idle_seconds=idle_seconds,
        app_category=app_category,
    )


def _wait_until(predicate, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class ActivityRuntimePipelineTest(unittest.TestCase):
    def test_pipeline_transitions(self) -> None:
        src = _FakeSource()
        rt = ActivityRuntime(src, idle_threshold=300, away_threshold=1800)
        src.push(sample(1.0, 0.0))
        src.push(sample(2.0, 400.0))
        src.push(sample(3.0, 2000.0))
        src.push(sample(4.0, 10.0))
        self.assertEqual(rt.collect_once().activity_mode, "focused")
        self.assertEqual(rt.collect_once().activity_mode, "idle")
        self.assertEqual(rt.collect_once().activity_mode, "away")
        self.assertEqual(rt.collect_once().activity_mode, "focused")
        self.assertEqual(rt.state.activity_mode, "focused")
        self.assertEqual(rt.state.focused_since, 4.0)
        self.assertEqual(rt.status.failure_count, 0)

    def test_unchanged_state_samples_do_not_emit(self) -> None:
        src = _FakeSource()
        rt = ActivityRuntime(src)
        src.push(sample(1.0, 10.0))
        src.push(sample(2.0, 20.0))
        src.push(sample(3.0, 30.0))
        rt.collect_once()
        rt.collect_once()
        rt.collect_once()
        self.assertEqual(rt.state.activity_mode, "focused")
        self.assertEqual(rt.state.updated_at, 1.0)
        self.assertEqual(rt.status.last_update_at, 1.0)
        self.assertEqual(rt.status.failure_count, 0)


class _CanaryCustomBoom(Exception):
    """allowlist に無いカスタム例外クラス (canary)。

    runtime は内部で型名を保持する (ログ/内部用途)。外部への写像は
    bootstrap の allowlist mapper が担う。
    """


class ActivityRuntimeSourceErrorTest(unittest.TestCase):
    def test_source_error_records_type_and_keeps_state(self) -> None:
        src = _FakeSource()
        rt = ActivityRuntime(src)
        src.push(sample(1.0, 0.0))
        src.push_error(OSError("boom raw detail"))
        rt.collect_once()
        self.assertEqual(rt.state.activity_mode, "focused")
        rt.collect_once()
        self.assertEqual(rt.status.failure_count, 1)
        self.assertEqual(rt.status.last_error_type, "OSError")
        self.assertEqual(rt.status.consecutive_failures, 1)
        self.assertEqual(rt.state.activity_mode, "focused")

    def test_canary_custom_exception_stored_as_type_name_internally(self) -> None:
        src = _FakeSource()
        rt = ActivityRuntime(src)
        src.push_error(_CanaryCustomBoom("secret canary detail"))
        rt.collect_once()
        # 内部 (runtime status) は型名を保持する。外部 payload の写像は
        # bootstrap の allowlist mapper が internal_error へ落とす。
        self.assertEqual(rt.status.last_error_type, "_CanaryCustomBoom")
        self.assertNotIn("secret canary detail", str(vars(rt)))
        from src.perception.bootstrap import sensor_error_code_from_name

        self.assertEqual(
            sensor_error_code_from_name(rt.status.last_error_type), "internal_error"
        )

    def test_source_error_message_is_not_retained(self) -> None:
        src = _FakeSource()
        rt = ActivityRuntime(src)
        src.push_error(OSError("secret raw detail"))
        rt.collect_once()
        self.assertEqual(rt.status.last_error_type, "OSError")
        for attr, value in vars(rt).items():
            if attr == "_source":
                continue
            self.assertNotIsInstance(value, OSError)

    def test_source_recovers_after_error(self) -> None:
        src = _FakeSource()
        rt = ActivityRuntime(src)
        src.push(sample(1.0, 0.0))
        src.push_error(RuntimeError("down"))
        src.push(sample(2.0, 400.0))
        rt.collect_once()
        rt.collect_once()
        rt.collect_once()
        self.assertEqual(rt.state.activity_mode, "idle")
        self.assertEqual(rt.status.failure_count, 1)
        self.assertEqual(rt.status.consecutive_failures, 0)

    def test_collect_once_returns_state_even_on_failure(self) -> None:
        src = _FakeSource()
        rt = ActivityRuntime(src)
        src.push_error(ValueError("x"))
        state = rt.collect_once()
        self.assertIsInstance(state, CompanionState)
        self.assertEqual(rt.status.failure_count, 1)

    def test_first_cycle_failure_exposes_consistent_initial_state(self) -> None:
        src = _FakeSource()
        rt = ActivityRuntime(src)
        src.push_error(OSError("source down at boot"))
        returned = rt.collect_once()
        self.assertIs(returned, rt.state)
        self.assertIs(returned, rt.status.last_state)
        self.assertEqual(returned.activity_mode, "idle")
        self.assertIsNone(rt.status.last_update_at)
        self.assertEqual(rt.status.failure_count, 1)
        self.assertEqual(rt.status.consecutive_failures, 1)
        self.assertEqual(rt.status.last_error_type, "OSError")


class ActivityRuntimeCallbackTest(unittest.TestCase):
    def test_callback_receives_state_on_every_cycle(self) -> None:
        received: list[CompanionState] = []
        src = _FakeSource()
        rt = ActivityRuntime(src, callback=received.append)
        src.push(sample(1.0, 0.0))
        src.push(sample(2.0, 20.0))
        rt.collect_once()
        rt.collect_once()
        self.assertEqual(len(received), 2)
        self.assertTrue(all(isinstance(s, CompanionState) for s in received))
        self.assertEqual(received[0].activity_mode, "focused")

    def test_callback_failure_does_not_kill_collection(self) -> None:
        seen: list[str] = []

        def cb(state: CompanionState) -> None:
            seen.append(state.activity_mode)
            if len(seen) == 1:
                raise RuntimeError("callback exploded")

        src = _FakeSource()
        rt = ActivityRuntime(src, callback=cb)
        src.push(sample(1.0, 0.0))
        src.push(sample(2.0, 400.0))
        rt.collect_once()
        rt.collect_once()
        self.assertEqual(seen, ["focused", "idle"])
        self.assertEqual(rt.state.activity_mode, "idle")
        self.assertEqual(rt.status.failure_count, 0)


class ActivityRuntimeLifecycleTest(unittest.TestCase):
    def test_construction_does_not_start_collection(self) -> None:
        rt = ActivityRuntime(_FakeSource())
        self.assertFalse(rt.is_running)
        self.assertIsNone(rt.state)
        self.assertFalse(rt.status.running)

    def test_start_stop_idempotent_and_restartable(self) -> None:
        rt = ActivityRuntime(_FakeSource())
        rt.start()
        self.assertTrue(rt.is_running)
        rt.start()
        self.assertTrue(rt.is_running)
        rt.stop(timeout=1.0)
        self.assertFalse(rt.is_running)
        rt.stop(timeout=1.0)
        self.assertFalse(rt.is_running)
        rt.start()
        self.assertTrue(rt.is_running)
        rt.stop(timeout=1.0)
        self.assertFalse(rt.is_running)

    def test_stop_before_start_is_noop(self) -> None:
        rt = ActivityRuntime(_FakeSource())
        rt.stop(timeout=1.0)
        self.assertFalse(rt.is_running)

    def test_prompt_stop_with_injected_event(self) -> None:
        src = _FakeSource()
        stop_event = threading.Event()
        rt = ActivityRuntime(src, poll_interval=3600.0, stop_event=stop_event)
        started = time.monotonic()
        rt.start()
        self.assertTrue(rt.is_running)
        stop_event.set()
        rt.stop(timeout=2.0)
        self.assertFalse(rt.is_running)
        self.assertLess(time.monotonic() - started, 5.0)

    def test_loop_survives_source_failure_and_recovers(self) -> None:
        src = _FlagSource()
        stop_event = threading.Event()
        rt = ActivityRuntime(src, poll_interval=0.01, stop_event=stop_event)
        src.failing = True
        rt.start()
        self.assertTrue(_wait_until(lambda: rt.status.failure_count >= 1))
        self.assertTrue(rt.is_running)
        src.failing = False
        ok = _wait_until(
            lambda: rt.status.consecutive_failures == 0 and rt.state is not None
        )
        self.assertTrue(ok)
        self.assertEqual(rt.state.activity_mode, "focused")
        stop_event.set()
        rt.stop(timeout=1.0)
        self.assertFalse(rt.is_running)


class ActivityRuntimeStatusTest(unittest.TestCase):
    def test_status_shape(self) -> None:
        rt = ActivityRuntime(_FakeSource())
        status = rt.status
        self.assertEqual(
            set(ActivityRuntimeStatus.__dataclass_fields__),
            {
                "running",
                "last_state",
                "last_update_at",
                "failure_count",
                "consecutive_failures",
                "last_error_type",
                "stop_failed",
                "stop_failed_at",
            },
        )
        self.assertFalse(status.running)
        self.assertIsNone(status.last_state)
        self.assertIsNone(status.last_update_at)
        self.assertEqual(status.failure_count, 0)
        self.assertEqual(status.consecutive_failures, 0)
        self.assertIsNone(status.last_error_type)
        self.assertFalse(status.stop_failed)
        self.assertIsNone(status.stop_failed_at)

    def test_status_tracks_running_and_updates(self) -> None:
        src = _FakeSource()
        rt = ActivityRuntime(src)
        src.push(sample(1.0, 0.0))
        rt.collect_once()
        self.assertTrue(rt.status.running is False)
        self.assertEqual(rt.status.last_update_at, 1.0)
        self.assertEqual(rt.status.last_state.activity_mode, "focused")


class ActivityRuntimePrivacyTest(unittest.TestCase):
    def test_no_raw_retention_after_cycles(self) -> None:
        src = _FakeSource()
        rt = ActivityRuntime(src)
        src.push(sample(1.0, 0.0))
        src.push(sample(2.0, 400.0))
        rt.collect_once()
        rt.collect_once()
        self.assertIsInstance(rt.state, CompanionState)
        for attr, value in vars(rt).items():
            if attr in ("_source", "_collector", "_aggregator"):
                continue
            self.assertNotIsInstance(
                value, (ActivitySample, PerceptionEvent),
                msg=f"raw object retained in {attr!r}",
            )

    def test_status_has_no_raw_fields(self) -> None:
        fields = set(ActivityRuntimeStatus.__dataclass_fields__)
        for forbidden in ("sample", "event", "raw", "text", "path", "pid", "window_title"):
            self.assertNotIn(forbidden, fields)

    def test_collector_retains_only_last_state(self) -> None:
        src = _FakeSource()
        rt = ActivityRuntime(src)
        src.push(sample(1.0, 0.0))
        rt.collect_once()
        self.assertEqual(vars(rt._collector)["_last_state"], "focused")
        self.assertNotIn("_last_sample", vars(rt._collector))


class ActivityRuntimeValidationTest(unittest.TestCase):
    def test_source_is_required(self) -> None:
        with self.assertRaises(TypeError):
            ActivityRuntime()  # type: ignore[call-arg]

    def test_poll_interval_must_be_positive(self) -> None:
        for bad in (0, -1, "5"):
            with self.assertRaises(ValueError):
                ActivityRuntime(_FakeSource(), poll_interval=bad)

    def test_thresholds_validated(self) -> None:
        with self.assertRaises(ValueError):
            ActivityRuntime(_FakeSource(), idle_threshold=-1)
        with self.assertRaises(ValueError):
            ActivityRuntime(_FakeSource(), away_threshold=10, idle_threshold=20)

    def test_invalid_callback_rejected(self) -> None:
        with self.assertRaises(TypeError):
            ActivityRuntime(_FakeSource(), callback="not-callable")


class ActivityRuntimeStopHardeningTest(unittest.TestCase):
    """ブロックする source に対する fail-safe / 冪等な停止の決定論的テスト。"""

    def _stop_blocked(
        self,
        rt: ActivityRuntime,
        release: threading.Event,
    ) -> None:
        """ブロックを解除し、スレッド終了を確認して所有権を解放する後始末。"""
        release.set()
        self.assertTrue(_wait_until(lambda: not rt.is_running, timeout=5.0))
        rt.stop(timeout=1.0)
        self.assertFalse(rt.is_running)
        self.assertFalse(rt.status.stop_failed)

    def test_stop_timeout_keeps_ownership_and_reports_fixed_failure(self) -> None:
        src = _BlockingSource()
        rt = ActivityRuntime(src, poll_interval=0.01, stop_event=threading.Event())
        rt.start()
        first = rt._thread
        self.assertTrue(rt.is_running)
        rt.stop(timeout=0.2)
        # 生存スレッドを停止済みと報告しない (所有権を保持、真実の running を返す)
        self.assertTrue(rt.is_running)
        self.assertTrue(rt.status.running)
        self.assertIs(rt._thread, first)
        # 固定の失敗状態のみ公開し、例外の内容は漏れない
        self.assertTrue(rt.status.stop_failed)
        self.assertIsNotNone(rt.status.stop_failed_at)
        self.assertIsNone(rt.status.last_error_type)
        self._stop_blocked(rt, src.release)

    def test_repeated_stop_is_idempotent_while_thread_live(self) -> None:
        src = _BlockingSource()
        rt = ActivityRuntime(src, poll_interval=0.01, stop_event=threading.Event())
        rt.start()
        first = rt._thread
        rt.stop(timeout=0.2)
        rt.stop(timeout=0.2)
        rt.stop(timeout=0.2)
        self.assertTrue(rt.status.stop_failed)
        self.assertIs(rt._thread, first)
        self.assertTrue(rt.is_running)
        alive = [
            t
            for t in threading.enumerate()
            if t.name == "activity-runtime" and t.is_alive()
        ]
        self.assertEqual(len(alive), 1)
        self._stop_blocked(rt, src.release)

    def test_restart_denied_while_thread_still_live(self) -> None:
        src = _BlockingSource()
        rt = ActivityRuntime(src, poll_interval=0.01, stop_event=threading.Event())
        rt.start()
        first = rt._thread
        rt.stop(timeout=0.2)
        self.assertTrue(rt.status.stop_failed)
        rt.start()
        # 再起動を拒否: 新しいスレッドを生成せず、複製コレクタを作らない
        self.assertIs(rt._thread, first)
        self.assertTrue(rt.is_running)
        alive = [
            t
            for t in threading.enumerate()
            if t.name == "activity-runtime" and t.is_alive()
        ]
        self.assertEqual(len(alive), 1)
        self._stop_blocked(rt, src.release)

    def test_restart_allowed_after_prior_thread_confirmed_dead(self) -> None:
        src = _BlockingSource()
        rt = ActivityRuntime(src, poll_interval=0.01, stop_event=threading.Event())
        rt.start()
        first = rt._thread
        rt.stop(timeout=0.2)
        self.assertTrue(rt.status.stop_failed)
        # ブロック解除 → 前スレッドが自ら終了
        src.release.set()
        self.assertTrue(_wait_until(lambda: not rt.is_running, timeout=5.0))
        # 終了確認後の start() はリソースを回収して再起動できる
        rt.start()
        self.assertIsNot(rt._thread, first)
        self.assertTrue(rt.is_running)
        self.assertFalse(rt.status.stop_failed)
        self.assertIsNone(rt.status.stop_failed_at)
        rt.stop(timeout=1.0)
        self.assertFalse(rt.is_running)

    def test_prompt_stop_normal_path_keeps_clean_state(self) -> None:
        src = _FlagSource()
        stop_event = threading.Event()
        rt = ActivityRuntime(src, poll_interval=0.01, stop_event=stop_event)
        rt.start()
        first = rt._thread
        stop_event.set()
        rt.stop(timeout=1.0)
        self.assertFalse(rt.is_running)
        self.assertFalse(rt.status.stop_failed)
        self.assertIsNone(rt.status.stop_failed_at)
        rt.start()
        self.assertIsNot(rt._thread, first)
        self.assertTrue(rt.is_running)
        rt.stop(timeout=1.0)
        self.assertFalse(rt.is_running)
        self.assertIsNone(rt._thread)


class ActivityRuntimeExportTest(unittest.TestCase):
    def test_package_exports_match_module(self) -> None:
        self.assertIs(ActivityRuntime, ModActivityRuntime)


class _FakeThread:
    """threading.Thread の決定論的代替。

    start 例外・即死・部分起動 (start 例外でも生存) をクラス属性で設定できる。
    made に生成順のインスタンスを残す。
    """

    start_error: BaseException | None = None
    die_immediately = False
    partial_alive = False
    made: list = []

    def __init__(
        self,
        *,
        target: object = None,
        name: str | None = None,
        daemon: bool | None = None,
        **kwargs: object,
    ) -> None:
        self.target = target
        self.name = name
        self.daemon = daemon
        self._started = False
        self._alive = False
        _FakeThread.made.append(self)

    def start(self) -> None:
        if _FakeThread.start_error is not None:
            if _FakeThread.partial_alive:
                self._alive = True
            raise _FakeThread.start_error
        self._started = True
        self._alive = not _FakeThread.die_immediately

    def is_alive(self) -> bool:
        return self._alive

    def join(self, timeout: float | None = None) -> None:
        if self._started:
            self._alive = False

    def kill(self) -> None:
        self._alive = False


class ActivityRuntimeStartHardeningTest(unittest.TestCase):
    """start の factory/start 例外・即死・所有権保持の決定論的テスト。"""

    def setUp(self) -> None:
        _FakeThread.made.clear()
        _FakeThread.start_error = None
        _FakeThread.die_immediately = False
        _FakeThread.partial_alive = False

    def tearDown(self) -> None:
        _FakeThread.start_error = None
        _FakeThread.die_immediately = False
        _FakeThread.partial_alive = False

    def test_factory_failure_returns_false_and_creates_no_worker(self) -> None:
        rt = ActivityRuntime(_FakeSource())
        with mock.patch.object(
            threading,
            "Thread",
            side_effect=RuntimeError("thread factory boom"),
        ):
            self.assertFalse(rt.start())
        self.assertIsNone(rt._thread)
        self.assertFalse(rt.is_running)
        self.assertFalse(rt.status.running)
        rt.stop(timeout=0.1)

    def test_start_failure_not_live_returns_false_and_clears_worker(self) -> None:
        _FakeThread.start_error = RuntimeError("start boom")
        rt = ActivityRuntime(_FakeSource())
        with mock.patch.object(threading, "Thread", new=_FakeThread):
            self.assertFalse(rt.start())
        self.assertEqual(len(_FakeThread.made), 1)
        self.assertIsNone(rt._thread)
        self.assertFalse(rt.is_running)
        self.assertFalse(rt.status.running)
        rt.stop(timeout=0.1)
        self.assertIsNone(rt._thread)
        self.assertFalse(rt.status.stop_failed)

    def test_start_failure_partially_live_returns_true_and_retains_worker(
        self,
    ) -> None:
        _FakeThread.start_error = RuntimeError("start boom")
        _FakeThread.partial_alive = True
        rt = ActivityRuntime(_FakeSource())
        with mock.patch.object(threading, "Thread", new=_FakeThread):
            self.assertTrue(rt.start())
        self.assertEqual(len(_FakeThread.made), 1)
        self.assertIs(rt._thread, _FakeThread.made[0])
        self.assertTrue(rt.is_running)
        self.assertTrue(rt.status.running)
        self.assertFalse(rt.status.stop_failed)
        self.assertIsNone(rt.status.stop_failed_at)
        # 生存スレッドがある限り再起動は拒否される
        with mock.patch.object(threading, "Thread", new=_FakeThread):
            self.assertFalse(rt.start())
        self.assertEqual(len(_FakeThread.made), 1)
        self.assertIs(rt._thread, _FakeThread.made[0])
        # 死亡確認後は所有権が解放され再起動できる
        _FakeThread.made[0].kill()
        _FakeThread.start_error = None
        _FakeThread.partial_alive = False
        with mock.patch.object(threading, "Thread", new=_FakeThread):
            self.assertTrue(rt.start())
        self.assertIsNot(rt._thread, _FakeThread.made[0])
        self.assertIs(rt._thread, _FakeThread.made[1])
        self.assertTrue(rt.is_running)
        rt.stop(timeout=0.1)
        self.assertFalse(rt.is_running)
        self.assertIsNone(rt._thread)

    def test_immediate_death_returns_false_and_clears_worker(self) -> None:
        _FakeThread.die_immediately = True
        rt = ActivityRuntime(_FakeSource())
        with mock.patch.object(threading, "Thread", new=_FakeThread):
            self.assertFalse(rt.start())
        self.assertEqual(len(_FakeThread.made), 1)
        self.assertIsNone(rt._thread)
        self.assertFalse(rt.is_running)
        self.assertFalse(rt.status.running)
        _FakeThread.die_immediately = False
        with mock.patch.object(threading, "Thread", new=_FakeThread):
            self.assertTrue(rt.start())
        self.assertIs(rt._thread, _FakeThread.made[1])
        self.assertTrue(rt.is_running)
        rt.stop(timeout=0.1)
        self.assertFalse(rt.is_running)

    def test_start_never_overwrites_live_worker(self) -> None:
        rt = ActivityRuntime(_FakeSource())
        with mock.patch.object(threading, "Thread", new=_FakeThread):
            self.assertTrue(rt.start())
        first = rt._thread
        self.assertTrue(rt.is_running)
        with mock.patch.object(threading, "Thread", new=_FakeThread):
            self.assertFalse(rt.start())
        self.assertIs(rt._thread, first)
        self.assertTrue(rt.is_running)
        rt.stop(timeout=0.1)
        self.assertFalse(rt.is_running)


if __name__ == "__main__":
    unittest.main()
