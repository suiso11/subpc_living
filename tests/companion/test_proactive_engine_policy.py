"""Tests for ProactiveEngine × DeterministicProactivePolicy wiring (defect-A fix).

DeterministicProactivePolicy.decide() が ProactiveEngine の
_check_schedule_remind / _check_break_suggest / _check_away_return から
実際に呼び出されることを検証する。
"""

from __future__ import annotations

import unittest
from unittest import mock

from src.companion.contracts import CompanionState
from src.companion.policy import DeterministicProactivePolicy
from src.persona.proactive import ProactiveEngine


def _make_state(
    *,
    activity_mode: str = "idle",
    present: bool = True,
    focused_since: float | None = None,
    interruptible: bool = True,
    updated_at: float = 0.0,
) -> CompanionState:
    return CompanionState(
        activity_mode=activity_mode,
        present=present,
        focused_since=focused_since,
        interruptible=interruptible,
        display_state=activity_mode,
        updated_at=updated_at,
    )


def _make_profile(*, name: str = "テストユーザー"):
    p = mock.MagicMock()
    p.name = name
    p.get_today_schedule.return_value = []
    return p


def _make_engine(
    companion_getter=None,
    companion_policy=None,
    calendar_source=None,
    profile=None,
):
    """Build a ProactiveEngine wired for testing (no threads)."""
    if profile is None:
        profile = _make_profile()
    if companion_policy is None:
        companion_policy = DeterministicProactivePolicy(
            long_work_seconds=600.0,
            schedule_lead_seconds=600.0,
            away_return_seconds=60.0,
        )
    return ProactiveEngine(
        profile=profile,
        companion_getter=companion_getter,
        companion_policy=companion_policy,
        calendar_source=calendar_source,
    )


class ScheduleRemindPolicyTest(unittest.TestCase):
    """_check_schedule_remind with companion_getter → policy.decide gating."""

    def test_focused_not_interruptible_schedule_approaching_fires(self):
        """a) focused + interruptible=False + schedule approaching: fires.

        ProactiveEngine の従来契約では schedule_remind は離席・不在のみ抑止し、
        focused (interruptible 不問) でも予定接近ならリマインドする。
        """
        now = 1000000.0
        state = _make_state(
            activity_mode="focused",
            interruptible=False,
            focused_since=now - 600,
            updated_at=now - 10,
        )
        # next_event_at is within schedule_lead_seconds → policy would see it
        # but focused + not interruptible → silent. Engine's legacy contract
        # still allows the remind for focused.
        cal = mock.MagicMock()
        cal.next_event.return_value = mock.MagicMock(start_at=now + 300, title="会議")
        engine = _make_engine(
            companion_getter=lambda: state,
            calendar_source=cal,
        )
        fired = []
        engine._callback = lambda kind, msg: fired.append((kind, msg))

        with mock.patch("src.persona.proactive.time") as mock_time:
            mock_time.time.return_value = now
            mock_time.sleep = mock.MagicMock()
            engine._check_schedule_remind()

        self.assertEqual(len(fired), 1)
        self.assertEqual(fired[0][0], "schedule_remind")

    def test_focused_interruptible_schedule_approaching_fires(self):
        """b) focused + interruptible=True + schedule approaching: fires."""
        now = 1000000.0
        state = _make_state(
            activity_mode="focused",
            interruptible=True,
            focused_since=now - 600,
            updated_at=now - 10,
        )
        cal = mock.MagicMock()
        cal.next_event.return_value = mock.MagicMock(start_at=now + 300, title="会議")
        engine = _make_engine(
            companion_getter=lambda: state,
            calendar_source=cal,
        )
        fired = []
        engine._callback = lambda kind, msg: fired.append((kind, msg))

        with mock.patch("src.persona.proactive.time") as mock_time:
            mock_time.time.return_value = now
            mock_time.sleep = mock.MagicMock()
            engine._check_schedule_remind()

        self.assertEqual(len(fired), 1)
        self.assertEqual(fired[0][0], "schedule_remind")

    def test_companion_getter_none_follows_legacy(self):
        """e) companion_getter is None: legacy path, policy.decide NOT called."""
        engine = _make_engine(companion_getter=None)
        fired = []
        engine._callback = lambda kind, msg: fired.append((kind, msg))
        # patch policy.decide to detect if it's called
        engine.companion_policy.decide = mock.MagicMock(side_effect=AssertionError("must not be called"))

        with mock.patch("src.persona.proactive.time") as mock_time:
            mock_time.time.return_value = 1000000.0
            mock_time.sleep = mock.MagicMock()
            engine._check_schedule_remind()

        engine.companion_policy.decide.assert_not_called()

    def test_companion_getter_returning_none_follows_legacy(self):
        """companion_getter returns None: falls through to legacy path."""
        engine = _make_engine(companion_getter=lambda: None)
        fired = []
        engine._callback = lambda kind, msg: fired.append((kind, msg))
        engine.companion_policy.decide = mock.MagicMock(side_effect=AssertionError("must not be called"))

        with mock.patch("src.persona.proactive.time") as mock_time:
            mock_time.time.return_value = 1000000.0
            mock_time.sleep = mock.MagicMock()
            engine._check_schedule_remind()

        engine.companion_policy.decide.assert_not_called()


class BreakSuggestPolicyTest(unittest.TestCase):
    """_check_break_suggest with companion_getter → policy.decide gating."""

    def test_idle_long_work_interruptible_fires(self):
        """c) idle + long work + interruptible: break_suggest fires."""
        now = 1000000.0
        state = _make_state(
            activity_mode="idle",
            present=True,
            focused_since=now - 3600,  # 1 hour > long_work_seconds(600)
            interruptible=True,
            updated_at=now - 200,  # older than away_return_seconds(60) to avoid away_return
        )
        engine = _make_engine(companion_getter=lambda: state)
        engine._session_start_time = now - 3600  # 1 hour session
        fired = []
        engine._callback = lambda kind, msg: fired.append((kind, msg))

        with mock.patch("src.persona.proactive.time") as mock_time:
            mock_time.time.return_value = now
            mock_time.sleep = mock.MagicMock()
            engine._check_break_suggest()

        self.assertEqual(len(fired), 1)
        self.assertEqual(fired[0][0], "break_suggest")

    def test_focused_not_interruptible_no_fire(self):
        """focused + interruptible=False: break_suggest does NOT fire."""
        now = 1000000.0
        state = _make_state(
            activity_mode="focused",
            interruptible=False,
            focused_since=now - 3600,
            updated_at=now - 10,
        )
        engine = _make_engine(companion_getter=lambda: state)
        fired = []
        engine._callback = lambda kind, msg: fired.append((kind, msg))

        with mock.patch("src.persona.proactive.time") as mock_time:
            mock_time.time.return_value = now
            mock_time.sleep = mock.MagicMock()
            engine._check_break_suggest()

        self.assertEqual(fired, [])


class AwayReturnTest(unittest.TestCase):
    """_check_away_return fires when policy decides away_return."""

    def test_away_return_fires(self):
        """d) 離席→復帰遷移 (直前ポーリングが away) で fires."""
        now = 1000000.0
        prev_away = _make_state(
            activity_mode="away",
            present=False,
            interruptible=False,
            updated_at=now - 120,
        )
        state = _make_state(
            activity_mode="idle",
            present=True,
            focused_since=None,
            interruptible=True,
            updated_at=now - 10,  # 10s ago < away_return_seconds(60)
        )
        engine = _make_engine(companion_getter=lambda: state)
        engine._last_companion_state = prev_away
        fired = []
        engine._callback = lambda kind, msg: fired.append((kind, msg))

        with mock.patch("src.persona.proactive.time") as mock_time:
            mock_time.time.return_value = now
            mock_time.sleep = mock.MagicMock()
            engine._check_away_return()

        self.assertEqual(len(fired), 1)
        self.assertEqual(fired[0][0], "away_return")
        self.assertIn("おかえりなさい", fired[0][1])

    def test_away_return_no_fire_when_ordinary_idle(self):
        """直前も present の通常 idle は復帰と誤認しない."""
        now = 1000000.0
        prev_idle = _make_state(
            activity_mode="idle",
            present=True,
            focused_since=None,
            interruptible=True,
            updated_at=now - 120,
        )
        state = _make_state(
            activity_mode="idle",
            present=True,
            focused_since=None,
            interruptible=True,
            updated_at=now - 10,
        )
        engine = _make_engine(companion_getter=lambda: state)
        engine._last_companion_state = prev_idle
        fired = []
        engine._callback = lambda kind, msg: fired.append((kind, msg))

        with mock.patch("src.persona.proactive.time") as mock_time:
            mock_time.time.return_value = now
            mock_time.sleep = mock.MagicMock()
            engine._check_away_return()

        self.assertEqual(fired, [])

    def test_away_return_no_fire_when_stale(self):
        """updated_at too old → policy says silent → no fire."""
        now = 1000000.0
        state = _make_state(
            activity_mode="idle",
            present=True,
            interruptible=True,
            updated_at=now - 300,  # 300s > away_return_seconds(60)
        )
        engine = _make_engine(companion_getter=lambda: state)
        fired = []
        engine._callback = lambda kind, msg: fired.append((kind, msg))

        with mock.patch("src.persona.proactive.time") as mock_time:
            mock_time.time.return_value = now
            mock_time.sleep = mock.MagicMock()
            engine._check_away_return()

        self.assertEqual(fired, [])

    def test_away_return_no_fire_when_still_away(self):
        """activity_mode=away → policy says silent → no fire."""
        now = 1000000.0
        state = _make_state(
            activity_mode="away",
            present=False,
            interruptible=False,
            updated_at=now - 5,
        )
        engine = _make_engine(companion_getter=lambda: state)
        fired = []
        engine._callback = lambda kind, msg: fired.append((kind, msg))

        with mock.patch("src.persona.proactive.time") as mock_time:
            mock_time.time.return_value = now
            mock_time.sleep = mock.MagicMock()
            engine._check_away_return()

        self.assertEqual(fired, [])

    def test_away_return_no_fire_when_companion_getter_none(self):
        """companion_getter is None: no-op."""
        engine = _make_engine(companion_getter=None)
        fired = []
        engine._callback = lambda kind, msg: fired.append((kind, msg))
        engine._check_away_return()
        self.assertEqual(fired, [])


class GreetingCompanionGateTest(unittest.TestCase):
    """f) Regression: _check_greeting still gates via _companion_gate."""

    def test_greeting_does_not_fire_when_focused(self):
        """focused (non-interruptible) → _companion_gate returns 'focused' → greeting blocked."""
        now = 1000000.0
        state = _make_state(
            activity_mode="focused",
            interruptible=False,
            focused_since=now - 600,
            updated_at=now - 10,
        )
        engine = _make_engine(companion_getter=lambda: state)
        engine._started_at = now - 5  # within 60s window
        fired = []
        engine._callback = lambda kind, msg: fired.append((kind, msg))

        with mock.patch("src.persona.proactive.time") as mock_time:
            mock_time.time.return_value = now
            mock_time.sleep = mock.MagicMock()
            engine._check_greeting()

        self.assertEqual(fired, [])


class AwayReturnCooldownTest(unittest.TestCase):
    """Verify away_return cooldown is registered in _cooldown dict."""

    def test_away_return_cooldown_exists(self):
        engine = _make_engine()
        self.assertIn("away_return", engine._cooldown)
        self.assertEqual(engine._cooldown["away_return"], 300.0)


class AwayReturnNamePartTest(unittest.TestCase):
    """Verify name part is included in the away_return message."""

    def test_name_included(self):
        now = 1000000.0
        prev_away = _make_state(
            activity_mode="away",
            present=False,
            interruptible=False,
            updated_at=now - 120,
        )
        state = _make_state(
            activity_mode="idle",
            present=True,
            interruptible=True,
            updated_at=now - 5,
        )
        engine = _make_engine(
            companion_getter=lambda: state,
            profile=_make_profile(name="太郎"),
        )
        engine._last_companion_state = prev_away
        fired = []
        engine._callback = lambda kind, msg: fired.append((kind, msg))

        with mock.patch("src.persona.proactive.time") as mock_time:
            mock_time.time.return_value = now
            mock_time.sleep = mock.MagicMock()
            engine._check_away_return()

        self.assertEqual(len(fired), 1)
        self.assertIn("太郎さん", fired[0][1])


class ScheduleRemindPolicyAdaptiveMessageTest(unittest.TestCase):
    """Defect-I fix: Policy path fires with adaptive message based on actual delta."""

    def _make_profile_with_event(self, time_str: str, title: str = "会議"):
        """Return a profile whose get_today_schedule contains one event."""
        profile = mock.MagicMock()
        profile.name = "テストユーザー"
        profile.get_today_schedule.return_value = [{"time": time_str, "title": title}]
        return profile

    def test_policy_delta_3min_no_prepare_text(self):
        """Policy should_act=True, delta=3 min: fires, message has title but NOT 準備は大丈夫ですか."""
        from datetime import datetime, timedelta
        now_dt = datetime.now()
        event_dt = now_dt + timedelta(minutes=3)
        time_str = event_dt.strftime("%H:%M")
        profile = self._make_profile_with_event(time_str, "会議")
        state = _make_state(activity_mode="idle")
        engine = _make_engine(companion_getter=lambda: state, profile=profile)
        fired = []
        engine._callback = lambda kind, msg: fired.append((kind, msg))

        engine._check_schedule_remind()

        self.assertEqual(len(fired), 1, f"Expected 1 fire, got {fired}")
        self.assertEqual(fired[0][0], "schedule_remind")
        self.assertIn("会議", fired[0][1])
        self.assertNotIn("準備は大丈夫ですか", fired[0][1])

    def test_policy_delta_8min_with_prepare_text(self):
        """Policy should_act=True, delta=8 min: fires, message contains 準備は大丈夫ですか."""
        from datetime import datetime, timedelta
        now_dt = datetime.now()
        event_dt = now_dt + timedelta(minutes=8)
        time_str = event_dt.strftime("%H:%M")
        profile = self._make_profile_with_event(time_str, "会議")
        state = _make_state(activity_mode="idle")
        engine = _make_engine(companion_getter=lambda: state, profile=profile)
        fired = []
        engine._callback = lambda kind, msg: fired.append((kind, msg))

        engine._check_schedule_remind()

        self.assertEqual(len(fired), 1, f"Expected 1 fire, got {fired}")
        self.assertEqual(fired[0][0], "schedule_remind")
        self.assertIn("会議", fired[0][1])
        self.assertIn("準備は大丈夫ですか", fired[0][1])

    def test_policy_empty_schedule_no_fire(self):
        """Policy should_act=True but profile.get_today_schedule returns []: no fire."""
        now = 1000000.0
        profile = _make_profile()
        profile.get_today_schedule.return_value = []
        state = _make_state(activity_mode="idle")
        engine = _make_engine(companion_getter=lambda: state, profile=profile)
        fired = []
        engine._callback = lambda kind, msg: fired.append((kind, msg))

        with mock.patch("src.persona.proactive.time") as mock_time:
            mock_time.time.return_value = now
            mock_time.sleep = mock.MagicMock()
            engine._check_schedule_remind()

        self.assertEqual(fired, [])


class BreakSuggestCompanionGetterNoneTest(unittest.TestCase):
    """_check_break_suggest with companion_getter=None follows legacy path."""

    def test_no_fire_legacy(self):
        engine = _make_engine(companion_getter=None)
        fired = []
        engine._callback = lambda kind, msg: fired.append((kind, msg))
        engine.companion_policy.decide = mock.MagicMock(side_effect=AssertionError("must not be called"))

        with mock.patch("src.persona.proactive.time") as mock_time:
            mock_time.time.return_value = 1000000.0
            mock_time.sleep = mock.MagicMock()
            engine._check_break_suggest()

        engine.companion_policy.decide.assert_not_called()


class _FakeThread:
    """実際には実行しない worker スレッド。start/join/is_alive を記録する。"""

    def __init__(self, target=None, daemon=True) -> None:
        self._target = target
        self._daemon = daemon
        self._alive = False
        self.started = 0
        self.join_timeouts: list[float | None] = []

    def start(self) -> None:
        self.started += 1
        self._alive = True

    def join(self, timeout=None) -> None:
        self.join_timeouts.append(timeout)

    def is_alive(self) -> bool:
        return self._alive

    def mark_dead(self) -> None:
        self._alive = False


class _DeadOnStartThread(_FakeThread):
    """start() 直後に即死する worker。"""

    def start(self) -> None:
        self.started += 1
        self._alive = False


class _PartialStartThread(_FakeThread):
    """start() が生存フラグを立てたまま例外を投げる worker。"""

    def start(self) -> None:
        self.started += 1
        self._alive = True
        raise RuntimeError("secret partial start detail")


class ProactiveEngineLifecycleTest(unittest.TestCase):
    """ProactiveEngine のスレッドライフサイクルを実スレッドなしで決定的に検証。

    スレッドは注入した thread_factory の偽物で代替し、ネットワーク・センサー・
    データ等の実リソースには一切触れない。
    """

    def _make_engine(self, factory=None):
        threads: list[_FakeThread] = []

        def _default_factory(target=None, daemon=True) -> _FakeThread:
            t = _FakeThread(target=target, daemon=daemon)
            threads.append(t)
            return t

        engine = ProactiveEngine(
            profile=_make_profile(),
            thread_factory=factory or _default_factory,
        )
        return engine, threads

    def test_start_success_and_is_running(self) -> None:
        engine, threads = self._make_engine()
        self.assertTrue(engine.start(callback=lambda k, m: None))
        self.assertTrue(engine.is_running)
        self.assertEqual(len(threads), 1)
        self.assertEqual(threads[0].started, 1)
        self.assertIs(engine._thread, threads[0])

    def test_duplicate_start_rejected(self) -> None:
        engine, threads = self._make_engine()
        self.assertTrue(engine.start(callback=lambda k, m: None))
        self.assertFalse(engine.start(callback=lambda k, m: None))
        self.assertEqual(len(threads), 1)
        self.assertEqual(threads[0].started, 1)

    def test_start_rejected_while_old_worker_live(self) -> None:
        engine, threads = self._make_engine()
        self.assertTrue(engine.start(callback=lambda k, m: None))
        engine.stop()  # タイムアウト相当で生存継続 → 所有権保持
        self.assertFalse(engine.is_running)
        self.assertFalse(engine.start(callback=lambda k, m: None))  # 旧 worker が残る
        self.assertEqual(len(threads), 1)

    def test_restart_allowed_after_reap(self) -> None:
        engine, threads = self._make_engine()
        self.assertTrue(engine.start(callback=lambda k, m: None))
        threads[0].mark_dead()
        engine.stop()
        self.assertIsNone(engine._thread)
        self.assertTrue(engine.start(callback=lambda k, m: None))
        self.assertTrue(engine.is_running)
        self.assertEqual(len(threads), 2)
        self.assertIs(engine._thread, threads[1])

    def test_thread_factory_failure_returns_false(self) -> None:
        engine, _ = self._make_engine()

        def bad_factory(target=None, daemon=True):
            raise RuntimeError("secret factory detail")

        engine._thread_factory = bad_factory
        started = engine.start(callback=lambda k, m: None)
        self.assertFalse(started)
        self.assertFalse(engine.is_running)
        self.assertIsNone(engine._thread)

    def test_thread_start_failure_returns_false(self) -> None:
        class _RaiseOnStart(_FakeThread):
            def start(self) -> None:
                self.started += 1
                raise RuntimeError("secret start detail")

        engine, _ = self._make_engine(factory=lambda target=None, daemon=True: _RaiseOnStart())
        started = engine.start(callback=lambda k, m: None)
        self.assertFalse(started)
        self.assertFalse(engine.is_running)
        self.assertIsNone(engine._thread)

    def test_partial_start_retains_live_ownership(self) -> None:
        engine, _ = self._make_engine(factory=lambda target=None, daemon=True: _PartialStartThread())
        started = engine.start(callback=lambda k, m: None)
        self.assertFalse(started)
        self.assertFalse(engine.is_running)
        thread = engine._thread
        self.assertIsNotNone(thread)
        self.assertTrue(thread.is_alive())
        # 生存 worker が残っている限り再起動は拒否される
        self.assertFalse(engine.start(callback=lambda k, m: None))
        self.assertIs(engine._thread, thread)
        # 死亡を確認できた時点で所有権が解放され再起動できる
        thread.mark_dead()
        engine.stop()
        self.assertIsNone(engine._thread)

    def test_immediate_death_rejected(self) -> None:
        engine, _ = self._make_engine(factory=lambda target=None, daemon=True: _DeadOnStartThread())
        started = engine.start(callback=lambda k, m: None)
        self.assertFalse(started)
        self.assertFalse(engine.is_running)

    def test_stop_retains_live_thread_on_timeout(self) -> None:
        engine, threads = self._make_engine()
        self.assertTrue(engine.start(callback=lambda k, m: None))
        engine.stop()  # join は即時返るが fake は生存したまま → 回収しない
        self.assertFalse(engine.is_running)
        self.assertIs(engine._thread, threads[0])

    def test_repeated_stop_reaps_after_death(self) -> None:
        engine, threads = self._make_engine()
        self.assertTrue(engine.start(callback=lambda k, m: None))
        engine.stop()  # 生存中は回収されない
        self.assertIs(engine._thread, threads[0])
        threads[0].mark_dead()
        engine.stop()  # 死亡確認で回収
        self.assertIsNone(engine._thread)

    def test_is_running_false_when_thread_dies_unexpectedly(self) -> None:
        engine, threads = self._make_engine()
        self.assertTrue(engine.start(callback=lambda k, m: None))
        threads[0].mark_dead()
        self.assertFalse(engine.is_running)

    def test_is_running_requires_requested_running(self) -> None:
        engine, threads = self._make_engine()
        self.assertTrue(engine.start(callback=lambda k, m: None))
        engine._running = False  # リクエスト停止状態 (生存 worker が残っても)
        self.assertFalse(engine.is_running)


class BridgeLifecycleCompatTest(unittest.TestCase):
    """ProactiveEngine の start が bool を返すようになっても、ブリッジ呼び出し
    (戻り値を無視・stop は例外を握りつぶす) が壊れないことを検証する。"""

    def test_start_and_stop_do_not_raise(self) -> None:
        from unittest import mock
        from src.discord_bot.proactive_bridge import ProactiveBridge, ProactiveConfig
        from src.persona.conversation_loop import ConversationLoopStore

        class _FakeBot:
            def get_channel(self, channel_id):
                return None

        config = ProactiveConfig(enabled=True, channel_id=1)
        store = ConversationLoopStore(
            None,
            base_interval_sec=3600,
            reply_timeout_sec=3600,
        )
        engine = mock.MagicMock()
        bridge = ProactiveBridge(
            bot=_FakeBot(),
            state=mock.MagicMock(llm=None, llm_profiles={}),
            config=config,
            profile=None,
            engine=engine,
            conversation_store=store,
        )
        bridge.start()
        engine.start.assert_called_once()
        bridge.stop()
        engine.stop.assert_called_once()


if __name__ == "__main__":
    unittest.main()
