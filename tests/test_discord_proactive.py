from __future__ import annotations

import asyncio
import json
import os
import tempfile
import threading
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.companion.contracts import CompanionState
from src.persona.proactive import CONVERSATION_PROMPTS, ProactiveEngine
from src.persona.conversation_loop import ConversationLoopStore
from src.discord_bot.proactive_bridge import (
    PROJECT_ROOT,
    ProactiveBridge,
    ProactiveConfig,
    create_proactive_bridge,
    rewrite_message,
)


class _FakeChannel:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send(self, text: str) -> None:
        self.sent.append(text)


class _FakeBot:
    def __init__(self, channel: _FakeChannel | None, loop=None) -> None:
        self._channel = channel
        self.loop = loop

    def get_channel(self, channel_id: int):
        return self._channel

    async def fetch_channel(self, channel_id: int):
        return self._channel


def _profile() -> SimpleNamespace:
    return SimpleNamespace(
        system_prompt="あなたは親しみやすいアシスタントです。",
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        repeat_penalty=1.1,
        num_ctx=4096,
        num_predict=None,
    )


def _state(llm=None, voice_tts=None) -> SimpleNamespace:
    return SimpleNamespace(
        llm=llm,
        llm_profiles={"default": _profile()},
        voice_tts=voice_tts,
    )


def _bridge(*, config, state, bot) -> ProactiveBridge:
    # engine を注入してバックグラウンドスレッドの起動を避ける
    return ProactiveBridge(
        bot=bot,
        state=state,
        config=config,
        profile=None,
        monitor_context=None,
        engine=MagicMock(),
        conversation_store=ConversationLoopStore(
            None,
            base_interval_sec=config.conversation_interval_hours * 3600,
            reply_timeout_sec=config.conversation_reply_timeout_hours * 3600,
            daily_limit=config.conversation_daily_limit,
            max_backoff_sec=config.conversation_max_backoff_hours * 3600,
        ),
    )


class ProactiveConfigTest(unittest.TestCase):
    def test_defaults(self) -> None:
        cfg = ProactiveConfig.from_env({})
        self.assertFalse(cfg.enabled)
        self.assertIsNone(cfg.channel_id)
        self.assertTrue(cfg.llm_rewrite)
        self.assertEqual(cfg.check_interval, 60.0)
        self.assertFalse(cfg.conversation_enabled)
        self.assertEqual(cfg.conversation_interval_hours, 6.0)
        self.assertEqual(cfg.conversation_idle_minutes, 60.0)
        self.assertEqual(cfg.conversation_daily_limit, 1)
        self.assertEqual(cfg.conversation_max_backoff_hours, 72.0)
        self.assertFalse(cfg.conversation_autoread)
        self.assertEqual(cfg.quiet_hours, (1, 9))

    def test_parses_values(self) -> None:
        cfg = ProactiveConfig.from_env(
            {
                "DISCORD_PROACTIVE_ENABLED": "true",
                "DISCORD_PROACTIVE_CHANNEL_ID": "12345",
                "DISCORD_PROACTIVE_LLM_REWRITE": "false",
                "DISCORD_PROACTIVE_CHECK_INTERVAL": "15",
                "DISCORD_PROACTIVE_CHAT_ENABLED": "true",
                "DISCORD_PROACTIVE_CHAT_INTERVAL_HOURS": "4",
                "DISCORD_PROACTIVE_CHAT_IDLE_MINUTES": "30",
                "DISCORD_PROACTIVE_CHAT_REPLY_TIMEOUT_HOURS": "8",
                "DISCORD_PROACTIVE_CHAT_DAILY_LIMIT": "2",
                "DISCORD_PROACTIVE_CHAT_MAX_BACKOFF_HOURS": "48",
                "DISCORD_PROACTIVE_CHAT_AUTOREAD": "true",
                "DISCORD_PROACTIVE_QUIET_HOURS": "23-7",
            }
        )
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.channel_id, 12345)
        self.assertFalse(cfg.llm_rewrite)
        self.assertEqual(cfg.check_interval, 15.0)
        self.assertTrue(cfg.conversation_enabled)
        self.assertEqual(cfg.conversation_interval_hours, 4.0)
        self.assertEqual(cfg.conversation_idle_minutes, 30.0)
        self.assertEqual(cfg.conversation_reply_timeout_hours, 8.0)
        self.assertEqual(cfg.conversation_daily_limit, 2)
        self.assertEqual(cfg.conversation_max_backoff_hours, 48.0)
        self.assertTrue(cfg.conversation_autoread)
        self.assertEqual(cfg.quiet_hours, (23, 7))

    def test_invalid_channel_and_interval_fall_back(self) -> None:
        cfg = ProactiveConfig.from_env(
            {
                "DISCORD_PROACTIVE_CHANNEL_ID": "not-an-int",
                "DISCORD_PROACTIVE_CHECK_INTERVAL": "abc",
                "DISCORD_PROACTIVE_QUIET_HOURS": "25-7",
            }
        )
        self.assertIsNone(cfg.channel_id)
        self.assertEqual(cfg.check_interval, 60.0)
        self.assertEqual(cfg.quiet_hours, (1, 9))


class RewriteMessageTest(unittest.TestCase):
    def test_uses_llm_output(self) -> None:
        llm = MagicMock()
        llm.generate.return_value = "  やっほー、休憩しよ！  "
        out = rewrite_message(llm, system_prompt="sys", message="休憩しませんか？")
        self.assertEqual(out, "やっほー、休憩しよ！")

    def test_empty_output_falls_back(self) -> None:
        llm = MagicMock()
        llm.generate.return_value = "   "
        out = rewrite_message(llm, system_prompt="sys", message="元の文")
        self.assertEqual(out, "元の文")

    def test_exception_falls_back(self) -> None:
        llm = MagicMock()
        llm.generate.side_effect = RuntimeError("boom")
        out = rewrite_message(llm, system_prompt="sys", message="元の文")
        self.assertEqual(out, "元の文")

    def test_none_llm_falls_back(self) -> None:
        out = rewrite_message(None, system_prompt="sys", message="元の文")
        self.assertEqual(out, "元の文")


class DeliverTest(unittest.TestCase):
    def test_deliver_rewrites_and_sends(self) -> None:
        channel = _FakeChannel()
        llm = MagicMock()
        llm.generate.return_value = "言い換え後のテキスト"
        voice_tts = MagicMock()
        voice_tts.autoread = MagicMock(
            side_effect=lambda *a, **k: asyncio.sleep(0)
        )
        state = _state(llm=llm, voice_tts=voice_tts)
        cfg = ProactiveConfig(enabled=True, channel_id=1, llm_rewrite=True)
        bridge = _bridge(config=cfg, state=state, bot=_FakeBot(channel))

        asyncio.run(bridge._deliver("break_suggest", "少し休憩しませんか？"))

        self.assertEqual(channel.sent, ["言い換え後のテキスト"])
        voice_tts.autoread.assert_called_once_with("言い換え後のテキスト", emotion="neutral")

    def test_deliver_without_rewrite_sends_original(self) -> None:
        channel = _FakeChannel()
        llm = MagicMock()
        state = _state(llm=llm)
        cfg = ProactiveConfig(enabled=True, channel_id=1, llm_rewrite=False)
        bridge = _bridge(config=cfg, state=state, bot=_FakeBot(channel))

        asyncio.run(bridge._deliver("greeting", "おはようございます"))

        self.assertEqual(channel.sent, ["おはようございます"])
        llm.generate.assert_not_called()

    def test_deliver_rewrite_failure_sends_original(self) -> None:
        channel = _FakeChannel()
        llm = MagicMock()
        llm.generate.side_effect = RuntimeError("boom")
        state = _state(llm=llm)
        cfg = ProactiveConfig(enabled=True, channel_id=1, llm_rewrite=True)
        bridge = _bridge(config=cfg, state=state, bot=_FakeBot(channel))

        asyncio.run(bridge._deliver("pc_alert", "PCが高温です"))

        self.assertEqual(channel.sent, ["PCが高温です"])

    def test_conversation_reply_keeps_prompt_context_once(self) -> None:
        channel = _FakeChannel()
        state = _state()
        cfg = ProactiveConfig(
            enabled=True,
            channel_id=1,
            llm_rewrite=False,
            conversation_enabled=True,
        )
        bridge = _bridge(config=cfg, state=state, bot=_FakeBot(channel))

        asyncio.run(bridge._deliver("conversation_start", "好きな音楽は？"))
        contextualized, found = bridge.contextualize_reply(1, "ジャズかな")
        plain, found_again = bridge.contextualize_reply(1, "もう一度")

        self.assertTrue(found)
        self.assertIn("好きな音楽は？", contextualized)
        self.assertIn("ジャズかな", contextualized)
        self.assertFalse(found_again)
        self.assertEqual(plain, "もう一度")

    def test_pending_conversation_survives_bridge_recreation(self) -> None:
        store = ConversationLoopStore(
            None,
            base_interval_sec=3600,
            reply_timeout_sec=3600,
        )
        cfg = ProactiveConfig(
            enabled=True,
            channel_id=1,
            llm_rewrite=False,
            conversation_enabled=True,
        )
        first = ProactiveBridge(
            bot=_FakeBot(_FakeChannel()),
            state=_state(),
            config=cfg,
            profile=None,
            engine=MagicMock(),
            conversation_store=store,
        )
        second = ProactiveBridge(
            bot=_FakeBot(None),
            state=_state(),
            config=cfg,
            profile=None,
            engine=MagicMock(),
            conversation_store=store,
        )

        asyncio.run(first._deliver("conversation_start", "最近の趣味は？"))
        contextualized, found = second.contextualize_reply(1, "カメラ")

        self.assertTrue(found)
        self.assertIn("最近の趣味は？", contextualized)
        self.assertIn("掘り下げ質問を1つまで", contextualized)

    def test_conversation_start_is_not_autoread_by_default(self) -> None:
        channel = _FakeChannel()
        voice_tts = MagicMock()
        cfg = ProactiveConfig(
            enabled=True,
            channel_id=1,
            llm_rewrite=False,
            conversation_enabled=True,
            conversation_autoread=False,
        )
        bridge = _bridge(
            config=cfg,
            state=_state(voice_tts=voice_tts),
            bot=_FakeBot(channel),
        )

        asyncio.run(bridge._deliver("conversation_start", "話す？"))

        voice_tts.autoread.assert_not_called()

    def test_pending_conversation_can_be_snoozed(self) -> None:
        channel = _FakeChannel()
        cfg = ProactiveConfig(
            enabled=True,
            channel_id=1,
            llm_rewrite=False,
            conversation_enabled=True,
        )
        bridge = _bridge(config=cfg, state=_state(), bot=_FakeBot(channel))
        asyncio.run(bridge._deliver("conversation_start", "話す？"))

        response = bridge.handle_conversation_control(1, "あとで")

        self.assertIn("3時間", response)
        self.assertFalse(bridge.conversation_store.has_pending(1))

    def test_expired_conversation_context_is_discarded(self) -> None:
        cfg = ProactiveConfig(
            enabled=True,
            channel_id=1,
            conversation_reply_timeout_hours=1.0,
        )
        bridge = _bridge(config=cfg, state=_state(), bot=_FakeBot(None))
        bridge.conversation_store.record_prompt(1, "古い質問", now=time.time() - 3601)

        text, found = bridge.contextualize_reply(1, "返答")

        self.assertFalse(found)
        self.assertEqual(text, "返答")

    def test_deliver_missing_channel_does_not_raise(self) -> None:
        state = _state()
        cfg = ProactiveConfig(enabled=True, channel_id=999, llm_rewrite=False)
        bridge = _bridge(config=cfg, state=state, bot=_FakeBot(None))
        # チャンネル未解決でも例外を出さない
        asyncio.run(bridge._deliver("greeting", "やあ"))


class OnTriggerBridgeTest(unittest.TestCase):
    def test_on_trigger_schedules_on_loop(self) -> None:
        loop = asyncio.new_event_loop()
        t = threading.Thread(target=loop.run_forever, daemon=True)
        t.start()
        try:
            channel = _FakeChannel()
            cfg = ProactiveConfig(enabled=True, channel_id=1, llm_rewrite=False)
            bridge = _bridge(
                config=cfg, state=_state(), bot=_FakeBot(channel, loop=loop)
            )
            # エンジンのスレッドを模して別スレッドからコールバックを呼ぶ
            caller = threading.Thread(
                target=bridge._on_trigger, args=("greeting", "こんにちは")
            )
            caller.start()
            caller.join()

            # ループ上で _deliver が完走するまで待つ
            done = threading.Event()
            loop.call_soon_threadsafe(lambda: None)
            for _ in range(200):
                if channel.sent:
                    done.set()
                    break
                threading.Event().wait(0.01)
            self.assertEqual(channel.sent, ["こんにちは"])
        finally:
            loop.call_soon_threadsafe(loop.stop)
            t.join(timeout=2)
            loop.close()

    def test_on_trigger_without_loop_is_noop(self) -> None:
        cfg = ProactiveConfig(enabled=True, channel_id=1)
        bridge = _bridge(config=cfg, state=_state(), bot=_FakeBot(None, loop=None))
        # loop が無くても例外を出さない
        bridge._on_trigger("greeting", "hi")


class ConversationStartTest(unittest.TestCase):
    def test_fires_after_idle_and_respects_cooldown(self) -> None:
        engine = ProactiveEngine(
            profile=SimpleNamespace(),
            conversation_enabled=True,
            conversation_interval=3600,
            conversation_idle=60,
            quiet_hours=(0, 0),
        )
        fired: list[tuple[str, str]] = []
        engine._callback = lambda trigger, message: fired.append((trigger, message))
        engine._last_user_activity = time.time() - 61

        engine._check_conversation_start()
        engine._check_conversation_start()

        self.assertEqual(len(fired), 1)
        self.assertEqual(fired[0][0], "conversation_start")
        self.assertIn(fired[0][1], CONVERSATION_PROMPTS)

    def test_waits_for_idle_time(self) -> None:
        engine = ProactiveEngine(
            profile=SimpleNamespace(),
            conversation_enabled=True,
            conversation_idle=60,
            quiet_hours=(0, 0),
        )
        fired = []
        engine._callback = lambda trigger, message: fired.append((trigger, message))

        engine._check_conversation_start()

        self.assertEqual(fired, [])

    def test_persistent_gate_can_suppress_conversation(self) -> None:
        gate = MagicMock(return_value=False)
        engine = ProactiveEngine(
            profile=SimpleNamespace(),
            conversation_enabled=True,
            conversation_idle=0,
            quiet_hours=(0, 0),
            conversation_gate=gate,
        )
        fired = []
        engine._callback = lambda trigger, message: fired.append((trigger, message))

        engine._check_conversation_start()

        gate.assert_called_once_with()
        self.assertEqual(fired, [])

    def test_quiet_hours_support_day_wrap(self) -> None:
        engine = ProactiveEngine(
            profile=SimpleNamespace(),
            conversation_enabled=True,
            quiet_hours=(23, 7),
        )

        self.assertTrue(engine._is_quiet_hour(23))
        self.assertTrue(engine._is_quiet_hour(3))
        self.assertFalse(engine._is_quiet_hour(12))


class WorkSessionTest(unittest.TestCase):
    def test_first_activity_starts_session_at_activity_time(self) -> None:
        engine = ProactiveEngine(profile=SimpleNamespace())

        with patch("src.persona.proactive.time.time", return_value=1000.0):
            engine.notify_user_activity()

        self.assertEqual(engine._session_start_time, 1000.0)
        self.assertEqual(engine._last_user_activity, 1000.0)

    def test_activity_after_30_minute_gap_starts_new_session(self) -> None:
        engine = ProactiveEngine(profile=SimpleNamespace())
        engine._session_start_time = 1000.0
        engine._last_user_activity = 1100.0

        with patch("src.persona.proactive.time.time", return_value=3000.0):
            engine.notify_user_activity()

        self.assertEqual(engine._session_start_time, 3000.0)
        self.assertEqual(engine._last_user_activity, 3000.0)

    def test_recent_activity_keeps_existing_session(self) -> None:
        engine = ProactiveEngine(profile=SimpleNamespace())
        engine._session_start_time = 1000.0
        engine._last_user_activity = 2000.0

        with patch("src.persona.proactive.time.time", return_value=2500.0):
            engine.notify_user_activity()

        self.assertEqual(engine._session_start_time, 1000.0)
        self.assertEqual(engine._last_user_activity, 2500.0)

    def test_service_uptime_alone_does_not_trigger_break(self) -> None:
        engine = ProactiveEngine(profile=SimpleNamespace())
        fired = []
        engine._callback = lambda trigger, message: fired.append((trigger, message))
        engine._last_user_activity = 200000.0

        with patch("src.persona.proactive.time.time", return_value=200100.0):
            engine._check_break_suggest()

        self.assertEqual(fired, [])


class NotifyActivityTest(unittest.TestCase):
    def test_notify_delegates_to_engine(self) -> None:
        engine = MagicMock()
        cfg = ProactiveConfig(enabled=True, channel_id=1)
        bridge = ProactiveBridge(
            bot=_FakeBot(None),
            state=_state(),
            config=cfg,
            profile=None,
            engine=engine,
        )
        bridge.notify_user_activity()
        engine.notify_user_activity.assert_called_once()


class CompanionGateTest(unittest.TestCase):
    def _state(
        self,
        activity_mode: str = "idle",
        present: bool = True,
        interruptible: bool = True,
    ) -> CompanionState:
        return CompanionState(
            activity_mode=activity_mode,
            present=present,
            focused_since=None,
            interruptible=interruptible,
            display_state=activity_mode,
            updated_at=time.time(),
        )

    def _profile(self, schedule: list | None = None) -> SimpleNamespace:
        return SimpleNamespace(
            name="",
            get_today_schedule=lambda: schedule or [],
        )

    def _schedule_profile(self) -> SimpleNamespace:
        t = (datetime.now() + timedelta(minutes=10)).strftime("%H:%M")
        return self._profile([{"time": t, "title": "会議", "note": ""}])

    def _engine(self, getter, *, profile=None, **kwargs) -> ProactiveEngine:
        return ProactiveEngine(
            profile=profile or self._profile(),
            companion_getter=getter,
            **kwargs,
        )

    def _break_fires(self, engine: ProactiveEngine) -> None:
        engine._session_start_time = time.time() - 3 * 3600
        engine._last_user_activity = time.time()

    def test_companion_gate_returns_modes(self) -> None:
        focused = self._engine(lambda: self._state("focused", interruptible=False))
        self.assertEqual(focused._companion_gate(), "focused")

        away = self._engine(lambda: self._state("away"))
        self.assertEqual(away._companion_gate(), "away")

        absent = self._engine(lambda: self._state("idle", present=False))
        self.assertEqual(absent._companion_gate(), "away")

        idle = self._engine(lambda: self._state("idle", interruptible=True))
        self.assertIsNone(idle._companion_gate())

        none = self._engine(None)
        self.assertIsNone(none._companion_gate())

    def test_focused_blocks_break_suggest(self) -> None:
        engine = self._engine(lambda: self._state("focused", interruptible=False))
        self._break_fires(engine)
        fired: list[tuple[str, str]] = []
        engine._callback = lambda trigger, message: fired.append((trigger, message))

        engine._check_break_suggest()

        self.assertEqual(fired, [])

    def test_focused_blocks_greeting(self) -> None:
        engine = self._engine(lambda: self._state("focused", interruptible=False))
        engine._started_at = time.time()
        fired: list[tuple[str, str]] = []
        engine._callback = lambda trigger, message: fired.append((trigger, message))

        with patch("src.persona.proactive.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2024, 1, 1, 7, 0, 0)
            engine._check_greeting()

        self.assertEqual(fired, [])

    def test_focused_blocks_conversation_start(self) -> None:
        engine = self._engine(
            lambda: self._state("focused", interruptible=False),
            conversation_enabled=True,
            conversation_idle=0,
            quiet_hours=(0, 0),
        )
        engine._last_user_activity = time.time() - 100
        fired: list[tuple[str, str]] = []
        engine._callback = lambda trigger, message: fired.append((trigger, message))

        engine._check_conversation_start()

        self.assertEqual(fired, [])

    def test_focused_blocks_pc_alert(self) -> None:
        monitor = MagicMock()
        monitor.get_status.return_value = {"running": True, "cpu_percent": 99}
        engine = ProactiveEngine(
            profile=self._profile(),
            monitor_context=monitor,
            companion_getter=lambda: self._state("focused", interruptible=False),
        )
        fired: list[tuple[str, str]] = []
        engine._callback = lambda trigger, message: fired.append((trigger, message))

        engine._check_pc_alert()

        self.assertEqual(fired, [])

    def test_focused_allows_schedule_remind(self) -> None:
        engine = self._engine(
            lambda: self._state("focused", interruptible=False),
            profile=self._schedule_profile(),
        )
        fired: list[tuple[str, str]] = []
        engine._callback = lambda trigger, message: fired.append((trigger, message))

        engine._check_schedule_remind()

        self.assertEqual([t for t, _ in fired], ["schedule_remind"])

    def test_away_blocks_schedule_remind(self) -> None:
        engine = self._engine(
            lambda: self._state("away"),
            profile=self._schedule_profile(),
        )
        fired: list[tuple[str, str]] = []
        engine._callback = lambda trigger, message: fired.append((trigger, message))

        engine._check_schedule_remind()

        self.assertEqual(fired, [])

    def test_away_blocks_break_suggest(self) -> None:
        engine = self._engine(lambda: self._state("away"))
        self._break_fires(engine)
        fired: list[tuple[str, str]] = []
        engine._callback = lambda trigger, message: fired.append((trigger, message))

        engine._check_break_suggest()

        self.assertEqual(fired, [])

    def test_idle_fires_conventionally(self) -> None:
        engine = self._engine(lambda: self._state("idle", interruptible=True))
        self._break_fires(engine)
        fired: list[tuple[str, str]] = []
        engine._callback = lambda trigger, message: fired.append((trigger, message))

        engine._check_break_suggest()

        self.assertEqual([t for t, _ in fired], ["break_suggest"])

    def test_getter_returns_none_no_gate(self) -> None:
        engine = self._engine(lambda: None)
        self._break_fires(engine)
        fired: list[tuple[str, str]] = []
        engine._callback = lambda trigger, message: fired.append((trigger, message))

        engine._check_break_suggest()

        self.assertEqual([t for t, _ in fired], ["break_suggest"])

    def test_getter_exception_falls_back(self) -> None:
        def bad_getter() -> CompanionState:
            raise RuntimeError("boom")

        engine = self._engine(bad_getter)
        self._break_fires(engine)
        fired: list[tuple[str, str]] = []
        engine._callback = lambda trigger, message: fired.append((trigger, message))

        engine._check_break_suggest()

        self.assertEqual([t for t, _ in fired], ["break_suggest"])


class CalendarScheduleTest(unittest.TestCase):
    def _profile(self, schedule: list | None = None) -> SimpleNamespace:
        return SimpleNamespace(
            name="",
            get_today_schedule=lambda: schedule or [],
        )

    def _calendar(self, next_event):
        calendar = MagicMock()
        calendar.next_event.return_value = next_event
        return calendar

    def _engine(self, calendar, *, policy=None, **kwargs) -> ProactiveEngine:
        return ProactiveEngine(
            profile=self._profile(),
            calendar_source=calendar,
            companion_policy=policy,
            **kwargs,
        )

    def _next(self, start_at, title="会議"):
        from src.companion.calendar import NextEvent

        return NextEvent(start_at=start_at, title=title)

    def test_fires_when_calendar_has_upcoming_event(self) -> None:
        now = time.time()
        calendar = self._calendar(self._next(now + 300))
        engine = self._engine(calendar)
        fired: list[tuple[str, str]] = []
        engine._callback = lambda trigger, message: fired.append((trigger, message))

        engine._check_schedule_remind()

        self.assertEqual([t for t, _ in fired], ["schedule_remind"])
        self.assertIn("会議", fired[0][1])
        # next_event は実行時の time.time() を now として呼ぶため、事前に取った now とは
        # 微小にズレうる。引数に now キーワードが渡されたことだけ検証する (フレーク防止)。
        assert calendar.next_event.call_args is not None
        assert calendar.next_event.call_args.kwargs.get("now") is not None
        assert calendar.next_event.call_args.kwargs["now"] >= now

    def test_does_not_fire_when_no_event(self) -> None:
        calendar = self._calendar(self._next(None))
        engine = self._engine(calendar)
        fired: list[tuple[str, str]] = []
        engine._callback = lambda trigger, message: fired.append((trigger, message))

        engine._check_schedule_remind()

        self.assertEqual(fired, [])

    def test_does_not_fire_for_past_event(self) -> None:
        calendar = self._calendar(self._next(time.time() - 60))
        engine = self._engine(calendar)
        fired: list[tuple[str, str]] = []
        engine._callback = lambda trigger, message: fired.append((trigger, message))

        engine._check_schedule_remind()

        self.assertEqual(fired, [])

    def test_does_not_fire_beyond_lead_seconds(self) -> None:
        calendar = self._calendar(self._next(time.time() + 3600))
        engine = self._engine(calendar)
        fired: list[tuple[str, str]] = []
        engine._callback = lambda trigger, message: fired.append((trigger, message))

        engine._check_schedule_remind()

        self.assertEqual(fired, [])

    def test_uses_policy_lead_seconds(self) -> None:
        from src.companion.policy import DeterministicProactivePolicy

        now = time.time()
        calendar = self._calendar(self._next(now + 900))
        policy = DeterministicProactivePolicy(schedule_lead_seconds=600)
        engine = self._engine(calendar, policy=policy)
        fired: list[tuple[str, str]] = []
        engine._callback = lambda trigger, message: fired.append((trigger, message))

        engine._check_schedule_remind()

        self.assertEqual(fired, [])

    def test_away_gate_blocks_calendar_remind(self) -> None:
        now = time.time()
        calendar = self._calendar(self._next(now + 300))
        away = CompanionState(
            activity_mode="away",
            present=False,
            focused_since=None,
            interruptible=False,
            display_state="away",
            updated_at=now,
        )
        engine = self._engine(calendar, companion_getter=lambda: away)
        fired: list[tuple[str, str]] = []
        engine._callback = lambda trigger, message: fired.append((trigger, message))

        engine._check_schedule_remind()

        self.assertEqual(fired, [])

    def test_without_calendar_uses_profile_schedule(self) -> None:
        t = (datetime.now() + timedelta(minutes=10)).strftime("%H:%M")
        engine = ProactiveEngine(
            profile=self._profile([{"time": t, "title": "打ち合わせ", "note": ""}]),
        )
        fired: list[tuple[str, str]] = []
        engine._callback = lambda trigger, message: fired.append((trigger, message))

        engine._check_schedule_remind()

        self.assertEqual([t for t, _ in fired], ["schedule_remind"])
        self.assertIn("打ち合わせ", fired[0][1])

    def test_calendar_exception_suppressed(self) -> None:
        calendar = MagicMock()
        calendar.next_event.side_effect = RuntimeError("boom")
        engine = self._engine(calendar)
        fired: list[tuple[str, str]] = []
        engine._callback = lambda trigger, message: fired.append((trigger, message))

        engine._check_schedule_remind()

        self.assertEqual(fired, [])


class CompanionWiringTest(unittest.TestCase):
    def test_create_proactive_bridge_forwards_companion_getter(self) -> None:
        getter = lambda: None
        enabled_cfg = ProactiveConfig(enabled=True, channel_id=1)
        with (
            patch(
                "src.discord_bot.proactive_bridge.ProactiveConfig.from_env",
                return_value=enabled_cfg,
            ),
            patch(
                "src.discord_bot.proactive_bridge._build_monitor_context",
                return_value=None,
            ),
            patch(
                "src.discord_bot.proactive_bridge.UserProfile",
                autospec=True,
            ) as mock_profile,
        ):
            mock_profile.return_value.load.return_value = None
            with patch(
                "src.discord_bot.proactive_bridge.ProactiveBridge",
                autospec=True,
            ) as mock_bridge_cls:
                result = create_proactive_bridge(
                    _state(), _FakeBot(None), companion_getter=getter
                )
                _, kwargs = mock_bridge_cls.call_args
                self.assertIs(kwargs["companion_getter"], getter)
                self.assertIs(result, mock_bridge_cls.return_value)

    def test_companion_getter_passed_into_proactive_engine(self) -> None:
        getter = lambda: None
        bridge = ProactiveBridge(
            bot=_FakeBot(None),
            state=_state(),
            config=ProactiveConfig(enabled=True, channel_id=1),
            profile=None,
            companion_getter=getter,
        )
        self.assertIs(bridge.engine.companion_getter, getter)

    def test_companion_getter_none_falls_back_to_default(self) -> None:
        bridge = ProactiveBridge(
            bot=_FakeBot(None),
            state=_state(),
            config=ProactiveConfig(enabled=True, channel_id=1),
            profile=None,
        )
        self.assertIsNone(bridge.engine.companion_getter)

    def test_bot_companion_state_returns_none_when_runtime_unset(self) -> None:
        with patch("src.discord_bot.bot.activity_runtime", None):
            from src.discord_bot.bot import _companion_state

            self.assertIsNone(_companion_state())

    def test_bot_companion_state_returns_state_when_runtime_present(self) -> None:
        state = CompanionState(
            activity_mode="focused",
            present=True,
            focused_since=None,
            interruptible=False,
            display_state="focused",
            updated_at=time.time(),
        )
        runtime = MagicMock()
        runtime.state = state
        with patch("src.discord_bot.bot.activity_runtime", runtime):
            from src.discord_bot.bot import _companion_state

            self.assertIs(_companion_state(), state)

    def test_bot_companion_state_returns_none_on_runtime_error(self) -> None:
        runtime = MagicMock()

        def _raise() -> CompanionState:
            raise RuntimeError("boom")

        type(runtime).state = property(_raise)
        with patch("src.discord_bot.bot.activity_runtime", runtime):
            from src.discord_bot.bot import _companion_state

            self.assertIsNone(_companion_state())

    def test_bridge_gates_proactive_when_companion_focused(self) -> None:
        focused = CompanionState(
            activity_mode="focused",
            present=True,
            focused_since=None,
            interruptible=False,
            display_state="focused",
            updated_at=time.time(),
        )
        engine = ProactiveEngine(
            profile=SimpleNamespace(name="", get_today_schedule=lambda: []),
            companion_getter=lambda: focused,
        )
        fired: list[tuple[str, str]] = []
        engine._callback = lambda trigger, message: fired.append((trigger, message))
        engine._session_start_time = time.time() - 3 * 3600
        engine._last_user_activity = time.time()

        engine._check_break_suggest()

        self.assertEqual(fired, [])

    def test_bridge_keeps_proactive_when_companion_idle(self) -> None:
        idle = CompanionState(
            activity_mode="idle",
            present=True,
            focused_since=None,
            interruptible=True,
            display_state="idle",
            updated_at=time.time(),
        )
        engine = ProactiveEngine(
            profile=SimpleNamespace(name="", get_today_schedule=lambda: []),
            companion_getter=lambda: idle,
        )
        fired: list[tuple[str, str]] = []
        engine._callback = lambda trigger, message: fired.append((trigger, message))
        engine._session_start_time = time.time() - 3 * 3600
        engine._last_user_activity = time.time()

        engine._check_break_suggest()

        self.assertEqual([t for t, _ in fired], ["break_suggest"])


class ProactivePersistenceTest(unittest.TestCase):
    def test_record_rejection_writes_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_path = str(Path(tmp) / "cooldown.json")
            engine = ProactiveEngine(profile=SimpleNamespace(), state_path=state_path)

            engine.record_rejection("conversation_start", now=12345.0)

            data = json.loads(Path(state_path).read_text(encoding="utf-8"))
            self.assertEqual(data, {"conversation_start": 12345.0})

    def test_new_engine_loads_cooldown_from_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_path = str(Path(tmp) / "cooldown.json")
            first = ProactiveEngine(profile=SimpleNamespace(), state_path=state_path)
            first.record_rejection("conversation_start", now=time.time())

            second = ProactiveEngine(profile=SimpleNamespace(), state_path=state_path)

            self.assertIn("conversation_start", second._last_fired)

    def test_record_rejection_sets_cooldown(self) -> None:
        engine = ProactiveEngine(
            profile=SimpleNamespace(),
            conversation_interval=3600,
        )
        with patch("src.persona.proactive.time.time", return_value=1000.0):
            engine.record_rejection("conversation_start")
            self.assertFalse(engine._can_fire("conversation_start"))

    def test_state_path_none_does_not_save(self) -> None:
        engine = ProactiveEngine(profile=SimpleNamespace())
        with patch.object(engine, "_save_state") as mock_save:
            engine.record_rejection("conversation_start")
            mock_save.assert_not_called()

    def test_bridge_passes_state_path_to_engine(self) -> None:
        cfg = ProactiveConfig(
            enabled=True,
            channel_id=1,
            conversation_state_path="data/proactive/conversation_state.json",
        )
        bridge = ProactiveBridge(
            bot=_FakeBot(None),
            state=_state(),
            config=cfg,
            profile=SimpleNamespace(name="", get_today_schedule=lambda: []),
        )
        expected = Path(PROJECT_ROOT) / "data/proactive/conversation_state.json.proactive.json"
        self.assertEqual(Path(bridge.engine.state_path).resolve(), expected.resolve())

    def test_injected_engine_state_path_not_overwritten(self) -> None:
        cfg = ProactiveConfig(
            enabled=True,
            channel_id=1,
            conversation_state_path="data/proactive/conversation_state.json",
        )
        engine = MagicMock()
        engine.state_path = None
        bridge = ProactiveBridge(
            bot=_FakeBot(None),
            state=_state(),
            config=cfg,
            profile=None,
            engine=engine,
            conversation_store=ConversationLoopStore(None, base_interval_sec=3600, reply_timeout_sec=3600),
        )
        self.assertIs(bridge.engine, engine)
        engine.record_rejection.assert_not_called()

    def test_later_command_records_rejection(self) -> None:
        cfg = ProactiveConfig(
            enabled=True,
            channel_id=1,
            conversation_enabled=True,
            conversation_state_path="data/proactive/conversation_state.json",
        )
        store = ConversationLoopStore(None, base_interval_sec=3600, reply_timeout_sec=3600)
        bridge = ProactiveBridge(
            bot=_FakeBot(None),
            state=_state(),
            config=cfg,
            profile=SimpleNamespace(name="", get_today_schedule=lambda: []),
            conversation_store=store,
        )
        store.record_prompt(1, "質問", now=time.time())
        with patch.object(bridge.engine, "record_rejection") as mock_rr:
            response = bridge.handle_conversation_control(1, "あとで")
            self.assertIn("3時間", response)
            mock_rr.assert_called_once_with("conversation_start")

    def test_quiet_today_command_records_rejection(self) -> None:
        cfg = ProactiveConfig(
            enabled=True,
            channel_id=1,
            conversation_enabled=True,
            conversation_state_path="data/proactive/conversation_state.json",
        )
        store = ConversationLoopStore(None, base_interval_sec=3600, reply_timeout_sec=3600)
        bridge = ProactiveBridge(
            bot=_FakeBot(None),
            state=_state(),
            config=cfg,
            profile=SimpleNamespace(name="", get_today_schedule=lambda: []),
            conversation_store=store,
        )
        store.record_prompt(1, "質問", now=time.time())
        with patch.object(bridge.engine, "record_rejection") as mock_rr:
            response = bridge.handle_conversation_control(1, "今日は静かに")
            self.assertIn("静かに", response)
            mock_rr.assert_called_once_with("conversation_start")


if __name__ == "__main__":
    unittest.main()