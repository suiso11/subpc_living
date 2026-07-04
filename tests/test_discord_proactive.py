from __future__ import annotations

import asyncio
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.discord_bot.proactive_bridge import (
    ProactiveBridge,
    ProactiveConfig,
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
    )


class ProactiveConfigTest(unittest.TestCase):
    def test_defaults(self) -> None:
        cfg = ProactiveConfig.from_env({})
        self.assertFalse(cfg.enabled)
        self.assertIsNone(cfg.channel_id)
        self.assertTrue(cfg.llm_rewrite)
        self.assertEqual(cfg.check_interval, 60.0)

    def test_parses_values(self) -> None:
        cfg = ProactiveConfig.from_env(
            {
                "DISCORD_PROACTIVE_ENABLED": "true",
                "DISCORD_PROACTIVE_CHANNEL_ID": "12345",
                "DISCORD_PROACTIVE_LLM_REWRITE": "false",
                "DISCORD_PROACTIVE_CHECK_INTERVAL": "15",
            }
        )
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.channel_id, 12345)
        self.assertFalse(cfg.llm_rewrite)
        self.assertEqual(cfg.check_interval, 15.0)

    def test_invalid_channel_and_interval_fall_back(self) -> None:
        cfg = ProactiveConfig.from_env(
            {
                "DISCORD_PROACTIVE_CHANNEL_ID": "not-an-int",
                "DISCORD_PROACTIVE_CHECK_INTERVAL": "abc",
            }
        )
        self.assertIsNone(cfg.channel_id)
        self.assertEqual(cfg.check_interval, 60.0)


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


if __name__ == "__main__":
    unittest.main()
