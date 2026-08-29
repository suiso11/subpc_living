"""Discord の screen_capture 政策ゲートのオフライン検証。

canonical 名 (SENSOR_SCREEN_CAPTURE_ENABLED) を優先し、Discord 固有の legacy 名
(DISCORD_SCREEN_CONTEXT_ENABLED) を canonical 未設定時のみ参照する。canonical の
false / 空 / 不正値は legacy の true を上書きする (fail closed)。Web の legacy
(WEB_SCREEN_CONTEXT_ENABLED) は Discord を有効化しない。
"""
from __future__ import annotations

import asyncio
import inspect
import os
import threading
import unittest
from datetime import datetime, timedelta, timezone

import discord
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock
from unittest.mock import MagicMock, patch

from src.chat.session import ChatSession
from src.discord_bot import bot as bot_module
from src.discord_bot.bot import (
    DISCORD_SCREEN_CAPTURE_LEGACY_ENV,
    DISCORD_VOICE_TRAINING_LOG_ENV,
    TTS_ERROR_FALLBACK,
    VOICE_LLM_ERROR_FALLBACK,
    VOICE_STT_ERROR_FALLBACK,
    VOICE_TTS_ERROR_FALLBACK,
    VOICE_TRAINING_SOURCE,
    DiscordConsoleState,
    discord_screen_capture_enabled,
    discord_voice_training_log_enabled,
)
from src.discord_bot import voice_stt as voice_stt_module
from src.discord_bot.voice_stt import (
    DiscordVoiceSTT,
    VoiceSTTConfig,
    VoiceSTTError,
    safe_stt_last_error,
    VOICE_STT_ERR_WORKER,
)
from src.discord_bot.voice_tts import (
    safe_tts_last_error,
    VOICE_TTS_ERR_AUTOREAD,
)
from src.perception.policy import resolve_sensor_policy
from src.screen import create_screen_context

CANONICAL = "SENSOR_SCREEN_CAPTURE_ENABLED"
LEGACY = DISCORD_SCREEN_CAPTURE_LEGACY_ENV
WEB_LEGACY = "WEB_SCREEN_CONTEXT_ENABLED"
MIC_CANONICAL = "SENSOR_MICROPHONE_ENABLED"
MONITOR_CANONICAL = "SENSOR_MONITOR_ENABLED"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class DiscordScreenCaptureResolverTest(unittest.TestCase):
    def test_defaults_all_off(self) -> None:
        self.assertFalse(discord_screen_capture_enabled({}))
        self.assertFalse(discord_screen_capture_enabled({WEB_LEGACY: "false"}))
        self.assertFalse(discord_screen_capture_enabled({LEGACY: "false"}))

    def test_canonical_true_enables(self) -> None:
        self.assertTrue(discord_screen_capture_enabled({CANONICAL: "true"}))
        self.assertTrue(discord_screen_capture_enabled({CANONICAL: " TRUE "}))

    def test_canonical_non_true_values_are_disabled(self) -> None:
        for value in ("", "false", "0", "1", "yes", "on", "no"):
            with self.subTest(value=value):
                self.assertFalse(discord_screen_capture_enabled({CANONICAL: value}))

    def test_canonical_presence_overrides_legacy_true(self) -> None:
        for value in ("false", "", "0", "1", "yes", "on", "no"):
            with self.subTest(value=value):
                self.assertFalse(
                    discord_screen_capture_enabled({CANONICAL: value, LEGACY: "true"})
                )

    def test_canonical_true_beats_legacy_false(self) -> None:
        self.assertTrue(
            discord_screen_capture_enabled({CANONICAL: "true", LEGACY: "false"})
        )

    def test_legacy_only_true_enables(self) -> None:
        self.assertTrue(discord_screen_capture_enabled({LEGACY: "true"}))

    def test_legacy_non_true_values_are_disabled(self) -> None:
        for value in ("", "false", "0", "1", "yes", "on", "no"):
            with self.subTest(value=value):
                self.assertFalse(discord_screen_capture_enabled({LEGACY: value}))

    def test_web_legacy_alias_never_enables_discord(self) -> None:
        self.assertFalse(discord_screen_capture_enabled({WEB_LEGACY: "true"}))
        self.assertFalse(
            discord_screen_capture_enabled({WEB_LEGACY: "true", LEGACY: "false"})
        )
        self.assertTrue(
            discord_screen_capture_enabled({WEB_LEGACY: "true", LEGACY: "true"})
        )

    def test_uses_os_environ_when_env_is_none(self) -> None:
        with mock.patch.dict(os.environ, {CANONICAL: "true"}):
            self.assertTrue(discord_screen_capture_enabled())
        with mock.patch.dict(os.environ, {CANONICAL: "false", LEGACY: "true"}):
            self.assertFalse(discord_screen_capture_enabled())
        with mock.patch.dict(os.environ, {WEB_LEGACY: "true"}):
            self.assertFalse(discord_screen_capture_enabled())


class DiscordVoiceTrainingLogResolverTest(unittest.TestCase):
    """通話 transcript 由来ターンの学習ログ保存 opt-in (fail closed)。

    通常の学習ログ (DISCORD_TRAINING_LOG_ENABLED) とは独立の別 env
    (DISCORD_VOICE_TRAINING_LOG_ENABLED)。明示 true のみ有効で、
    既定 false / 不正値・空・false は無効に倒れる (fail closed)。
    """

    ENV = DISCORD_VOICE_TRAINING_LOG_ENV

    def test_default_off(self) -> None:
        self.assertFalse(discord_voice_training_log_enabled({}))
        self.assertFalse(DiscordConsoleState().voice_training_log_enabled)

    def test_true_enables(self) -> None:
        self.assertTrue(discord_voice_training_log_enabled({self.ENV: "true"}))
        self.assertTrue(discord_voice_training_log_enabled({self.ENV: " TRUE "}))

    def test_non_true_values_are_disabled(self) -> None:
        for value in ("", "false", "0", "1", "yes", "on", "no"):
            with self.subTest(value=value):
                self.assertFalse(discord_voice_training_log_enabled({self.ENV: value}))

    def test_uses_os_environ_when_env_is_none(self) -> None:
        with mock.patch.dict(os.environ, {self.ENV: "true"}):
            self.assertTrue(discord_voice_training_log_enabled())
        with mock.patch.dict(os.environ, {self.ENV: "false"}):
            self.assertFalse(discord_voice_training_log_enabled())


class DiscordVoiceTrainingGateTest(unittest.TestCase):
    """record_auto_reply の二重同意ゲートを検証する。

    通話 transcript 由来ターン (source=VOICE_TRAINING_SOURCE) は、
    DISCORD_VOICE_TRAINING_LOG_ENABLED=true (voice_training_log_enabled) を
    明示しない限り training_log.record_turn へ渡らない。テキスト自動返信
    (discord_auto_reply) は従来どおり常に記録される。TTS 読み上げ・チャンネル
    投稿はこのゲートの対象外 (record_auto_reply だけがスキップされる)。
    """

    def _make_state(self, voice_enabled: bool) -> DiscordConsoleState:
        state = DiscordConsoleState()
        state.config = MagicMock()
        state.llm_profiles = {
            "default": SimpleNamespace(
                name="default",
                model="qwen2.5:7b-instruct-q4_K_M",
                num_ctx=2048,
                num_predict=None,
                temperature=0.7,
            )
        }
        state.channel_profile_map = {}
        state.training_log = MagicMock()
        state.voice_training_log_enabled = voice_enabled
        return state

    @staticmethod
    def _make_message(content: str) -> SimpleNamespace:
        return SimpleNamespace(
            id=101,
            content=content,
            author=SimpleNamespace(id=2),
            channel=SimpleNamespace(id=100),
            guild=SimpleNamespace(id=200),
        )

    def _record(self, state: DiscordConsoleState, message: SimpleNamespace, source: str) -> None:
        state.record_auto_reply(
            message,
            "返答",
            [SimpleNamespace(id=501)],
            user_text=message.content,
            source=source,
        )

    def test_voice_source_disabled_by_default_skips_record(self) -> None:
        state = self._make_state(voice_enabled=False)
        self._record(state, self._make_message("[12:34:56] speaker: こんにちは"), VOICE_TRAINING_SOURCE)
        state.training_log.record_turn.assert_not_called()

    def test_voice_source_disabled_with_training_log_present_skips_record(self) -> None:
        # 通常の学習ログが有効 (training_log が存在) でも、音声 opt-in が無ければ保存しない。
        state = self._make_state(voice_enabled=False)
        self._record(state, self._make_message("[12:34:56] speaker: こんにちは"), VOICE_TRAINING_SOURCE)
        state.training_log.record_turn.assert_not_called()

    def test_voice_source_enabled_records(self) -> None:
        state = self._make_state(voice_enabled=True)
        self._record(state, self._make_message("[12:34:56] speaker: こんにちは"), VOICE_TRAINING_SOURCE)
        state.training_log.record_turn.assert_called_once()
        call = state.training_log.record_turn.call_args.kwargs
        self.assertEqual(call["source"], VOICE_TRAINING_SOURCE)
        self.assertEqual(call["user_text"], "[12:34:56] speaker: こんにちは")

    def test_text_auto_reply_records_regardless_of_voice_opt_in(self) -> None:
        state = self._make_state(voice_enabled=False)
        self._record(state, self._make_message("通常のテキスト投稿"), "discord_auto_reply")
        state.training_log.record_turn.assert_called_once()
        call = state.training_log.record_turn.call_args.kwargs
        self.assertEqual(call["source"], "discord_auto_reply")


class DiscordVoiceTranscriptAdmissionTest(unittest.TestCase):
    """STT transcript admission is limited to the current listening session."""

    STARTED_AT = datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc)

    def _make_state(self) -> DiscordConsoleState:
        state = DiscordConsoleState()
        state.voice_stt = SimpleNamespace(
            listening=True,
            started_at=self.STARTED_AT,
            transcript_channel_id=42,
        )
        state.voice_reply_gate = SimpleNamespace(active=True)
        state.auto_reply_channel_ids = {42}
        return state

    def _message(self, created_at) -> SimpleNamespace:
        return SimpleNamespace(
            content="[12:00:01] speaker: current transcript",
            created_at=created_at,
            author=SimpleNamespace(id=99),
            channel=SimpleNamespace(id=42),
        )

    def test_before_equal_and_after_session_start(self) -> None:
        state = self._make_state()
        for created_at, expected in (
            (self.STARTED_AT - timedelta(microseconds=1), False),
            (self.STARTED_AT, True),
            (self.STARTED_AT + timedelta(microseconds=1), True),
        ):
            with self.subTest(created_at=created_at):
                self.assertEqual(
                    state.is_allowed_voice_transcript_message(
                        self._message(created_at), bot_user_id=99
                    ),
                    expected,
                )

    def test_restart_rejects_message_from_previous_session(self) -> None:
        state = self._make_state()
        old_message = self._message(self.STARTED_AT + timedelta(seconds=1))
        state.voice_stt.started_at = self.STARTED_AT + timedelta(seconds=10)
        self.assertFalse(
            state.is_allowed_voice_transcript_message(old_message, bot_user_id=99)
        )
        self.assertTrue(
            state.is_allowed_voice_transcript_message(
                self._message(self.STARTED_AT + timedelta(seconds=10)), bot_user_id=99
            )
        )

    def test_missing_or_naive_timestamps_are_rejected(self) -> None:
        state = self._make_state()
        cases = (
            (None, self.STARTED_AT),
            (self.STARTED_AT.replace(tzinfo=None), self.STARTED_AT),
            (self.STARTED_AT, None),
            (self.STARTED_AT, self.STARTED_AT.replace(tzinfo=None)),
            (self.STARTED_AT, "2026-07-01T12:00:00Z"),
        )
        for started_at, created_at in cases:
            with self.subTest(started_at=started_at, created_at=created_at):
                state.voice_stt.started_at = started_at
                self.assertFalse(
                    state.is_allowed_voice_transcript_message(
                        self._message(created_at), bot_user_id=99
                    )
                )

    def test_inactive_or_not_listening_is_rejected(self) -> None:
        state = self._make_state()
        message = self._message(self.STARTED_AT)
        state.voice_stt.listening = False
        self.assertFalse(
            state.is_allowed_voice_transcript_message(message, bot_user_id=99)
        )
        state.voice_stt.listening = True
        state.voice_reply_gate.active = False
        self.assertFalse(
            state.is_allowed_voice_transcript_message(message, bot_user_id=99)
        )


class DiscordScreenContextInitTest(unittest.TestCase):
    def _init_with_env(self, env: dict[str, str]) -> tuple[DiscordConsoleState, MagicMock]:
        state = DiscordConsoleState()
        with mock.patch.dict(os.environ, env, clear=True):
            with patch.object(bot_module, "create_screen_context") as create:
                state._init_screen_context()
        return state, create

    def test_default_env_constructs_nothing(self) -> None:
        state, create = self._init_with_env({})
        create.assert_not_called()
        self.assertIsNone(state.screen_context)

    def test_canonical_true_constructs_and_starts(self) -> None:
        fake = MagicMock()
        fake.start.return_value = True
        state = DiscordConsoleState()
        with mock.patch.dict(os.environ, {CANONICAL: "true"}, clear=True):
            with patch.object(
                bot_module, "create_screen_context", return_value=fake
            ) as create:
                state._init_screen_context()
        create.assert_called_once_with()
        fake.start.assert_called_once_with()
        self.assertIs(state.screen_context, fake)

    def test_canonical_false_overrides_legacy_true_constructs_nothing(self) -> None:
        state, create = self._init_with_env({CANONICAL: "false", LEGACY: "true"})
        create.assert_not_called()
        self.assertIsNone(state.screen_context)

    def test_canonical_invalid_overrides_legacy_true_constructs_nothing(self) -> None:
        state, create = self._init_with_env({CANONICAL: "yes", LEGACY: "true"})
        create.assert_not_called()
        self.assertIsNone(state.screen_context)

    def test_legacy_only_true_constructs_and_starts(self) -> None:
        fake = MagicMock()
        fake.start.return_value = True
        state = DiscordConsoleState()
        with mock.patch.dict(os.environ, {LEGACY: "true"}, clear=True):
            with patch.object(
                bot_module, "create_screen_context", return_value=fake
            ) as create:
                state._init_screen_context()
        create.assert_called_once_with()
        fake.start.assert_called_once_with()
        self.assertIs(state.screen_context, fake)

    def test_web_legacy_alone_constructs_nothing(self) -> None:
        state, create = self._init_with_env({WEB_LEGACY: "true"})
        create.assert_not_called()
        self.assertIsNone(state.screen_context)

    def test_construction_failure_leaves_none(self) -> None:
        state = DiscordConsoleState()
        with mock.patch.dict(os.environ, {CANONICAL: "true"}, clear=True):
            with patch.object(
                bot_module,
                "create_screen_context",
                side_effect=RuntimeError("boom"),
            ) as create:
                state._init_screen_context()
        create.assert_called_once_with()
        self.assertIsNone(state.screen_context)

    def test_start_failure_leaves_none(self) -> None:
        fake = MagicMock()
        fake.start.return_value = False
        state = DiscordConsoleState()
        with mock.patch.dict(os.environ, {CANONICAL: "true"}, clear=True):
            with patch.object(
                bot_module, "create_screen_context", return_value=fake
            ) as create:
                state._init_screen_context()
        create.assert_called_once_with()
        fake.start.assert_called_once_with()
        self.assertIsNone(state.screen_context)

    def test_close_stops_screen_context(self) -> None:
        fake = MagicMock()
        state = DiscordConsoleState()
        state.screen_context = fake
        state.task_calendar_sync = None
        state.calendar_pull_worker = None
        state.close()
        fake.stop.assert_called_once_with()

    def test_close_without_screen_context_is_safe(self) -> None:
        state = DiscordConsoleState()
        state.task_calendar_sync = None
        state.calendar_pull_worker = None
        state.close()
        self.assertIsNone(state.screen_context)


class DiscordScreenPartialStartCleanupTest(unittest.TestCase):
    """Discord _init_screen_context の start 失敗時 best-effort stop と exactly-once 検証。

    start が false / 例外で失敗したときは構築済み context を破棄前に stop() を
    ちょうど1回呼ぶ。成功して保持した context は stop() しない (終了時の二重 stop
    防止)。cleanup 例外は生の詳細を漏らさず型名のみをログする。
    """

    CANARY = "canary raw boom C:\\Users\\secret\\screen.jpg"

    @staticmethod
    def _run(env, fake) -> None:
        state = DiscordConsoleState()
        with mock.patch.dict(os.environ, env, clear=True):
            with patch.object(bot_module, "create_screen_context", return_value=fake):
                state._init_screen_context()
        return state

    def test_start_false_stops_once(self) -> None:
        fake = MagicMock()
        fake.start.return_value = False
        state = self._run({CANONICAL: "true"}, fake)
        self.assertIsNone(state.screen_context)
        fake.stop.assert_called_once_with()

    def test_start_raise_stops_once(self) -> None:
        fake = MagicMock()
        fake.start.side_effect = RuntimeError(self.CANARY)
        state = self._run({CANONICAL: "true"}, fake)
        self.assertIsNone(state.screen_context)
        fake.stop.assert_called_once_with()

    def test_start_true_does_not_stop(self) -> None:
        fake = MagicMock()
        fake.start.return_value = True
        state = self._run({CANONICAL: "true"}, fake)
        self.assertIs(state.screen_context, fake)
        fake.stop.assert_not_called()

    def test_stop_raise_logs_type_only_and_keeps_none(self) -> None:
        fake = MagicMock()
        fake.start.return_value = False
        fake.stop.side_effect = RuntimeError(self.CANARY)
        state = DiscordConsoleState()
        with mock.patch.dict(os.environ, {CANONICAL: "true"}, clear=True):
            with patch.object(
                bot_module, "create_screen_context", return_value=fake
            ), patch.object(bot_module.logger, "error") as mock_error:
                state._init_screen_context()
        self.assertIsNone(state.screen_context)
        fake.stop.assert_called_once_with()
        error_texts = " ".join(str(a) for call in mock_error.call_args_list for a in call.args)
        self.assertIn("RuntimeError", error_texts)
        self.assertNotIn("secret", error_texts)
        self.assertNotIn(".jpg", error_texts)
        self.assertNotIn(self.CANARY, error_texts)


class DiscordRemoteScreenIngestGateTest(unittest.TestCase):
    """remote モードの二重ゲートを文書化する。

    Discord 側の screen_capture ゲート (canonical / Discord legacy) に加え、
    SCREEN_CONTEXT_MODE=remote では共有 SensorPolicy.screen_ingest が無ければ
    RemoteScreenContext.start() が False になり、Discord は画面なしで続行する。
    """

    def test_remote_without_ingest_policy_leaves_screen_unavailable(self):
        state = DiscordConsoleState()
        with mock.patch.dict(os.environ, {
            CANONICAL: "true",
            "SCREEN_CONTEXT_MODE": "remote",
        }, clear=True):
            with patch.object(
                bot_module, "create_screen_context", side_effect=create_screen_context
            ) as create:
                state._init_screen_context()
        create.assert_called_once_with()
        self.assertIsNone(state.screen_context)

    def test_remote_with_ingest_policy_starts_screen(self):
        fake = MagicMock()
        fake.start.return_value = True
        state = DiscordConsoleState()
        with mock.patch.dict(os.environ, {
            CANONICAL: "true",
            "SCREEN_CONTEXT_MODE": "remote",
            "SENSOR_SCREEN_INGEST_ENABLED": "true",
        }, clear=True):
            with patch.object(
                bot_module, "create_screen_context", return_value=fake
            ) as create:
                state._init_screen_context()
        create.assert_called_once_with()
        fake.start.assert_called_once_with()
        self.assertIs(state.screen_context, fake)

    def test_remote_canonical_ingest_false_overrides_capture_legacy(self):
        state = DiscordConsoleState()
        with mock.patch.dict(os.environ, {
            LEGACY: "true",
            "SCREEN_CONTEXT_MODE": "remote",
            "SENSOR_SCREEN_INGEST_ENABLED": "false",
        }, clear=True):
            with patch.object(
                bot_module, "create_screen_context", side_effect=create_screen_context
            ) as create:
                state._init_screen_context()
        create.assert_called_once_with()
        self.assertIsNone(state.screen_context)


class DiscordVoiceSTTGateDocumentationTest(unittest.TestCase):
    """voice STT のゲート契約を文書化する。

    管理コンテナの構築は常に行い、実 receive / STT の開始は二重ゲート
    (DISCORD_VOICE_STT_ENABLED と共有 SensorPolicy.microphone) を通過した
    ときだけ /voice join|start で行われる。自動起動はしない。
    """

    def test_constructor_does_not_auto_start(self) -> None:
        with mock.patch.dict(os.environ, {"DISCORD_VOICE_STT_ENABLED": "true"}, clear=True):
            stt = DiscordVoiceSTT(VoiceSTTConfig.from_env(PROJECT_ROOT))
        self.assertTrue(stt.config.enabled)
        self.assertIsNone(stt.voice_client)
        self.assertFalse(stt.connected)
        self.assertFalse(stt.listening)
        self.assertIsNone(stt.started_at)

    def test_disabled_config_blocks_join_before_any_network(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            stt = DiscordVoiceSTT(VoiceSTTConfig.from_env(PROJECT_ROOT))
        self.assertFalse(stt.config.enabled)
        with self.assertRaises(VoiceSTTError):
            asyncio.run(stt.join(MagicMock()))


class _FakeVoiceRecv:
    class VoiceRecvClient:
        pass


class _FakeVoiceChannel(discord.VoiceChannel):
    def __init__(self, guild: SimpleNamespace) -> None:
        self.id = 111
        self.name = "test-vc"
        self.guild = guild

    async def connect(self, *, cls: Any, self_deaf: bool) -> "_FakeVoiceClient":
        return _FakeVoiceClient(self)


class _FakeVoiceClient:
    def __init__(self, channel: _FakeVoiceChannel) -> None:
        self.channel = channel

    def is_connected(self) -> bool:
        return True

    def is_listening(self) -> bool:
        return False

    def listen(self, sink: Any, *, after: Any = None) -> None:
        pass

    async def disconnect(self, *, force: bool = False) -> None:
        pass

    async def move_to(self, channel: _FakeVoiceChannel) -> None:
        self.channel = channel


def _fake_voice_interaction() -> SimpleNamespace:
    guild = SimpleNamespace(voice_client=None)
    channel = _FakeVoiceChannel(guild)
    return SimpleNamespace(
        guild=guild,
        user=SimpleNamespace(voice=SimpleNamespace(channel=channel)),
        channel_id=channel.id,
    )


class DiscordVoiceSTTGateMatrixTest(unittest.TestCase):
    """voice receive/STT の二重ゲート回帰テスト (実 Discord・実音声なし)。

    /voice join / start は接続前に二重ゲートを検証する:
      gate1 = DISCORD_VOICE_STT_ENABLED (feature opt-in)
      gate2 = 共有 SensorPolicy.microphone (canonical: SENSOR_MICROPHONE_ENABLED)
    両方が true のときだけ接続を許す。canonical 名が存在すればその値が確定で、
    false / 空 / 不正値は fail closed (明示 true のみ有効)。
    """

    FEATURE = "DISCORD_VOICE_STT_ENABLED"
    MIC = MIC_CANONICAL

    @staticmethod
    def _make_stt(env: dict[str, str]) -> DiscordVoiceSTT:
        with mock.patch.dict(os.environ, env, clear=True):
            return DiscordVoiceSTT(VoiceSTTConfig.from_env(PROJECT_ROOT))

    def _assert_rejected(self, env: dict[str, str], stt: DiscordVoiceSTT) -> str:
        with self.assertRaises(VoiceSTTError) as ctx:
            asyncio.run(stt.join(MagicMock()))
        self.assertIsNone(stt.voice_client)
        return str(ctx.exception)

    def test_default_both_gates_off_rejects_before_connect(self) -> None:
        stt = self._make_stt({})
        self.assertFalse(stt.config.enabled)
        self.assertFalse(stt.config.microphone_enabled)
        self._assert_rejected({}, stt)

    def test_feature_gate_only_rejects_before_connect(self) -> None:
        stt = self._make_stt({self.FEATURE: "true"})
        self.assertTrue(stt.config.enabled)
        self.assertFalse(stt.config.microphone_enabled)
        msg = self._assert_rejected({self.FEATURE: "true"}, stt)
        self.assertIn(self.MIC, msg)

    def test_microphone_gate_only_rejects_before_connect(self) -> None:
        stt = self._make_stt({self.MIC: "true"})
        self.assertFalse(stt.config.enabled)
        self.assertTrue(stt.config.microphone_enabled)
        msg = self._assert_rejected({self.MIC: "true"}, stt)
        self.assertIn(self.FEATURE, msg)

    def test_start_also_rejects_before_connect_when_gate_false(self) -> None:
        stt = self._make_stt({self.FEATURE: "true"})
        with self.assertRaises(VoiceSTTError):
            asyncio.run(stt.start(MagicMock()))
        self.assertIsNone(stt.voice_client)

    def test_canonical_non_true_overrides_feature_enabled(self) -> None:
        for value in ("", "false", "0", "1", "yes", "on", "no"):
            with self.subTest(value=value):
                stt = self._make_stt({self.FEATURE: "true", self.MIC: value})
                self.assertFalse(stt.config.microphone_enabled)
                self._assert_rejected({self.FEATURE: "true", self.MIC: value}, stt)

    def test_both_gates_true_join_with_fake_discord(self) -> None:
        stt = self._make_stt({self.FEATURE: "true", self.MIC: "true"})
        self.assertTrue(stt.config.enabled)
        self.assertTrue(stt.config.microphone_enabled)
        interaction = _fake_voice_interaction()
        with patch.object(voice_stt_module, "voice_recv", _FakeVoiceRecv):
            result = asyncio.run(stt.join(interaction))
        self.assertIn("参加しました", result)
        self.assertIsNotNone(stt.voice_client)
        self.assertTrue(stt.connected)

    def test_shared_resolver_canonical_precedence_for_microphone(self) -> None:
        self.assertTrue(resolve_sensor_policy({self.MIC: "true"}).microphone)
        self.assertFalse(resolve_sensor_policy({self.MIC: "false"}).microphone)
        for value in ("", "0", "1", "yes", "on", "no"):
            with self.subTest(value=value):
                self.assertFalse(resolve_sensor_policy({self.MIC: value}).microphone)


class SensorPolicyMonitorGateDocumentationTest(unittest.TestCase):
    """共有 SensorPolicy.monitor の canonical precedence を文書化する。

    SENSOR_MONITOR_ENABLED の明示 true のみが監視センサーを有効化する。
    false / 空 / 不正値は fail closed。monitor には legacy 別名が無いため、
    canonical の存在だけで確定値が決まる。
    """

    MONITOR = MONITOR_CANONICAL

    def test_canonical_true_enables_monitor(self) -> None:
        self.assertTrue(resolve_sensor_policy({self.MONITOR: "true"}).monitor)
        self.assertTrue(resolve_sensor_policy({self.MONITOR: " TRUE "}).monitor)

    def test_canonical_non_true_values_disable_monitor(self) -> None:
        for value in ("", "false", "0", "1", "yes", "on", "no"):
            with self.subTest(value=value):
                self.assertFalse(resolve_sensor_policy({self.MONITOR: value}).monitor)

    def test_default_off(self) -> None:
        self.assertFalse(resolve_sensor_policy({}).monitor)
        self.assertFalse(resolve_sensor_policy().monitor)


class DiscordVoiceSanitizationCanaryTest(unittest.TestCase):
    """voice STT/TTS の sanitization canary (ゲート維持 + 固定コード保証)。

    last_error は既知の固定コードのみを表示し、canary として注入した生の
    例外文字列・パスが status へ表面化しないことを保証する。参加のための
    consent ゲートは従来どおり VoiceSTTError の固定文言で拒否される。
    """

    CANARY = "canary raw boom C:\\Users\\secret\\voice.wav"

    def test_status_last_error_only_shows_known_codes(self) -> None:
        stt = DiscordVoiceSTT(
            VoiceSTTConfig(
                enabled=True,
                transcript_channel_id=None,
                transcript_dir=Path(PROJECT_ROOT),
            )
        )
        stt.last_error = self.CANARY
        self.assertEqual(safe_stt_last_error(stt.last_error), "-")
        self.assertNotIn(self.CANARY, stt.status_text())
        stt.last_error = VOICE_STT_ERR_WORKER
        self.assertEqual(safe_stt_last_error(stt.last_error), VOICE_STT_ERR_WORKER)
        self.assertIn("voice_last_error: worker_failure", stt.status_text())

    def test_tts_last_error_only_shows_known_codes(self) -> None:
        self.assertEqual(safe_tts_last_error(self.CANARY), "-")
        self.assertEqual(safe_tts_last_error(VOICE_TTS_ERR_AUTOREAD), VOICE_TTS_ERR_AUTOREAD)

    def test_slash_fallback_messages_are_static(self) -> None:
        for text in (VOICE_STT_ERROR_FALLBACK, VOICE_TTS_ERROR_FALLBACK, TTS_ERROR_FALLBACK):
            self.assertTrue(text)
            self.assertNotIn("`", text)
            self.assertNotIn("{", text)

    def test_gate_rejection_keeps_fixed_messages(self) -> None:
        with mock.patch.dict(os.environ, {MIC_CANONICAL: "true"}, clear=True):
            stt = DiscordVoiceSTT(VoiceSTTConfig.from_env(PROJECT_ROOT))
        self.assertTrue(stt.config.microphone_enabled)
        self.assertFalse(stt.config.enabled)
        with self.assertRaises(VoiceSTTError) as ctx:
            asyncio.run(stt.join(MagicMock()))
        msg = str(ctx.exception)
        self.assertIn("DISCORD_VOICE_STT_ENABLED", msg)
        self.assertNotIn(self.CANARY, msg)


class DiscordScreenSanitizationCanaryTest(unittest.TestCase):
    """Discord ScreenContext の sanitization canary (ゲート維持 + 固定診断保証)。

    ScreenContext の構築・start 失敗時に、生の例外文字列・パス・URL・モデル名が
    ログへ表面化しないことを保証する。ゲート (無効時は構築しない / 失敗時は
    screen 機能のみ無効化) は従来どおり維持する。
    """

    CANARY = "canary raw boom C:\\Users\\secret\\screen.jpg https://x.invalid model-name"

    def test_init_failure_logs_no_raw_content_and_preserves_gate(self) -> None:
        state = DiscordConsoleState()
        with mock.patch.dict(os.environ, {CANONICAL: "true"}, clear=True):
            with patch.object(
                bot_module,
                "create_screen_context",
                side_effect=RuntimeError(self.CANARY),
            ) as create:
                with patch.object(bot_module.logger, "error") as mock_error:
                    state._init_screen_context()
        create.assert_called_once_with()
        self.assertIsNone(state.screen_context)
        for call in mock_error.call_args_list:
            text = " ".join(str(a) for a in call.args)
            self.assertNotIn("secret", text)
            self.assertNotIn(".jpg", text)
            self.assertNotIn(".invalid", text)
            self.assertNotIn(self.CANARY, text)

    def test_disabled_policy_never_logs_raw_failure(self) -> None:
        state = DiscordConsoleState()
        with mock.patch.dict(os.environ, {CANONICAL: "false"}, clear=True):
            with patch.object(bot_module, "create_screen_context") as create:
                with patch.object(bot_module.logger, "error") as mock_error:
                    state._init_screen_context()
        create.assert_not_called()
        self.assertIsNone(state.screen_context)
        mock_error.assert_not_called()


class DiscordVoiceLLMReplySanitizationTest(unittest.TestCase):
    """通話 transcript へのLLM返信失敗の sanitization canary。

    handle_voice_reply は失敗時に生の例外テキスト・transcript・パス・URL・モデル名・
    トークン情報を reply へ載せず、固定の非秘密日本語フォールバックのみを使う。
    内部ログは例外型名のみを記録し、生の詳細を漏らさない。
    """

    CANARY = "canary raw boom C:\\Users\\secret\\token https://x.invalid model-name"

    def test_fallback_message_is_static_and_nonsecret(self) -> None:
        self.assertTrue(VOICE_LLM_ERROR_FALLBACK)
        self.assertNotIn("`", VOICE_LLM_ERROR_FALLBACK)
        self.assertNotIn("{", VOICE_LLM_ERROR_FALLBACK)
        self.assertNotIn("error", VOICE_LLM_ERROR_FALLBACK.lower())
        self.assertNotIn(self.CANARY, VOICE_LLM_ERROR_FALLBACK)

    def test_handle_voice_reply_uses_fallback_not_raw_exception(self) -> None:
        source = inspect.getsource(bot_module)
        # 通話返信の失敗経路は固定フォールバック + 型名のみのログへ置き換わっている。
        self.assertIn("await message.reply(VOICE_LLM_ERROR_FALLBACK", source)
        self.assertIn("type(e).__name__", source)

    def test_text_chat_error_path_unchanged(self) -> None:
        # 通常テキスト自動返信の生例外表示は変更しない (scope外)。
        source = inspect.getsource(bot_module)
        self.assertIn("LLM error: `{e}`", source)


class DiscordVoiceReplyRevocationWiringTest(unittest.TestCase):
    """Discord 音声返信の取り消し (revocation) 配線の静的検証。

    - /voice start 成功時のみゲートを activate する
    - /voice stop|leave は debouncer.cancel_all + gate.revoke を STT 停止より先に行う
    - close は同期 best-effort で revoke_sync + cancel_all する
    - on_message は STT listening かつ gate active のときだけ submit する
    - on_flush は世代を捕捉して返信タスクを追跡し、handle_voice_reply は各副作用の
      前に世代チェックを行う
    - テキスト自動返信経路は変更しない
    """

    @staticmethod
    def _sub_source(name: str, *, after: str) -> str:
        source = inspect.getsource(bot_module)
        start = source.index(name)
        return source[start : source.index(after, start)]

    def test_start_activates_only_after_success(self) -> None:
        block = self._sub_source("async def voice_start(", after="async def voice_stop(")
        self.assertLess(block.index("await state.voice_stt.start("), block.index("voice_reply_gate.activate()"))

    def test_stop_cancels_and_revokes_before_stt_stop(self) -> None:
        block = self._sub_source("async def voice_stop(", after="async def voice_leave(")
        cancel_pos = block.index("voice_reply_debouncer.cancel_all()")
        revoke_pos = block.index("await state.voice_reply_gate.revoke()")
        stop_pos = block.index("await state.voice_stt.stop()")
        self.assertLess(cancel_pos, stop_pos)
        self.assertLess(revoke_pos, stop_pos)

    def test_leave_cancels_and_revokes_before_stt_leave(self) -> None:
        block = self._sub_source("async def voice_leave(", after='@voice_group.command(name="say"')
        cancel_pos = block.index("voice_reply_debouncer.cancel_all()")
        revoke_pos = block.index("await state.voice_reply_gate.revoke()")
        leave_pos = block.index("await state.voice_stt.leave()")
        self.assertLess(cancel_pos, leave_pos)
        self.assertLess(revoke_pos, leave_pos)

    def test_close_revokes_synchronously_before_stt_close(self) -> None:
        block = self._sub_source("def close(self)", after="def is_allowed(")
        revoke_pos = block.index("voice_reply_gate.revoke_sync()")
        cancel_pos = block.index("voice_reply_debouncer.cancel_all()")
        stt_close_pos = block.index("self.voice_stt.close()")
        self.assertLess(revoke_pos, stt_close_pos)
        self.assertLess(cancel_pos, stt_close_pos)

    def test_on_message_uses_single_admission_gate_before_debouncer(self) -> None:
        block = self._sub_source(
            "async def on_message(", after="if state.proactive_bridge is not None:"
        )
        gate_pos = block.index(
            "state.is_allowed_voice_transcript_message(message, bot_user_id=bot_user_id)"
        )
        submit_pos = block.index("voice_reply_debouncer.submit(")
        self.assertLess(gate_pos, submit_pos)
        self.assertNotIn("state.voice_stt.listening", block)
        self.assertNotIn("state.voice_reply_gate.active", block)
        self.assertNotIn("register_quick_task", block)
        self.assertNotIn("try_register_event_text", block)
        self.assertNotIn('source="voice"', block)

    def test_admission_gate_checks_current_session_timestamp(self) -> None:
        block = self._sub_source(
            "def is_allowed_voice_transcript_message(", after="def extract_voice_transcript("
        )
        self.assertIn("stt.started_at", block)
        self.assertIn("message.created_at", block)
        self.assertIn("created_at >= started_at", block)
        self.assertIn("_is_valid_aware_datetime", block)

    def test_handle_voice_reply_uses_separated_generation(self) -> None:
        # 音声LLMはセッション履歴と切り離した ask_voice_transcript を使い、
        # 履歴を汚す ask_message_text は使わない。
        block = self._sub_source(
            "async def handle_voice_reply(", after="def _schedule_voice_reply("
        )
        self.assertIn("state.ask_voice_transcript", block)
        self.assertNotIn("state.ask_message_text", block)

    def test_handle_voice_reply_commits_after_sync_recheck_without_await(self) -> None:
        # 生成返却後は await を挟まず世代を同期再チェックしてから、ロック下で履歴
        # コミットする。再チェックとコミットの間に await が無ければ、revoke は
        # コミットへ割り込めない。
        block = self._sub_source(
            "async def handle_voice_reply(", after="def _schedule_voice_reply("
        )
        commit_pos = block.index("session.add_user_message")
        prefix = block[:commit_pos]
        recheck_pos = prefix.rindex("if not _active():")
        between = block[recheck_pos:commit_pos]
        self.assertNotIn("await", between)
        self.assertIn("with lock:", between)

    def test_state_ask_voice_transcript_removes_temp_user(self) -> None:
        source = inspect.getsource(bot_module)
        self.assertIn("def ask_voice_transcript(", source)
        block = self._sub_source("def ask_voice_transcript(", after="def synthesize(")
        self.assertIn("session._messages.append(", block)
        self.assertIn("finally:", block)
        self.assertIn("session._messages.pop()", block)

    def test_voice_reply_tasks_are_tracked_by_gate(self) -> None:
        # 返信タスクと音声自動読み上げタスクの双方が、取り消しゲートに追跡される。
        # 実装内の出現順や周辺の具体的な文言には依存しない。
        schedule_block = self._sub_source(
            "def _schedule_voice_reply(", after="# 断片ごとの即返信を避け"
        )
        handle_block = self._sub_source(
            "async def handle_voice_reply(", after="def _schedule_voice_reply("
        )

        self.assertIn("asyncio.create_task(handle_voice_reply", schedule_block)
        self.assertIn("gate.track(task)", schedule_block)
        self.assertIn("autoread_task = asyncio.create_task(", handle_block)
        self.assertIn("gate.track(autoread_task)", handle_block)

    def test_handle_voice_reply_checks_generation_before_each_side_effect(self) -> None:
        source = inspect.getsource(bot_module)
        self.assertIn("def handle_voice_reply(", source)
        self.assertIn("gate.is_active(generation)", source)
        # 副作用 (LLM前・to_thread後・返信・学習・TTS・リアクション) ごとの
        # 世代チェックが複数箇所に存在する。
        self.assertGreaterEqual(source.count("if not _active():"), 6)

    def test_state_initializes_gate_and_imports_class(self) -> None:
        source = inspect.getsource(bot_module)
        self.assertIn("VoiceReplyGenerationGate", source)
        self.assertIn("self.voice_reply_gate = VoiceReplyGenerationGate()", source)

    def test_text_chat_auto_reply_unchanged(self) -> None:
        source = inspect.getsource(bot_module)
        self.assertIn("LLM error: `{e}`", source)


class DiscordVoiceHistoryBarrierCanaryTest(unittest.TestCase):
    """音声transcriptのLLM生成がセッション履歴へ漏れないことを保証する canary バリア。

    ask_voice_transcript はセッションロック下で transcript を一時的に user メッセージ
    として要求構築に使うだけに留め、正常・例外を問わず返却前に必ず除去する。履歴コミット
    は handle_voice_reply が世代チェック後に別途行うため、生成時点では永続履歴・RAG・
    成長台帳へ何も書き込まれない。
    """

    CANARY = "canary voice secret history leak 8848"

    def _make_state(self) -> DiscordConsoleState:
        state = DiscordConsoleState()
        state.llm = MagicMock()
        state.config = MagicMock()
        state.config.emotion_tag_enabled = False
        state.llm_profiles = {
            "default": SimpleNamespace(
                name="default",
                model="m",
                num_ctx=2048,
                num_predict=None,
                temperature=0.7,
            )
        }
        service = MagicMock()
        service.respond.return_value = (SimpleNamespace(text="こんにちは"), None)
        state._service_for_profile = MagicMock(return_value=service)
        return state

    def test_generation_leaves_no_history_and_returns_clean_response(self) -> None:
        state = self._make_state()
        session = ChatSession(system_prompt="system", max_history_turns=20)
        lock = threading.Lock()
        response, emotion = state.ask_voice_transcript(
            session, lock, state.llm_profiles["default"], self.CANARY
        )
        self.assertEqual(response, "こんにちは")
        self.assertEqual(emotion, "neutral")
        self.assertEqual(session.messages, [])
        self.assertNotIn(self.CANARY, str(session.messages))

    def test_generation_exception_leaves_no_history(self) -> None:
        state = self._make_state()
        state._service_for_profile.return_value.respond.side_effect = RuntimeError("boom")
        session = ChatSession(system_prompt="system", max_history_turns=20)
        lock = threading.Lock()
        with self.assertRaises(RuntimeError):
            state.ask_voice_transcript(
                session, lock, state.llm_profiles["default"], self.CANARY
            )
        self.assertEqual(session.messages, [])
        self.assertNotIn(self.CANARY, str(session.messages))

    def test_generation_preserves_existing_history(self) -> None:
        state = self._make_state()
        session = ChatSession(system_prompt="system", max_history_turns=20)
        session.add_user_message("既存の発言")
        session.add_assistant_message("既存の返答")
        before = list(session.messages)
        lock = threading.Lock()
        response, _emotion = state.ask_voice_transcript(
            session, lock, state.llm_profiles["default"], self.CANARY
        )
        self.assertEqual(response, "こんにちは")
        self.assertEqual(session.messages, before)
        self.assertNotIn(self.CANARY, str(session.messages))


if __name__ == "__main__":
    unittest.main()