"""Voice/CLI の SensorPolicy + CLI 同意ゲートのオフライン検証。

実マイク・カメラ・画面・プロセス・モデル・ネットワークを使わず、
parser・env合成・エントリポイント・パイプラインコンストラクタのゲートを検証する。
"""
from __future__ import annotations

import io
import queue
import unittest
from contextlib import ExitStack, redirect_stdout
from types import SimpleNamespace
from unittest import mock

import numpy as np

from src.audio import main as audio_main
from src.audio.pipeline import MicrophoneInputError, VoicePipeline
from src.chat.config import ChatConfig
from src.integrations import google_calendar as google_calendar_module
from src.llm.providers.fake import FakeProvider
from src.llm.registry import ProviderRegistry


class InitProvider(FakeProvider):
    """initialize() の backend チェックに使う fake provider。"""

    def __init__(self) -> None:
        super().__init__(available=True)

    def has_model(self) -> bool:
        return True

    def list_models(self) -> list[str]:
        return ["fake"]


class SensorParserTest(unittest.TestCase):
    """--microphone / --camera / --monitor / --screen の parser 検証。"""

    def _parse(self, *argv):
        return audio_main.build_parser().parse_args(list(argv))

    def test_defaults_all_sensors_false(self):
        args = self._parse()
        self.assertFalse(args.microphone)
        self.assertFalse(args.camera)
        self.assertFalse(args.monitor)
        self.assertFalse(args.screen)
        self.assertFalse(args.no_vision)
        self.assertFalse(args.no_monitor)
        self.assertFalse(args.text_mode)

    def test_affirmative_flags_set_true(self):
        args = self._parse("--microphone", "--camera", "--monitor", "--screen")
        self.assertTrue(args.microphone)
        self.assertTrue(args.camera)
        self.assertTrue(args.monitor)
        self.assertTrue(args.screen)

    def test_deprecated_no_flags_remain_accepted(self):
        args = self._parse("--no-vision", "--no-monitor")
        self.assertTrue(args.no_vision)
        self.assertTrue(args.no_monitor)
        self.assertFalse(args.camera)
        self.assertFalse(args.monitor)


class SensorFlagResolutionTest(unittest.TestCase):
    """CLIフラグ + env (SensorPolicy) の合成と disable override。"""

    def _args(self, **overrides):
        base = dict(
            microphone=False,
            camera=False,
            monitor=False,
            screen=False,
            no_vision=False,
            no_monitor=False,
        )
        base.update(overrides)
        return SimpleNamespace(**base)

    def test_empty_env_and_no_flags_all_false(self):
        flags = audio_main._resolve_sensor_flags(self._args(), env={})
        self.assertEqual(
            flags,
            {
                "microphone": False,
                "camera": False,
                "screen_capture": False,
                "monitor": False,
                "activity": False,
            },
        )

    def test_env_true_enables_all_sensors(self):
        env = {
            "SENSOR_MICROPHONE_ENABLED": "true",
            "SENSOR_CAMERA_ENABLED": "true",
            "SENSOR_SCREEN_CAPTURE_ENABLED": "true",
            "SENSOR_MONITOR_ENABLED": "true",
            "SENSOR_ACTIVITY_ENABLED": "true",
        }
        flags = audio_main._resolve_sensor_flags(self._args(), env=env)
        self.assertTrue(all(flags.values()))

    def test_flags_enable_sensors(self):
        args = self._args(microphone=True, camera=True, monitor=True, screen=True)
        flags = audio_main._resolve_sensor_flags(args, env={})
        self.assertTrue(flags["microphone"])
        self.assertTrue(flags["camera"])
        self.assertTrue(flags["screen_capture"])
        self.assertTrue(flags["monitor"])

    def test_deprecated_no_flags_override_flags(self):
        args = self._args(camera=True, monitor=True, no_vision=True, no_monitor=True)
        flags = audio_main._resolve_sensor_flags(args, env={})
        self.assertFalse(flags["camera"])
        self.assertFalse(flags["monitor"])
        self.assertFalse(flags["microphone"])
        self.assertFalse(flags["screen_capture"])

    def test_deprecated_no_flags_override_env(self):
        env = {"SENSOR_CAMERA_ENABLED": "true", "SENSOR_MONITOR_ENABLED": "true"}
        args = self._args(no_vision=True, no_monitor=True)
        flags = audio_main._resolve_sensor_flags(args, env=env)
        self.assertFalse(flags["camera"])
        self.assertFalse(flags["monitor"])


class VoiceEntrypointGateTest(unittest.TestCase):
    """main() のマイク同意ゲートと resolved boolean の受け渡し。"""

    def _main(self, argv, env=None):
        out = io.StringIO()
        with (
            mock.patch.dict("os.environ", dict(env or {}), clear=True),
            mock.patch.object(audio_main, "_start_companion_activity_runtime") as start,
            mock.patch.object(audio_main, "_stop_companion_activity_runtime"),
            mock.patch.object(audio_main, "run_text_to_speech_mode") as tts_mode,
            mock.patch.object(audio_main, "run_voice_mode") as voice_mode,
            mock.patch("sys.argv", ["audio-main"] + list(argv)),
            redirect_stdout(out),
        ):
            audio_main.main()
        return start, tts_mode, voice_mode

    def test_text_mode_never_requires_or_opens_microphone(self):
        start, tts_mode, voice_mode = self._main(["--text-mode"], env={})
        start.assert_not_called()
        tts_mode.assert_called_once()
        voice_mode.assert_not_called()

    def test_text_mode_skips_activity_runtime_even_when_activity_enabled(self):
        start, tts_mode, voice_mode = self._main(
            ["--text-mode"], env={"SENSOR_ACTIVITY_ENABLED": "true"}
        )
        start.assert_not_called()
        tts_mode.assert_called_once()
        voice_mode.assert_not_called()

    def test_text_mode_never_constructs_activity_runtime(self):
        """text mode では活動収集の構築パスが1回も呼ばれないことを保証する。"""
        with (
            mock.patch.dict(
                "os.environ", {"SENSOR_ACTIVITY_ENABLED": "true"}, clear=True
            ),
            mock.patch.object(
                audio_main,
                "_start_companion_activity_runtime",
                side_effect=AssertionError(
                    "activity runtime must not start in text mode"
                ),
            ),
            mock.patch.object(audio_main, "_stop_companion_activity_runtime"),
            mock.patch.object(audio_main, "run_text_to_speech_mode") as tts_mode,
            mock.patch.object(audio_main, "run_voice_mode") as voice_mode,
            mock.patch("sys.argv", ["audio-main", "--text-mode"]),
            redirect_stdout(io.StringIO()),
        ):
            audio_main.main()
        tts_mode.assert_called_once()
        voice_mode.assert_not_called()

    def test_voice_mode_runs_activity_lifecycle_around_voice(self):
        calls: list[str] = []
        with (
            mock.patch.dict("os.environ", {}, clear=True),
            mock.patch.object(
                audio_main,
                "_start_companion_activity_runtime",
                side_effect=lambda *a, **k: calls.append("start"),
            ),
            mock.patch.object(
                audio_main,
                "_stop_companion_activity_runtime",
                side_effect=lambda *a, **k: calls.append("stop"),
            ),
            mock.patch.object(
                audio_main,
                "run_voice_mode",
                side_effect=lambda *a, **k: calls.append("voice"),
            ),
            mock.patch.object(audio_main, "run_text_to_speech_mode"),
            mock.patch("sys.argv", ["audio-main", "--microphone"]),
            redirect_stdout(io.StringIO()),
        ):
            audio_main.main()
        self.assertEqual(calls, ["start", "voice", "stop"])

    def test_voice_mode_starts_activity_runtime_when_activity_enabled(self):
        start, _, voice_mode = self._main(
            ["--microphone"], env={"SENSOR_ACTIVITY_ENABLED": "true"}
        )
        start.assert_called_once()
        voice_mode.assert_called_once()
        _, kwargs = voice_mode.call_args
        self.assertTrue(kwargs["sensor_flags"]["activity"])

    def test_voice_mode_microphone_flag_proceeds_with_resolved_flags(self):
        start, _, voice_mode = self._main(["--microphone"], env={})
        start.assert_called_once()
        self.assertEqual(voice_mode.call_count, 1)
        _, kwargs = voice_mode.call_args
        flags = kwargs["sensor_flags"]
        self.assertTrue(flags["microphone"])
        self.assertFalse(flags["camera"])
        self.assertFalse(flags["monitor"])
        self.assertFalse(flags["activity"])

    def test_voice_mode_microphone_from_env_proceeds(self):
        _, _, voice_mode = self._main(
            [], env={"SENSOR_MICROPHONE_ENABLED": "true"}
        )
        voice_mode.assert_called_once()

    def test_voice_mode_without_microphone_fails_before_activity_and_pipeline(self):
        with self.assertRaises(SystemExit) as raised:
            with (
                mock.patch.dict("os.environ", {}, clear=True),
                mock.patch.object(
                    audio_main, "_start_companion_activity_runtime"
                ) as start,
                mock.patch.object(audio_main, "run_voice_mode") as voice_mode,
                mock.patch.object(audio_main, "run_text_to_speech_mode"),
                mock.patch("sys.argv", ["audio-main"]),
                redirect_stdout(io.StringIO()),
            ):
                audio_main.main()
        self.assertEqual(raised.exception.code, 1)
        start.assert_not_called()
        voice_mode.assert_not_called()

    def test_wakeword_requires_microphone(self):
        with self.assertRaises(SystemExit) as raised:
            with (
                mock.patch.dict("os.environ", {}, clear=True),
                mock.patch.object(
                    audio_main, "_start_companion_activity_runtime"
                ) as start,
                mock.patch.object(audio_main, "run_voice_mode"),
                mock.patch("sys.argv", ["audio-main", "--wakeword"]),
                redirect_stdout(io.StringIO()),
            ):
                audio_main.main()
        self.assertEqual(raised.exception.code, 1)
        start.assert_not_called()


class SensorSummaryTest(unittest.TestCase):
    """privacy-safe な起動サマリー (有効センサー名のみ)。"""

    def test_summary_lists_enabled_sensor_names_only(self):
        flags = {
            "microphone": True,
            "camera": False,
            "screen_capture": True,
            "monitor": False,
            "activity": True,
        }
        out = io.StringIO()
        with redirect_stdout(out):
            audio_main._print_sensor_summary(flags, text_mode=False)
        text = out.getvalue()
        self.assertIn("microphone", text)
        self.assertIn("screen_capture", text)
        self.assertIn("activity", text)
        self.assertNotIn("camera", text)
        self.assertNotIn("monitor", text)
        self.assertNotIn("SENSOR_", text)

    def test_summary_none_shown_when_all_disabled(self):
        flags = {
            "microphone": False,
            "camera": False,
            "screen_capture": False,
            "monitor": False,
            "activity": False,
        }
        out = io.StringIO()
        with redirect_stdout(out):
            audio_main._print_sensor_summary(flags, text_mode=False)
        self.assertIn("なし", out.getvalue())

    def test_summary_text_mode_no_sensors(self):
        out = io.StringIO()
        with redirect_stdout(out):
            audio_main._print_sensor_summary({}, text_mode=True)
        self.assertIn("テキストモード", out.getvalue())


def run_initialize(pipeline: VoicePipeline) -> tuple[bool, str]:
    output = io.StringIO()
    with redirect_stdout(output):
        ok = pipeline.initialize()
    return ok, output.getvalue()


def _build_sensor_pipeline(
    failing_factory: str | None = None,
    config: ChatConfig | None = None,
    **kwargs,
) -> tuple[VoicePipeline, dict, str]:
    """センサー factory をモックして VoicePipeline を構築する。

    failing_factory (vision/screen/monitor) 指定時はその factory が canary 例外を
    投げる。config を渡さない場合は既定 (ollama) を使う。構築中の標準出力も返す。
    """
    config = config or ChatConfig()
    provider = InitProvider()
    registry = ProviderRegistry()
    registry.register(config.resolved_local_provider_id(), provider, local=True)
    service = mock.Mock()
    mocks: dict = {}
    canary = SensorFailureSanitizationTest.SensorCanaryError()
    out = io.StringIO()
    with ExitStack() as stack:
        for target in (
            "src.audio.pipeline.WhisperSTT",
            "src.audio.pipeline.create_tts_backend",
            "src.audio.pipeline.create_vad",
            "src.audio.pipeline.AudioRecorder",
            "src.audio.pipeline.AudioPlayer",
            "src.audio.pipeline.create_web_search_context",
            "src.audio.pipeline.ChatSession",
            "src.audio.pipeline.IdleManager",
            "src.audio.pipeline.GrowthTracker",
            "src.tasks.store.TaskStore",
            "src.tasks.calendar_sync.CalendarContext",
        ):
            stack.enter_context(mock.patch(target))
        stack.enter_context(
            mock.patch(
                "src.audio.pipeline.build_local_service",
                return_value=(service, registry),
            )
        )
        factory_targets = {
            "vision": "src.audio.pipeline.VisionContext",
            "screen": "src.audio.pipeline.create_screen_context",
            "monitor": "src.audio.pipeline.MonitorContext",
        }
        for name, attr in factory_targets.items():
            if name == failing_factory:
                mocks[name] = stack.enter_context(
                    mock.patch(attr, side_effect=canary)
                )
            else:
                mocks[name] = stack.enter_context(mock.patch(attr))
        with redirect_stdout(out):
            pipeline = VoicePipeline(
                chat_config=config,
                enable_rag=False,
                enable_persona=False,
                enable_vision=kwargs.get("enable_vision", False),
                enable_screen=kwargs.get("enable_screen", False),
                enable_monitor=kwargs.get("enable_monitor", False),
            )
    return pipeline, mocks, out.getvalue()


def _build_calendar_pipeline(
    env: dict | None = None, **kwargs
) -> tuple[VoicePipeline, dict, str]:
    """VoicePipeline を env パッチ + GoogleCalendarMCPClient モック付きで構築する。

    Calendar 書き込み opt-in (VOICE_CALENDAR_WRITE_ENABLED) の解決と、外部
    GoogleCalendarMCPClient の構築有無をオフラインで検証するためのヘルパー。
    """
    config = ChatConfig()
    provider = InitProvider()
    registry = ProviderRegistry()
    registry.register(config.resolved_local_provider_id(), provider, local=True)
    service = mock.Mock()
    mocks: dict = {}
    out = io.StringIO()
    with (
        mock.patch.dict("os.environ", dict(env or {}), clear=True),
        ExitStack() as stack,
    ):
        for target in (
            "src.audio.pipeline.WhisperSTT",
            "src.audio.pipeline.create_tts_backend",
            "src.audio.pipeline.create_vad",
            "src.audio.pipeline.AudioRecorder",
            "src.audio.pipeline.AudioPlayer",
            "src.audio.pipeline.create_web_search_context",
            "src.audio.pipeline.ChatSession",
            "src.audio.pipeline.IdleManager",
            "src.audio.pipeline.GrowthTracker",
            "src.tasks.store.TaskStore",
            "src.tasks.calendar_sync.CalendarContext",
        ):
            mocks[target.rsplit(".", 1)[-1]] = stack.enter_context(mock.patch(target))
        mocks["build"] = stack.enter_context(
            mock.patch(
                "src.audio.pipeline.build_local_service",
                return_value=(service, registry),
            )
        )
        mocks["gcal"] = stack.enter_context(
            mock.patch.object(google_calendar_module, "GoogleCalendarMCPClient")
        )
        with redirect_stdout(out):
            pipeline = VoicePipeline(
                chat_config=config,
                enable_rag=False,
                enable_persona=False,
                **kwargs,
            )
    return pipeline, mocks, out.getvalue()


class VoicePipelineSensorDefaultsTest(unittest.TestCase):
    """VoicePipeline コンストラクタ: 既定は全センサー無効で文脈を構築しない。"""

    def _build(self, config=None, provider=None, **kwargs):
        config = config or ChatConfig()
        provider = provider or InitProvider()
        registry = ProviderRegistry()
        registry.register(config.resolved_local_provider_id(), provider, local=True)
        service = mock.Mock()
        mocks: dict = {}
        with ExitStack() as stack:
            mocks["stt"] = stack.enter_context(
                mock.patch("src.audio.pipeline.WhisperSTT")
            )
            mocks["tts"] = stack.enter_context(
                mock.patch("src.audio.pipeline.create_tts_backend")
            )
            mocks["vad"] = stack.enter_context(
                mock.patch("src.audio.pipeline.create_vad")
            )
            mocks["recorder"] = stack.enter_context(
                mock.patch("src.audio.pipeline.AudioRecorder")
            )
            mocks["player"] = stack.enter_context(
                mock.patch("src.audio.pipeline.AudioPlayer")
            )
            mocks["build"] = stack.enter_context(
                mock.patch(
                    "src.audio.pipeline.build_local_service",
                    return_value=(service, registry),
                )
            )
            mocks["web"] = stack.enter_context(
                mock.patch("src.audio.pipeline.create_web_search_context")
            )
            mocks["session"] = stack.enter_context(
                mock.patch("src.audio.pipeline.ChatSession")
            )
            mocks["idle"] = stack.enter_context(
                mock.patch("src.audio.pipeline.IdleManager")
            )
            mocks["growth"] = stack.enter_context(
                mock.patch("src.audio.pipeline.GrowthTracker")
            )
            mocks["tasks"] = stack.enter_context(
                mock.patch("src.tasks.store.TaskStore")
            )
            mocks["calendar"] = stack.enter_context(
                mock.patch("src.tasks.calendar_sync.CalendarContext")
            )
            mocks["vision"] = stack.enter_context(
                mock.patch("src.audio.pipeline.VisionContext")
            )
            mocks["screen"] = stack.enter_context(
                mock.patch("src.audio.pipeline.create_screen_context")
            )
            mocks["monitor"] = stack.enter_context(
                mock.patch("src.audio.pipeline.MonitorContext")
            )
            with redirect_stdout(io.StringIO()):
                pipeline = VoicePipeline(
                    chat_config=config,
                    enable_rag=False,
                    enable_persona=False,
                    **kwargs,
                )
        return pipeline, mocks

    def test_defaults_disable_vision_screen_monitor(self):
        pipeline, mocks = self._build()
        self.assertFalse(pipeline.enable_vision)
        self.assertFalse(pipeline.enable_screen)
        self.assertFalse(pipeline.enable_monitor)
        self.assertIsNone(pipeline.vision_context)
        self.assertIsNone(pipeline.screen_context)
        self.assertIsNone(pipeline.monitor_context)
        mocks["vision"].assert_not_called()
        mocks["screen"].assert_not_called()
        mocks["monitor"].assert_not_called()

    def test_disabled_contexts_never_started_or_sampled(self):
        pipeline, _ = self._build()
        self.assertIsNone(pipeline.vision_context)
        self.assertIsNone(pipeline.screen_context)
        self.assertIsNone(pipeline.monitor_context)
        pipeline.stt = mock.Mock()
        pipeline.tts = mock.Mock()
        pipeline.vad = mock.Mock()
        ok, output = run_initialize(pipeline)
        self.assertTrue(ok)
        self.assertIn("Vision (video input) skipped", output)
        self.assertIn("Monitor (PC log collection) skipped", output)

    def test_enabled_fakes_construct_contexts(self):
        pipeline, mocks = self._build(
            enable_vision=True, enable_screen=True, enable_monitor=True
        )
        self.assertTrue(pipeline.enable_vision)
        self.assertTrue(pipeline.enable_screen)
        self.assertTrue(pipeline.enable_monitor)
        self.assertIsNotNone(pipeline.vision_context)
        self.assertIsNotNone(pipeline.screen_context)
        self.assertIsNotNone(pipeline.monitor_context)
        mocks["vision"].assert_called_once()
        mocks["screen"].assert_called_once()
        mocks["monitor"].assert_called_once()

    def test_enabled_fakes_start_during_initialize(self):
        pipeline, mocks = self._build(
            enable_vision=True, enable_screen=True, enable_monitor=True
        )
        mocks["vision"].return_value.start.return_value = True
        mocks["vision"].return_value.get_status.return_value = {
            "emotion_detection": True
        }
        mocks["screen"].return_value.start.return_value = True
        mocks["screen"].return_value.get_status.return_value = {
            "model": "fake",
            "analysis_interval": 90.0,
        }
        mocks["monitor"].return_value.start.return_value = True
        pipeline.stt = mock.Mock()
        pipeline.tts = mock.Mock()
        pipeline.vad = mock.Mock()
        with mock.patch("time.sleep"):
            ok, _ = run_initialize(pipeline)
        self.assertTrue(ok)
        mocks["vision"].return_value.start.assert_called()
        mocks["screen"].return_value.start.assert_called()
        mocks["monitor"].return_value.start.assert_called()


class SensorFailureSanitizationTest(unittest.TestCase):
    """Vision/Screen/Monitor の構築・起動失敗時のログ健全性 (canary) 検証。

    例外の経路・デバイス・モデル・URL・内容をログへ漏らさず、
    固定メッセージと例外型名だけを出すことを保証する。
    失敗時はセンサーを無効化して残りの初期化を継続する。
    """

    _SENSOR_SECRETS = ("/secret/camera", "model-xyz", "http://secret", "hidden-content")

    class SensorCanaryError(RuntimeError):
        def __str__(self) -> str:
            return " | ".join(SensorFailureSanitizationTest._SENSOR_SECRETS)

    def _build(
        self, failing_factory: str | None = None, **kwargs
    ) -> tuple[VoicePipeline, dict, str]:
        """_build_sensor_pipeline の薄いラッパー。"""
        return _build_sensor_pipeline(failing_factory, **kwargs)

    def assert_sanitized(self, output: str, fixed_fragment: str) -> None:
        self.assertIn(fixed_fragment, output)
        self.assertIn(self.SensorCanaryError.__name__, output)
        for secret in self._SENSOR_SECRETS:
            self.assertNotIn(secret, output)

    def test_vision_construction_failure_is_sanitized(self):
        pipeline, _, out = self._build("vision", enable_vision=True)
        self.assertIsNone(pipeline.vision_context)
        self.assert_sanitized(out, "Vision init skipped")

    def test_screen_construction_failure_is_sanitized(self):
        pipeline, _, out = self._build("screen", enable_screen=True)
        self.assertIsNone(pipeline.screen_context)
        self.assert_sanitized(out, "Screen init skipped")

    def test_monitor_construction_failure_is_sanitized(self):
        pipeline, _, out = self._build("monitor", enable_monitor=True)
        self.assertIsNone(pipeline.monitor_context)
        self.assert_sanitized(out, "Monitor init skipped")

    def test_vision_start_failure_is_sanitized_and_continues(self):
        pipeline, mocks, _ = self._build(
            enable_vision=True, enable_screen=True, enable_monitor=True
        )
        mocks["vision"].return_value.start.side_effect = self.SensorCanaryError()

        ok, out = run_initialize(pipeline)

        self.assertTrue(ok)
        self.assertIsNone(pipeline.vision_context)
        self.assertIsNone(pipeline.session.vision_context)
        self.assert_sanitized(out, "Vision init failed")
        mocks["vision"].return_value.stop.assert_called_once()
        mocks["screen"].return_value.start.assert_called()
        mocks["monitor"].return_value.start.assert_called()

    def test_screen_start_failure_is_sanitized_and_continues(self):
        pipeline, mocks, _ = self._build(
            enable_vision=True, enable_screen=True, enable_monitor=True
        )
        mocks["screen"].return_value.start.side_effect = self.SensorCanaryError()

        ok, out = run_initialize(pipeline)

        self.assertTrue(ok)
        self.assertIsNone(pipeline.screen_context)
        self.assertIsNone(pipeline.session.screen_context)
        self.assert_sanitized(out, "Screen init failed")
        mocks["screen"].return_value.stop.assert_called_once()
        mocks["monitor"].return_value.start.assert_called()

    def test_monitor_start_failure_is_sanitized_and_continues(self):
        pipeline, mocks, _ = self._build(
            enable_vision=True, enable_screen=True, enable_monitor=True
        )
        mocks["monitor"].return_value.start.side_effect = self.SensorCanaryError()

        ok, out = run_initialize(pipeline)

        self.assertTrue(ok)
        self.assertIsNone(pipeline.monitor_context)
        self.assertIsNone(pipeline.session.monitor_context)
        self.assert_sanitized(out, "Monitor init failed")
        mocks["monitor"].return_value.stop.assert_called_once()

    def test_vision_start_false_stops_partial_context_once(self):
        """start() が False を返す場合も部分起動した Vision は一度だけ停止する。"""
        pipeline, mocks, _ = self._build(
            enable_vision=True, enable_screen=True, enable_monitor=True
        )
        mocks["vision"].return_value.start.return_value = False
        mocks["screen"].return_value.start.return_value = True
        mocks["screen"].return_value.get_status.return_value = {
            "model": "fake",
            "analysis_interval": 90.0,
        }
        mocks["monitor"].return_value.start.return_value = True

        ok, out = run_initialize(pipeline)

        self.assertTrue(ok)
        self.assertIsNone(pipeline.vision_context)
        self.assertIsNone(pipeline.session.vision_context)
        self.assertIn("camera could not be opened", out)
        mocks["vision"].return_value.stop.assert_called_once()
        mocks["screen"].return_value.stop.assert_not_called()
        mocks["monitor"].return_value.stop.assert_not_called()

    def test_screen_start_false_stops_partial_context_once(self):
        """start() が False を返す場合も部分起動した Screen は一度だけ停止する。"""
        pipeline, mocks, _ = self._build(
            enable_vision=True, enable_screen=True, enable_monitor=True
        )
        mocks["vision"].return_value.start.return_value = True
        mocks["vision"].return_value.get_status.return_value = {
            "emotion_detection": True
        }
        mocks["screen"].return_value.start.return_value = False
        mocks["monitor"].return_value.start.return_value = True

        ok, out = run_initialize(pipeline)

        self.assertTrue(ok)
        self.assertIsNone(pipeline.screen_context)
        self.assertIsNone(pipeline.session.screen_context)
        self.assertIn("screen could not be captured", out)
        mocks["screen"].return_value.stop.assert_called_once()
        mocks["monitor"].return_value.stop.assert_not_called()

    def test_monitor_start_false_stops_partial_context_once(self):
        """start() が False を返す場合も部分起動した Monitor は一度だけ停止する。"""
        pipeline, mocks, _ = self._build(
            enable_vision=True, enable_screen=True, enable_monitor=True
        )
        mocks["vision"].return_value.start.return_value = True
        mocks["vision"].return_value.get_status.return_value = {
            "emotion_detection": True
        }
        mocks["screen"].return_value.start.return_value = True
        mocks["screen"].return_value.get_status.return_value = {
            "model": "fake",
            "analysis_interval": 90.0,
        }
        mocks["monitor"].return_value.start.return_value = False

        ok, out = run_initialize(pipeline)

        self.assertTrue(ok)
        self.assertIsNone(pipeline.monitor_context)
        self.assertIsNone(pipeline.session.monitor_context)
        self.assertIn("Monitor start failed", out)
        mocks["monitor"].return_value.stop.assert_called_once()


class SensorDiagnosticCp932Test(unittest.TestCase):
    """センサー診断出力が全て cp932 で符号化可能であることの検証。

    Windows開発機 (cp932 コンソール) への移植性のため、Vision/Screen/Monitor の
    診断出力は ASCII のみで構成され、必ず cp932 でエンコードできる。
    """

    _MARKERS = ("Vision", "Screen", "Monitor", "camera", "emotion", "metric")

    def _sensor_lines(self, output: str) -> list[str]:
        return [
            line.strip()
            for line in output.splitlines()
            if any(m in line for m in self._MARKERS)
        ]

    def _assert_cp932(self, lines: list[str]) -> None:
        self.assertTrue(lines, "no sensor diagnostics captured")
        for line in lines:
            try:
                line.encode("cp932")
            except UnicodeEncodeError as exc:
                self.fail(f"not cp932 encodable: {exc} (line={line!r})")

    def _run_init(self, pipeline: VoicePipeline, mocks: dict, mode: str) -> str:
        """initialize をセンサー成功/失敗/例外モードで実行し、出力を返す。"""
        if mode == "ok":
            mocks["vision"].return_value.start.return_value = True
            mocks["vision"].return_value.get_status.return_value = {
                "emotion_detection": True
            }
            mocks["screen"].return_value.start.return_value = True
            mocks["screen"].return_value.get_status.return_value = {
                "model": "fake",
                "analysis_interval": 90.0,
            }
            mocks["monitor"].return_value.start.return_value = True
        elif mode == "false":
            mocks["vision"].return_value.start.return_value = False
            mocks["screen"].return_value.start.return_value = False
            mocks["monitor"].return_value.start.return_value = False
        else:
            canary = SensorFailureSanitizationTest.SensorCanaryError()
            mocks["vision"].return_value.start.side_effect = canary
            mocks["screen"].return_value.start.side_effect = canary
            mocks["monitor"].return_value.start.side_effect = canary
        pipeline.stt = mock.Mock()
        pipeline.tts = mock.Mock()
        pipeline.vad = mock.Mock()
        with mock.patch("time.sleep"):
            ok, out = run_initialize(pipeline)
        self.assertTrue(ok)
        return out

    def test_all_sensor_diagnostics_encode_as_cp932(self) -> None:
        collected: list[str] = []

        # 構築時の失敗診断 (Vision/Screen/Monitor スキップ)
        for factory in ("vision", "screen", "monitor"):
            _, _, out = _build_sensor_pipeline(
                factory, enable_vision=True, enable_screen=True, enable_monitor=True
            )
            collected.extend(self._sensor_lines(out))

        # Screen は Ollama backend 専用 (非 Ollama はスキップ) 診断
        openai_config = ChatConfig(
            local_provider_kind="openai_compatible", model="fake"
        )
        _, _, out = _build_sensor_pipeline(config=openai_config, enable_screen=True)
        collected.extend(self._sensor_lines(out))

        # 起動の成功 / False / 例外 の initialize 診断
        for mode in ("ok", "false", "raise"):
            pipeline, mocks, _ = _build_sensor_pipeline(
                enable_vision=True, enable_screen=True, enable_monitor=True
            )
            collected.extend(self._sensor_lines(self._run_init(pipeline, mocks, mode)))

        # 既定 (全センサー無効) のスキップ診断
        pipeline, _, _ = _build_sensor_pipeline()
        pipeline.stt = mock.Mock()
        pipeline.tts = mock.Mock()
        pipeline.vad = mock.Mock()
        with mock.patch("time.sleep"):
            ok, out = run_initialize(pipeline)
        self.assertTrue(ok)
        collected.extend(self._sensor_lines(out))

        # 非 Ollama backend での Screen スキップ診断
        pipeline, _, _ = _build_sensor_pipeline(
            config=openai_config, enable_screen=True
        )
        pipeline.stt = mock.Mock()
        pipeline.tts = mock.Mock()
        pipeline.vad = mock.Mock()
        with mock.patch("time.sleep"):
            ok, out = run_initialize(pipeline)
        self.assertTrue(ok)
        collected.extend(self._sensor_lines(out))

        self._assert_cp932(collected)


class VadCalibrationSanitizationTest(unittest.TestCase):
    """VAD キャリブレーション診断のサニタイズ + cp932 検証 (canary)。

    マイク録音・VAD calibrate の失敗で raw パス・デバイス・モデル・音声・エラー内容を
    漏らさず、固定メッセージと例外型名だけを出力し、cp932 で符号化できることを保証する。
    失敗時は既定閾値フォールバックで initialize を継続して True を返す。
    """

    _VAD_SECRETS = ("/secret/vad", "vad-model-xyz", "http://vad-secret", "vad-hidden-audio")

    class VadCanaryError(RuntimeError):
        def __str__(self) -> str:
            return " | ".join(VadCalibrationSanitizationTest._VAD_SECRETS)

    def _assert_cp932(self, output: str) -> None:
        for line in output.splitlines():
            if "VAD" in line:
                try:
                    line.encode("cp932")
                except UnicodeEncodeError as exc:
                    self.fail(f"not cp932 encodable: {exc} (line={line!r})")

    def test_vad_calibration_failure_is_sanitized_and_falls_back(self):
        pipeline, _, _ = _build_sensor_pipeline()
        pipeline.vad = mock.Mock()
        pipeline.vad.calibrate.side_effect = self.VadCanaryError()
        pipeline.stt = mock.Mock()
        pipeline.tts = mock.Mock()

        ok, out = run_initialize(pipeline)

        self.assertTrue(ok)
        self.assertIn("VAD calibration failed (using default threshold)", out)
        self.assertIn(self.VadCanaryError.__name__, out)
        for secret in self._VAD_SECRETS:
            self.assertNotIn(secret, out)
        self._assert_cp932(out)

    def test_vad_success_diagnostics_are_cp932(self):
        pipeline, _, _ = _build_sensor_pipeline()
        pipeline.vad = mock.Mock()
        pipeline.stt = mock.Mock()
        pipeline.tts = mock.Mock()

        ok, out = run_initialize(pipeline)

        self.assertTrue(ok)
        self.assertIn("VAD OK", out)
        self.assertNotIn("calibration failed", out)
        self._assert_cp932(out)


class RunVoiceModeFlagPassingTest(unittest.TestCase):
    """run_voice_mode が resolved boolean を VoicePipeline へ渡す検証。"""

    def _args(self, **overrides):
        base = dict(
            stt_model="small",
            tts_voice="jf_alpha",
            vad="auto",
            no_streaming_tts=False,
            no_rag=False,
            microphone=True,
            camera=True,
            monitor=True,
            no_vision=False,
            camera_id=0,
            screen=False,
            no_monitor=False,
            no_persona=False,
            wakeword=False,
            wakeword_model="hey_jarvis",
            wakeword_threshold=0.5,
        )
        base.update(overrides)
        return SimpleNamespace(**base)

    def _run(self, args):
        config = ChatConfig()
        pipeline = mock.Mock()
        pipeline.initialize.return_value = True
        pipeline.run_interactive.return_value = None
        with (
            mock.patch("src.chat.config.ChatConfig.load", return_value=config),
            mock.patch(
                "src.audio.pipeline.VoicePipeline", return_value=pipeline
            ) as vp,
            redirect_stdout(io.StringIO()),
        ):
            audio_main.run_voice_mode(args)
        return vp

    def test_resolved_flags_reach_voice_pipeline(self):
        vp = self._run(self._args())
        _, kwargs = vp.call_args
        self.assertTrue(kwargs["enable_vision"])
        self.assertTrue(kwargs["enable_monitor"])
        self.assertFalse(kwargs["enable_screen"])

    def test_deprecated_no_vision_overrides_camera(self):
        vp = self._run(self._args(no_vision=True))
        _, kwargs = vp.call_args
        self.assertFalse(kwargs["enable_vision"])


class MicrophoneInputErrorBoundaryTest(unittest.TestCase):
    """マイク入力の open/enter/read ループ失敗が MicrophoneInputError へ正規化される検証。

    実マイク・sounddevice を使わず、ローカルな fake recorder/stream だけで
    生例外 (canary) が漏れず、固定メッセージのみになり、ストリーム解放
    (cleanup) が一度だけ実行されることを保証する。
    """

    _CANARY = "raw-mic-canary-secret"

    class CanaryError(RuntimeError):
        def __str__(self) -> str:
            return MicrophoneInputErrorBoundaryTest._CANARY

    class _FakeStream:
        def __init__(self, enter_error=None):
            self.enter_error = enter_error
            self.exit_count = 0

        def __enter__(self):
            if self.enter_error is not None:
                raise self.enter_error
            return self

        def __exit__(self, exc_type, exc, tb):
            self.exit_count += 1
            return False

    class _FakeRecorder:
        def __init__(self, open_error=None, stream=None):
            self.open_error = open_error
            self.stream = stream

        def open_stream(self, callback=None, frame_size=None):
            if self.open_error is not None:
                raise self.open_error
            return self.stream

    class _RaisingQueue:
        def __init__(self, error):
            self.error = error

        def get(self, timeout=None):
            raise self.error

    def _make_pipeline(self):
        pipeline, _, _ = _build_sensor_pipeline()
        return pipeline

    def _assert_normalized(self, call):
        with self.assertRaises(MicrophoneInputError) as raised:
            call()
        self.assertNotIn(self._CANARY, str(raised.exception))
        self.assertEqual(str(raised.exception), MicrophoneInputError._FIXED_MESSAGE)

    def test_open_stream_failure_is_normalized(self):
        pipeline = self._make_pipeline()
        pipeline.recorder = self._FakeRecorder(open_error=self.CanaryError())
        self._assert_normalized(lambda: pipeline._listen_for_speech())

    def test_stream_enter_failure_is_normalized(self):
        pipeline = self._make_pipeline()
        stream = self._FakeStream(enter_error=self.CanaryError())
        pipeline.recorder = self._FakeRecorder(stream=stream)
        self._assert_normalized(lambda: pipeline._listen_for_speech())

    def test_audio_queue_loop_failure_is_normalized(self):
        pipeline = self._make_pipeline()
        stream = self._FakeStream()
        pipeline.recorder = self._FakeRecorder(stream=stream)
        pipeline._audio_queue = self._RaisingQueue(self.CanaryError())
        self._assert_normalized(lambda: pipeline._listen_for_speech())
        self.assertEqual(stream.exit_count, 1)

    def test_wakeword_wait_loop_failure_is_normalized(self):
        pipeline = self._make_pipeline()
        pipeline.enable_wakeword = True
        pipeline.wakeword_detector = mock.Mock()
        stream = self._FakeStream()
        pipeline.recorder = self._FakeRecorder(stream=stream)
        pipeline._running = True
        with mock.patch(
            "src.audio.pipeline.time.sleep", side_effect=self.CanaryError()
        ):
            self._assert_normalized(lambda: pipeline._wait_for_wakeword())
        self.assertEqual(stream.exit_count, 1)

    def test_run_interactive_prints_fixed_message_and_cleanup_once(self):
        pipeline = self._make_pipeline()
        pipeline.recorder = self._FakeRecorder(open_error=self.CanaryError())
        pipeline.cleanup = mock.Mock()
        out = io.StringIO()
        with redirect_stdout(out):
            pipeline.run_interactive()
        self.assertFalse(pipeline._running)
        self.assertIn(MicrophoneInputError._FIXED_MESSAGE, out.getvalue())
        self.assertNotIn(self._CANARY, out.getvalue())
        pipeline.cleanup.assert_called_once()


class SttSanitizationTest(unittest.TestCase):
    """STT ロード・転写失敗診断のサニタイズ (canary) 検証。

    STT のロード失敗・1ターンの転写失敗では、モデル・デバイス・転写テキスト・
    例外内容を漏らさず、固定メッセージ (ASCII) と例外型名だけを出すことを保証する。
    転写失敗は idle 通知を一度だけ行い、idle へ戻して None を返し、
    対話ループをトレースバックなしで継続させる。
    """

    _STT_SECRETS = ("/secret/model", "model-xyz", "device=cuda:7", "raw-transcript-text")

    class SttCanaryError(RuntimeError):
        def __str__(self) -> str:
            return " | ".join(SttSanitizationTest._STT_SECRETS)

    def test_stt_load_failure_is_ascii_type_only_and_returns_false(self):
        pipeline, _, _ = _build_sensor_pipeline()
        pipeline.stt.load.side_effect = self.SttCanaryError()

        ok, out = run_initialize(pipeline)

        self.assertFalse(ok)
        self.assertIn("STT load failed", out)
        self.assertIn(self.SttCanaryError.__name__, out)
        for secret in self._STT_SECRETS:
            self.assertNotIn(secret, out)
        for line in out.splitlines():
            if "STT" in line:
                line.encode("ascii")

    def test_stt_transcribe_failure_recovers_turn_without_leak(self):
        pipeline = VoicePipeline.__new__(VoicePipeline)
        pipeline._state = VoicePipeline.STATE_IDLE
        pipeline.vad = SimpleNamespace(sample_rate=10)
        pipeline._listen_for_speech = lambda: np.ones(4, dtype=np.float32)
        pipeline.stt = mock.Mock()
        pipeline.stt.transcribe.side_effect = self.SttCanaryError()
        pipeline.idle_manager = mock.Mock()
        pipeline._try_register_event = lambda text: None

        out = io.StringIO()
        with redirect_stdout(out):
            result = pipeline.process_voice_turn()

        self.assertIsNone(result)
        self.assertEqual(pipeline.state, VoicePipeline.STATE_IDLE)
        self.assertIn("STT transcription failed", out.getvalue())
        self.assertIn(self.SttCanaryError.__name__, out.getvalue())
        for secret in self._STT_SECRETS:
            self.assertNotIn(secret, out.getvalue())
        pipeline.idle_manager.notify_inference_start.assert_called_once_with(
            wait_for_gpu=True
        )
        pipeline.idle_manager.notify_inference_end.assert_called_once()

    def test_stt_transcribe_failure_does_not_raise(self):
        pipeline = VoicePipeline.__new__(VoicePipeline)
        pipeline._state = VoicePipeline.STATE_IDLE
        pipeline.vad = SimpleNamespace(sample_rate=10)
        pipeline._listen_for_speech = lambda: np.ones(4, dtype=np.float32)
        pipeline.stt = mock.Mock()
        pipeline.stt.transcribe.side_effect = self.SttCanaryError()
        pipeline.idle_manager = None
        pipeline._try_register_event = lambda text: None

        with redirect_stdout(io.StringIO()):
            result = pipeline.process_voice_turn()

        self.assertIsNone(result)
        self.assertEqual(pipeline.state, VoicePipeline.STATE_IDLE)

    def test_default_pipeline_diagnostics_hide_recognized_transcript(self):
        pipeline = VoicePipeline.__new__(VoicePipeline)
        pipeline.config = ChatConfig(emotion_tag_enabled=False)
        pipeline._assistant_service = mock.Mock()
        pipeline._assistant_service.respond_stream.return_value = iter(["応答"])
        pipeline._provider_registry = None
        pipeline.session = self._MinSession()
        pipeline.tts = mock.Mock()
        pipeline.tts.synthesize.return_value = b""
        pipeline.player = mock.Mock()
        pipeline._tts_queue = queue.Queue()
        pipeline._state = VoicePipeline.STATE_IDLE
        pipeline.streaming_tts = False
        pipeline.vad = SimpleNamespace(sample_rate=10)
        pipeline._listen_for_speech = lambda: np.ones(4, dtype=np.float32)
        pipeline.stt = SimpleNamespace(
            transcribe=lambda audio: "secret-user-transcript-xyz"
        )
        pipeline.idle_manager = None
        pipeline._try_register_event = lambda text: None

        out = io.StringIO()
        with redirect_stdout(out):
            response = pipeline.process_voice_turn()

        self.assertEqual(response, "応答")
        self.assertNotIn("secret-user-transcript-xyz", out.getvalue())

    class _MinSession:
        def __init__(self) -> None:
            self.system_prompt = ""
            self._messages = []

        def add_user_message(self, content: str) -> None:
            self._messages.append({"role": "user", "content": content})

        def add_assistant_message(self, content: str) -> None:
            self._messages.append({"role": "assistant", "content": content})

        def build_blocks(self):
            return []


class VoiceCalendarWriteGateTest(unittest.TestCase):
    """音声発話からのカレンダー書き込み独立 opt-in (VOICE_CALENDAR_WRITE_ENABLED)。

    - 既定 / 未設定 / 不正値は fail closed (False)。マイク同意とは独立で、
      マイク同意 (SENSOR_MICROPHONE_ENABLED / --microphone) だけではカレンダーへ
      書き込まない。
    - 解決はコンストラクタで一度だけ行われ、bool 注入でも決まる。
    - 無効時は process_voice_turn が _try_register_event を呼ばず、外部カレンダー
      クライアントも構築しない (通常の LLM / セッション経路へ進む)。
    - 有効時は従来の直接登録挙動を維持する (クライアント構築は従来通り
      TASKS_CALENDAR_SYNC_ENABLED=true も必要)。
    """

    def test_default_false_and_no_client_even_when_tasks_sync_true(self):
        pipeline, mocks, _ = _build_calendar_pipeline(
            env={"TASKS_CALENDAR_SYNC_ENABLED": "true"}
        )
        self.assertFalse(pipeline.voice_calendar_write_enabled)
        self.assertIsNone(pipeline.calendar_client)
        mocks["gcal"].from_env.assert_not_called()

    def test_non_true_values_fail_closed(self):
        for raw in ("1", "yes", "on", "false", "TRUE2", ""):
            pipeline, mocks, _ = _build_calendar_pipeline(
                env={
                    "VOICE_CALENDAR_WRITE_ENABLED": raw,
                    "TASKS_CALENDAR_SYNC_ENABLED": "true",
                }
            )
            self.assertFalse(pipeline.voice_calendar_write_enabled, msg=raw)
            mocks["gcal"].from_env.assert_not_called()

    def test_exact_true_forms_enable(self):
        for raw in ("true", "TRUE", " true "):
            pipeline, _, _ = _build_calendar_pipeline(
                env={"VOICE_CALENDAR_WRITE_ENABLED": raw}
            )
            self.assertTrue(pipeline.voice_calendar_write_enabled, msg=raw)

    def test_env_true_constructs_client_when_tasks_sync_enabled(self):
        pipeline, mocks, _ = _build_calendar_pipeline(
            env={
                "VOICE_CALENDAR_WRITE_ENABLED": "true",
                "TASKS_CALENDAR_SYNC_ENABLED": "true",
            }
        )
        self.assertTrue(pipeline.voice_calendar_write_enabled)
        self.assertIsNotNone(pipeline.calendar_client)
        mocks["gcal"].from_env.assert_called_once()

    def test_env_true_without_tasks_sync_does_not_construct_client(self):
        pipeline, mocks, _ = _build_calendar_pipeline(
            env={"VOICE_CALENDAR_WRITE_ENABLED": "true"}
        )
        self.assertTrue(pipeline.voice_calendar_write_enabled)
        self.assertIsNone(pipeline.calendar_client)
        mocks["gcal"].from_env.assert_not_called()

    def test_bool_injection_overrides_env(self):
        pipeline, mocks, _ = _build_calendar_pipeline(
            env={"VOICE_CALENDAR_WRITE_ENABLED": "true"},
            voice_calendar_write_enabled=False,
        )
        self.assertFalse(pipeline.voice_calendar_write_enabled)
        mocks["gcal"].from_env.assert_not_called()

        pipeline, mocks, _ = _build_calendar_pipeline(
            env={}, voice_calendar_write_enabled=True
        )
        self.assertTrue(pipeline.voice_calendar_write_enabled)
        mocks["gcal"].from_env.assert_not_called()

    # --- process_voice_turn のゲート ---

    class _MinSession:
        def __init__(self) -> None:
            self.system_prompt = ""
            self._messages = []
            self.assistant_flags = []

        def add_user_message(self, content: str) -> None:
            self._messages.append({"role": "user", "content": content})

        def add_assistant_message(
            self, content: str, *, store_memory: bool = True, record_growth: bool = True
        ) -> None:
            self._messages.append({"role": "assistant", "content": content})
            self.assistant_flags.append(
                {"store_memory": store_memory, "record_growth": record_growth}
            )

        def build_blocks(self):
            return []

    def _turn_pipeline(self, enabled=None, event_reply=None):
        pipeline = VoicePipeline.__new__(VoicePipeline)
        pipeline.config = ChatConfig(emotion_tag_enabled=False)
        pipeline._assistant_service = mock.Mock()
        pipeline._assistant_service.respond_stream.return_value = iter(["応答"])
        pipeline._provider_registry = None
        pipeline.session = self._MinSession()
        pipeline.tts = mock.Mock()
        pipeline.tts.synthesize.return_value = b""
        pipeline.player = mock.Mock()
        pipeline._tts_queue = queue.Queue()
        pipeline._state = VoicePipeline.STATE_IDLE
        pipeline.streaming_tts = False
        pipeline.vad = SimpleNamespace(sample_rate=10)
        pipeline._listen_for_speech = lambda: np.ones(4, dtype=np.float32)
        pipeline.stt = SimpleNamespace(
            transcribe=lambda audio: "明日15時に歯医者の予定入れて"
        )
        pipeline.idle_manager = None
        if enabled is not None:
            pipeline.voice_calendar_write_enabled = enabled
        pipeline._try_register_event = mock.Mock(return_value=event_reply)
        return pipeline

    def test_disabled_skips_event_registration_and_proceeds_to_llm(self):
        pipeline = self._turn_pipeline(enabled=False)
        with redirect_stdout(io.StringIO()):
            response = pipeline.process_voice_turn()
        self.assertEqual(response, "応答")
        pipeline._try_register_event.assert_not_called()
        self.assertEqual(
            pipeline.session._messages,
            [
                {"role": "user", "content": "明日15時に歯医者の予定入れて"},
                {"role": "assistant", "content": "応答"},
            ],
        )

    def test_missing_attribute_defaults_to_disabled(self):
        # __new__ 構築 (constructor 非経由) のインスタンスは既定で無効 (fail closed)。
        pipeline = self._turn_pipeline(enabled=None)
        with redirect_stdout(io.StringIO()):
            response = pipeline.process_voice_turn()
        self.assertEqual(response, "応答")
        pipeline._try_register_event.assert_not_called()

    def test_enabled_calls_event_registration_and_short_circuits(self):
        pipeline = self._turn_pipeline(
            enabled=True, event_reply="予定を登録しました: 8/29 15:00 歯医者"
        )
        with redirect_stdout(io.StringIO()):
            response = pipeline.process_voice_turn()
        self.assertEqual(response, "予定を登録しました: 8/29 15:00 歯医者")
        pipeline._try_register_event.assert_called_once_with(
            "明日15時に歯医者の予定入れて"
        )
        self.assertEqual(
            pipeline.session._messages,
            [
                {"role": "user", "content": "明日15時に歯医者の予定入れて"},
                {
                    "role": "assistant",
                    "content": "予定を登録しました: 8/29 15:00 歯医者",
                },
            ],
        )
        self.assertEqual(
            pipeline.session.assistant_flags,
            [{"store_memory": False, "record_growth": False}],
        )


if __name__ == "__main__":
    unittest.main()