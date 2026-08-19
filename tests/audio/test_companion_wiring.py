from __future__ import annotations

import io
import sys
import unittest
from contextlib import redirect_stdout
from unittest import mock

from src.audio import main as audio_main
from src.audio.pipeline import VoicePipeline
from src.chat.config import ChatConfig
from src.perception.activity import ActivitySample


class _FakeSource:
    def __init__(self) -> None:
        self.calls = 0

    def sample(self) -> ActivitySample:
        self.calls += 1
        return ActivitySample(
            timestamp=float(self.calls),
            idle_seconds=0.0,
            app_category="work",
        )


_ENV_TRUE = {
    "COMPANION_ACTIVITY_ENABLED": "true",
    "COMPANION_ACTIVITY_POLL_INTERVAL_SECONDS": "5",
    "COMPANION_ACTIVITY_IDLE_THRESHOLD_SECONDS": "300",
    "COMPANION_ACTIVITY_AWAY_THRESHOLD_SECONDS": "1800",
}


class CompanionWiringTest(unittest.TestCase):
    def tearDown(self) -> None:
        runtime = audio_main._activity_runtime
        audio_main._activity_runtime = None
        if runtime is not None:
            runtime.stop(timeout=1.0)

    def test_start_enabled_returns_started_runtime(self) -> None:
        fake = _FakeSource()
        with mock.patch(
            "src.perception.bootstrap.create_activity_source", return_value=fake
        ):
            runtime = audio_main._start_companion_activity_runtime(_ENV_TRUE)
        self.assertIsNotNone(runtime)
        self.assertIs(runtime, audio_main._activity_runtime)
        self.assertTrue(runtime.is_running)

    def test_start_disabled_returns_none(self) -> None:
        for env in ({}, {"COMPANION_ACTIVITY_ENABLED": "false"}):
            with self.subTest(env=env), mock.patch(
                "src.perception.bootstrap.create_activity_source",
                side_effect=AssertionError("must not create source"),
            ):
                self.assertIsNone(audio_main._start_companion_activity_runtime(env))
                self.assertIsNone(audio_main._activity_runtime)

    def test_stop_calls_runtime_stop_and_clears_reference(self) -> None:
        runtime = mock.Mock()
        with mock.patch(
            "src.perception.create_activity_runtime_from_env", return_value=runtime
        ):
            audio_main._start_companion_activity_runtime(_ENV_TRUE)
        self.assertIs(audio_main._activity_runtime, runtime)
        audio_main._stop_companion_activity_runtime()
        runtime.stop.assert_called_once_with()
        self.assertIsNone(audio_main._activity_runtime)

    def test_stop_is_noop_when_no_runtime(self) -> None:
        audio_main._activity_runtime = None
        audio_main._stop_companion_activity_runtime()
        self.assertIsNone(audio_main._activity_runtime)

    def test_main_starts_and_stops_runtime(self) -> None:
        calls = []
        with mock.patch.object(
            audio_main,
            "_start_companion_activity_runtime",
            side_effect=lambda: calls.append("start"),
        ), mock.patch.object(
            audio_main,
            "_stop_companion_activity_runtime",
            side_effect=lambda: calls.append("stop"),
        ), mock.patch.object(audio_main, "run_text_to_speech_mode"), mock.patch(
            "sys.argv", ["audio-main", "--text-mode"]
        ):
            with redirect_stdout(io.StringIO()):
                audio_main.main()
        self.assertEqual(calls, ["start", "stop"])

    def test_voice_pipeline_stores_activity_runtime(self) -> None:
        runtime = mock.Mock()
        config = ChatConfig(emotion_tag_enabled=False)
        provider = mock.Mock()
        registry = mock.Mock()
        registry.get.return_value = mock.Mock(provider=provider)
        service = mock.Mock()
        with mock.patch("src.audio.pipeline.WhisperSTT"), \
             mock.patch("src.audio.pipeline.create_tts_backend"), \
             mock.patch("src.audio.pipeline.create_vad"), \
             mock.patch("src.audio.pipeline.AudioRecorder"), \
             mock.patch("src.audio.pipeline.AudioPlayer"), \
             mock.patch(
                 "src.audio.pipeline.build_local_service",
                 return_value=(service, registry),
             ), \
             mock.patch("src.audio.pipeline.create_web_search_context"), \
             mock.patch("src.audio.pipeline.ChatSession"), \
             mock.patch("src.audio.pipeline.IdleManager"), \
             mock.patch("src.audio.pipeline.GrowthTracker"), \
             mock.patch("src.tasks.store.TaskStore"), \
             mock.patch("src.tasks.calendar_sync.CalendarContext"):
            pipeline = VoicePipeline(
                chat_config=config,
                enable_rag=False,
                enable_vision=False,
                enable_monitor=False,
                enable_persona=False,
                activity_runtime=runtime,
            )
        self.assertIs(pipeline.activity_runtime, runtime)


if __name__ == "__main__":
    unittest.main()
