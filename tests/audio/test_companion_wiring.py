from __future__ import annotations

import io
import sys
import unittest
from contextlib import redirect_stdout
from unittest import mock

from src.audio import main as audio_main
from src.audio.pipeline import VoicePipeline
from src.chat.config import ChatConfig
from src.companion.contracts import CompanionState
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

    # ------------------------------------------------------------------ #
    # ProactiveEngine companion_getter wiring tests
    # ------------------------------------------------------------------ #

    def _build_pipeline(self, *, enable_persona: bool, activity_runtime=None):
        """Helper: build a VoicePipeline with all heavy deps mocked.

        Returns (pipeline, proactive_mock) when enable_persona=True,
        otherwise (pipeline, None).
        """
        config = ChatConfig(emotion_tag_enabled=False)
        provider = mock.Mock()
        registry = mock.Mock()
        registry.get.return_value = mock.Mock(provider=provider)
        service = mock.Mock()
        proactive_patcher = mock.patch("src.audio.pipeline.ProactiveEngine")
        base_patches = [
            mock.patch("src.audio.pipeline.WhisperSTT"),
            mock.patch("src.audio.pipeline.create_tts_backend"),
            mock.patch("src.audio.pipeline.create_vad"),
            mock.patch("src.audio.pipeline.AudioRecorder"),
            mock.patch("src.audio.pipeline.AudioPlayer"),
            mock.patch(
                "src.audio.pipeline.build_local_service",
                return_value=(service, registry),
            ),
            mock.patch("src.audio.pipeline.create_web_search_context"),
            mock.patch("src.audio.pipeline.ChatSession"),
            mock.patch("src.audio.pipeline.IdleManager"),
            mock.patch("src.audio.pipeline.GrowthTracker"),
            mock.patch("src.tasks.store.TaskStore"),
            mock.patch("src.tasks.calendar_sync.CalendarContext"),
        ]
        persona_patches = [
            mock.patch("src.audio.pipeline.UserProfile"),
            mock.patch("src.audio.pipeline.ConversationSummarizer"),
            mock.patch("src.audio.pipeline.SessionPreloader"),
        ]
        with __import__("contextlib").ExitStack() as stack:
            for cm in base_patches:
                stack.enter_context(cm)
            for cm in persona_patches:
                stack.enter_context(cm)
            pe_mock = stack.enter_context(proactive_patcher) if enable_persona else None
            pipeline = VoicePipeline(
                chat_config=config,
                enable_rag=False,
                enable_vision=False,
                enable_monitor=False,
                enable_persona=enable_persona,
                activity_runtime=activity_runtime,
            )
        return pipeline, pe_mock

    def test_companion_getter_wired_through_persona(self) -> None:
        """enable_persona=True: ProactiveEngine receives _companion_state as companion_getter."""
        state = CompanionState(
            activity_mode="idle",
            present=True,
            focused_since=None,
            interruptible=True,
            display_state="idle",
            updated_at=1000.0,
        )
        runtime = mock.Mock()
        runtime.state = state
        pipeline, pe_mock = self._build_pipeline(
            enable_persona=True, activity_runtime=runtime,
        )
        # Verify the method returns the runtime's state
        self.assertIs(pipeline._companion_state(), state)
        # Verify ProactiveEngine was constructed with the right companion_getter
        self.assertIsNotNone(pe_mock)
        pe_mock.assert_called_once()
        _, kwargs = pe_mock.call_args
        # bound-method identity differs on each access; compare underlying function
        self.assertIs(
            kwargs["companion_getter"].__func__,
            VoicePipeline._companion_state,
        )
        # Also confirm calling the getter works end-to-end
        self.assertIs(kwargs["companion_getter"](), state)

    def test_companion_state_returns_runtime_state(self) -> None:
        """_companion_state() returns the runtime's .state value."""
        state = CompanionState(
            activity_mode="focused",
            present=True,
            focused_since=900.0,
            interruptible=False,
            display_state="focused",
            updated_at=1000.0,
        )
        runtime = mock.Mock()
        runtime.state = state
        pipeline, _ = self._build_pipeline(
            enable_persona=False, activity_runtime=runtime,
        )
        self.assertIs(pipeline._companion_state(), state)

    def test_companion_state_none_when_no_runtime(self) -> None:
        """_companion_state() returns None when activity_runtime is None."""
        pipeline, _ = self._build_pipeline(
            enable_persona=False, activity_runtime=None,
        )
        self.assertIsNone(pipeline._companion_state())

    def test_companion_state_none_when_runtime_raises(self) -> None:
        """_companion_state() returns None defensively when runtime.state raises."""
        runtime = mock.Mock()
        type(runtime).state = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("broken"))
        )
        pipeline, _ = self._build_pipeline(
            enable_persona=False, activity_runtime=runtime,
        )
        self.assertIsNone(pipeline._companion_state())


if __name__ == "__main__":
    unittest.main()
