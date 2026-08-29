"""Voice diagnostics must not disclose transcripts, paths, or raw exceptions."""
from __future__ import annotations

import io
import logging
import sys
import types
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import src.audio.main as audio_main
import src.tasks.event_intent as event_intent_module

from src.audio.pipeline import VoicePipeline
from src.audio.tts import KokoroTTS
from src.audio.wakeword import WakeWordDetector


class DiagnosticPrivacyTest(unittest.TestCase):
    def test_wakeword_load_failure_and_model_status_are_ascii_and_private(self) -> None:
        secret_path = r"C:\private\wakeword\secret.onnx"
        secret_error = f"provider=CUDA model=secret-model path={secret_path}"

        class FailingModel:
            def __init__(self, **kwargs) -> None:
                raise RuntimeError(secret_error)

        wakeword = types.ModuleType("openwakeword")
        wakeword.models = {"secret-model": {"model_path": secret_path}}
        wakeword_model = types.ModuleType("openwakeword.model")
        wakeword_model.Model = FailingModel

        detector = WakeWordDetector(model_names=["secret-model"])
        with patch.dict(
            sys.modules,
            {"openwakeword": wakeword, "openwakeword.model": wakeword_model},
        ), self.assertLogs("src.audio.wakeword", level=logging.INFO) as captured:
            self.assertFalse(detector.load())

        output = "\n".join(captured.output)
        self.assertIn("wakeword model load failed: RuntimeError", output)
        self.assertNotIn(secret_error, output)
        self.assertNotIn(secret_path, output)
        output.encode("cp932")

    def test_wakeword_missing_and_complete_status_do_not_emit_model_names(self) -> None:
        secret_path = r"C:\private\wakeword\secret.onnx"
        secret_model = "secret-model"

        class WorkingModel:
            def __init__(self, **kwargs) -> None:
                self.models = {secret_model: object()}

        wakeword = types.ModuleType("openwakeword")
        wakeword.models = {}
        wakeword_model = types.ModuleType("openwakeword.model")
        wakeword_model.Model = WorkingModel
        detector = WakeWordDetector(model_names=[secret_model])
        complete_detector = WakeWordDetector()

        with patch.dict(
            sys.modules,
            {"openwakeword": wakeword, "openwakeword.model": wakeword_model},
        ), self.assertLogs("src.audio.wakeword", level=logging.INFO) as captured:
            self.assertFalse(detector.load())
            wakeword.models = {secret_model: {"model_path": secret_path}}
            self.assertTrue(complete_detector.load())

        output = "\n".join(captured.output)
        self.assertIn("wakeword model load complete", output)
        self.assertNotIn(secret_model, output)
        self.assertNotIn(secret_path, output)
        output.encode("cp932")

    def test_tts_chunk_failure_and_success_diagnostics_are_private(self) -> None:
        secret_transcript = "秘密の音声 transcript"
        secret_path = r"C:\private\tts\secret.onnx"
        secret_error = f"failed at {secret_path}: {secret_transcript}"

        class FakeKokoro:
            sess = types.SimpleNamespace(
                get_providers=staticmethod(lambda: ["secret-provider"])
            )

            def __init__(self, *args, **kwargs) -> None:
                pass

        kokoro_module = types.ModuleType("kokoro_onnx")
        kokoro_module.Kokoro = FakeKokoro
        load_tts = KokoroTTS(models_dir=secret_path, lang="en")
        load_output = io.StringIO()
        with patch.dict(sys.modules, {"kokoro_onnx": kokoro_module}), patch.object(
            load_tts, "is_installed", return_value=True
        ), redirect_stdout(load_output):
            load_tts.load()
        load_diagnostics = load_output.getvalue()
        self.assertIn("model load complete", load_diagnostics)
        self.assertNotIn(secret_path, load_diagnostics)
        self.assertNotIn("secret-provider", load_diagnostics)
        load_diagnostics.encode("cp932")

        tts = KokoroTTS(models_dir=secret_path)
        tts.load = lambda: None
        tts._create_chunk = lambda chunk: (_ for _ in ()).throw(
            RuntimeError(secret_error)
        )

        failed_output = io.StringIO()
        with redirect_stdout(failed_output):
            tts.synthesize(secret_transcript)
        failure_diagnostics = failed_output.getvalue()
        self.assertIn("chunk synthesis failed: RuntimeError", failure_diagnostics)
        self.assertIn("all chunks failed", failure_diagnostics)
        self.assertNotIn(secret_transcript, failure_diagnostics)
        self.assertNotIn(secret_path, failure_diagnostics)
        self.assertNotIn(secret_error, failure_diagnostics)
        failure_diagnostics.encode("cp932")

        tts._create_chunk = lambda chunk: (np.ones(100, dtype=np.float32), 24000)
        success_output = io.StringIO()
        with redirect_stdout(success_output):
            tts.synthesize(secret_transcript)
        success_diagnostics = success_output.getvalue()
        self.assertIn("synthesis complete", success_diagnostics)
        self.assertIn("chunks=1", success_diagnostics)
        self.assertNotIn(secret_transcript, success_diagnostics)
        success_diagnostics.encode("cp932")

    def test_session_summary_reports_only_presence_and_count(self) -> None:
        secret_summary = "秘密の要約 transcript"
        secret_fact = "秘密の抽出事実"

        class FakeSummarizer:
            def process_session_end(self, **kwargs):
                return {"summary": secret_summary, "extracted_facts": [secret_fact]}

        class FakeSession:
            turn_count = 2
            messages = [{"role": "user", "content": "secret transcript"}]
            session_id = "secret-session"

        pipeline = VoicePipeline.__new__(VoicePipeline)
        pipeline.summarizer = FakeSummarizer()
        pipeline.session = FakeSession()
        pipeline.llm = object()
        pipeline.profile = None

        output = io.StringIO()
        with redirect_stdout(output):
            pipeline._summarize_session()
        diagnostics = output.getvalue()
        self.assertIn("present=True", diagnostics)
        self.assertIn("extracted_facts=1", diagnostics)
        self.assertNotIn(secret_summary, diagnostics)
        self.assertNotIn(secret_fact, diagnostics)
        diagnostics.encode("cp932")

    @staticmethod
    def _turn_pipeline(*, transcript: str, response: str, event_reply=None):
        class Session:
            def __init__(self) -> None:
                self.session_id = "audio-canary-session"
                self.system_prompt = ""
                self._messages = []
                self.assistant_flags = []

            @property
            def messages(self):
                return list(self._messages)

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

        class Stream:
            def __iter__(self):
                return iter((response,))

            def close(self) -> None:
                pass

        session = Session()
        pipeline = VoicePipeline.__new__(VoicePipeline)
        pipeline.config = SimpleNamespace(emotion_tag_enabled=False)
        pipeline._assistant_service = SimpleNamespace(
            respond_stream=lambda *args, **kwargs: Stream()
        )
        pipeline.session = session
        pipeline.tts = SimpleNamespace(synthesize=lambda text, **kwargs: b"wav")
        pipeline.player = SimpleNamespace(play_wav=lambda *args, **kwargs: None)
        pipeline._state = VoicePipeline.STATE_IDLE
        pipeline.streaming_tts = False
        pipeline.vad = SimpleNamespace(sample_rate=10)
        pipeline._listen_for_speech = lambda: np.ones(4, dtype=np.float32)
        pipeline.stt = SimpleNamespace(transcribe=lambda audio: transcript)
        pipeline.idle_manager = None
        pipeline.voice_calendar_write_enabled = event_reply is not None
        pipeline._try_register_event = lambda text: event_reply
        return pipeline, session

    def test_turn_preserves_transcript_and_response_without_printing_content(self) -> None:
        transcript = "audio transcript canary 8231"
        response = "assistant response canary 8231"
        pipeline, session = self._turn_pipeline(
            transcript=transcript, response=response
        )

        output = io.StringIO()
        with redirect_stdout(output):
            result = pipeline.process_voice_turn()

        self.assertEqual(result, response)
        self.assertEqual(session._messages, [
            {"role": "user", "content": transcript},
            {"role": "assistant", "content": response},
        ])
        diagnostics = output.getvalue()
        self.assertNotIn(transcript, diagnostics)
        self.assertNotIn(response, diagnostics)

    def test_event_reply_is_returned_and_played_without_printing_content(self) -> None:
        transcript = "event transcript canary 8231"
        event_reply = "event response canary 8231"
        pipeline, session = self._turn_pipeline(
            transcript=transcript, response="unused", event_reply=event_reply
        )
        spoken = []
        pipeline.tts.synthesize = lambda text, **kwargs: spoken.append(text) or b"wav"

        output = io.StringIO()
        with redirect_stdout(output):
            result = pipeline.process_voice_turn()

        self.assertEqual(result, event_reply)
        self.assertEqual(spoken, [event_reply])
        self.assertEqual(session._messages, [
            {"role": "user", "content": transcript},
            {"role": "assistant", "content": event_reply},
        ])
        self.assertEqual(
            session.assistant_flags,
            [{"store_memory": False, "record_growth": False}],
        )
        diagnostics = output.getvalue()
        self.assertNotIn(transcript, diagnostics)
        self.assertNotIn(event_reply, diagnostics)

    def test_proactive_content_is_delivered_and_failure_is_type_only(self) -> None:
        proactive_text = "proactive response canary 8231"
        pipeline = VoicePipeline.__new__(VoicePipeline)
        pipeline._running = True
        pipeline._state = VoicePipeline.STATE_IDLE
        spoken = []
        pipeline.tts = SimpleNamespace(
            synthesize=lambda text: spoken.append(text) or b"wav"
        )
        played = []
        pipeline.player = SimpleNamespace(
            play_wav=lambda wav, **kwargs: played.append(wav)
        )

        output = io.StringIO()
        with redirect_stdout(output):
            pipeline._on_proactive_trigger("idle", proactive_text)
        self.assertEqual(spoken, [proactive_text])
        self.assertEqual(played, [b"wav"])
        self.assertNotIn(proactive_text, output.getvalue())

        raw = "raw proactive exception C:\\private\\audio.wav"
        pipeline.tts = SimpleNamespace(
            synthesize=lambda text: (_ for _ in ()).throw(RuntimeError(raw))
        )
        output = io.StringIO()
        with redirect_stdout(output):
            pipeline._on_proactive_trigger("idle", proactive_text)
        diagnostics = output.getvalue()
        self.assertIn("RuntimeError", diagnostics)
        self.assertNotIn(raw, diagnostics)
        self.assertNotIn(proactive_text, diagnostics)

    def test_event_exception_diagnostic_omits_raw_details(self) -> None:
        raw = "raw event exception C:\\private\\calendar.json event canary"
        pipeline = VoicePipeline.__new__(VoicePipeline)
        pipeline.calendar_client = object()
        pipeline.tasks_calendar_id = "calendar-canary"
        pipeline.tasks_timezone = "UTC"
        with patch.object(
            event_intent_module,
            "try_register_event",
            side_effect=RuntimeError(raw),
        ), redirect_stdout(io.StringIO()) as output:
            result = pipeline._try_register_event("event transcript canary")

        self.assertIsNone(result)
        diagnostics = output.getvalue()
        self.assertIn("RuntimeError", diagnostics)
        self.assertNotIn(raw, diagnostics)
        self.assertNotIn("event transcript canary", diagnostics)

    def test_text_mode_tts_failure_keeps_response_and_hides_raw_exception(self) -> None:
        response = "main response canary 8231"
        raw = "raw TTS exception C:\\private\\tts.onnx transcript canary"
        config = SimpleNamespace(
            model="fake-model",
            history_dir="data/test-history",
            max_history_turns=4,
            effective_system_prompt=lambda: "",
            resolved_local_provider_id=lambda: "ollama",
            emotion_tag_enabled=False,
        )

        class FakeTTS:
            voice = "fake-voice"

            def __init__(self) -> None:
                self.seen = []

            def load(self) -> None:
                pass

            def synthesize(self, text: str) -> bytes:
                self.seen.append(text)
                raise RuntimeError(raw)

        class FakeSession:
            session_id = "main-canary-session"
            system_prompt = ""
            turn_count = 1

            def add_user_message(self, text: str) -> None:
                pass

            def add_assistant_message(self, text: str) -> None:
                pass

            def build_blocks(self):
                return []

            def save(self):
                return "saved"

        class FakeClient:
            def is_available(self) -> bool:
                return True

            def has_model(self) -> bool:
                return True

        tts = FakeTTS()
        client = FakeClient()
        registry = SimpleNamespace(
            get=lambda provider_id: SimpleNamespace(provider=client),
            close=lambda: None,
        )
        service = SimpleNamespace(respond_stream=lambda *args, **kwargs: iter((response,)))
        session = FakeSession()
        args = SimpleNamespace(tts_voice="fake-voice")

        with patch("src.chat.config.ChatConfig.load", return_value=config):
            with patch("src.chat.web_search.create_web_search_context", return_value=None), \
                 patch("src.audio.tts_factory.create_tts_backend", return_value=tts), \
                 patch("src.audio.audio_io.AudioPlayer"), \
                 patch("src.assistant.factory.build_local_service", return_value=(service, registry)), \
                 patch("src.chat.session.ChatSession", return_value=session), \
                 patch("src.growth.tracker.GrowthTracker", return_value=None), \
                 patch("builtins.input", side_effect=["main transcript canary", "/quit"]), \
                 redirect_stdout(io.StringIO()) as output:
                audio_main.run_text_to_speech_mode(args)

        diagnostics = output.getvalue()
        self.assertIn(response, diagnostics)
        self.assertNotIn("main transcript canary", diagnostics)
        self.assertEqual(tts.seen, [response])
        self.assertIn("RuntimeError", diagnostics)
        self.assertNotIn(raw, diagnostics)


if __name__ == "__main__":
    unittest.main()
