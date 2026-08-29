"""CLI / Voice 経路の AssistantRequest ファクトリ利用と context/history 描画のオフライン parity テスト。

- 検証付きファクトリ (src.assistant.requests.create_request) の挙動
- CLI 経路 (src.chat.main) と Voice 経路 (src.audio.pipeline / src.audio.main) が
  明示的な channel / profile / privacy を渡すこと
- build_blocks → ContextBuilder の描画が CLI と Voice で同一であること
- audio text モードの StreamResult が finally で必ず1回閉じられ、主経路の例外を
  マスクしないこと
- Voice カレンダー直接返信のコミットが store_memory=False / record_growth=False であること
"""
from __future__ import annotations

import io
import queue
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest import mock

import numpy as np

from src.assistant.contracts import AssistantRequest
from src.assistant.requests import create_request
from src.audio.main import _stream_assistant_response, run_text_to_speech_mode
from src.audio.pipeline import VoicePipeline
from src.chat.emotion import EMOTION_TAG_INSTRUCTION
from src.chat.main import run_chat_loop
from src.chat.session import ChatSession
from src.context.builder import ContextBuilder
from src.context.contracts import ContextBlock, ContextMessage
from src.llm.errors import ProviderRequestError


class FakeStreamResult:
    """Offline StreamResult 風オブジェクト。close() 回数を記録する。"""

    def __init__(self, chunks) -> None:
        self.chunks = chunks
        self.close_count = 0

    def __iter__(self):
        yield from self.chunks

    def close(self) -> None:
        self.close_count += 1


class FailingStreamResult(FakeStreamResult):
    """反復途中で例外を送出する StreamResult 風オブジェクト。"""

    def __init__(self, chunks, exc) -> None:
        super().__init__(chunks)
        self.exc = exc

    def __iter__(self):
        yield from self.chunks
        raise self.exc


class RecordingFactory:
    """create_request を記録しつつ実ファクトリへ委譲するスパイ。"""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(dict(kwargs))
        return create_request(**kwargs)


class InputSequence:
    def __init__(self, *values: str) -> None:
        self._values = iter(values)

    def __call__(self, prompt: str) -> str:
        try:
            return next(self._values)
        except StopIteration as exc:
            raise EOFError from exc


class NoopTTS:
    def synthesize(self, text, *, style=None) -> bytes:
        return b""


class NoopPlayer:
    def play_wav(self, wav_data, *, blocking=True) -> None:
        pass


class RecordingSession:
    """コミット引数 (store_memory / record_growth) を記録するセッション風オブジェクト。"""

    def __init__(self, session_id: str = "voice-session", system_prompt: str = "sys") -> None:
        self.session_id = session_id
        self.system_prompt = system_prompt
        self._messages: list[dict] = []
        self.user_messages: list[str] = []
        self.assistant_commits: list[dict] = []

    def add_user_message(self, content: str) -> None:
        self.user_messages.append(content)

    def add_assistant_message(
        self,
        content: str,
        *,
        store_memory: bool = True,
        record_growth: bool = True,
    ) -> None:
        self.assistant_commits.append(
            {"content": content, "store_memory": store_memory, "record_growth": record_growth}
        )

    def build_blocks(self):
        return ()


class StubBlocksSession:
    """CLI 経路のブロック受け渡しを固定するためのスタブセッション。"""

    def __init__(
        self,
        session_id: str = "cli-session",
        system_prompt: str = "sys",
    ) -> None:
        self.session_id = session_id
        self.system_prompt = system_prompt
        self._messages: list[dict] = []
        self.blocks = (
            ContextBlock(
                source="history",
                content=(
                    ContextMessage(role="user", content="こんにちは"),
                    ContextMessage(role="assistant", content="はい"),
                ),
                local_only=True,
            ),
        )

    def add_user_message(self, content: str) -> None:
        self._messages.append({"role": "user", "content": content})

    def add_assistant_message(
        self, content: str, *, store_memory: bool = True, record_growth: bool = True
    ) -> None:
        self._messages.append({"role": "assistant", "content": content})

    def build_blocks(self):
        return self.blocks

    def get_summary(self) -> str:
        return "summary"

    def clear(self) -> None:
        self._messages.clear()

    def save(self) -> str:
        return "saved"


def _make_session(
    history_dir: str,
    system_prompt: str = "sys",
    *,
    emotion_tags: bool = False,
    conversation_source: str = "chat",
) -> ChatSession:
    session = ChatSession(
        system_prompt=system_prompt,
        max_history_turns=20,
        history_dir=history_dir,
        emotion_tags=emotion_tags,
        conversation_source=conversation_source,
    )
    session.add_user_message("こんにちは")
    session.add_assistant_message("はい、こんにちは", store_memory=False, record_growth=False)
    session.add_user_message("今日の予定は？")
    return session


class CreateRequestTest(unittest.TestCase):
    def test_returns_validated_request_with_explicit_metadata(self) -> None:
        request = create_request(
            text="こんにちは",
            conversation_id="cid-1",
            channel="cli",
            profile="chat_auto",
            privacy="local_only",
        )
        self.assertIsInstance(request, AssistantRequest)
        self.assertEqual(request.text, "こんにちは")
        self.assertEqual(request.conversation_id, "cid-1")
        self.assertEqual(request.channel, "cli")
        self.assertEqual(request.profile, "chat_auto")
        self.assertEqual(request.privacy, "local_only")
        # ファクトリは requested_provider / requested_model / allow_cloud を発明しない。
        self.assertIsNone(request.requested_provider)
        self.assertIsNone(request.requested_model)
        self.assertFalse(request.allow_cloud)
        self.assertIsNone(request.request_id)

    def test_rejects_invalid_channel(self) -> None:
        with self.assertRaises(ValueError):
            create_request(
                text="x",
                conversation_id="c",
                channel="bogus",
                profile="chat_auto",
                privacy="local_only",
            )

    def test_rejects_invalid_profile(self) -> None:
        with self.assertRaises(ValueError):
            create_request(
                text="x",
                conversation_id="c",
                channel="cli",
                profile="bogus",
                privacy="local_only",
            )

    def test_rejects_invalid_privacy(self) -> None:
        with self.assertRaises(ValueError):
            create_request(
                text="x",
                conversation_id="c",
                channel="cli",
                profile="chat_auto",
                privacy="bogus",
            )


class CliRequestParityTest(unittest.TestCase):
    def test_run_chat_loop_uses_factory_with_explicit_cli_metadata(self) -> None:
        session = StubBlocksSession()
        config = SimpleNamespace(stream=False)
        service = mock.Mock()
        service.respond.return_value = (SimpleNamespace(text="はい"), None)
        recording = RecordingFactory()

        with mock.patch("src.chat.main.create_request", recording):
            with redirect_stdout(io.StringIO()):
                run_chat_loop(
                    config, session, service, read_input=InputSequence("こんにちは")
                )

        self.assertEqual(len(recording.calls), 1)
        call = recording.calls[0]
        self.assertEqual(call["text"], "こんにちは")
        self.assertEqual(call["conversation_id"], "cli-session")
        self.assertEqual(call["channel"], "cli")
        self.assertEqual(call["profile"], "chat_auto")
        self.assertEqual(call["privacy"], "local_only")

        # CLI は session.build_blocks() の blocks をそのまま service.respond へ渡す。
        service.respond.assert_called_once()
        call_args = service.respond.call_args
        self.assertIs(call_args.args[1], session.blocks)
        self.assertEqual(call_args.kwargs["base_system"], session.system_prompt)

        # 渡された blocks を request.privacy=local_only で描画した共通文脈が正しい。
        rendered = ContextBuilder(session.system_prompt).build_messages(
            session.blocks, privacy="local_only", target_local=True
        )
        self.assertEqual(
            rendered,
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "こんにちは"},
                {"role": "assistant", "content": "はい"},
            ],
        )


class VoiceRequestParityTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    @staticmethod
    def _make_pipeline(service, session) -> VoicePipeline:
        pipeline = VoicePipeline.__new__(VoicePipeline)
        pipeline.config = SimpleNamespace(emotion_tag_enabled=False)
        pipeline._assistant_service = service
        pipeline.session = session
        pipeline.tts = NoopTTS()
        pipeline.player = NoopPlayer()
        pipeline._tts_queue = queue.Queue()
        pipeline._state = VoicePipeline.STATE_IDLE
        return pipeline

    def test_sequential_and_streaming_use_factory_with_voice_metadata(self) -> None:
        session = _make_session(self._tmp.name)
        blocks = session.build_blocks()
        service = mock.Mock()
        service.respond_stream.return_value = FakeStreamResult(("はい", "です", "。"))
        pipeline = self._make_pipeline(service, session)
        recording = RecordingFactory()

        with mock.patch("src.audio.pipeline.create_request", recording):
            with redirect_stdout(io.StringIO()):
                sequential = pipeline._sequential_llm_then_tts(blocks, "こんにちは")
                streaming = pipeline._stream_llm_with_tts(blocks, "こんにちは")

        self.assertEqual(sequential, "はいです。")
        self.assertEqual(streaming, "はいです。")
        self.assertEqual(len(recording.calls), 2)
        for call in recording.calls:
            self.assertEqual(call["text"], "こんにちは")
            self.assertEqual(call["conversation_id"], session.session_id)
            self.assertEqual(call["channel"], "voice")
            self.assertEqual(call["profile"], "voice_fast")
            self.assertEqual(call["privacy"], "local_only")

        # 両経路とも同一の blocks を respond_stream へ渡す。
        self.assertEqual(service.respond_stream.call_count, 2)
        for call in service.respond_stream.call_args_list:
            self.assertIs(call.args[1], blocks)
            self.assertEqual(call.kwargs["base_system"], session.system_prompt)

        # request.privacy=local_only で描画した共通文脈が build_messages と一致する。
        rendered = ContextBuilder(session.system_prompt).build_messages(
            blocks, privacy="local_only", target_local=True
        )
        self.assertEqual(rendered, session.build_messages())


class RenderedContextParityTest(unittest.TestCase):
    """CLI 相当と Voice の build_blocks → ContextBuilder 描画が同一であること。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def test_cli_and_voice_render_identical_common_context_history(self) -> None:
        cli_session = _make_session(self._tmp.name)
        voice_session = _make_session(self._tmp.name)

        cli_blocks = cli_session.build_blocks()
        voice_blocks = voice_session.build_blocks()

        cli_messages = ContextBuilder(cli_session.system_prompt).build_messages(
            cli_blocks, privacy="local_only", target_local=True
        )
        voice_messages = ContextBuilder(voice_session.system_prompt).build_messages(
            voice_blocks, privacy="local_only", target_local=True
        )

        self.assertEqual([b.source for b in cli_blocks], ["history"])
        self.assertEqual([b.source for b in voice_blocks], ["history"])
        self.assertEqual(cli_messages, voice_messages)
        self.assertEqual(cli_messages, cli_session.build_messages())
        self.assertEqual(voice_messages, voice_session.build_messages())

    def test_cli_audio_text_voice_render_identical_common_context_history_emotion(self) -> None:
        sessions = (
            _make_session(
                self._tmp.name, emotion_tags=True, conversation_source="cli"
            ),
            _make_session(
                self._tmp.name, emotion_tags=True, conversation_source="audio_text"
            ),
            _make_session(
                self._tmp.name, emotion_tags=True, conversation_source="voice"
            ),
        )

        block_sources: list[list[str]] = []
        rendered_messages: list[list[dict]] = []
        for session in sessions:
            blocks = session.build_blocks()
            block_sources.append([b.source for b in blocks])
            rendered = ContextBuilder(session.system_prompt).build_messages(
                blocks, privacy="local_only", target_local=True
            )
            rendered_messages.append(rendered)
            self.assertEqual(rendered, session.build_messages())

        # emotion は common context の一部であり、CLI / audio-text / Voice の
        # いずれでも history と同一順で描画される。
        for source_list in block_sources:
            self.assertEqual(source_list, ["emotion", "history"])
        self.assertEqual(rendered_messages[0], rendered_messages[1])
        self.assertEqual(rendered_messages[1], rendered_messages[2])
        self.assertIn(EMOTION_TAG_INSTRUCTION, rendered_messages[0][0]["content"])


class AudioTextStreamCloseTest(unittest.TestCase):
    """audio text モードの _stream_assistant_response の close / error 挙動。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def test_normal_completion_closes_exactly_once_and_returns_text(self) -> None:
        stream = FakeStreamResult(("音", "声", "です"))
        service = mock.Mock()
        service.respond_stream.return_value = stream
        tokens: list[str] = []

        response = _stream_assistant_response(
            service, object(), [], base_system="sys", on_token=tokens.append
        )

        self.assertEqual(response, "音声です")
        self.assertEqual(tokens, ["音", "声", "です"])
        self.assertEqual(stream.close_count, 1)

    def test_empty_response_closes_exactly_once(self) -> None:
        stream = FakeStreamResult(())
        service = mock.Mock()
        service.respond_stream.return_value = stream

        response = _stream_assistant_response(service, object(), [], base_system="sys")

        self.assertEqual(response, "")
        self.assertEqual(stream.close_count, 1)

    def test_generation_error_closes_exactly_once_and_propagates(self) -> None:
        exc = ProviderRequestError("fake", "respond_stream", "planned failure")
        stream = FailingStreamResult(("一部",), exc)
        service = mock.Mock()
        service.respond_stream.return_value = stream

        with self.assertRaises(ProviderRequestError):
            _stream_assistant_response(service, object(), [], base_system="sys")

        self.assertEqual(stream.close_count, 1)

    def test_respond_stream_failure_propagates_without_close(self) -> None:
        exc = ProviderRequestError("fake", "respond_stream", "start failure")
        service = mock.Mock()
        service.respond_stream.side_effect = exc

        with self.assertRaises(ProviderRequestError):
            _stream_assistant_response(service, object(), [], base_system="sys")

    def test_run_text_to_speech_loop_uses_factory_and_closes_stream(self) -> None:
        config = SimpleNamespace(
            model="test-model",
            effective_system_prompt=lambda: "sys",
            max_history_turns=20,
            history_dir=self._tmp.name,
            resolved_local_provider_id=lambda: "ollama",
            emotion_tag_enabled=True,
        )
        session = _make_session(self._tmp.name)
        stream = FakeStreamResult(("音声", "です"))
        service = mock.Mock()
        service.respond_stream.return_value = stream
        client = SimpleNamespace(is_available=lambda: True, has_model=lambda: True)
        registry = mock.Mock()
        registry.get.return_value = SimpleNamespace(provider=client)
        tts = SimpleNamespace(
            load=lambda: None, synthesize=lambda text, style=None: b"", voice="fake_voice"
        )
        player = NoopPlayer()
        recording = RecordingFactory()
        inputs = iter(["こんにちは"])

        def _fake_input(prompt: str = "") -> str:
            try:
                return next(inputs)
            except StopIteration:
                raise KeyboardInterrupt

        with (
            mock.patch("src.chat.config.ChatConfig.load", return_value=config),
            mock.patch("src.chat.config.validate_local_provider_kind", return_value="ollama"),
            mock.patch("src.chat.web_search.create_web_search_context", return_value=None),
            mock.patch("src.audio.tts_factory.create_tts_backend", return_value=tts),
            mock.patch("src.audio.tts_factory.backend_name", return_value="fake"),
            mock.patch("src.audio.audio_io.AudioPlayer", return_value=player),
            mock.patch(
                "src.assistant.factory.build_local_service", return_value=(service, registry)
            ),
            mock.patch(
                "src.growth.tracker.GrowthTracker", side_effect=RuntimeError("offline")
            ),
            mock.patch("src.chat.session.ChatSession", return_value=session) as mock_chat_session,
            mock.patch("builtins.input", side_effect=_fake_input),
            mock.patch("src.assistant.requests.create_request", recording),
        ):
            with redirect_stdout(io.StringIO()):
                run_text_to_speech_mode(SimpleNamespace(tts_voice="fake_voice"))

        self.assertEqual(mock_chat_session.call_args.kwargs["emotion_tags"], True)

        self.assertEqual(len(recording.calls), 1)
        call = recording.calls[0]
        self.assertEqual(call["text"], "こんにちは")
        self.assertEqual(call["conversation_id"], session.session_id)
        self.assertEqual(call["channel"], "voice")
        self.assertEqual(call["profile"], "voice_fast")
        self.assertEqual(call["privacy"], "local_only")
        self.assertEqual(stream.close_count, 1)


class ImmediateFailStream:
    """反復開始直後に例外を送出する StreamResult 風オブジェクト (部分出力を出さない)。"""

    def __init__(self, exc) -> None:
        self.exc = exc
        self.close_count = 0

    def __iter__(self):
        raise self.exc

    def close(self) -> None:
        self.close_count += 1


class TTSSession:
    """run_text_to_speech_mode のエラー制御テスト用フェイクセッション。"""

    def __init__(self, *, save_error: Exception | None = None) -> None:
        self.session_id = "tts-error-session"
        self.system_prompt = "sys"
        self._messages: list[dict] = []
        self.turn_count = 0
        self.rollback_calls = 0
        self.save_calls = 0
        self._save_error = save_error

    @property
    def messages(self) -> list[dict]:
        return list(self._messages)

    def add_user_message(self, content: str) -> None:
        self._messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str, **kwargs) -> None:
        self._messages.append({"role": "assistant", "content": content})
        self.turn_count += 1

    def build_blocks(self):
        return ()

    def rollback_last_user_message(self) -> bool:
        if self._messages and self._messages[-1].get("role") == "user":
            self._messages.pop()
            self.rollback_calls += 1
            return True
        return False

    def save(self) -> str:
        self.save_calls += 1
        if self._save_error is not None:
            raise self._save_error
        return "saved"


class InputSequenceWithFinal:
    """入力を順に返し、尽きたら final_exc (既定は KeyboardInterrupt) を送出する。"""

    def __init__(self, *values, final_exc=None) -> None:
        self._values = iter(values)
        self._final_exc = final_exc

    def __call__(self, prompt: str = "") -> str:
        try:
            return next(self._values)
        except StopIteration:
            if self._final_exc is not None:
                raise self._final_exc
            raise KeyboardInterrupt


class AudioTextErrorContainmentTest(unittest.TestCase):
    """run_text_to_speech_mode の生成エラー封じ込めと終了時クリーンアップのオフライン検証。"""

    def _run(self, service, registry, session, inputs) -> str:
        config = SimpleNamespace(
            model="test-model",
            effective_system_prompt=lambda: "sys",
            max_history_turns=20,
            history_dir="data/test-history",
            resolved_local_provider_id=lambda: "ollama",
            emotion_tag_enabled=False,
        )
        tts = SimpleNamespace(
            load=lambda: None, synthesize=lambda text, style=None: b"", voice="fake_voice"
        )
        output = io.StringIO()
        with (
            mock.patch("src.chat.config.ChatConfig.load", return_value=config),
            mock.patch("src.chat.config.validate_local_provider_kind", return_value="ollama"),
            mock.patch("src.chat.web_search.create_web_search_context", return_value=None),
            mock.patch("src.audio.tts_factory.create_tts_backend", return_value=tts),
            mock.patch("src.audio.tts_factory.backend_name", return_value="fake"),
            mock.patch("src.audio.audio_io.AudioPlayer", return_value=NoopPlayer()),
            mock.patch(
                "src.assistant.factory.build_local_service", return_value=(service, registry)
            ),
            mock.patch("src.growth.tracker.GrowthTracker", side_effect=RuntimeError("offline")),
            mock.patch("src.chat.session.ChatSession", return_value=session),
            mock.patch("builtins.input", side_effect=inputs),
            redirect_stdout(output),
        ):
            run_text_to_speech_mode(SimpleNamespace(tts_voice="fake_voice"))
        return output.getvalue()

    @staticmethod
    def _ok_client() -> SimpleNamespace:
        return SimpleNamespace(is_available=lambda: True, has_model=lambda: True)

    @staticmethod
    def _registry(client) -> mock.Mock:
        registry = mock.Mock()
        registry.get.return_value = SimpleNamespace(provider=client)
        return registry

    def test_generation_failure_then_next_turn_succeeds(self) -> None:
        raw = "generation exploded C:\\private\\raw"
        first_stream = ImmediateFailStream(
            ProviderRequestError("fake", "respond_stream", raw)
        )
        second_stream = FakeStreamResult(("返事", "です"))
        service = mock.Mock()
        service.respond_stream.side_effect = [first_stream, second_stream]
        registry = self._registry(self._ok_client())
        session = TTSSession()
        output = self._run(
            service,
            registry,
            session,
            InputSequenceWithFinal("こんにちは", "次"),
        )

        self.assertEqual(
            session.messages,
            [
                {"role": "user", "content": "次"},
                {"role": "assistant", "content": "返事です"},
            ],
        )
        self.assertEqual(session.rollback_calls, 1)
        self.assertEqual(session.save_calls, 1)
        self.assertEqual(service.respond_stream.call_count, 2)
        self.assertEqual(first_stream.close_count, 1)
        self.assertEqual(second_stream.close_count, 1)
        self.assertEqual(registry.close.call_count, 1)
        self.assertIn("AI応答の生成に失敗しました", output)
        self.assertIn("ProviderRequestError", output)
        self.assertNotIn(raw, output)
        self.assertNotIn("こんにちは", output)

    def test_request_start_failure_rolls_back_and_continues(self) -> None:
        raw = "respond start failed C:\\private\\raw"
        service = mock.Mock()
        service.respond_stream.side_effect = ProviderRequestError(
            "fake", "respond_stream", raw
        )
        registry = self._registry(self._ok_client())
        session = TTSSession()
        output = self._run(
            service,
            registry,
            session,
            InputSequenceWithFinal("こんにちは"),
        )

        self.assertEqual(session.messages, [])
        self.assertEqual(session.rollback_calls, 1)
        self.assertEqual(session.save_calls, 0)
        self.assertEqual(service.respond_stream.call_count, 1)
        self.assertEqual(registry.close.call_count, 1)
        self.assertIn("AI応答の生成に失敗しました", output)
        self.assertIn("ProviderRequestError", output)
        self.assertNotIn(raw, output)
        self.assertNotIn("こんにちは", output)

    def test_keyboard_interrupt_commits_success_and_closes_registry_once(self) -> None:
        service = mock.Mock()
        service.respond_stream.return_value = FakeStreamResult(("返事",))
        registry = self._registry(self._ok_client())
        session = TTSSession()
        output = self._run(
            service,
            registry,
            session,
            InputSequenceWithFinal("こんにちは"),
        )

        self.assertEqual(
            session.messages,
            [
                {"role": "user", "content": "こんにちは"},
                {"role": "assistant", "content": "返事"},
            ],
        )
        self.assertEqual(session.save_calls, 1)
        self.assertEqual(service.respond_stream.return_value.close_count, 1)
        self.assertEqual(registry.close.call_count, 1)
        self.assertIn("終了します", output)
        self.assertIn("会話を保存しました: saved", output)

    def test_save_failure_is_best_effort_and_closes_registry(self) -> None:
        raw = "save exploded C:\\private\\db"
        service = mock.Mock()
        service.respond_stream.return_value = FakeStreamResult(("返事",))
        registry = self._registry(self._ok_client())
        session = TTSSession(save_error=RuntimeError(raw))
        output = self._run(
            service,
            registry,
            session,
            InputSequenceWithFinal("こんにちは"),
        )

        self.assertEqual(session.save_calls, 1)
        self.assertEqual(registry.close.call_count, 1)
        self.assertIn("会話の保存に失敗しました", output)
        self.assertIn("RuntimeError", output)
        self.assertNotIn(raw, output)
        self.assertNotIn("こんにちは", output)

    def test_unexpected_input_failure_closes_registry_once_and_propagates(self) -> None:
        raw = "unexpected input failure C:\\private\\raw"
        service = mock.Mock()
        service.respond_stream.return_value = FakeStreamResult(("返事",))
        registry = self._registry(self._ok_client())
        session = TTSSession()

        with self.assertRaises(RuntimeError):
            self._run(
                service,
                registry,
                session,
                InputSequenceWithFinal("こんにちは", final_exc=RuntimeError(raw)),
            )

        self.assertEqual(session.save_calls, 1)
        self.assertEqual(registry.close.call_count, 1)


class VoiceCalendarCommitTest(unittest.TestCase):
    def test_calendar_direct_reply_commits_without_memory_or_growth(self) -> None:
        pipeline = VoicePipeline.__new__(VoicePipeline)
        pipeline.voice_calendar_write_enabled = True
        pipeline._try_register_event = lambda text: "登録しました"
        pipeline.vad = SimpleNamespace(sample_rate=10)
        pipeline._listen_for_speech = lambda: np.ones(4, dtype=np.float32)
        pipeline.stt = SimpleNamespace(transcribe=lambda audio: "明日会議を入れて")
        pipeline.tts = NoopTTS()
        pipeline.player = NoopPlayer()
        pipeline.idle_manager = None
        pipeline._state = VoicePipeline.STATE_IDLE
        session = RecordingSession()
        pipeline.session = session

        with redirect_stdout(io.StringIO()):
            reply = pipeline.process_voice_turn()

        self.assertEqual(reply, "登録しました")
        self.assertEqual(session.user_messages, ["明日会議を入れて"])
        self.assertEqual(
            session.assistant_commits,
            [
                {
                    "content": "登録しました",
                    "store_memory": False,
                    "record_growth": False,
                }
            ],
        )

    def test_calendar_direct_reply_not_attempted_when_write_disabled(self) -> None:
        pipeline = VoicePipeline.__new__(VoicePipeline)
        pipeline.voice_calendar_write_enabled = False
        event_calls: list[str] = []
        pipeline._try_register_event = lambda text: event_calls.append(text) or None
        pipeline.vad = SimpleNamespace(sample_rate=10)
        pipeline._listen_for_speech = lambda: np.ones(4, dtype=np.float32)
        pipeline.stt = SimpleNamespace(transcribe=lambda audio: "こんにちは")
        pipeline.tts = NoopTTS()
        pipeline.player = NoopPlayer()
        pipeline.idle_manager = None
        pipeline._state = VoicePipeline.STATE_IDLE
        pipeline.streaming_tts = False
        pipeline.config = SimpleNamespace(emotion_tag_enabled=False)
        session = RecordingSession()
        pipeline.session = session
        service = mock.Mock()
        service.respond_stream.return_value = FakeStreamResult(("こんにちは",))
        pipeline._assistant_service = service

        with redirect_stdout(io.StringIO()):
            reply = pipeline.process_voice_turn()

        self.assertEqual(reply, "こんにちは")
        self.assertEqual(event_calls, [])
        self.assertEqual(
            session.assistant_commits,
            [
                {
                    "content": "こんにちは",
                    "store_memory": True,
                    "record_growth": True,
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()