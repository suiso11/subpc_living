from contextlib import redirect_stdout
import io
import queue
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from src.assistant.factory import build_local_service
from src.audio.pipeline import VoicePipeline
from src.chat.config import ChatConfig
from src.context.contracts import ContextBlock, ContextMessage
from src.llm.errors import ProviderRequestError
from src.llm.providers.fake import FakeProvider
from src.llm.routing.static import StaticRouter


class RecordingRouter:
    def __init__(self, delegate) -> None:
        self.delegate = delegate
        self.requests = []

    def route(self, request):
        self.requests.append(request)
        return self.delegate.route(request)


class NoopTTS:
    def synthesize(self, text, *, style=None):
        return b""


class NoopPlayer:
    def play_wav(self, wav_data, *, blocking=True) -> None:
        pass


class TestSession:
    def __init__(self, session_id: str = "voice-session", system_prompt: str = "") -> None:
        self.session_id = session_id
        self.system_prompt = system_prompt
        self._messages = []

    def add_user_message(self, content: str) -> None:
        self._messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        self._messages.append({"role": "assistant", "content": content})

    def rollback_last_user_message(self) -> bool:
        if not self._messages or self._messages[-1].get("role") != "user":
            return False
        self._messages.pop()
        return True

    def build_messages(self):
        return [dict(message) for message in self._messages]

    def build_blocks(self):
        if not self._messages:
            return []
        return [ContextBlock(
            source="history",
            content=tuple(
                ContextMessage(role=m["role"], content=m["content"])
                for m in self._messages
            ),
            local_only=True,
        )]


class IdleManagerStub:
    def __init__(self) -> None:
        self.started = 0
        self.ended = 0

    def notify_inference_start(self, *, wait_for_gpu: bool) -> None:
        self.started += 1

    def notify_inference_end(self) -> None:
        self.ended += 1


class FailOnceProvider(FakeProvider):
    def __init__(self) -> None:
        super().__init__(stream_chunks=("次", "ターン"))
        self.attempts = 0

    def generate_stream(self, messages, **options):
        self.attempts += 1
        self.calls.append(
            {
                "kind": "generate_stream",
                "messages": [dict(message) for message in messages],
                "options": dict(options),
            }
        )
        if self.attempts == 1:
            raise ProviderRequestError(
                "fake", "generate_stream", "planned first-turn failure"
            )
        yield from self.stream_chunks


class CountingCloseProvider(FakeProvider):
    def __init__(self) -> None:
        super().__init__()
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1
        super().close()


class FakeStreamResult:
    """Offline StreamResult stand-in that records close() calls."""

    def __init__(self, chunks) -> None:
        self.chunks = chunks
        self.close_count = 0

    def __iter__(self):
        yield from self.chunks

    def close(self) -> None:
        self.close_count += 1


class FailingStreamResult(FakeStreamResult):
    """反復途中で例外を送出するStreamResult風オブジェクト。"""

    def __init__(self, chunks, exc) -> None:
        super().__init__(chunks)
        self.exc = exc

    def __iter__(self):
        yield from self.chunks
        raise self.exc


class VoicePipelineLLMTest(unittest.TestCase):
    @staticmethod
    def make_pipeline(provider, config=None, session=None):
        config = config or ChatConfig(emotion_tag_enabled=False)
        service, registry = build_local_service(config, provider=provider)
        recording_router = RecordingRouter(
            StaticRouter(registry, default_provider_id="ollama")
        )
        service._router = recording_router

        pipeline = VoicePipeline.__new__(VoicePipeline)
        pipeline.config = config
        pipeline._assistant_service = service
        pipeline._provider_registry = registry
        pipeline.llm = provider
        pipeline.session = session or TestSession()
        pipeline.tts = NoopTTS()
        pipeline.player = NoopPlayer()
        pipeline._tts_queue = queue.Queue()
        pipeline._state = pipeline.STATE_IDLE
        return pipeline, recording_router

    def test_streaming_tts_generation_preserves_provider_tokens(self) -> None:
        provider = FakeProvider(stream_chunks=("スト", "リーム", "。"))
        pipeline, router = self.make_pipeline(provider)
        pipeline.session.add_user_message("話して")
        blocks = pipeline.session.build_blocks()
        user_text = "話して"

        with redirect_stdout(io.StringIO()):
            response = pipeline._stream_llm_with_tts(blocks, user_text)

        self.assertEqual(response, "ストリーム。")
        self.assertEqual(len(router.requests), 1)
        self.assertEqual(provider.calls[0]["kind"], "generate_stream")

    def test_fake_provider_stream_and_voice_request_preserve_config(self) -> None:
        config = ChatConfig(
            temperature=0.23,
            top_p=0.81,
            top_k=17,
            repeat_penalty=1.37,
            num_ctx=3456,
            num_predict=123,
            emotion_tag_enabled=False,
        )
        provider = FakeProvider(stream_chunks=("応", "答", "です"))
        pipeline, router = self.make_pipeline(provider, config=config)
        pipeline.session.add_user_message("最初の質問")
        pipeline.session.add_assistant_message("最初の応答")
        pipeline.session.add_user_message("最後の質問")
        blocks = pipeline.session.build_blocks()
        user_text = "最後の質問"

        with redirect_stdout(io.StringIO()):
            response = pipeline._sequential_llm_then_tts(blocks, user_text)

        self.assertEqual(response, "応答です")
        self.assertEqual(len(router.requests), 1)
        request = router.requests[0]
        self.assertEqual(request.text, "最後の質問")
        self.assertEqual(request.conversation_id, "voice-session")
        self.assertEqual(request.channel, "voice")
        self.assertEqual(request.profile, "voice_fast")
        self.assertEqual(request.privacy, "local_only")
        self.assertEqual(
            provider.calls[0]["options"],
            {
                "temperature": 0.23,
                "top_p": 0.81,
                "top_k": 17,
                "repeat_penalty": 1.37,
                "num_ctx": 3456,
                "num_predict": 123,
            },
        )

    def test_generation_failure_returns_to_idle_and_next_turn_succeeds(self) -> None:
        provider = FailOnceProvider()
        session = TestSession()
        pipeline, _ = self.make_pipeline(provider, session=session)
        transcriptions = iter(("失敗するターン", "成功するターン"))
        pipeline.streaming_tts = False
        pipeline.vad = SimpleNamespace(sample_rate=10)
        pipeline._listen_for_speech = lambda: np.ones(4, dtype=np.float32)
        pipeline.stt = SimpleNamespace(transcribe=lambda audio: next(transcriptions))
        pipeline.idle_manager = IdleManagerStub()
        pipeline._try_register_event = lambda text: None

        with redirect_stdout(io.StringIO()):
            first_response = pipeline.process_voice_turn()
            second_response = pipeline.process_voice_turn()

        self.assertIsNone(first_response)
        self.assertEqual(second_response, "次ターン")
        self.assertEqual(pipeline.state, pipeline.STATE_IDLE)
        self.assertEqual(
            session._messages,
            [
                {"role": "user", "content": "成功するターン"},
                {"role": "assistant", "content": "次ターン"},
            ],
        )
        self.assertEqual(pipeline.idle_manager.started, 2)
        self.assertEqual(pipeline.idle_manager.ended, 2)

    def _make_turn_pipeline(self, provider, session=None, *, transcription="ターン"):
        session = session or TestSession()
        pipeline, _ = self.make_pipeline(provider, session=session)
        pipeline.streaming_tts = False
        pipeline.vad = SimpleNamespace(sample_rate=10)
        pipeline._listen_for_speech = lambda: np.ones(4, dtype=np.float32)
        pipeline.stt = SimpleNamespace(transcribe=lambda audio: transcription)
        pipeline._try_register_event = lambda text: None
        pipeline.idle_manager = None
        return pipeline, session

    def test_idle_manager_none_successful_turn_returns_to_idle(self) -> None:
        provider = FakeProvider(stream_chunks=("成功", "応答"))
        session = TestSession()
        pipeline, _ = self._make_turn_pipeline(provider, session=session)

        with redirect_stdout(io.StringIO()):
            response = pipeline.process_voice_turn()

        self.assertEqual(response, "成功応答")
        self.assertEqual(pipeline.state, pipeline.STATE_IDLE)
        self.assertEqual(
            session._messages,
            [
                {"role": "user", "content": "ターン"},
                {"role": "assistant", "content": "成功応答"},
            ],
        )

    def test_idle_manager_none_empty_response_turn_returns_to_idle(self) -> None:
        provider = FakeProvider(stream_chunks=())
        pipeline, _ = self._make_turn_pipeline(provider)

        with redirect_stdout(io.StringIO()):
            response = pipeline.process_voice_turn()

        self.assertIsNone(response)
        self.assertEqual(pipeline.state, pipeline.STATE_IDLE)

    def test_idle_manager_none_provider_error_turn_returns_to_idle_and_rolls_back(self) -> None:
        provider = FailOnceProvider()
        session = TestSession()
        pipeline, _ = self._make_turn_pipeline(provider, session=session)

        with redirect_stdout(io.StringIO()):
            response = pipeline.process_voice_turn()

        self.assertIsNone(response)
        self.assertEqual(pipeline.state, pipeline.STATE_IDLE)
        self.assertEqual(session._messages, [])

    def test_idle_manager_enabled_notifies_start_and_end_exactly_once(self) -> None:
        provider = FakeProvider(stream_chunks=("応", "答"))
        pipeline, _ = self._make_turn_pipeline(provider)
        idle = IdleManagerStub()
        pipeline.idle_manager = idle

        with redirect_stdout(io.StringIO()):
            response = pipeline.process_voice_turn()

        self.assertEqual(response, "応答")
        self.assertEqual(pipeline.state, pipeline.STATE_IDLE)
        self.assertEqual(idle.started, 1)
        self.assertEqual(idle.ended, 1)

    def test_provider_registry_close_is_safe_when_repeated_or_missing(self) -> None:
        provider = CountingCloseProvider()
        pipeline, _ = self.make_pipeline(provider)

        pipeline._close_provider_registry()
        pipeline._close_provider_registry()

        self.assertEqual(provider.close_count, 1)
        self.assertIsNone(pipeline._provider_registry)

    def _make_stream_pipeline(self, stream):
        pipeline, _ = self.make_pipeline(FakeProvider())
        service = mock.Mock()
        service.respond_stream.return_value = stream
        pipeline._assistant_service = service
        return pipeline

    def test_stream_result_closed_exactly_once_on_normal_completion(self) -> None:
        stream = FakeStreamResult(("スト", "リーム", "。"))
        pipeline = self._make_stream_pipeline(stream)

        with redirect_stdout(io.StringIO()):
            response = pipeline._stream_llm_with_tts([], "話して")

        self.assertEqual(response, "ストリーム。")
        self.assertEqual(stream.close_count, 1)

    def test_stream_result_closed_exactly_once_on_repeated_output_abort(self) -> None:
        stream = FakeStreamResult(tuple("あ。あ。あ。あ。あ。あ。あ。"))
        pipeline = self._make_stream_pipeline(stream)

        with redirect_stdout(io.StringIO()):
            response = pipeline._stream_llm_with_tts([], "話して")

        self.assertEqual(response, "出力が乱れたので止めます。")
        self.assertEqual(stream.close_count, 1)

    def test_stream_result_closed_exactly_once_on_generation_exception(self) -> None:
        exc = ProviderRequestError("fake", "respond_stream", "planned failure")
        stream = FailingStreamResult(("スト", "リーム", "。"), exc)
        pipeline = self._make_stream_pipeline(stream)

        with redirect_stdout(io.StringIO()):
            with self.assertRaises(ProviderRequestError):
                pipeline._stream_llm_with_tts([], "話して")

        self.assertEqual(stream.close_count, 1)

    def test_respond_stream_failure_propagates_without_close(self) -> None:
        exc = ProviderRequestError("fake", "respond_stream", "start failure")
        pipeline, _ = self.make_pipeline(FakeProvider())
        service = mock.Mock()
        service.respond_stream.side_effect = exc
        pipeline._assistant_service = service

        with redirect_stdout(io.StringIO()):
            with self.assertRaises(ProviderRequestError):
                pipeline._stream_llm_with_tts([], "話して")

    def test_sequential_stream_result_closed_exactly_once_on_normal_completion(self) -> None:
        stream = FakeStreamResult(("応", "答", "。"))
        pipeline = self._make_stream_pipeline(stream)

        with redirect_stdout(io.StringIO()):
            response = pipeline._sequential_llm_then_tts([], "話して")

        self.assertEqual(response, "応答。")
        self.assertEqual(stream.close_count, 1)

    def test_sequential_stream_result_closed_exactly_once_on_empty_response(self) -> None:
        stream = FakeStreamResult(())
        pipeline = self._make_stream_pipeline(stream)

        with redirect_stdout(io.StringIO()):
            response = pipeline._sequential_llm_then_tts([], "話して")

        self.assertEqual(response, "")
        self.assertEqual(stream.close_count, 1)

    def test_sequential_stream_result_closed_exactly_once_on_generation_exception(self) -> None:
        exc = ProviderRequestError("fake", "respond_stream", "planned failure")
        stream = FailingStreamResult(("応", "答", "。"), exc)
        pipeline = self._make_stream_pipeline(stream)

        with redirect_stdout(io.StringIO()):
            with self.assertRaises(ProviderRequestError):
                pipeline._sequential_llm_then_tts([], "話して")

        self.assertEqual(stream.close_count, 1)

    def test_respond_stream_called_with_blocks_and_base_system(self) -> None:
        """respond_stream receives blocks (not messages) and base_system from session."""
        provider = FakeProvider(stream_chunks=("応", "答"))
        session = TestSession(system_prompt="You are helpful.")
        pipeline, _ = self.make_pipeline(provider, session=session)
        pipeline.streaming_tts = False
        pipeline.session.add_user_message("テスト")
        blocks = pipeline.session.build_blocks()
        user_text = "テスト"

        mock_service = mock.Mock()
        mock_service.respond_stream.return_value = iter(["応", "答"])
        pipeline._assistant_service = mock_service

        with redirect_stdout(io.StringIO()):
            pipeline._sequential_llm_then_tts(blocks, user_text)

        mock_service.respond_stream.assert_called_once()
        call_args = mock_service.respond_stream.call_args
        # First positional arg is request, second is blocks
        self.assertIs(call_args.args[1], blocks)
        self.assertEqual(call_args.kwargs["base_system"], "You are helpful.")


if __name__ == "__main__":
    unittest.main()
