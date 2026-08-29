from __future__ import annotations

import asyncio
import contextlib
import inspect
import io
import os
import queue
import tempfile
import threading
import time
import unittest
import wave
from concurrent.futures import Future
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from src.discord_bot import bot as bot_module
from src.discord_bot import voice_stt as voice_stt_module
from src.discord_bot.voice_stt import (
    CompletedSpeech,
    DiscordSTTSink,
    DiscordVoiceSTT,
    SpeechChunk,
    SpeechSegmenter,
    VoiceSTTConfig,
    _DEBUG_AUDIO_TTL_DEFAULT_SEC,
    _DEBUG_AUDIO_TTL_MAX_SEC,
    _is_likely_hallucination,
    _sweep_debug_audio,
    pcm48_stereo_to_16k_mono,
    safe_stt_last_error,
    VOICE_STT_ERR_LISTEN,
    VOICE_STT_ERR_SEND,
    VOICE_STT_ERR_WORKER,
)
from src.discord_bot.voice_tts import (
    VoiceTTSConfig,
    VoiceTTSError,
    VoiceTTSPlayer,
    VOICE_TTS_ERR_AUTOREAD,
    VOICE_TTS_ERR_PLAYBACK,
)


class DiscordVoiceSTTTest(unittest.TestCase):
    def test_converts_discord_pcm_to_16k_mono(self) -> None:
        frames_48k = 960
        stereo = np.full((frames_48k, 2), 3277, dtype=np.int16)

        audio = pcm48_stereo_to_16k_mono(stereo.tobytes())

        self.assertEqual(audio.dtype, np.float32)
        self.assertEqual(audio.shape, (320,))
        self.assertAlmostEqual(float(audio.mean()), 0.1, places=3)

    def test_segmenter_emits_after_silence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = VoiceSTTConfig(
                enabled=True,
                transcript_channel_id=None,
                transcript_dir=Path(tmp),
                energy_threshold=0.01,
                silence_duration_ms=60,
                speech_pad_ms=0,
                min_speech_duration_ms=60,
                max_segment_seconds=5.0,
            )
            segmenter = SpeechSegmenter(config)
            now = datetime.now(timezone.utc)
            speech_frame = np.full(segmenter.frame_size, 0.1, dtype=np.float32)
            silence_frame = np.zeros(segmenter.frame_size, dtype=np.float32)

            emitted = []
            for frame in [speech_frame, speech_frame, speech_frame, silence_frame, silence_frame]:
                emitted.extend(segmenter.add_audio(frame, now))

            self.assertEqual(len(emitted), 1)
            self.assertGreaterEqual(emitted[0].audio.size, segmenter.frame_size * 3)
            self.assertEqual(emitted[0].reason, "silence")

    def test_hallucination_filter_drops_known_junk(self) -> None:
        self.assertTrue(_is_likely_hallucination("ご視聴ありがとうございました"))
        self.assertTrue(_is_likely_hallucination("おつかれさまでした"))
        self.assertTrue(_is_likely_hallucination("すっ"))
        self.assertTrue(_is_likely_hallucination("ん"))

    def test_hallucination_filter_keeps_real_speech(self) -> None:
        self.assertFalse(_is_likely_hallucination("今日はいい天気ですね"))
        self.assertFalse(_is_likely_hallucination("うん、分かりました"))
        self.assertFalse(_is_likely_hallucination(""))
        self.assertFalse(_is_likely_hallucination("OK"))

    def test_feature_and_transcript_save_flags_require_literal_true(self) -> None:
        for raw, expected in (("true", True), ("TRUE", True), (" true ", True),
                              ("yes", False), ("1", False), ("on", False),
                              ("false", False)):
            with self.subTest(raw=raw):
                with mock.patch.dict(
                    os.environ,
                    {
                        "DISCORD_VOICE_STT_ENABLED": raw,
                        "DISCORD_VOICE_STT_SAVE_TRANSCRIPTS": raw,
                    },
                    clear=True,
                ):
                    config = VoiceSTTConfig.from_env(Path("."))
                self.assertEqual(config.enabled, expected)
                self.assertEqual(config.save_transcripts, expected)


class DiscordVoiceSTTSanitizationCanaryTest(unittest.TestCase):
    """raw 例外・パス・名前が last_error / ログへ漏れないことの canary 検証。"""

    CANARY = "canary raw boom C:\\Users\\secret\\voice.wav"

    def _make_stt(self) -> DiscordVoiceSTT:
        return DiscordVoiceSTT(
            VoiceSTTConfig(
                enabled=True,
                transcript_channel_id=None,
                transcript_dir=Path("."),
            )
        )

    def test_worker_error_records_fixed_code(self) -> None:
        stt = self._make_stt()
        stt.last_error = self.CANARY
        stt._record_worker_error()
        self.assertEqual(stt.last_error, VOICE_STT_ERR_WORKER)

    def test_send_error_records_fixed_code(self) -> None:
        stt = self._make_stt()
        future = Future()
        future.set_exception(RuntimeError(self.CANARY))
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            stt._record_future_error(future)
        self.assertEqual(stt.last_error, VOICE_STT_ERR_SEND)
        self.assertNotIn("canary raw boom", buf.getvalue())
        self.assertNotIn("voice.wav", buf.getvalue())

    def test_listen_error_callback_records_fixed_code(self) -> None:
        stt = self._make_stt()
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            stt._after_listening(RuntimeError(self.CANARY))
        self.assertEqual(stt.last_error, VOICE_STT_ERR_LISTEN)
        self.assertNotIn("canary raw boom", buf.getvalue())
        self.assertNotIn("voice.wav", buf.getvalue())

    def test_debug_audio_success_does_not_log_path_or_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stt = DiscordVoiceSTT(
                VoiceSTTConfig(
                    enabled=True,
                    transcript_channel_id=None,
                    transcript_dir=Path(tmp),
                    debug_audio_dir=Path(tmp),
                )
            )
            chunk = _make_chunk()
            with contextlib.redirect_stdout(io.StringIO()) as buf:
                stt._dump_debug_audio(chunk)
            out = buf.getvalue()
            self.assertNotIn(tmp, out)
            self.assertNotIn("secret-user", out)
            self.assertNotIn(".wav", out)
            self.assertIn("debug audio saved", out)

    def test_debug_audio_failure_does_not_leak_error_or_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stt = DiscordVoiceSTT(
                VoiceSTTConfig(
                    enabled=True,
                    transcript_channel_id=None,
                    transcript_dir=Path(tmp),
                    debug_audio_dir=Path(tmp),
                )
            )
            chunk = _make_chunk()
            with mock.patch.object(
                voice_stt_module,
                "_write_debug_wav",
                side_effect=RuntimeError(self.CANARY),
            ):
                with contextlib.redirect_stdout(io.StringIO()) as buf:
                    stt._dump_debug_audio(chunk)
            out = buf.getvalue()
            self.assertNotIn("canary raw boom", out)
            self.assertNotIn(tmp, out)
            self.assertIn("debug audio dump failed", out)

    def test_status_text_hides_channel_name_and_raw_error(self) -> None:
        stt = self._make_stt()
        stt.last_error = self.CANARY
        fake_vc = SimpleNamespace(channel=SimpleNamespace(name="secret-vc"))
        fake_vc.is_connected = lambda: True
        fake_vc.is_listening = lambda: False
        stt.voice_client = fake_vc
        out = stt.status_text()
        self.assertIn("voice_connected: True", out)
        self.assertIn("voice_listening: False", out)
        self.assertNotIn("secret-vc", out)
        self.assertNotIn("canary raw boom", out)
        self.assertNotIn("voice.wav", out)
        self.assertIn("voice_last_error: -", out)

    def test_status_text_shows_known_code(self) -> None:
        stt = self._make_stt()
        stt.last_error = VOICE_STT_ERR_WORKER
        self.assertIn("voice_last_error: worker_failure", stt.status_text())

    def test_safe_last_error_whitelist(self) -> None:
        self.assertEqual(safe_stt_last_error(self.CANARY), "-")
        self.assertEqual(safe_stt_last_error(VOICE_STT_ERR_WORKER), VOICE_STT_ERR_WORKER)
        self.assertEqual(safe_stt_last_error(VOICE_STT_ERR_SEND), VOICE_STT_ERR_SEND)
        self.assertEqual(safe_stt_last_error(VOICE_STT_ERR_LISTEN), VOICE_STT_ERR_LISTEN)


class DiscordVoiceSTTDebugRetentionTest(unittest.TestCase):
    """Debug-WAV bounded retention: TTL expiry, non-WAV/symlink preservation,
    fail-closed invalid TTL, and write-failure sanitization."""

    @staticmethod
    def _make_wav(path: Path) -> None:
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00\x00" * 160)

    def test_sweep_deletes_only_expired_wav_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            debug_dir = Path(tmp)
            old = debug_dir / "old.wav"
            new = debug_dir / "new.wav"
            self._make_wav(old)
            self._make_wav(new)
            now = datetime.now(timezone.utc)
            os.utime(old, (now.timestamp() - 7200, now.timestamp() - 7200))
            os.utime(new, (now.timestamp() - 60, now.timestamp() - 60))

            _sweep_debug_audio(debug_dir, ttl_sec=3600, now=now)

            self.assertFalse(old.exists())
            self.assertTrue(new.exists())

    def test_sweep_preserves_non_wav_files_and_subdirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            debug_dir = Path(tmp)
            notes = debug_dir / "notes.txt"
            notes.write_text("keep me", encoding="utf-8")
            sub = debug_dir / "subdir"
            sub.mkdir()
            nested = sub / "nested.wav"
            self._make_wav(nested)
            now = datetime.now(timezone.utc)
            for p in (notes, sub, nested):
                os.utime(p, (now.timestamp() - 7200, now.timestamp() - 7200))

            _sweep_debug_audio(debug_dir, ttl_sec=3600, now=now)

            self.assertTrue(notes.exists())
            self.assertTrue(nested.exists())

    def test_sweep_never_follows_or_deletes_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            debug_dir = Path(tmp)
            target = debug_dir / "target.wav"
            self._make_wav(target)
            sub = debug_dir / "subdir"
            sub.mkdir()
            nested = sub / "nested.wav"
            self._make_wav(nested)
            link = debug_dir / "link.wav"
            dirlink = debug_dir / "dirlink.wav"
            try:
                os.symlink(target.name, link)
                os.symlink(sub.name, dirlink)
            except OSError:
                self.skipTest("symlink creation unavailable")
            now = datetime.now(timezone.utc)
            for p in (target, sub, nested, link, dirlink):
                os.utime(p, (now.timestamp() - 7200, now.timestamp() - 7200))

            _sweep_debug_audio(debug_dir, ttl_sec=3600, now=now)

            self.assertTrue(target.exists())
            self.assertTrue(link.exists())
            self.assertTrue(link.is_symlink())
            self.assertTrue(dirlink.exists())
            self.assertTrue(dirlink.is_symlink())
            self.assertTrue(nested.exists())

    def test_sweep_does_not_recurse_or_delete_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            debug_dir = Path(tmp)
            sub = debug_dir / "nested"
            sub.mkdir()
            nested_wav = sub / "nested.wav"
            self._make_wav(nested_wav)
            fake_dir = debug_dir / "dir.wav"
            fake_dir.mkdir()
            now = datetime.now(timezone.utc)
            os.utime(nested_wav, (now.timestamp() - 7200, now.timestamp() - 7200))
            os.utime(fake_dir, (now.timestamp() - 7200, now.timestamp() - 7200))

            _sweep_debug_audio(debug_dir, ttl_sec=3600, now=now)

            self.assertTrue(nested_wav.exists())
            self.assertTrue(fake_dir.is_dir())

    def test_unset_ttl_uses_conservative_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(
                os.environ,
                {"DISCORD_VOICE_STT_DEBUG_AUDIO_DIR": "data/discord_voice/debug_audio"},
                clear=False,
            ):
                config = VoiceSTTConfig.from_env(tmp)
            self.assertIsNotNone(config.debug_audio_dir)
            self.assertEqual(config.debug_audio_ttl_sec, _DEBUG_AUDIO_TTL_DEFAULT_SEC)

    def test_valid_ttl_is_bounded_and_parsed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(
                os.environ,
                {
                    "DISCORD_VOICE_STT_DEBUG_AUDIO_DIR": "data/discord_voice/debug_audio",
                    "DISCORD_VOICE_STT_DEBUG_AUDIO_TTL_SEC": "120",
                },
                clear=False,
            ):
                config = VoiceSTTConfig.from_env(tmp)
            self.assertIsNotNone(config.debug_audio_dir)
            self.assertEqual(config.debug_audio_ttl_sec, 120)

    def test_invalid_ttl_fails_closed_and_disables_debug_dir(self) -> None:
        bad_values = ("0", "-5", "abc", "1.5", str(_DEBUG_AUDIO_TTL_MAX_SEC + 1))
        with tempfile.TemporaryDirectory() as tmp:
            for bad in bad_values:
                with mock.patch.dict(
                    os.environ,
                    {
                        "DISCORD_VOICE_STT_DEBUG_AUDIO_DIR": "data/discord_voice/debug_audio",
                        "DISCORD_VOICE_STT_DEBUG_AUDIO_TTL_SEC": bad,
                    },
                    clear=False,
                ):
                    config = VoiceSTTConfig.from_env(tmp)
                self.assertIsNone(config.debug_audio_dir, msg=f"ttl={bad}")

    def test_invalid_ttl_skips_debug_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stt = DiscordVoiceSTT(
                VoiceSTTConfig(
                    enabled=True,
                    transcript_channel_id=None,
                    transcript_dir=Path(tmp),
                    debug_audio_dir=Path(tmp),
                    debug_audio_ttl_sec=0,
                )
            )
            with mock.patch.object(
                voice_stt_module,
                "_write_debug_wav",
                side_effect=AssertionError("must not be called"),
            ) as write_mock:
                with contextlib.redirect_stdout(io.StringIO()) as buf:
                    stt._dump_debug_audio(_make_chunk())
            write_mock.assert_not_called()
            self.assertNotIn("debug audio", buf.getvalue())
            self.assertEqual(list(Path(tmp).rglob("*.wav")), [])

    def test_write_failure_sanitizes_and_still_sweeps_expired(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            debug_dir = Path(tmp)
            old = debug_dir / "old.wav"
            self._make_wav(old)
            now = datetime.now(timezone.utc)
            os.utime(old, (now.timestamp() - 7200, now.timestamp() - 7200))
            stt = DiscordVoiceSTT(
                VoiceSTTConfig(
                    enabled=True,
                    transcript_channel_id=None,
                    transcript_dir=Path(tmp),
                    debug_audio_dir=debug_dir,
                )
            )
            with mock.patch.object(
                voice_stt_module,
                "_write_debug_wav",
                side_effect=RuntimeError(
                    "canary raw boom C:\\Users\\secret\\voice.wav"
                ),
            ):
                with contextlib.redirect_stdout(io.StringIO()) as buf:
                    stt._dump_debug_audio(_make_chunk())
            out = buf.getvalue()
            self.assertNotIn("canary raw boom", out)
            self.assertNotIn(tmp, out)
            self.assertNotIn(".wav", out)
            self.assertIn("debug audio dump failed", out)
            self.assertFalse(old.exists())


def _make_chunk(user_name: str = "secret-user") -> SpeechChunk:
    now = datetime.now(timezone.utc)
    return SpeechChunk(
        guild_id=None,
        voice_channel_id=None,
        user_id=7,
        user_name=user_name,
        audio=np.zeros(16000, dtype=np.float32),
        started_at=now,
        ended_at=now,
        reason="silence",
    )


class DiscordVoiceTTSSanitizationCanaryTest(unittest.TestCase):
    """voice TTS の sanitization canary 検証。"""

    CANARY = "canary raw boom C:\\Users\\secret\\voice.wav"

    @staticmethod
    def _silence_wav() -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00\x00" * 1600)
        return buf.getvalue()

    def _make_player(self, voice_client: object) -> VoiceTTSPlayer:
        return VoiceTTSPlayer(
            config=VoiceTTSConfig(autoread=True),
            synthesize=lambda *args, **kwargs: self._silence_wav(),
            get_voice_client=lambda: voice_client,
        )

    def test_autoread_records_fixed_code_without_leak(self) -> None:
        player = self._make_player(SimpleNamespace(is_connected=lambda: True))
        with mock.patch.object(player, "say", side_effect=RuntimeError(self.CANARY)):
            with contextlib.redirect_stdout(io.StringIO()) as buf:
                asyncio.run(player.autoread("こんにちは"))
        self.assertEqual(player.last_error, VOICE_TTS_ERR_AUTOREAD)
        self.assertNotIn("canary raw boom", buf.getvalue())
        self.assertNotIn("voice.wav", buf.getvalue())

    def test_say_cancellation_stops_voice_client_and_does_not_count(self) -> None:
        started = asyncio.Event()

        class _FakeVC:
            stop_calls = 0

            def is_connected(self) -> bool:
                return True

            def is_playing(self) -> bool:
                return False

            def stop(self) -> None:
                self.stop_calls += 1

            def play(self, source: object, after=None) -> None:
                started.set()

        vc = _FakeVC()
        player = VoiceTTSPlayer(
            config=VoiceTTSConfig(autoread=False),
            synthesize=lambda *args, **kwargs: self._silence_wav(),
            get_voice_client=lambda: vc,
        )

        async def scenario() -> None:
            task = asyncio.create_task(player.say("テスト"))
            await asyncio.wait_for(started.wait(), timeout=1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        asyncio.run(scenario())
        self.assertEqual(vc.stop_calls, 1)
        self.assertEqual(player.played_count, 0)

    def test_playback_error_raises_fixed_message(self) -> None:
        canary = self.CANARY

        class _FakeVC:
            def __init__(self) -> None:
                self.after = None

            def is_connected(self) -> bool:
                return True

            def is_playing(self) -> bool:
                return False

            def stop(self) -> None:
                pass

            def play(self, source: object, after=None) -> None:
                self.after = after
                after(RuntimeError(canary))

        vc = _FakeVC()
        player = VoiceTTSPlayer(
            config=VoiceTTSConfig(autoread=False),
            synthesize=lambda *args, **kwargs: self._silence_wav(),
            get_voice_client=lambda: vc,
        )

        async def run() -> str:
            with self.assertRaises(VoiceTTSError) as ctx:
                await player.say("テスト")
            return str(ctx.exception)

        msg = asyncio.run(run())
        self.assertEqual(msg, "音声再生中にエラーが発生しました。")
        self.assertEqual(player.last_error, VOICE_TTS_ERR_PLAYBACK)


class DiscordVoiceReplyWiringTest(unittest.TestCase):
    """通話返信の履歴コミットと自動読み上げタスクの配線を検証する。"""

    @staticmethod
    def _voice_reply_source() -> str:
        source = inspect.getsource(bot_module)
        start = source.index("async def handle_voice_reply(")
        end = source.index("def _schedule_voice_reply(", start)
        return source[start:end]

    def test_voice_commit_disables_memory_and_growth(self) -> None:
        block = self._voice_reply_source()
        self.assertIn(
            "session.add_assistant_message(response, store_memory=False, record_growth=False)",
            block,
        )

    def test_autoread_child_is_tracked_by_generation_gate(self) -> None:
        block = self._voice_reply_source()
        task_pos = block.index("autoread_task = asyncio.create_task")
        track_pos = block.index("gate.track(autoread_task)")
        self.assertLess(task_pos, track_pos)


class DiscordVoiceSTTStopContractTest(unittest.TestCase):
    """Consent withdrawal must stop all future processing/posting promptly.

    Deterministic blocking-transcriber tests: a fake transcriber blocks on a
    threading.Event so we can force a stop while the worker is mid-transcription
    and observe that it discards (never writes debug WAV / transcript or
    schedules a send after stop), that the join is bounded, that stop is
    idempotent, and that the worker can be restarted only after it has died.
    """

    def _make_stt(self, tmp: str) -> DiscordVoiceSTT:
        return DiscordVoiceSTT(
            VoiceSTTConfig(
                enabled=True,
                transcript_channel_id=None,
                transcript_dir=Path(tmp),
                save_transcripts=True,
                debug_audio_dir=Path(tmp),
                hallucination_filter=False,
            )
        )

    def _make_voice_client(self) -> SimpleNamespace:
        return SimpleNamespace(
            is_listening=lambda: False,
            stop_listening=lambda: None,
            is_connected=lambda: False,
        )

    @staticmethod
    def _wait_until(cond, timeout: float = 2.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if cond():
                return True
            time.sleep(0.01)
        return cond()

    @staticmethod
    def _feed_frame(sink: DiscordSTTSink, user: SimpleNamespace, value: float = 0.1) -> None:
        frame_size = int(16000 * 30 / 1000)
        pcm = np.full((frame_size * 3, 2), int(value * 32767), dtype=np.int16).tobytes()
        sink.write(user, SimpleNamespace(pcm=pcm))

    def test_stop_discards_in_progress_buffers_without_enqueue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stt = self._make_stt(tmp)
            stt.voice_client = self._make_voice_client()
            sink = DiscordSTTSink(
                config=stt.config,
                output_queue=stt._queue,
                guild_id=None,
                voice_channel_id=None,
            )
            stt.sink = sink
            user = SimpleNamespace(id=7, bot=False, display_name="alice")
            for _ in range(5):
                self._feed_frame(sink, user)
            self.assertEqual(stt._queue.qsize(), 0)

            result = asyncio.run(stt.stop())
            self.assertIn("停止", result)

            self.assertEqual(stt._queue.qsize(), 0)
            self.assertEqual(sink.dropped_segments, 0)
            self.assertFalse(stt.worker_alive)
            self.assertFalse(stt.stop_pending)
            # sink closed: later audio is dropped, never queued
            self._feed_frame(sink, user)
            self.assertEqual(stt._queue.qsize(), 0)

    def test_stop_during_transcription_prevents_writes_and_sends(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stt = self._make_stt(tmp)
            entered = threading.Event()
            release = threading.Event()

            def blocking_transcribe(audio: np.ndarray) -> str:
                entered.set()
                release.wait(2)
                return "こんにちは"

            try:
                with mock.patch.object(stt, "_transcribe", side_effect=blocking_transcribe):
                    with mock.patch.object(stt, "_schedule_transcript_send") as sched:
                        async def scenario() -> None:
                            try:
                                stt._loop = asyncio.get_running_loop()
                                self.assertTrue(stt._ensure_worker())
                                stt._queue.put(_make_chunk())
                                self.assertTrue(entered.wait(timeout=2))
                                with mock.patch.object(
                                    voice_stt_module, "WORKER_JOIN_TIMEOUT", 0.2
                                ):
                                    await stt.stop()
                            finally:
                                release.set()

                        asyncio.run(scenario())
            finally:
                release.set()

            self.assertTrue(self._wait_until(lambda: not stt.worker_alive))
            self.assertEqual(stt.transcript_count, 0)
            sched.assert_not_called()
            self.assertEqual(list(Path(tmp).rglob("*.wav")), [])
            self.assertEqual(list(Path(tmp).rglob("*.jsonl")), [])

    def test_stop_join_timeout_is_bounded_and_keeps_truthful_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stt = self._make_stt(tmp)
            entered = threading.Event()
            block = threading.Event()

            def blocking_transcribe(audio: np.ndarray) -> str:
                entered.set()
                block.wait(2)
                return "x"

            try:
                async def scenario() -> None:
                    try:
                        stt._loop = asyncio.get_running_loop()
                        with mock.patch.object(stt, "_transcribe", side_effect=blocking_transcribe):
                            self.assertTrue(stt._ensure_worker())
                            stt._queue.put(_make_chunk())
                            self.assertTrue(entered.wait(timeout=2))
                            start = time.monotonic()
                            with mock.patch.object(
                                voice_stt_module, "WORKER_JOIN_TIMEOUT", 0.2
                            ):
                                await stt.stop()
                            elapsed = time.monotonic() - start
                        # bounded: did not block the event loop for the full default timeout
                        self.assertLess(elapsed, 3.0)
                        self.assertTrue(stt.stop_pending)
                        self.assertTrue(stt.worker_alive)
                    finally:
                        block.set()

                asyncio.run(scenario())
            finally:
                block.set()
            self.assertTrue(self._wait_until(lambda: not stt.worker_alive))
            self.assertFalse(stt.stop_pending)

    def test_stop_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stt = self._make_stt(tmp)
            stt.voice_client = self._make_voice_client()

            async def scenario() -> None:
                r1 = await stt.stop()
                r2 = await stt.stop()
                self.assertIn("停止", r1)
                self.assertIn("停止", r2)
                stt.close()

            asyncio.run(scenario())
            self.assertFalse(stt.worker_alive)
            self.assertFalse(stt.stop_pending)
            self.assertIsNone(stt.sink)

    def test_prevents_new_worker_while_old_alive_and_restarts_after_death(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stt = self._make_stt(tmp)
            entered = threading.Event()
            release = threading.Event()

            def blocking_transcribe(audio: np.ndarray) -> str:
                entered.set()
                release.wait(2)
                return "x"

            try:
                with mock.patch.object(stt, "_transcribe", side_effect=blocking_transcribe):
                    self.assertTrue(stt._ensure_worker())
                    stt._queue.put(_make_chunk())
                    self.assertTrue(entered.wait(timeout=2))
                    # worker alive: a new worker must NOT be started
                    self.assertFalse(stt._ensure_worker())
                    self.assertTrue(stt.worker_alive)

                    with mock.patch.object(
                        voice_stt_module, "WORKER_JOIN_TIMEOUT", 0.2
                    ):
                        asyncio.run(stt.stop())
                    self.assertTrue(stt.stop_pending)
                    self.assertTrue(stt.worker_alive)

                    release.set()
                    self.assertTrue(self._wait_until(lambda: not stt.worker_alive))
                    self.assertFalse(stt.stop_pending)

                    # death confirmed: a fresh worker can now be started
                    self.assertTrue(stt._ensure_worker())
                    self.assertTrue(stt.worker_alive)
                    stt._clear_queue()
                    with mock.patch.object(
                        voice_stt_module, "WORKER_JOIN_TIMEOUT", 2.0
                    ):
                        stt.close()
                    self.assertFalse(stt.worker_alive)
                    self.assertFalse(stt.stop_pending)
            finally:
                release.set()


class DiscordVoiceSTTConsentRaceTest(unittest.TestCase):
    """Deterministic consent-withdrawal race tests.

    A processing generation plus a consent lock make stop/close deterministic
    against in-flight transcribe/write/schedule/send work, and the sink's
    closed check shares discard's lock so a closed sink never enqueues after
    consent is withdrawn.
    """

    def _make_stt(self, tmp: str) -> DiscordVoiceSTT:
        return DiscordVoiceSTT(
            VoiceSTTConfig(
                enabled=True,
                transcript_channel_id=None,
                transcript_dir=Path(tmp),
                save_transcripts=True,
                debug_audio_dir=Path(tmp),
                hallucination_filter=False,
            )
        )

    @staticmethod
    def _feed_frame(sink: DiscordSTTSink, user: SimpleNamespace, value: float = 0.1) -> None:
        frame_size = int(16000 * 30 / 1000)
        pcm = np.full((frame_size * 3, 2), int(value * 32767), dtype=np.int16).tobytes()
        sink.write(user, SimpleNamespace(pcm=pcm))

    @staticmethod
    def _wait_until(cond, timeout: float = 2.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if cond():
                return True
            time.sleep(0.01)
        return cond()

    def test_generation_capture_and_revocation_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stt = self._make_stt(tmp)
            g1 = stt._capture_generation()
            self.assertTrue(stt._is_generation_active(g1))
            self.assertEqual(stt._revoke_generation(), g1 + 1)
            # revoked generation is deterministically inactive
            self.assertFalse(stt._is_generation_active(g1))
            # generation is monotonically increasing
            g2 = stt._capture_generation()
            self.assertEqual(g2, g1 + 1)
            self.assertGreater(g2, g1)
            self.assertTrue(stt._is_generation_active(g2))

    def test_revoke_cancels_retained_send_futures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stt = self._make_stt(tmp)
            fut = Future()
            with stt._consent_lock:
                stt._send_futures.add(fut)
            self.assertFalse(fut.cancelled())
            stt._revoke_generation()
            self.assertTrue(fut.cancelled())
            # cancelled sends record no error
            stt._record_future_error(fut)
            self.assertEqual(stt.last_error, "")

    def test_send_transcript_rechecks_generation_before_send(self) -> None:
        class _Rec:
            def __init__(self) -> None:
                self.sent: list[str] = []

            async def send(self, text: str) -> None:
                self.sent.append(text)

        with tempfile.TemporaryDirectory() as tmp:
            stt = self._make_stt(tmp)
            stt._bot = object()
            stt.transcript_channel_id = 123
            chunk = _make_chunk(user_name="alice")

            async def run_active() -> _Rec:
                rec = _Rec()
                with mock.patch.object(
                    stt, "_resolve_transcript_channel", new_callable=mock.AsyncMock, return_value=rec
                ):
                    gen = stt._capture_generation()
                    await stt._send_transcript(chunk, "こんにちは", gen)
                return rec

            async def run_revoked() -> _Rec:
                rec = _Rec()
                with mock.patch.object(
                    stt, "_resolve_transcript_channel", new_callable=mock.AsyncMock, return_value=rec
                ):
                    gen = stt._capture_generation()
                    stt._revoke_generation()
                    await stt._send_transcript(chunk, "こんにちは", gen)
                return rec

            active_rec = asyncio.run(run_active())
            self.assertEqual(len(active_rec.sent), 1)
            self.assertIn("alice: こんにちは", active_rec.sent[0])

            revoked_rec = asyncio.run(run_revoked())
            self.assertEqual(revoked_rec.sent, [])

    def test_enqueue_rejects_once_sink_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = VoiceSTTConfig(
                enabled=True,
                transcript_channel_id=None,
                transcript_dir=Path(tmp),
                energy_threshold=0.01,
                silence_duration_ms=60,
                speech_pad_ms=0,
                min_speech_duration_ms=60,
                max_segment_seconds=5.0,
            )
            q: "queue.Queue[SpeechChunk]" = queue.Queue()
            sink = DiscordSTTSink(
                config=config, output_queue=q, guild_id=None, voice_channel_id=None
            )
            user = SimpleNamespace(id=7, bot=False, display_name="alice")
            now = datetime.now(timezone.utc)
            speech = CompletedSpeech(
                audio=np.ones(16000, dtype=np.float32),
                started_at=now,
                ended_at=now,
                reason="stop",
            )
            sink._enqueue(user, speech)
            self.assertEqual(q.qsize(), 1)
            sink.discard_all()
            self.assertTrue(sink._closed)
            sink._enqueue(user, speech)
            self.assertEqual(q.qsize(), 1)

    def test_sink_enqueue_and_discard_are_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = VoiceSTTConfig(
                enabled=True,
                transcript_channel_id=None,
                transcript_dir=Path(tmp),
                energy_threshold=0.01,
                silence_duration_ms=60,
                speech_pad_ms=0,
                min_speech_duration_ms=60,
                max_segment_seconds=5.0,
            )
            q: "queue.Queue[SpeechChunk]" = queue.Queue()
            sink = DiscordSTTSink(
                config=config, output_queue=q, guild_id=None, voice_channel_id=None
            )
            user = SimpleNamespace(id=7, bot=False, display_name="alice")
            now = datetime.now(timezone.utc)
            speech = CompletedSpeech(
                audio=np.ones(16000, dtype=np.float32),
                started_at=now,
                ended_at=now,
                reason="stop",
            )

            entered = threading.Event()
            release = threading.Event()
            real_put = q.put_nowait

            def blocking_put(chunk: SpeechChunk) -> None:
                entered.set()
                release.wait(2)
                real_put(chunk)

            q.put_nowait = blocking_put  # type: ignore[assignment]

            enqueue_thread = threading.Thread(
                target=sink._enqueue, args=(user, speech), daemon=True
            )
            discard_thread: threading.Thread | None = None
            discarded = threading.Event()
            enqueue_thread.start()
            try:
                self.assertTrue(entered.wait(2))

                def do_discard() -> None:
                    sink.discard_all()
                    discarded.set()

                discard_thread = threading.Thread(target=do_discard, daemon=True)
                discard_thread.start()
                time.sleep(0.2)
                # discard is serialized behind the in-flight enqueue (same lock)
                self.assertFalse(discarded.is_set())
            finally:
                release.set()
                enqueue_thread.join(2)
                if discard_thread is not None:
                    discard_thread.join(2)
            self.assertTrue(discarded.is_set())
            # the chunk queued strictly before discard; discard only closed
            self.assertEqual(q.qsize(), 1)
            self.assertTrue(sink._closed)

            # closed sink never enqueues new chunks after discard
            sink._enqueue(user, speech)
            self.assertEqual(q.qsize(), 1)

    def test_stop_cancels_retained_send_futures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stt = self._make_stt(tmp)
            scheduled = threading.Event()
            wait_evt = asyncio.Event()
            original_schedule = stt._schedule_transcript_send

            def recording_schedule(chunk: SpeechChunk, text: str, generation: int) -> None:
                original_schedule(chunk, text, generation)
                scheduled.set()

            async def blocking_resolve():
                await wait_evt.wait()
                return None

            with mock.patch.object(stt, "_transcribe", return_value="こんにちは"):
                with mock.patch.object(
                    stt, "_schedule_transcript_send", side_effect=recording_schedule
                ):
                    with mock.patch.object(
                        stt, "_resolve_transcript_channel", side_effect=blocking_resolve
                    ):
                        with mock.patch.object(
                            stt, "_send_notice", new_callable=mock.AsyncMock
                        ):
                            async def scenario() -> None:
                                try:
                                    stt._loop = asyncio.get_running_loop()
                                    self.assertTrue(stt._ensure_worker())
                                    stt._queue.put(_make_chunk())
                                    self.assertTrue(scheduled.wait(2))
                                    futures = list(stt._send_futures)
                                    self.assertEqual(len(futures), 1)
                                    fut = futures[0]
                                    await stt.stop()
                                    self.assertTrue(fut.cancelled())
                                finally:
                                    wait_evt.set()

                            asyncio.run(scenario())

            self.assertEqual(stt.last_error, "")
            self.assertEqual(len(stt._send_futures), 0)
            self.assertTrue(self._wait_until(lambda: not stt.worker_alive))

    def test_stop_after_transcribe_old_generation_cannot_persist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stt = self._make_stt(tmp)
            entered = threading.Event()
            release = threading.Event()

            def blocking_transcribe(audio: np.ndarray) -> str:
                entered.set()
                release.wait(2)
                return "こんにちは"

            try:
                with mock.patch.object(stt, "_transcribe", side_effect=blocking_transcribe):
                    with mock.patch.object(stt, "_schedule_transcript_send") as sched:
                        async def scenario() -> None:
                            try:
                                stt._loop = asyncio.get_running_loop()
                                self.assertTrue(stt._ensure_worker())
                                stt._queue.put(_make_chunk())
                                self.assertTrue(entered.wait(timeout=2))
                                with mock.patch.object(
                                    voice_stt_module, "WORKER_JOIN_TIMEOUT", 0.2
                                ):
                                    await stt.stop()
                            finally:
                                release.set()

                        asyncio.run(scenario())
            finally:
                release.set()

            self.assertTrue(self._wait_until(lambda: not stt.worker_alive))
            # old generation finished transcribing but could not persist/post/count
            self.assertEqual(stt.transcript_count, 0)
            sched.assert_not_called()
            self.assertEqual(list(Path(tmp).rglob("*.wav")), [])
            self.assertEqual(list(Path(tmp).rglob("*.jsonl")), [])

    def test_schedule_revoked_generation_closes_coroutine_retains_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stt = self._make_stt(tmp)
            stt._loop = object()  # never reached: revoked generation short-circuits
            old_gen = stt._capture_generation()
            stt._revoke_generation()

            async def fake_send() -> None:
                return None

            coro = fake_send()
            with mock.patch.object(
                stt, "_send_transcript", new=lambda chunk, text, generation: coro
            ):
                stt._schedule_transcript_send(_make_chunk(), "こんにちは", old_gen)
            # the revoked generation must not schedule a send, must not retain a
            # future, and must close the coroutine so it never "leaks".
            self.assertIsNone(coro.cr_frame)
            self.assertEqual(len(stt._send_futures), 0)

    def test_schedule_closes_coroutine_when_loop_closed(self) -> None:
        loop = asyncio.new_event_loop()
        loop.close()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                stt = self._make_stt(tmp)
                stt._loop = loop
                gen = stt._capture_generation()

                async def fake_send() -> None:
                    return None

                coro = fake_send()
                with mock.patch.object(
                    stt, "_send_transcript", new=lambda chunk, text, generation: coro
                ):
                    stt._schedule_transcript_send(_make_chunk(), "こんにちは", gen)
                # a closed loop can never run the coroutine; it must be closed to
                # avoid a "never awaited" leak and nothing may be retained.
                self.assertIsNone(coro.cr_frame)
                self.assertEqual(len(stt._send_futures), 0)
        finally:
            loop.close()

    def test_schedule_rejects_open_but_not_running_loop(self) -> None:
        loop = asyncio.new_event_loop()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                stt = self._make_stt(tmp)
                stt._loop = loop  # open yet NOT running: no loop is pumping
                gen = stt._capture_generation()

                async def fake_send() -> None:
                    return None

                coro = fake_send()
                with mock.patch.object(
                    stt, "_send_transcript", new=lambda chunk, text, generation: coro
                ):
                    stt._schedule_transcript_send(_make_chunk(), "こんにちは", gen)
                # an inert loop can never run the coroutine: it must be closed
                # and nothing may be retained.
                self.assertIsNone(coro.cr_frame)
                self.assertEqual(len(stt._send_futures), 0)
        finally:
            loop.close()

    def test_send_notice_is_bounded_and_best_effort(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stt = self._make_stt(tmp)

            async def slow_resolve():
                await asyncio.sleep(30)
                return None

            async def scenario() -> None:
                with mock.patch.object(
                    stt, "_resolve_transcript_channel", side_effect=slow_resolve
                ):
                    with mock.patch.object(voice_stt_module, "NOTICE_SEND_TIMEOUT", 0.05):
                        with contextlib.redirect_stdout(io.StringIO()) as buf:
                            await stt._send_notice("[voice] STT stopped.")
                self.assertIn("notice send timed out", buf.getvalue())
                # best-effort: a stuck notice must not record a send failure
                self.assertEqual(stt.last_error, "")

            asyncio.run(scenario())
if __name__ == "__main__":
    unittest.main()
