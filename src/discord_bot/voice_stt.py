"""Discord voice-channel speech-to-text bridge."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import stat
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import discord
import numpy as np
from discord.opus import OpusError

from src.audio.stt import WhisperSTT
from src.perception.policy import parse_opt_in, resolve_sensor_policy

try:
    from discord.ext import voice_recv
except Exception:  # pragma: no cover - exercised when optional dependency is absent
    voice_recv = None


# discord-ext-voice_recv's reader logs every non-ReceiverReport RTCP packet at
# INFO, which is noisy on real Discord voice streams. Quiet it down so STT
# diagnostics stay readable.
logging.getLogger("discord.ext.voice_recv.reader").setLevel(logging.WARNING)


# Module-level counter for swallowed Opus decode errors. The monkeypatched
# decoder runs deep inside discord-ext-voice_recv with no handle back to our
# sink, so we track the count here and let the sink/status read it. Diagnostic
# only; not reset between sessions.
_OPUS_DECODE_ERRORS = 0


def opus_decode_error_count() -> int:
    return _OPUS_DECODE_ERRORS


def _patch_voice_recv_opus_decoder() -> None:
    """Make PacketDecoder._decode_packet swallow OpusError instead of crashing the reader.

    discord-ext-voice_recv's jitter buffer and FEC handling already absorb most
    packet loss, but a corrupted/missing reference frame can still raise
    OpusError from Decoder.decode. The library propagates that out of the RTP
    callback and stops the voice reader entirely, which kills STT mid-session.

    We must NOT drop the frame (empty PCM) or swap in a fresh decoder here:
    Opus is stateful, so both splice discontinuities into the stream and the
    resulting audio sounds like buzzing/clicking to Whisper. Instead ask the
    decoder to conceal the loss (decode(None)), which emits a plausible 20 ms of
    PCM and keeps decoder state intact.
    """
    if voice_recv is None:
        return
    try:
        from discord.ext.voice_recv.opus import PacketDecoder
    except Exception:
        return
    if getattr(PacketDecoder._decode_packet, "_subpc_patched", False):
        return

    original = PacketDecoder._decode_packet

    def safe_decode(self, packet):  # type: ignore[no-untyped-def]
        global _OPUS_DECODE_ERRORS
        try:
            return original(self, packet)
        except OpusError:
            _OPUS_DECODE_ERRORS += 1
            decoder = getattr(self, "_decoder", None)
            concealed = b""
            if decoder is not None:
                try:
                    # Packet loss concealment: emit ~20 ms of interpolated PCM
                    # without discarding decoder state.
                    concealed = decoder.decode(None, fec=False)
                except Exception:
                    concealed = b""
            if _OPUS_DECODE_ERRORS == 1 or _OPUS_DECODE_ERRORS % 50 == 0:
                print(
                    "[DiscordVoiceSTT] opus decode error concealed "
                    f"count={_OPUS_DECODE_ERRORS}"
                )
            return packet, concealed

    safe_decode._subpc_patched = True  # type: ignore[attr-defined]
    PacketDecoder._decode_packet = safe_decode  # type: ignore[assignment]


_patch_voice_recv_opus_decoder()


# Discord voice is end-to-end encrypted with the DAVE protocol (mandatory:
# advertising max_dave_protocol_version=0 gets the connection closed with 4017).
# discord.py 2.7 implements the MLS handshake and *send-side* frame encryption,
# but discord-ext-voice-recv knows nothing about DAVE, so received frames stay
# E2EE after transport decryption and reach the Opus decoder as random bytes —
# ~40% raise "corrupted stream" and the rest decode into buzzing noise, which
# Whisper transcribes as 「ブーブー」「パカッ」. Fix: run each received frame
# through discord.py's own DaveSession (which holds the MLS group keys) right
# after transport decryption.
_DAVE_DECRYPT_FAILURES = 0
_DAVE_DECRYPT_OK = 0


def dave_decrypt_failure_count() -> int:
    return _DAVE_DECRYPT_FAILURES


def _dave_decrypt_frame(voice_client: Any, ssrc: int, data: bytes) -> bytes:
    """Decrypt one received DAVE media frame; return input unchanged if N/A.

    Falls back to the original bytes when the session isn't ready, the sender
    is unknown, or decryption fails (e.g. unencrypted Opus silence frames) —
    the Opus decoder then either handles them or the loss-concealment path
    kicks in.
    """
    global _DAVE_DECRYPT_FAILURES, _DAVE_DECRYPT_OK
    if not data:
        return data
    try:
        session = getattr(getattr(voice_client, "_connection", None), "dave_session", None)
        if session is None or not session.ready:
            return data
        user_id = voice_client._get_id_from_ssrc(ssrc)
        if user_id is None:
            return data
        import davey

        frame = session.decrypt(user_id, davey.MediaType.audio, bytes(data))
        if not frame:
            return data
        _DAVE_DECRYPT_OK += 1
        if _DAVE_DECRYPT_OK == 1:
            print("[DiscordVoiceSTT] DAVE E2EE フレーム復号 OK (受信音声は正常に復号されます)")
        return bytes(frame)
    except Exception:
        _DAVE_DECRYPT_FAILURES += 1
        if _DAVE_DECRYPT_FAILURES <= 3 or _DAVE_DECRYPT_FAILURES % 200 == 0:
            print(
                "[DiscordVoiceSTT] DAVE decrypt failed "
                f"count={_DAVE_DECRYPT_FAILURES}"
            )
        return data


def _patch_voice_recv_dave_decrypt() -> None:
    """Insert DAVE frame decryption right after transport decryption.

    AudioReader binds `decrypt_rtp` as an instance attribute on its
    PacketDecryptor, so wrap it per-reader from a patched AudioReader.__init__.
    This point sees every packet exactly once (including ones later used for
    FEC), before any Opus decoding.
    """
    if voice_recv is None:
        return
    try:
        from discord.ext.voice_recv.reader import AudioReader
    except Exception:
        return
    if getattr(AudioReader.__init__, "_subpc_dave_patched", False):
        return

    original_init = AudioReader.__init__

    def patched_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        original_init(self, *args, **kwargs)
        voice_client = self.voice_client
        inner_decrypt = self.decryptor.decrypt_rtp

        def decrypt_rtp_with_dave(packet):  # type: ignore[no-untyped-def]
            return _dave_decrypt_frame(voice_client, packet.ssrc, inner_decrypt(packet))

        self.decryptor.decrypt_rtp = decrypt_rtp_with_dave

    patched_init._subpc_dave_patched = True  # type: ignore[attr-defined]
    AudioReader.__init__ = patched_init  # type: ignore[assignment]


_patch_voice_recv_dave_decrypt()


DISCORD_PCM_SAMPLE_RATE = 48000
DISCORD_PCM_CHANNELS = 2
STT_SAMPLE_RATE = 16000
MAX_DISCORD_MESSAGE = 1900

# Bounded wait for the STT worker to exit after a stop. Kept deliberately short
# so async stop never wedges the event loop; if the worker is stuck mid-LLM/STT
# it is left alive and reported truthfully via worker_alive/stop_pending.
WORKER_JOIN_TIMEOUT = 5.0

# Bounded best-effort wait for the "[voice] STT started/stopped." notice post.
# A stuck transcript channel (slow fetch/send) must never wedge start/stop on
# the event loop, so the notice is wrapped in wait_for and failures swallowed.
NOTICE_SEND_TIMEOUT = 5.0

# Bounded retention for the opt-in debug WAV dump. Positive seconds only;
# unset uses a conservative default so debug audio never accumulates without
# bound. Zero / negative / non-numeric / over-max TTL values are invalid and
# fail closed by disabling debug writes entirely.
_DEBUG_AUDIO_TTL_DEFAULT_SEC = 3600
_DEBUG_AUDIO_TTL_MAX_SEC = 86400


# Whisper on near-silent or short Japanese audio reliably emits these filler
# phrases. Drop them so the transcript channel stays useful.
_WHISPER_HALLUCINATION_PHRASES = (
    "ご視聴ありがとうございました",
    "ご清聴ありがとうございました",
    "ありがとうございました",
    "おつかれさまでした",
    "お疲れ様でした",
    "以上で終わります",
    "ご視聴ありがとうございました。",
)

_HIRAGANA = set(
    "ぁぃぅぇぉゃゅょっゎあいうえおかきくけこさしすせそたちつてとなにぬねの"
    "はひふへほまみむめもやゆよらりるれろわをんー"
)
_KATAKANA = set(
    "ァィゥェォャュョッヮアイウエオカキクケコサシスセソタチツテトナニヌネノ"
    "ハヒフヘホマミムメモヤユヨラリルレロワヲンー"
)
_HALLUCINATION_PUNCT = set(" 　、。！？!?,.")


def _is_likely_hallucination(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return False
    for phrase in _WHISPER_HALLUCINATION_PHRASES:
        if phrase in stripped:
            return True
    if len(stripped) <= 2:
        chars = set(stripped) - _HALLUCINATION_PUNCT
        if chars and chars <= (_HIRAGANA | _KATAKANA):
            return True
    return False


class VoiceSTTError(RuntimeError):
    """Raised for user-facing voice STT command failures."""


# last_error に設定する固定型コード。生の例外テキスト・パス・URL・端末情報は
# ログ / slash reply / status へ一切載せず、診断はこれらのコードで行う。
VOICE_STT_ERR_WORKER = "worker_failure"
VOICE_STT_ERR_SEND = "transcript_send_failed"
VOICE_STT_ERR_LISTEN = "listen_failed"

_KNOWN_STT_ERR_CODES = frozenset({VOICE_STT_ERR_WORKER, VOICE_STT_ERR_SEND, VOICE_STT_ERR_LISTEN})


def safe_stt_last_error(value: str) -> str:
    """last_error は既知の固定コードだけ表示し、未知の文字列は '-' に落とす。"""
    return value if value in _KNOWN_STT_ERR_CODES else "-"


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_int(value: str | None, default: int) -> int:
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _parse_optional_int(value: str | None) -> int | None:
    if value is None or not value.strip():
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _parse_float(value: str | None, default: float) -> float:
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _resolve_debug_audio_ttl_sec() -> int | None:
    """Resolve the debug-audio retention TTL, or None on invalid value.

    Positive bounded seconds (1.._DEBUG_AUDIO_TTL_MAX_SEC). Unset -> the
    conservative default. Zero / negative / non-numeric / over-max -> None,
    which callers treat as fail closed (debug writes disabled).
    """
    raw = os.environ.get("DISCORD_VOICE_STT_DEBUG_AUDIO_TTL_SEC", "").strip()
    if not raw:
        return _DEBUG_AUDIO_TTL_DEFAULT_SEC
    try:
        ttl = int(raw)
    except ValueError:
        return None
    if ttl <= 0 or ttl > _DEBUG_AUDIO_TTL_MAX_SEC:
        return None
    return ttl


def _resolve_debug_audio_dir(project_root: Path) -> Path | None:
    """Optional dir for dumping the exact 16 kHz mono audio sent to Whisper.

    Set DISCORD_VOICE_STT_DEBUG_AUDIO_DIR to a path to enable. Diagnostic only:
    lets us inspect/listen to what STT actually receives when transcripts look
    like noise. Unset (default) means no dump. An invalid
    DISCORD_VOICE_STT_DEBUG_AUDIO_TTL_SEC (zero / negative / non-numeric /
    over-max) fails closed and disables debug writes entirely.
    """
    raw = os.environ.get("DISCORD_VOICE_STT_DEBUG_AUDIO_DIR", "").strip()
    if not raw:
        return None
    if _resolve_debug_audio_ttl_sec() is None:
        return None
    path = Path(raw)
    return path if path.is_absolute() else project_root / path


def _write_debug_wav(path: Path, audio: np.ndarray) -> None:
    import wave

    pcm16 = np.clip(audio, -1.0, 1.0)
    pcm16 = (pcm16 * 32767.0).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(STT_SAMPLE_RATE)
        wav.writeframes(pcm16.tobytes())


def _sweep_debug_audio(
    debug_dir: Path,
    ttl_sec: int,
    now: datetime | None = None,
) -> None:
    """Best-effort delete regular *.wav files directly in debug_dir older than TTL.

    Diagnostics must never break the worker, so every failure is swallowed.
    Candidates are only regular .wav files in the top level of debug_dir: no
    recursion, no symlink following, no other file types, and nothing outside
    the configured directory is ever touched. Never exposes paths/filenames.
    """
    if ttl_sec <= 0:
        return
    try:
        cutoff = (now or datetime.now(timezone.utc)).timestamp() - ttl_sec
        for entry in debug_dir.iterdir():
            try:
                st = os.lstat(entry)
                if not stat.S_ISREG(st.st_mode):
                    continue
                if entry.suffix.lower() != ".wav":
                    continue
                if st.st_mtime < cutoff:
                    entry.unlink(missing_ok=True)
            except OSError:
                continue
    except OSError:
        return


def _split_message(text: str) -> list[str]:
    if not text:
        return ["(empty)"]
    chunks = []
    while text:
        chunks.append(text[:MAX_DISCORD_MESSAGE])
        text = text[MAX_DISCORD_MESSAGE:]
    return chunks


def pcm48_stereo_to_16k_mono(pcm: bytes) -> np.ndarray:
    """Convert Discord decoded PCM to float32 mono 16 kHz audio."""
    if not pcm:
        return np.empty(0, dtype=np.float32)

    samples = np.frombuffer(pcm, dtype="<i2")
    frame_count = samples.size // DISCORD_PCM_CHANNELS
    if frame_count == 0:
        return np.empty(0, dtype=np.float32)

    samples = samples[: frame_count * DISCORD_PCM_CHANNELS]
    stereo = samples.reshape((-1, DISCORD_PCM_CHANNELS)).astype(np.float32)
    mono48 = stereo.mean(axis=1) / 32768.0

    factor = DISCORD_PCM_SAMPLE_RATE // STT_SAMPLE_RATE
    target_count = mono48.size // factor
    if target_count == 0:
        return np.empty(0, dtype=np.float32)

    # Discord voice is 48 kHz and Whisper input here is 16 kHz. Averaging each
    # 3-sample group is cheap and good enough for speech transcription.
    return mono48[: target_count * factor].reshape((-1, factor)).mean(axis=1).astype(np.float32)


@dataclass(frozen=True)
class VoiceSTTConfig:
    enabled: bool
    transcript_channel_id: int | None
    transcript_dir: Path
    microphone_enabled: bool = False
    timezone: str = "Asia/Tokyo"
    language: str = "ja"
    model_size: str = "auto"
    device: str = "auto"
    compute_type: str = "auto"
    device_index: int = 0
    energy_threshold: float = 0.008
    silence_duration_ms: int = 700
    speech_pad_ms: int = 240
    min_speech_duration_ms: int = 400
    max_segment_seconds: float = 12.0
    max_queue_size: int = 16
    save_transcripts: bool = False
    hallucination_filter: bool = True
    debug_audio_dir: Path | None = None
    debug_audio_ttl_sec: int = _DEBUG_AUDIO_TTL_DEFAULT_SEC

    @classmethod
    def from_env(cls, project_root: str | Path, *, timezone: str = "Asia/Tokyo") -> "VoiceSTTConfig":
        root = Path(project_root)
        transcript_dir = Path(
            os.environ.get(
                "DISCORD_VOICE_TRANSCRIPT_DIR",
                "data/discord_voice/transcripts",
            )
        )
        if not transcript_dir.is_absolute():
            transcript_dir = root / transcript_dir

        return cls(
            enabled=parse_opt_in(os.environ.get("DISCORD_VOICE_STT_ENABLED")),
            microphone_enabled=resolve_sensor_policy(os.environ).microphone,
            transcript_channel_id=_parse_optional_int(os.environ.get("DISCORD_VOICE_TRANSCRIPT_CHANNEL_ID")),
            transcript_dir=transcript_dir,
            timezone=os.environ.get("DISCORD_VOICE_TIMEZONE", timezone).strip() or timezone,
            language=os.environ.get("DISCORD_VOICE_STT_LANGUAGE", "ja").strip() or "ja",
            model_size=os.environ.get("DISCORD_VOICE_STT_MODEL", "auto").strip() or "auto",
            device=os.environ.get("DISCORD_VOICE_STT_DEVICE", "auto").strip() or "auto",
            compute_type=(
                os.environ.get("DISCORD_VOICE_STT_COMPUTE_TYPE", "auto").strip()
                or "auto"
            ),
            device_index=_parse_int(os.environ.get("DISCORD_VOICE_STT_DEVICE_INDEX"), 0),
            energy_threshold=_parse_float(
                os.environ.get("DISCORD_VOICE_STT_ENERGY_THRESHOLD"),
                0.008,
            ),
            silence_duration_ms=_parse_int(
                os.environ.get("DISCORD_VOICE_STT_SILENCE_MS"),
                700,
            ),
            speech_pad_ms=_parse_int(os.environ.get("DISCORD_VOICE_STT_PAD_MS"), 240),
            min_speech_duration_ms=_parse_int(
                os.environ.get("DISCORD_VOICE_STT_MIN_MS"),
                400,
            ),
            max_segment_seconds=_parse_float(
                os.environ.get("DISCORD_VOICE_STT_MAX_SECONDS"),
                12.0,
            ),
            max_queue_size=_parse_int(os.environ.get("DISCORD_VOICE_STT_QUEUE_SIZE"), 16),
            save_transcripts=parse_opt_in(
                os.environ.get("DISCORD_VOICE_STT_SAVE_TRANSCRIPTS")
            ),
            hallucination_filter=_parse_bool(
                os.environ.get("DISCORD_VOICE_STT_HALLUCINATION_FILTER"),
                default=True,
            ),
            debug_audio_dir=_resolve_debug_audio_dir(root),
            debug_audio_ttl_sec=_resolve_debug_audio_ttl_sec()
            or _DEBUG_AUDIO_TTL_DEFAULT_SEC,
        )


@dataclass(frozen=True)
class CompletedSpeech:
    audio: np.ndarray
    started_at: datetime
    ended_at: datetime
    reason: str


@dataclass(frozen=True)
class SpeechChunk:
    guild_id: int | None
    voice_channel_id: int | None
    user_id: int
    user_name: str
    audio: np.ndarray
    started_at: datetime
    ended_at: datetime
    reason: str

    @property
    def duration_sec(self) -> float:
        return self.audio.size / STT_SAMPLE_RATE


class SpeechSegmenter:
    """Small energy-based VAD for per-speaker Discord PCM chunks."""

    def __init__(self, config: VoiceSTTConfig):
        self.sample_rate = STT_SAMPLE_RATE
        self.frame_size = int(self.sample_rate * 30 / 1000)
        self.energy_threshold = config.energy_threshold
        self.silence_frames = max(1, int(config.silence_duration_ms / 30))
        self.pad_frames = max(0, int(config.speech_pad_ms / 30))
        self.min_samples = int(self.sample_rate * config.min_speech_duration_ms / 1000)
        self.max_samples = int(self.sample_rate * config.max_segment_seconds)

        self._pending = np.empty(0, dtype=np.float32)
        self._pre_buffer: deque[np.ndarray] = deque(maxlen=self.pad_frames)
        self._speech_frames: list[np.ndarray] = []
        self._is_speaking = False
        self._silence_count = 0
        self._started_at: datetime | None = None

    def add_audio(self, audio: np.ndarray, now: datetime) -> list[CompletedSpeech]:
        if audio.size == 0:
            return []
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        self._pending = np.concatenate([self._pending, audio])
        completed: list[CompletedSpeech] = []
        while self._pending.size >= self.frame_size:
            frame = self._pending[: self.frame_size]
            self._pending = self._pending[self.frame_size :]
            speech = self._process_frame(frame, now)
            if speech is not None:
                completed.append(speech)
        return completed

    def discard(self) -> None:
        """Drop all buffered (not yet emitted) audio without producing a segment."""
        self._pending = np.empty(0, dtype=np.float32)
        self._pre_buffer.clear()
        self._speech_frames = []
        self._is_speaking = False
        self._silence_count = 0
        self._started_at = None

    def flush(self, now: datetime, *, reason: str = "speaking_stop") -> CompletedSpeech | None:
        if self._pending.size and self._is_speaking:
            self._speech_frames.append(self._pending.copy())
        self._pending = np.empty(0, dtype=np.float32)
        if not self._is_speaking and not self._speech_frames:
            self._pre_buffer.clear()
            return None
        return self._emit(now, reason)

    def _process_frame(self, frame: np.ndarray, now: datetime) -> CompletedSpeech | None:
        rms = float(np.sqrt(np.mean(frame ** 2))) if frame.size else 0.0
        is_speech = rms > self.energy_threshold

        if not self._is_speaking:
            self._pre_buffer.append(frame.copy())
            if is_speech:
                self._is_speaking = True
                self._silence_count = 0
                self._started_at = now
                self._speech_frames = list(self._pre_buffer)
                self._speech_frames.append(frame.copy())
                self._pre_buffer.clear()
            return None

        self._speech_frames.append(frame.copy())
        if is_speech:
            self._silence_count = 0
        else:
            self._silence_count += 1
            if self._silence_count >= self.silence_frames:
                return self._emit(now, "silence")

        total_samples = sum(item.size for item in self._speech_frames)
        if total_samples >= self.max_samples:
            return self._emit(now, "max_duration")
        return None

    def _emit(self, now: datetime, reason: str) -> CompletedSpeech | None:
        started_at = self._started_at or now
        frames = self._speech_frames
        self._speech_frames = []
        self._pre_buffer.clear()
        self._is_speaking = False
        self._silence_count = 0
        self._started_at = None

        if not frames:
            return None
        audio = np.concatenate(frames).astype(np.float32)
        if audio.size < self.min_samples:
            return None
        return CompletedSpeech(
            audio=audio,
            started_at=started_at,
            ended_at=now,
            reason=reason,
        )


_AudioSinkBase = voice_recv.AudioSink if voice_recv is not None else object


def _sink_listener(name: str | None = None):
    if voice_recv is None:
        return lambda func: func
    return voice_recv.AudioSink.listener(name) if name is not None else voice_recv.AudioSink.listener()


class DiscordSTTSink(_AudioSinkBase):
    """Receives Discord PCM packets and emits speech chunks to a worker queue."""

    def __init__(
        self,
        *,
        config: VoiceSTTConfig,
        output_queue: "queue.Queue[SpeechChunk]",
        guild_id: int | None,
        voice_channel_id: int | None,
    ):
        super().__init__()
        self.config = config
        self.output_queue = output_queue
        self.guild_id = guild_id
        self.voice_channel_id = voice_channel_id
        self.dropped_segments = 0
        self.received_packets = 0
        self.received_audio_seconds = 0.0
        self._buffers: dict[int, SpeechSegmenter] = {}
        self._lock = threading.RLock()
        self._closed = False

    def wants_opus(self) -> bool:
        return False

    def write(self, user: discord.User | discord.Member | None, data: Any) -> None:
        if user is None or getattr(user, "bot", False):
            return
        pcm = getattr(data, "pcm", b"")
        if not pcm:
            return

        with self._lock:
            if self._closed:
                return
            now = datetime.now(timezone.utc)
            audio = pcm48_stereo_to_16k_mono(pcm)
            self.received_packets += 1
            self.received_audio_seconds += audio.size / STT_SAMPLE_RATE
            if self.received_packets == 1 or self.received_packets % 250 == 0:
                print(
                    "[DiscordVoiceSTT] receiving audio "
                    f"packets={self.received_packets} "
                    f"audio_sec={self.received_audio_seconds:.1f}"
                )
            segmenter = self._buffers.setdefault(user.id, SpeechSegmenter(self.config))
            completed = segmenter.add_audio(audio, now)
        for speech in completed:
            self._enqueue(user, speech)

    @_sink_listener()
    def on_voice_member_speaking_stop(self, member: discord.Member) -> None:
        self.flush_member(member)

    @_sink_listener()
    def on_voice_member_disconnect(self, member: discord.Member, ssrc: int | None = None) -> None:
        self.flush_member(member)

    def flush_member(self, member: discord.Member | discord.User) -> None:
        now = datetime.now(timezone.utc)
        with self._lock:
            segmenter = self._buffers.get(member.id)
            speech = segmenter.flush(now) if segmenter is not None else None
        if speech is not None:
            self._enqueue(member, speech)

    def flush_all(self) -> None:
        now = datetime.now(timezone.utc)
        pending: list[tuple[discord.User | discord.Member, CompletedSpeech]] = []
        with self._lock:
            for user_id, segmenter in self._buffers.items():
                speech = segmenter.flush(now, reason="stop")
                if speech is None:
                    continue
                user = self._user_from_id(user_id)
                if user is not None:
                    pending.append((user, speech))
        for user, speech in pending:
            self._enqueue(user, speech)

    def discard_all(self) -> None:
        """Drop every segmenter's raw buffer and close the sink without enqueueing.

        Used on consent withdrawal so in-progress audio is never turned into a
        queued SpeechChunk (and therefore never transcribed/posted).
        """
        with self._lock:
            self._closed = True
            for segmenter in self._buffers.values():
                segmenter.discard()
            self._buffers.clear()

    def cleanup(self) -> None:
        self.discard_all()

    def _enqueue(self, user: discord.User | discord.Member, speech: CompletedSpeech) -> None:
        chunk = SpeechChunk(
            guild_id=self.guild_id,
            voice_channel_id=self.voice_channel_id,
            user_id=user.id,
            user_name=getattr(user, "display_name", None) or getattr(user, "name", None) or str(user),
            audio=speech.audio,
            started_at=speech.started_at,
            ended_at=speech.ended_at,
            reason=speech.reason,
        )
        # The closed check must share discard_all's lock: a discard (consent
        # withdrawal) that races an in-flight enqueue either wins the lock first
        # (chunk rejected) or loses (chunk queued strictly before discard), so a
        # closed sink never emits a new chunk after discard returns.
        with self._lock:
            if self._closed:
                print("[DiscordVoiceSTT] sink closed; speech segment dropped")
                return
            try:
                self.output_queue.put_nowait(chunk)
                print(
                    "[DiscordVoiceSTT] speech segment queued "
                    f"duration={chunk.duration_sec:.2f}s "
                    f"reason={chunk.reason} queue={self.output_queue.qsize()}"
                )
            except queue.Full:
                self.dropped_segments += 1
                print(
                    "[DiscordVoiceSTT] speech segment dropped "
                    f"dropped={self.dropped_segments}"
                )

    def _user_from_id(self, user_id: int) -> discord.User | discord.Member | None:
        voice_client = getattr(self, "voice_client", None)
        if voice_client is None:
            return None
        guild = getattr(voice_client, "guild", None)
        if guild is not None:
            member = guild.get_member(user_id)
            if member is not None:
                return member
        client = getattr(voice_client, "client", None)
        return client.get_user(user_id) if client is not None else None


class DiscordVoiceSTT:
    """Manage Discord voice receive, STT worker, transcript posting, and logs."""

    def __init__(self, config: VoiceSTTConfig):
        self.config = config
        self.voice_client: Any | None = None
        self.sink: DiscordSTTSink | None = None
        self.transcript_channel_id: int | None = config.transcript_channel_id

        self._bot: discord.Client | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: "queue.Queue[SpeechChunk]" = queue.Queue(maxsize=config.max_queue_size)
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._stop_pending = False
        self._stt: WhisperSTT | None = None
        self._stt_lock = threading.Lock()
        self._file_lock = threading.Lock()

        # Monotonically increasing processing generation. A worker or a pending
        # send captures the generation at the moment it starts work; stop/close
        # revokes it (bumps under the consent lock) so anything still in flight
        # from the revoked generation can finish transcribing but can never
        # persist a transcript, dump debug audio, or post/schedule a send.
        self._generation = 0
        self._consent_lock = threading.RLock()
        self._send_futures: set[Any] = set()

        self.started_at: datetime | None = None
        self.transcript_count = 0
        self.last_transcript_at: datetime | None = None
        self.last_error: str = ""

    @property
    def available(self) -> bool:
        return voice_recv is not None

    @property
    def connected(self) -> bool:
        return bool(self.voice_client is not None and self.voice_client.is_connected())

    @property
    def listening(self) -> bool:
        return bool(
            self.voice_client is not None
            and hasattr(self.voice_client, "is_listening")
            and self.voice_client.is_listening()
        )

    @property
    def worker_alive(self) -> bool:
        """True only while the STT worker thread is actually running."""
        return self._worker is not None and self._worker.is_alive()

    @property
    def stop_pending(self) -> bool:
        """True only while a stop is requested but the worker is still alive."""
        return self._stop_pending and self.worker_alive

    async def join(self, interaction: discord.Interaction) -> str:
        self._validate_enabled()
        if interaction.guild is None:
            raise VoiceSTTError("voice STT はサーバー内でだけ使えます。")
        channel = self._user_voice_channel(interaction)

        current = interaction.guild.voice_client
        if current is not None and current.is_connected():
            if getattr(current, "channel", None) and current.channel.id != channel.id:
                await current.move_to(channel)
            self.voice_client = current
        else:
            assert voice_recv is not None
            self.voice_client = await channel.connect(cls=voice_recv.VoiceRecvClient, self_deaf=False)

        if not hasattr(self.voice_client, "listen"):
            await self.voice_client.disconnect(force=True)
            assert voice_recv is not None
            self.voice_client = await channel.connect(cls=voice_recv.VoiceRecvClient, self_deaf=False)

        return (
            f"voice channel に参加しました: {channel.name}\n"
            "文字起こしを始めるには /voice start を実行してください。"
        )

    async def start(
        self,
        interaction: discord.Interaction,
        *,
        transcript_channel: discord.TextChannel | None = None,
    ) -> str:
        await self.join(interaction)
        if self.voice_client is None:
            raise VoiceSTTError("voice client が初期化されていません。")
        if self.listening:
            return "voice STT はすでに開始済みです。"

        self.last_error = ""
        self._bot = interaction.client
        self._loop = asyncio.get_running_loop()
        self.transcript_channel_id = (
            transcript_channel.id
            if transcript_channel is not None
            else self.config.transcript_channel_id or interaction.channel_id
        )
        if not self._ensure_worker():
            raise VoiceSTTError(
                "voice STT は前回の処理がまだ停止していません。少し待ってから再試行してください。"
            )

        guild_id = interaction.guild_id
        voice_channel_id = getattr(getattr(self.voice_client, "channel", None), "id", None)
        self.sink = DiscordSTTSink(
            config=self.config,
            output_queue=self._queue,
            guild_id=guild_id,
            voice_channel_id=voice_channel_id,
        )
        self.voice_client.listen(self.sink, after=self._after_listening)
        self.started_at = datetime.now(timezone.utc)
        print(
            "[DiscordVoiceSTT] listening started "
            f"guild={guild_id} voice_channel={voice_channel_id} "
            f"transcript_channel={self.transcript_channel_id}"
        )
        await self._send_notice(
            f"[voice] STT started. transcript_channel_id={self.transcript_channel_id}"
        )
        return "voice STT を開始しました。通話音声を文字起こしします。"

    async def stop(self) -> str:
        await self._halt_processing()
        await self._send_notice("[voice] STT stopped.")
        return "voice STT を停止しました。"

    async def _halt_processing(self) -> None:
        """Consent withdrawal: stop all future processing and posting promptly.

        Signals the worker first so anything already in flight is discarded
        rather than transcribed/posted. The processing generation is revoked
        (and retained send futures cancelled) before the sink is discarded, so
        an in-flight worker of the old generation can finish transcribing but
        cannot persist/post/count. Then stops listening, discards the sink's
        raw buffers (never flushing them to the queue), clears queued chunks,
        and bounded-joins the worker without blocking the event loop.
        """
        self._stop_event.set()
        # Revoke off the event loop: the worker's commit barrier holds the
        # consent lock while persisting, and waiting on that lock (plus
        # cancelling retained futures) must never block the event loop during
        # stop.
        await asyncio.to_thread(self._revoke_generation)
        if self.voice_client is not None and hasattr(self.voice_client, "is_listening"):
            try:
                if self.voice_client.is_listening():
                    self.voice_client.stop_listening()
            except Exception:
                print("[DiscordVoiceSTT] failed to stop listening")
        if self.sink is not None:
            sink = self.sink
            sink.discard_all()
            print(
                "[DiscordVoiceSTT] listening stopped "
                f"packets={sink.received_packets} "
                f"audio_sec={sink.received_audio_seconds:.1f} "
                f"transcripts={self.transcript_count}"
            )
            self.sink = None
        self.started_at = None
        self._clear_queue()
        await self._join_worker_async()

    def _clear_queue(self) -> None:
        """Drain queued SpeechChunks, discarding each with its task_done call."""
        while True:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                return

    def _capture_generation(self) -> int:
        """Read the current processing generation atomically w.r.t. revoke."""
        with self._consent_lock:
            return self._generation

    def _is_generation_active(self, generation: int) -> bool:
        """True while `generation` is still the live processing generation.

        Synchronized with _revoke_generation via the consent lock, so a revoked
        worker/send deterministically sees an inactive generation and a worker
        that already passed this check persists only while holding the lock.
        """
        with self._consent_lock:
            return generation == self._generation

    def _revoke_generation(self) -> int:
        """Revoke the current processing generation and cancel retained sends.

        Bumps the generation under the consent lock and cancels every retained
        run_coroutine_threadsafe send future. Cancelling is done after releasing
        the lock so a cancel's done callback never runs while holding it; a
        cancelled future records no error (see _record_future_error).
        """
        with self._consent_lock:
            self._generation += 1
            futures = list(self._send_futures)
            self._send_futures.clear()
        for future in futures:
            future.cancel()
        return self._generation

    def _join_worker(self, timeout: float | None = None) -> None:
        """Bounded join. Release thread ownership only when death is confirmed."""
        if timeout is None:
            timeout = WORKER_JOIN_TIMEOUT
        worker = self._worker
        if worker is None or not worker.is_alive():
            self._worker = None
            self._stop_pending = False
            return
        worker.join(timeout)
        if worker.is_alive():
            self._stop_pending = True
        else:
            self._worker = None
            self._stop_pending = False

    async def _join_worker_async(self, timeout: float | None = None) -> None:
        """Bounded join off the event loop so async stop never blocks it."""
        if timeout is None:
            timeout = WORKER_JOIN_TIMEOUT
        if self._worker is None or not self._worker.is_alive():
            self._worker = None
            self._stop_pending = False
            return
        await asyncio.to_thread(self._join_worker, timeout)

    async def leave(self) -> str:
        await self.stop()
        if self.voice_client is not None and self.voice_client.is_connected():
            await self.voice_client.disconnect(force=True)
        self.voice_client = None
        return "voice channel から退出しました。"

    def close(self) -> None:
        """Synchronous shutdown: revoke, discard, clear, signal, bounded-join worker."""
        self._stop_event.set()
        self._revoke_generation()
        if self.voice_client is not None and hasattr(self.voice_client, "is_listening"):
            try:
                if self.voice_client.is_listening():
                    self.voice_client.stop_listening()
            except Exception:
                pass
        if self.sink is not None:
            self.sink.discard_all()
            self.sink = None
        self.started_at = None
        self._clear_queue()
        self._join_worker()

    def status_text(self) -> str:
        dropped = self.sink.dropped_segments if self.sink is not None else 0
        packets = self.sink.received_packets if self.sink is not None else 0
        audio_seconds = self.sink.received_audio_seconds if self.sink is not None else 0.0
        decode_errors = opus_decode_error_count()
        return (
            f"voice_stt_enabled: {self.config.enabled}\n"
            f"voice_microphone_enabled: {self.config.microphone_enabled}\n"
            f"voice_recv_available: {self.available}\n"
            f"voice_connected: {self.connected}\n"
            f"voice_listening: {self.listening}\n"
            f"voice_worker_alive: {self.worker_alive}\n"
            f"voice_stop_pending: {self.stop_pending}\n"
            f"voice_transcript_channel_id: {self.transcript_channel_id or '-'}\n"
            f"voice_queue_size: {self._queue.qsize()}\n"
            f"voice_received_packets: {packets}\n"
            f"voice_received_audio_sec: {audio_seconds:.1f}\n"
            f"voice_transcripts: {self.transcript_count}\n"
            f"voice_decode_errors: {decode_errors}\n"
            f"voice_dave_decrypt_failures: {dave_decrypt_failure_count()}\n"
            f"voice_dropped_segments: {dropped}\n"
            f"voice_last_error: {safe_stt_last_error(self.last_error)}"
        )

    def _validate_enabled(self) -> None:
        # 接続前に二重ゲートを検証する (fail closed)。両方 true のときだけ実
        # receive / STT の構築・開始を許す。canonical 名 (SENSOR_MICROPHONE_ENABLED)
        # は共有 SensorPolicy の resolve_sensor_policy で解決済みで、canonical が
        # 存在すればその値が確定 (false/空/不正値は fail closed)。
        if not self.config.enabled:
            raise VoiceSTTError(
                "voice STT は無効です。DISCORD_VOICE_STT_ENABLED=true を設定してください。"
            )
        if not self.config.microphone_enabled:
            raise VoiceSTTError(
                "voice STT のマイク政策が無効です。共有 SensorPolicy の "
                "SENSOR_MICROPHONE_ENABLED=true を設定してください。"
            )
        if voice_recv is None:
            raise VoiceSTTError(
                "discord-ext-voice-recv が未導入です。requirements を更新してインストールしてください。"
            )

    @staticmethod
    def _user_voice_channel(interaction: discord.Interaction) -> discord.VoiceChannel:
        voice_state = getattr(interaction.user, "voice", None)
        channel = getattr(voice_state, "channel", None)
        if channel is None:
            raise VoiceSTTError("先にDiscordの通話チャンネルへ入ってから実行してください。")
        if not isinstance(channel, discord.VoiceChannel):
            raise VoiceSTTError("通常のvoice channelで実行してください。")
        return channel

    def _ensure_worker(self) -> bool:
        """Start a fresh STT worker unless one is still alive.

        Returns True when a fresh worker was started. Returns False (and does
        NOT start a new worker) while the previous one remains alive, so a
        restart never runs two workers at once.
        """
        if self._worker is not None and self._worker.is_alive():
            return False
        self._stop_event.clear()
        self._stop_pending = False
        self._worker = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="discord-voice-stt-worker",
        )
        self._worker.start()
        return True

    def _record_worker_error(self) -> None:
        """Set last_error to a fixed type code without leaking exception text."""
        self.last_error = VOICE_STT_ERR_WORKER
        print("[DiscordVoiceSTT] worker failure")

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                chunk = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                if self._stop_event.is_set():
                    continue
                generation = self._capture_generation()
                text = self._transcribe(chunk.audio)
                # Final generation-active check synchronized with revoke. All
                # persist/post/count work happens under the consent lock so a
                # revoke either wins first (nothing is written/scheduled) or
                # runs after this block completes; it can never interleave.
                with self._consent_lock:
                    if generation != self._generation:
                        continue
                    if self.config.debug_audio_dir is not None:
                        self._dump_debug_audio(chunk)
                    if text and self.config.hallucination_filter and _is_likely_hallucination(text):
                        print(
                            "[DiscordVoiceSTT] filtered hallucination "
                            f"duration={chunk.duration_sec:.2f}s"
                        )
                        text = ""
                    if text:
                        self.transcript_count += 1
                        self.last_transcript_at = datetime.now(timezone.utc)
                        self._write_transcript(chunk, text)
                        self._schedule_transcript_send(chunk, text, generation)
                    else:
                        print(
                            "[DiscordVoiceSTT] empty transcript "
                            f"duration={chunk.duration_sec:.2f}s"
                        )
            except Exception:
                self._record_worker_error()
            finally:
                self._queue.task_done()

    def _dump_debug_audio(self, chunk: SpeechChunk) -> None:
        if self.config.debug_audio_ttl_sec <= 0:
            # invalid retention TTL must fail closed: never write debug audio
            return
        try:
            local_dt = chunk.ended_at.astimezone(ZoneInfo(self.config.timezone))
            stamp = local_dt.strftime("%H%M%S_%f")[:-3]
            name = f"{local_dt.date().isoformat()}_{stamp}_{chunk.user_name}_{chunk.reason}.wav"
            safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
            path = self.config.debug_audio_dir / safe  # type: ignore[operator]
            _sweep_debug_audio(
                self.config.debug_audio_dir,  # type: ignore[arg-type]
                self.config.debug_audio_ttl_sec,
            )
            _write_debug_wav(path, chunk.audio)
            _sweep_debug_audio(
                self.config.debug_audio_dir,  # type: ignore[arg-type]
                self.config.debug_audio_ttl_sec,
            )
            print(
                f"[DiscordVoiceSTT] debug audio saved "
                f"duration={chunk.duration_sec:.2f}s"
            )
        except Exception:  # diagnostics must never break the worker
            print("[DiscordVoiceSTT] debug audio dump failed")

    def _transcribe(self, audio: np.ndarray) -> str:
        with self._stt_lock:
            if self._stt is None:
                self._stt = WhisperSTT(
                    model_size=self.config.model_size,
                    language=self.config.language,
                    device=self.config.device,
                    compute_type=self.config.compute_type,
                    device_index=self.config.device_index,
                )
            return self._stt.transcribe(audio, sample_rate=STT_SAMPLE_RATE)

    def _schedule_transcript_send(self, chunk: SpeechChunk, text: str, generation: int) -> None:
        loop = self._loop
        if loop is None:
            return
        coro = self._send_transcript(chunk, text, generation)
        # Atomic under the consent lock: the generation check, the loop
        # eligibility check, the run_coroutine_threadsafe submission, and the
        # retained-future insertion are one critical section so a concurrent
        # revoke can never interleave (which would leave a
        # retained-but-never-cancelled future or let a revoked generation
        # schedule a send). run_coroutine_threadsafe only schedules on the loop
        # and never blocks on it, so holding the lock here is safe.
        with self._consent_lock:
            if generation != self._generation:
                coro.close()
                return
            # The loop must exist, be open, and be running before submission so
            # a run_coroutine_threadsafe future is never stranded on a dead or
            # inert loop. Test doubles that lack is_closed/is_running are
            # rejected (treated as ineligible) rather than guessed at. Every
            # rejection closes the coroutine so it never dangles un-awaited.
            is_closed = getattr(loop, "is_closed", None)
            is_running = getattr(loop, "is_running", None)
            if (
                is_closed is None
                or is_running is None
                or is_closed()
                or not is_running()
            ):
                coro.close()
                return
            try:
                future = asyncio.run_coroutine_threadsafe(coro, loop)
            except RuntimeError:
                # Loop closed/shut down between the eligibility check and the
                # submission: nothing will ever run the coroutine, so close it
                # to avoid a "never awaited" leak.
                coro.close()
                return
            self._send_futures.add(future)
        future.add_done_callback(self._on_send_future_done)

    def _on_send_future_done(self, future: Any) -> None:
        with self._consent_lock:
            self._send_futures.discard(future)
        self._record_future_error(future)

    def _record_future_error(self, future: Any) -> None:
        if future.cancelled():
            return
        try:
            future.result()
        except Exception:
            self.last_error = VOICE_STT_ERR_SEND
            print("[DiscordVoiceSTT] transcript send failed")

    async def _send_transcript(self, chunk: SpeechChunk, text: str, generation: int) -> None:
        channel = await self._resolve_transcript_channel()
        if channel is None:
            return
        local_dt = chunk.ended_at.astimezone(ZoneInfo(self.config.timezone))
        prefix = f"[{local_dt.strftime('%H:%M:%S')}] {chunk.user_name}: "
        for part in _split_message(prefix + text):
            # Recheck generation immediately before each send: the channel
            # resolve above yielded, so a revoke may have run meanwhile. The
            # check is synchronized with revoke and, for a send already running
            # on the loop, the revoke's future.cancel() is delivered at this
            # same await, so a revoked generation never posts.
            if not self._is_generation_active(generation):
                return
            await channel.send(part)

    async def _send_notice(self, text: str) -> None:
        # Best-effort and bounded: a stuck transcript channel (slow fetch/send)
        # must never wedge start/stop on the event loop. wait_for cancels the
        # inner coroutine on timeout, so nothing dangles.
        try:
            await asyncio.wait_for(self._post_notice(text), timeout=NOTICE_SEND_TIMEOUT)
        except asyncio.TimeoutError:
            print("[DiscordVoiceSTT] notice send timed out")
        except Exception:
            print("[DiscordVoiceSTT] notice send failed")

    async def _post_notice(self, text: str) -> None:
        channel = await self._resolve_transcript_channel()
        if channel is not None:
            await channel.send(text)

    async def _resolve_transcript_channel(self) -> Any | None:
        if self._bot is None or self.transcript_channel_id is None:
            return None
        channel = self._bot.get_channel(self.transcript_channel_id)
        if channel is None:
            channel = await self._bot.fetch_channel(self.transcript_channel_id)
        return channel if hasattr(channel, "send") else None

    def _write_transcript(self, chunk: SpeechChunk, text: str) -> None:
        if not self.config.save_transcripts:
            return
        local_dt = chunk.ended_at.astimezone(ZoneInfo(self.config.timezone))
        path = self.config.transcript_dir / f"{local_dt.date().isoformat()}.jsonl"
        item = {
            "created_at": local_dt.isoformat(timespec="seconds"),
            "started_at_utc": chunk.started_at.isoformat(timespec="seconds"),
            "ended_at_utc": chunk.ended_at.isoformat(timespec="seconds"),
            "guild_id": chunk.guild_id,
            "voice_channel_id": chunk.voice_channel_id,
            "transcript_channel_id": self.transcript_channel_id,
            "user_id": chunk.user_id,
            "user_name": chunk.user_name,
            "text": text,
            "duration_sec": round(chunk.duration_sec, 2),
            "reason": chunk.reason,
        }
        with self._file_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def _after_listening(self, error: Exception | None) -> None:
        if error is not None:
            self.last_error = VOICE_STT_ERR_LISTEN
            print("[DiscordVoiceSTT] listen stopped with error")
