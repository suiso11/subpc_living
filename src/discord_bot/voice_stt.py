"""Discord voice-channel speech-to-text bridge."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
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
        except OpusError as exc:
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
                    f"count={_OPUS_DECODE_ERRORS} ssrc={getattr(self, 'ssrc', '?')}: {exc}"
                )
            return packet, concealed

    safe_decode._subpc_patched = True  # type: ignore[attr-defined]
    PacketDecoder._decode_packet = safe_decode  # type: ignore[assignment]


_patch_voice_recv_opus_decoder()


DISCORD_PCM_SAMPLE_RATE = 48000
DISCORD_PCM_CHANNELS = 2
STT_SAMPLE_RATE = 16000
MAX_DISCORD_MESSAGE = 1900


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


def _resolve_debug_audio_dir(project_root: Path) -> Path | None:
    """Optional dir for dumping the exact 16 kHz mono audio sent to Whisper.

    Set DISCORD_VOICE_STT_DEBUG_AUDIO_DIR to a path to enable. Diagnostic only:
    lets us inspect/listen to what STT actually receives when transcripts look
    like noise. Unset (default) means no dump.
    """
    raw = os.environ.get("DISCORD_VOICE_STT_DEBUG_AUDIO_DIR", "").strip()
    if not raw:
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
    save_transcripts: bool = True
    hallucination_filter: bool = True
    debug_audio_dir: Path | None = None

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
            enabled=_parse_bool(os.environ.get("DISCORD_VOICE_STT_ENABLED"), default=False),
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
            save_transcripts=_parse_bool(
                os.environ.get("DISCORD_VOICE_STT_SAVE_TRANSCRIPTS"),
                default=True,
            ),
            hallucination_filter=_parse_bool(
                os.environ.get("DISCORD_VOICE_STT_HALLUCINATION_FILTER"),
                default=True,
            ),
            debug_audio_dir=_resolve_debug_audio_dir(root),
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
        if self._closed or user is None or getattr(user, "bot", False):
            return
        pcm = getattr(data, "pcm", b"")
        if not pcm:
            return

        now = datetime.now(timezone.utc)
        audio = pcm48_stereo_to_16k_mono(pcm)
        with self._lock:
            self.received_packets += 1
            self.received_audio_seconds += audio.size / STT_SAMPLE_RATE
            if self.received_packets == 1 or self.received_packets % 250 == 0:
                print(
                    "[DiscordVoiceSTT] receiving audio "
                    f"packets={self.received_packets} "
                    f"audio_sec={self.received_audio_seconds:.1f} "
                    f"user={getattr(user, 'display_name', None) or user}"
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

    def cleanup(self) -> None:
        self.flush_all()
        self._closed = True

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
        try:
            self.output_queue.put_nowait(chunk)
            print(
                "[DiscordVoiceSTT] speech segment queued "
                f"user={chunk.user_name} duration={chunk.duration_sec:.2f}s "
                f"reason={chunk.reason} queue={self.output_queue.qsize()}"
            )
        except queue.Full:
            self.dropped_segments += 1
            print(
                "[DiscordVoiceSTT] speech segment dropped "
                f"user={chunk.user_name} dropped={self.dropped_segments}"
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
        self._stt: WhisperSTT | None = None
        self._stt_lock = threading.Lock()
        self._file_lock = threading.Lock()

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
        self._ensure_worker()

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
        if self.voice_client is not None and hasattr(self.voice_client, "is_listening"):
            if self.voice_client.is_listening():
                self.voice_client.stop_listening()
        if self.sink is not None:
            self.sink.flush_all()
            print(
                "[DiscordVoiceSTT] listening stopped "
                f"packets={self.sink.received_packets} "
                f"audio_sec={self.sink.received_audio_seconds:.1f} "
                f"transcripts={self.transcript_count}"
            )
        self.sink = None
        self.started_at = None
        self._stop_event.set()
        await self._send_notice("[voice] STT stopped.")
        return "voice STT を停止しました。"

    async def leave(self) -> str:
        await self.stop()
        if self.voice_client is not None and self.voice_client.is_connected():
            await self.voice_client.disconnect(force=True)
        self.voice_client = None
        return "voice channel から退出しました。"

    def close(self) -> None:
        if self.sink is not None:
            self.sink.flush_all()
        self._stop_event.set()

    def status_text(self) -> str:
        channel_name = "-"
        if self.voice_client is not None and getattr(self.voice_client, "channel", None) is not None:
            channel_name = getattr(self.voice_client.channel, "name", "-")
        dropped = self.sink.dropped_segments if self.sink is not None else 0
        packets = self.sink.received_packets if self.sink is not None else 0
        audio_seconds = self.sink.received_audio_seconds if self.sink is not None else 0.0
        decode_errors = opus_decode_error_count()
        return (
            f"voice_stt_enabled: {self.config.enabled}\n"
            f"voice_recv_available: {self.available}\n"
            f"voice_connected: {self.connected}\n"
            f"voice_listening: {self.listening}\n"
            f"voice_channel: {channel_name}\n"
            f"voice_transcript_channel_id: {self.transcript_channel_id or '-'}\n"
            f"voice_queue_size: {self._queue.qsize()}\n"
            f"voice_received_packets: {packets}\n"
            f"voice_received_audio_sec: {audio_seconds:.1f}\n"
            f"voice_transcripts: {self.transcript_count}\n"
            f"voice_decode_errors: {decode_errors}\n"
            f"voice_dropped_segments: {dropped}\n"
            f"voice_last_error: {self.last_error or '-'}"
        )

    def _validate_enabled(self) -> None:
        if not self.config.enabled:
            raise VoiceSTTError("voice STT は無効です。DISCORD_VOICE_STT_ENABLED=true を設定してください。")
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

    def _ensure_worker(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            self._stop_event.clear()
            return
        self._stop_event.clear()
        self._worker = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="discord-voice-stt-worker",
        )
        self._worker.start()

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                chunk = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                if self.config.debug_audio_dir is not None:
                    self._dump_debug_audio(chunk)
                text = self._transcribe(chunk.audio)
                if text and self.config.hallucination_filter and _is_likely_hallucination(text):
                    print(
                        "[DiscordVoiceSTT] filtered hallucination "
                        f"user={chunk.user_name} duration={chunk.duration_sec:.2f}s "
                        f"text={text!r}"
                    )
                    text = ""
                if text:
                    self.transcript_count += 1
                    self.last_transcript_at = datetime.now(timezone.utc)
                    self._write_transcript(chunk, text)
                    self._schedule_transcript_send(chunk, text)
                else:
                    print(
                        "[DiscordVoiceSTT] empty transcript "
                        f"user={chunk.user_name} duration={chunk.duration_sec:.2f}s"
                    )
            except Exception as exc:
                self.last_error = str(exc)
                print(f"[DiscordVoiceSTT] worker error: {exc}")
            finally:
                self._queue.task_done()

    def _dump_debug_audio(self, chunk: SpeechChunk) -> None:
        try:
            local_dt = chunk.ended_at.astimezone(ZoneInfo(self.config.timezone))
            stamp = local_dt.strftime("%H%M%S_%f")[:-3]
            name = f"{local_dt.date().isoformat()}_{stamp}_{chunk.user_name}_{chunk.reason}.wav"
            safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
            path = self.config.debug_audio_dir / safe  # type: ignore[operator]
            _write_debug_wav(path, chunk.audio)
            print(
                f"[DiscordVoiceSTT] debug audio saved {path} "
                f"duration={chunk.duration_sec:.2f}s"
            )
        except Exception as exc:  # diagnostics must never break the worker
            print(f"[DiscordVoiceSTT] debug audio dump failed: {exc}")

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

    def _schedule_transcript_send(self, chunk: SpeechChunk, text: str) -> None:
        if self._loop is None:
            return
        future = asyncio.run_coroutine_threadsafe(self._send_transcript(chunk, text), self._loop)
        future.add_done_callback(self._record_future_error)

    def _record_future_error(self, future: Any) -> None:
        try:
            future.result()
        except Exception as exc:
            self.last_error = str(exc)
            print(f"[DiscordVoiceSTT] send error: {exc}")

    async def _send_transcript(self, chunk: SpeechChunk, text: str) -> None:
        channel = await self._resolve_transcript_channel()
        if channel is None:
            return
        local_dt = chunk.ended_at.astimezone(ZoneInfo(self.config.timezone))
        prefix = f"[{local_dt.strftime('%H:%M:%S')}] {chunk.user_name}: "
        for part in _split_message(prefix + text):
            await channel.send(part)

    async def _send_notice(self, text: str) -> None:
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
            self.last_error = str(error)
            print(f"[DiscordVoiceSTT] listen stopped with error: {error}")
