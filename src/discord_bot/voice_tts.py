"""Discord voice-channel TTS playback.

Synthesizes text with KokoroTTS and plays it into the voice channel the bot is
connected to (via DiscordVoiceSTT's voice_client). Playback is serialized so
overlapping /voice say calls and auto-read replies queue instead of clashing.

Send-side DAVE (E2EE) encryption is handled by discord.py itself, so unlike
the receive path no extra patching is needed here.
"""
from __future__ import annotations

import asyncio
import io
import os
import wave
from dataclasses import dataclass
from typing import Any, Callable

import discord
import numpy as np

DISCORD_SAMPLE_RATE = 48000
DISCORD_CHANNELS = 2


class VoiceTTSError(RuntimeError):
    """Raised for user-facing voice TTS failures."""


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None or not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_float(value: str | None, default: float) -> float:
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _parse_int(value: str | None, default: int) -> int:
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class VoiceTTSConfig:
    voice: str = "jf_alpha"
    speed: float = 1.0
    autoread: bool = True
    max_chars: int = 500

    @classmethod
    def from_env(cls) -> "VoiceTTSConfig":
        return cls(
            voice=os.environ.get("DISCORD_VOICE_TTS_VOICE", "jf_alpha").strip() or "jf_alpha",
            speed=_parse_float(os.environ.get("DISCORD_VOICE_TTS_SPEED"), 1.0),
            autoread=_parse_bool(os.environ.get("DISCORD_VOICE_TTS_AUTOREAD"), True),
            max_chars=_parse_int(os.environ.get("DISCORD_VOICE_TTS_MAX_CHARS"), 500),
        )


def wav_to_discord_pcm(wav_data: bytes) -> bytes:
    """Convert a mono WAV (any sample rate) to 48 kHz stereo s16le PCM."""
    with wave.open(io.BytesIO(wav_data)) as wf:
        rate = wf.getframerate()
        channels = wf.getnchannels()
        raw = wf.readframes(wf.getnframes())

    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape((-1, channels)).mean(axis=1)
    if samples.size == 0:
        return b""

    if rate != DISCORD_SAMPLE_RATE:
        # Linear interpolation resample. TTS output is 24 kHz so this is an
        # exact 2x upsample; good enough for speech playback.
        src_t = np.arange(samples.size, dtype=np.float64) / rate
        dst_count = int(round(samples.size * DISCORD_SAMPLE_RATE / rate))
        dst_t = np.arange(dst_count, dtype=np.float64) / DISCORD_SAMPLE_RATE
        samples = np.interp(dst_t, src_t, samples).astype(np.float32)

    pcm16 = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    stereo = np.repeat(pcm16, DISCORD_CHANNELS).tobytes()

    # discord.PCMAudio drops a trailing partial 20 ms frame; pad with silence
    # so the end of the utterance isn't clipped.
    frame_bytes = int(DISCORD_SAMPLE_RATE * 0.02) * DISCORD_CHANNELS * 2
    remainder = len(stereo) % frame_bytes
    if remainder:
        stereo += b"\x00" * (frame_bytes - remainder)
    return stereo


class VoiceTTSPlayer:
    """Serialize TTS synthesis + playback into the connected voice channel."""

    def __init__(
        self,
        *,
        config: VoiceTTSConfig,
        synthesize: Callable[[str, str, float], bytes],
        get_voice_client: Callable[[], Any | None],
    ):
        self.config = config
        self._synthesize = synthesize
        self._get_voice_client = get_voice_client
        self._play_lock = asyncio.Lock()
        self.played_count = 0
        self.last_error: str = ""
        # /voice tts on|off で実行中に切り替えられる。初期値は env 設定。
        self.autoread_enabled = config.autoread

    def _connected_voice_client(self) -> Any:
        voice_client = self._get_voice_client()
        if voice_client is None or not voice_client.is_connected():
            raise VoiceTTSError(
                "ボットが通話チャンネルに接続していません。先に /voice join を実行してください。"
            )
        return voice_client

    async def say(
        self,
        text: str,
        *,
        voice: str | None = None,
        speed: float | None = None,
    ) -> float:
        """Speak text in the connected voice channel. Returns audio seconds."""
        text = (text or "").strip()
        if not text:
            raise VoiceTTSError("読み上げるテキストが空です。")
        if len(text) > self.config.max_chars:
            text = text[: self.config.max_chars] + " 以下省略"

        self._connected_voice_client()

        wav_data = await asyncio.to_thread(
            self._synthesize,
            text,
            voice or self.config.voice,
            speed if speed is not None else self.config.speed,
        )
        pcm = await asyncio.to_thread(wav_to_discord_pcm, wav_data)
        if not pcm:
            raise VoiceTTSError("音声合成結果が空でした。")

        duration = len(pcm) / (DISCORD_SAMPLE_RATE * DISCORD_CHANNELS * 2)

        async with self._play_lock:
            voice_client = self._connected_voice_client()
            if voice_client.is_playing():
                voice_client.stop()

            loop = asyncio.get_running_loop()
            done = asyncio.Event()
            play_error: list[Exception] = []

            def _after(exc: Exception | None) -> None:
                if exc is not None:
                    play_error.append(exc)
                loop.call_soon_threadsafe(done.set)

            source = discord.PCMAudio(io.BytesIO(pcm))
            voice_client.play(source, after=_after)
            try:
                # Playback runs in discord.py's player thread; add margin so a
                # stalled player can't hang the queue forever.
                await asyncio.wait_for(done.wait(), timeout=duration + 10.0)
            except asyncio.TimeoutError:
                voice_client.stop()
                raise VoiceTTSError("再生がタイムアウトしました。")

            if play_error:
                self.last_error = str(play_error[0])
                raise VoiceTTSError(f"再生エラー: {play_error[0]}")

        self.played_count += 1
        return duration

    async def autoread(self, text: str) -> None:
        """Best-effort auto readout for LLM replies; never raises."""
        if not self.autoread_enabled:
            return
        voice_client = self._get_voice_client()
        if voice_client is None or not voice_client.is_connected():
            return
        try:
            await self.say(text)
        except Exception as exc:
            self.last_error = str(exc)
            print(f"[DiscordVoiceTTS] autoread failed: {exc}")
