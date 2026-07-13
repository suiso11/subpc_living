"""TTS backend selection."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.audio.style_bert_vits2_tts import StyleBertVITS2TTS
from src.audio.tts import KokoroTTS


TTSBackend = Any


def normalize_tts_backend(value: str | None) -> str:
    backend = (value or "kokoro").strip().lower().replace("-", "_")
    if backend in {"stylebertvits2", "style_bert", "style_bert_vits2", "sbv2"}:
        return "style_bert_vits2"
    return "kokoro"


def default_voice_for_backend(backend: str) -> str:
    if normalize_tts_backend(backend) == "style_bert_vits2":
        return "jvnv-F1-jp"
    return "jf_alpha"


def all_tts_voices() -> dict[str, str]:
    voices = KokoroTTS.list_ja_voices()
    voices.update(StyleBertVITS2TTS.list_ja_voices())
    return voices


def create_tts_backend(
    *,
    backend: str | None = None,
    env_prefix: str | None = None,
    models_dir: str | Path | None = None,
    voice: str | None = None,
    speed: float = 1.0,
) -> TTSBackend:
    selected = backend
    if selected is None and env_prefix:
        selected = os.environ.get(f"{env_prefix}_TTS_BACKEND")
    if selected is None:
        selected = os.environ.get("TTS_BACKEND")

    selected = normalize_tts_backend(selected)
    if selected == "style_bert_vits2":
        # voice=None は from_env に委ねて SBV2_TTS_VOICE を拾わせる
        if voice is not None and voice not in StyleBertVITS2TTS.JA_VOICES:
            voice = default_voice_for_backend(selected)
        tts = StyleBertVITS2TTS.from_env(voice=voice, speed=speed)
        if tts.voice not in StyleBertVITS2TTS.JA_VOICES:
            tts.voice = default_voice_for_backend(selected)
        return tts

    if voice not in KokoroTTS.JA_VOICES:
        voice = default_voice_for_backend(selected)
    return KokoroTTS(
        models_dir=models_dir or Path("models") / "tts" / "kokoro",
        voice=voice,
        speed=speed,
    )


def backend_name(tts: TTSBackend | None) -> str:
    if tts is None:
        return "-"
    return getattr(tts, "backend_name", "kokoro")
