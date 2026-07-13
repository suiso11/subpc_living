"""HTTP client for the dedicated Style-Bert-VITS2 TTS server."""
from __future__ import annotations

import os
from dataclasses import dataclass

import httpx


@dataclass
class StyleBertVITS2TTS:
    base_url: str = "http://127.0.0.1:50121"
    voice: str = "jvnv-F1-jp"
    speed: float = 1.0
    style: str = "Neutral"
    style_weight: float = 1.0
    timeout: float = 120.0

    JA_VOICES = {
        "jvnv-F1-jp": "Style-Bert-VITS2 JVNV F1 JP",
        "tsukuyomi-chan": "Style-Bert-VITS2 Tsukuyomi-chan",
    }
    backend_name = "style_bert_vits2"

    @classmethod
    def from_env(cls, *, voice: str | None = None, speed: float = 1.0) -> "StyleBertVITS2TTS":
        return cls(
            base_url=os.environ.get("SBV2_TTS_URL", "http://127.0.0.1:50121").rstrip("/"),
            voice=voice or os.environ.get("SBV2_TTS_VOICE", "jvnv-F1-jp"),
            speed=speed,
            style=os.environ.get("SBV2_TTS_STYLE", "Neutral"),
            style_weight=_parse_float(os.environ.get("SBV2_TTS_STYLE_WEIGHT"), 1.0),
            timeout=_parse_float(os.environ.get("SBV2_TTS_TIMEOUT"), 120.0),
        )

    def is_installed(self) -> bool:
        try:
            self._health()
            return True
        except Exception:
            return False

    def load(self) -> None:
        self._health()

    def is_loaded(self) -> bool:
        try:
            return bool(self._health().get("ok"))
        except Exception:
            return False

    def set_voice(self, voice: str) -> None:
        if voice not in self.JA_VOICES:
            raise ValueError(f"Unknown Style-Bert-VITS2 voice: {voice}")
        self.voice = voice

    @classmethod
    def list_ja_voices(cls) -> dict[str, str]:
        return cls.JA_VOICES.copy()

    def synthesize(
        self,
        text: str,
        style: str | None = None,
        style_weight: float | None = None,
    ) -> bytes:
        # None のときは起動時の既定スタイルを使う。
        use_style = style if style is not None else self.style
        use_weight = style_weight if style_weight is not None else self.style_weight
        # tsukuyomi-chan は Neutral スタイルのみ。常に Neutral を送る。
        if self.voice == "tsukuyomi-chan":
            use_style = "Neutral"
        payload = {
            "text": text,
            "voice": self.voice,
            "speed": self.speed,
            "style": use_style,
            "style_weight": use_weight,
        }
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(f"{self.base_url}/synthesize", json=payload)
            response.raise_for_status()
            return response.content

    def _health(self) -> dict:
        with httpx.Client(timeout=min(self.timeout, 10.0)) as client:
            response = client.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()


def _parse_float(value: str | None, default: float) -> float:
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError:
        return default
