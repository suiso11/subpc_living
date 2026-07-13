#!/usr/bin/env python3
"""Small HTTP server for Style-Bert-VITS2 synthesis.

Run this with the dedicated Style-Bert-VITS2 virtualenv. The main app keeps its
existing dependency set and talks to this server over localhost HTTP.
"""
from __future__ import annotations

import io
import json
import logging
import os
import sys
import threading
import time
import wave
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import numpy as np
from style_bert_vits2.constants import Languages
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.tts_model import TTSModel


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _setup_logging() -> logging.Logger:
    """stdout (journald) と logs/subpc-sbv2-tts.log の両方に出す。"""
    root = logging.getLogger()
    root.setLevel(getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO))
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-7s [%(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    root.addHandler(stream)
    try:
        log_dir = PROJECT_ROOT / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_dir / "subpc-sbv2-tts.log",
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except OSError:
        pass
    return logging.getLogger("sbv2")


logger = _setup_logging()
DEFAULT_MODEL_ROOT = PROJECT_ROOT / "models" / "tts" / "style_bert_vits2" / "model_assets"
DEFAULT_MODEL_NAME = "jvnv-F1-jp"
DEFAULT_BERT_MODEL = "ku-nlp/deberta-v2-large-japanese-char-wwm"
DEFAULT_MODEL_FILE = "jvnv-F1-jp_e160_s14000.safetensors"
DEFAULT_CONFIG_FILE = "config.json"
DEFAULT_STYLE_VEC_FILE = "style_vectors.npy"


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _audio_to_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    samples = np.asarray(audio).reshape(-1)
    if samples.dtype.kind == "f":
        pcm16 = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    else:
        pcm16 = np.clip(samples, -32768, 32767).astype("<i2")

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return buffer.getvalue()


class StyleBertVITS2Engine:
    def __init__(self) -> None:
        self.model_name = os.environ.get("SBV2_MODEL_NAME", DEFAULT_MODEL_NAME).strip() or DEFAULT_MODEL_NAME
        self.model_root = Path(os.environ.get("SBV2_MODEL_ROOT", str(DEFAULT_MODEL_ROOT))).expanduser()
        self.bert_model = os.environ.get("SBV2_BERT_MODEL", DEFAULT_BERT_MODEL).strip() or DEFAULT_BERT_MODEL
        self.device = os.environ.get("SBV2_DEVICE", "cuda:1").strip() or "cuda:1"
        self.default_style = os.environ.get("SBV2_STYLE", "Neutral").strip() or "Neutral"
        self.default_style_weight = _env_float("SBV2_STYLE_WEIGHT", 1.0)
        self.default_split_interval = _env_float("SBV2_SPLIT_INTERVAL", 0.3)
        self.max_chars = _env_int("SBV2_MAX_CHARS", 500)
        self.model_file = os.environ.get("SBV2_MODEL_FILE", DEFAULT_MODEL_FILE).strip() or DEFAULT_MODEL_FILE
        self.config_file = os.environ.get("SBV2_CONFIG_FILE", DEFAULT_CONFIG_FILE).strip() or DEFAULT_CONFIG_FILE
        self.style_vec_file = os.environ.get("SBV2_STYLE_VEC_FILE", DEFAULT_STYLE_VEC_FILE).strip() or DEFAULT_STYLE_VEC_FILE
        self._model: TTSModel | None = None
        self._lock = threading.Lock()

    @property
    def voices(self) -> dict[str, str]:
        return {self.model_name: f"Style-Bert-VITS2 {self.model_name}"}

    def load(self) -> None:
        if self._model is not None:
            return

        logger.info(
            "loading model=%s file=%s bert=%s device=%s",
            self.model_name, self.model_file, self.bert_model, self.device,
        )
        start = time.time()
        bert_models.load_model(Languages.JP, self.bert_model)
        bert_models.load_tokenizer(Languages.JP, self.bert_model)
        model_dir = self.model_root / self.model_name
        self._model = TTSModel(
            model_path=model_dir / self.model_file,
            config_path=model_dir / self.config_file,
            style_vec_path=model_dir / self.style_vec_file,
            device=self.device,
        )
        logger.info("loaded (%.2fs)", time.time() - start)

    def synthesize(self, payload: dict[str, Any]) -> tuple[bytes, dict[str, str]]:
        self.load()
        assert self._model is not None

        text = str(payload.get("text", "")).strip()
        if not text:
            raise ValueError("text is required")
        if len(text) > self.max_chars:
            text = text[: self.max_chars]

        voice = str(payload.get("voice") or self.model_name)
        if voice != self.model_name:
            raise ValueError(f"unknown voice: {voice}")

        raw_speed = payload.get("speed", 1.0)
        try:
            speed = float(raw_speed)
        except (TypeError, ValueError):
            speed = 1.0
        speed = max(0.5, min(speed, 1.5))
        length = 1.0 / speed

        style = str(payload.get("style") or self.default_style)
        style_weight = float(payload.get("style_weight") or self.default_style_weight)
        split_interval = float(payload.get("split_interval") or self.default_split_interval)

        start = time.time()
        with self._lock:
            sample_rate, audio = self._model.infer(
                text=text,
                language=Languages.JP,
                length=length,
                line_split=True,
                split_interval=split_interval,
                style=style,
                style_weight=style_weight,
            )
        elapsed = time.time() - start
        duration = len(audio) / sample_rate
        logger.info(
            "synthesized (%.2fs, audio=%.1fs, voice=%s, speed=%.2f, style=%s) %s",
            elapsed, duration, voice, speed, style, text[:30],
        )

        return _audio_to_wav(audio, sample_rate), {
            "X-SBV2-Elapsed": f"{elapsed:.3f}",
            "X-SBV2-Duration": f"{duration:.3f}",
            "X-SBV2-Sample-Rate": str(sample_rate),
        }

    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "loaded": self._model is not None,
            "model": self.model_name,
            "model_file": self.model_file,
            "device": self.device,
            "voices": self.voices,
        }


ENGINE = StyleBertVITS2Engine()


class Handler(BaseHTTPRequestHandler):
    server_version = "SubpcStyleBertVITS2/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        logger.info("%s %s", self.address_string(), fmt % args)

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(HTTPStatus.OK, ENGINE.health())
            return
        if self.path == "/voices":
            self._send_json(HTTPStatus.OK, {"voices": ENGINE.voices})
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/synthesize":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
            wav_data, headers = ENGINE.synthesize(payload)
        except ValueError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        except Exception as exc:
            logger.error("synthesize error: %s", exc)
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
            return

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(wav_data)))
        for key, value in headers.items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(wav_data)


def main() -> None:
    host = os.environ.get("SBV2_HOST", "127.0.0.1")
    port = _env_int("SBV2_PORT", 50121)
    ENGINE.load()

    warmup_text = os.environ.get("SBV2_WARMUP_TEXT", "こんにちは。").strip()
    if warmup_text:
        try:
            ENGINE.synthesize({"text": warmup_text})
        except Exception as exc:
            logger.warning("warmup failed: %s", exc)

    server = ThreadingHTTPServer((host, port), Handler)
    logger.info("listening on http://%s:%s", host, port)
    server.serve_forever()


if __name__ == "__main__":
    main()
