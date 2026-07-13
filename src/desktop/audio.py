"""Native microphone capture used by the QML push-to-talk button."""
from __future__ import annotations

import io
import threading
import wave
from typing import Any


class NativeAudioRecorder:
    def __init__(self, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate
        self._capture_rate = sample_rate
        self._stream: Any = None
        self._chunks: list[Any] = []
        self._lock = threading.Lock()

    @property
    def recording(self) -> bool:
        return self._stream is not None

    def start(self) -> None:
        if self.recording:
            return
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError("sounddevice がインストールされていません") from exc
        self._chunks = []

        def callback(indata, frames, time_info, status) -> None:  # noqa: ANN001
            del frames, time_info, status
            with self._lock:
                self._chunks.append(indata.copy())

        def open_stream(rate: int):
            stream = sd.InputStream(
                samplerate=rate,
                channels=1,
                dtype="int16",
                callback=callback,
            )
            try:
                stream.start()
            except Exception:
                stream.close()
                raise
            return stream

        self._capture_rate = self.sample_rate
        try:
            stream = open_stream(self.sample_rate)
        except Exception as requested_error:
            try:
                device = sd.query_devices(kind="input")
                fallback_rate = int(round(float(device["default_samplerate"])))
                if fallback_rate <= 0 or fallback_rate == self.sample_rate:
                    raise requested_error
                self._chunks = []
                stream = open_stream(fallback_rate)
                self._capture_rate = fallback_rate
            except Exception:
                raise requested_error
        self._stream = stream

    def stop(self) -> bytes:
        stream = self._stream
        if stream is None:
            return b""
        self._stream = None
        stream.stop()
        stream.close()
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("numpy がインストールされていません") from exc
        with self._lock:
            chunks = self._chunks
            self._chunks = []
        if not chunks:
            return b""
        samples = np.concatenate(chunks, axis=0).reshape(-1).astype(np.float64)
        if self._capture_rate != self.sample_rate and samples.size > 1:
            output_size = max(1, int(round(samples.size * self.sample_rate / self._capture_rate)))
            source_positions = np.arange(samples.size, dtype=np.float64)
            target_positions = np.linspace(0, samples.size - 1, output_size)
            samples = np.interp(target_positions, source_positions, samples)
        pcm = np.clip(np.rint(samples), -32768, 32767).astype("<i2").tobytes()
        output = io.BytesIO()
        with wave.open(output, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.sample_rate)
            wav.writeframes(pcm)
        return output.getvalue()
