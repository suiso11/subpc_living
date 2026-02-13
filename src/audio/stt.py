"""
音声認識 (STT) モジュール
Phase 3: faster-whisper を使用した音声→テキスト変換
Phase 9: device="auto" でGPU自動検出 (P40: cuda/float16, GTX 1060: cpu/int8)
"""
import io
import time
import numpy as np
from pathlib import Path
from typing import Optional


class WhisperSTT:
    """faster-whisper ベースの音声認識クラス"""

    def __init__(
        self,
        model_size: str = "auto",
        language: str = "ja",
        device: str = "auto",
        compute_type: str = "auto",
        beam_size: int = 5,
        vad_filter: bool = True,
    ):
        # device="auto" の場合、gpu_config から最適設定を取得
        resolved_device, resolved_compute, resolved_model = self._resolve_config(
            device, compute_type, model_size,
        )
        self.model_size = resolved_model
        self.language = language
        self.device = resolved_device
        self.compute_type = resolved_compute
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self._model = None
        self._last_info = None

    @staticmethod
    def _resolve_config(device: str, compute_type: str, model_size: str) -> tuple[str, str, str]:
        """device/compute_type/model_size の auto を解決する"""
        if device != "auto" and compute_type != "auto" and model_size != "auto":
            return device, compute_type, model_size
        try:
            from src.service.gpu_config import resolve_stt_config
            return resolve_stt_config(device, compute_type, model_size)
        except ImportError:
            # gpu_config が無い場合のフォールバック
            return (
                "cpu" if device == "auto" else device,
                "int8" if compute_type == "auto" else compute_type,
                "small" if model_size == "auto" else model_size,
            )

    def load(self) -> None:
        """モデルをロード（初回は自動ダウンロード）"""
        if self._model is not None:
            return
        from faster_whisper import WhisperModel

        print(f"[STT] モデル '{self.model_size}' をロード中 (device={self.device}, compute_type={self.compute_type})...")
        start = time.time()
        self._model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )
        elapsed = time.time() - start
        print(f"[STT] モデルロード完了 ({elapsed:.1f}秒)")

    def transcribe(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """
        音声データをテキストに変換

        Args:
            audio_data: float32 の音声データ (モノラル, 16kHz推奨)
            sample_rate: サンプルレート

        Returns:
            認識されたテキスト
        """
        self.load()

        # float32に正規化
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        # -1.0 ~ 1.0 に正規化
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val

        start = time.time()
        segments, info = self._model.transcribe(
            audio_data,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
        )
        self._last_info = info

        # セグメントをテキストに結合
        text = ""
        for segment in segments:
            text += segment.text

        elapsed = time.time() - start
        text = text.strip()
        if text:
            print(f"[STT] 認識完了 ({elapsed:.1f}秒): {text[:50]}{'...' if len(text) > 50 else ''}")

        return text

    def transcribe_file(self, filepath: str | Path) -> str:
        """音声ファイルからテキストに変換"""
        self.load()

        segments, info = self._model.transcribe(
            str(filepath),
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
        )
        self._last_info = info
        return "".join(s.text for s in segments).strip()

    @property
    def last_info(self):
        """最後の認識の情報（言語検出結果等）"""
        return self._last_info

    def is_loaded(self) -> bool:
        return self._model is not None
