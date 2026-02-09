"""
音声認識 (STT) モジュール
Phase 3: faster-whisper を使用した音声→テキスト変換
CPU実行 (i7-8700, int8量子化)
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
        model_size: str = "small",
        language: str = "ja",
        device: str = "cpu",
        compute_type: str = "int8",
        beam_size: int = 5,
        vad_filter: bool = True,
    ):
        self.model_size = model_size
        self.language = language
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self._model = None
        self._last_info = None

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
