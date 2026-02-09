"""
音声合成 (TTS) モジュール
Phase 3: kokoro-onnx を使用したテキスト→音声変換
CPU実行、日本語対応（Kokoro 82Mモデル）
"""
import wave
import io
import time
import numpy as np
from pathlib import Path
from typing import Optional


class KokoroTTS:
    """kokoro-onnx ベースの音声合成クラス"""

    # 利用可能な日本語ボイス
    JA_VOICES = {
        "jf_alpha": "日本語 女性 (Alpha)",
        "jf_gongitsune": "日本語 女性 (Gongitsune)",
        "jf_nezumi": "日本語 女性 (Nezumi)",
        "jf_tebukuro": "日本語 女性 (Tebukuro)",
        "jm_kumo": "日本語 男性 (Kumo)",
    }

    # HuggingFaceリポジトリ
    HF_REPO = "fastrtc/kokoro-onnx"
    MODEL_FILE = "kokoro-v1.0.onnx"
    VOICES_FILE = "voices-v1.0.bin"

    def __init__(
        self,
        models_dir: str | Path = "models/tts/kokoro",
        voice: str = "jf_alpha",
        speed: float = 1.0,
        lang: str = "ja",
    ):
        self.models_dir = Path(models_dir)
        self.voice = voice
        self.speed = speed
        self.lang = lang
        self.sample_rate = 24000  # kokoro出力は24kHz

        self._model_path = self.models_dir / self.MODEL_FILE
        self._voices_path = self.models_dir / self.VOICES_FILE
        self._kokoro = None

    def is_installed(self) -> bool:
        """モデルファイルがダウンロード済みか確認"""
        return self._model_path.exists() and self._voices_path.exists()

    def install(self) -> None:
        """モデルファイルをHuggingFaceからダウンロード"""
        from huggingface_hub import hf_hub_download

        self.models_dir.mkdir(parents=True, exist_ok=True)

        print("[TTS] kokoro-onnx モデルをダウンロード中...")
        hf_hub_download(
            self.HF_REPO, self.MODEL_FILE,
            local_dir=str(self.models_dir),
        )
        hf_hub_download(
            self.HF_REPO, self.VOICES_FILE,
            local_dir=str(self.models_dir),
        )
        print("[TTS] ✅ モデルダウンロード完了")

    def load(self) -> None:
        """モデルをロード"""
        if self._kokoro is not None:
            return

        if not self.is_installed():
            print("[TTS] モデルが見つかりません。ダウンロードします...")
            self.install()

        from kokoro_onnx import Kokoro

        print("[TTS] kokoro-onnx モデルをロード中...")
        start = time.time()
        self._kokoro = Kokoro(str(self._model_path), str(self._voices_path))
        elapsed = time.time() - start
        print(f"[TTS] モデルロード完了 ({elapsed:.1f}秒)")

    def synthesize(self, text: str) -> bytes:
        """
        テキストを音声に変換し、WAVバイトデータを返す

        Args:
            text: 合成するテキスト

        Returns:
            WAV形式のバイトデータ
        """
        self.load()

        start = time.time()
        samples, sr = self._kokoro.create(
            text,
            voice=self.voice,
            speed=self.speed,
            lang=self.lang,
        )
        elapsed = time.time() - start
        self.sample_rate = sr

        # float32 → int16 → WAV
        audio_int16 = (samples * 32767).astype(np.int16)
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(audio_int16.tobytes())

        wav_data = wav_buffer.getvalue()
        duration = len(samples) / sr
        print(f"[TTS] 合成完了 ({elapsed:.2f}秒, 音声{duration:.1f}秒): {text[:30]}{'...' if len(text) > 30 else ''}")

        return wav_data

    def synthesize_to_file(self, text: str, filepath: str | Path) -> Path:
        """テキストを音声に変換し、WAVファイルに保存"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        wav_data = self.synthesize(text)
        with open(filepath, "wb") as f:
            f.write(wav_data)
        return filepath

    def synthesize_to_numpy(self, text: str) -> tuple[np.ndarray, int]:
        """
        テキストを音声に変換し、numpy配列で返す

        Returns:
            (audio_data as float32, sample_rate)
        """
        self.load()
        samples, sr = self._kokoro.create(
            text, voice=self.voice, speed=self.speed, lang=self.lang,
        )
        return samples, sr

    def set_voice(self, voice: str) -> None:
        """ボイスを変更"""
        self.voice = voice

    @classmethod
    def list_ja_voices(cls) -> dict[str, str]:
        """利用可能な日本語ボイス一覧"""
        return cls.JA_VOICES.copy()

    def is_loaded(self) -> bool:
        return self._kokoro is not None
