"""
音声合成 (TTS) モジュール
Phase 3: kokoro-onnx を使用したテキスト→音声変換
ONNX Runtime 実行、日本語対応（Kokoro 82Mモデル）
"""
import wave
import io
import os
import re
import time
import numpy as np
from pathlib import Path
from typing import Optional

# kokoro-onnx の音素上限 (510) を超えないよう、テキストを文単位で分割する。
# 日本語1文字 ≒ 3〜8音素に展開されるため、安全マージンを取って50文字以下に。
_MAX_CHUNK_CHARS = 50
_CHUNK_PAUSE_SEC = 0.18
_SENTENCE_SPLIT = re.compile(r'(?<=[。！？\n.!?])')

# 絵文字・特殊文字のパターン
_EMOJI_PATTERN = re.compile(
    r'[\U0001F600-\U0001F64F'  # Emoticons
    r'\U0001F300-\U0001F5FF'   # Misc Symbols and Pictographs
    r'\U0001F680-\U0001F6FF'   # Transport and Map
    r'\U0001F1E0-\U0001F1FF'   # Flags
    r'\U00002702-\U000027B0'   # Dingbats
    r'\U0001F900-\U0001F9FF'   # Supplemental Symbols and Pictographs
    r'\U0001FA00-\U0001FA6F'   # Chess Symbols
    r'\U0001FA70-\U0001FAFF'   # Symbols and Pictographs Extended-A
    r'\U00002600-\U000026FF'   # Misc symbols
    r'\U0000FE00-\U0000FE0F'   # Variation Selectors
    r'\U0000200D'              # Zero Width Joiner
    r'\U000020E3'              # Combining Enclosing Keycap
    r']+'
)

# 一般的な絵文字（頻出するもの）
_COMMON_EMOJI = re.compile(r'[✅❌⚠️🔊🎤🔄👤🤖💬✨🎙️ℹ️🔥💡🎯🚀⭐🌟💫🎵🎶🎨📚💻🔧⚙️🛠️📱📞☎️📧✉️🔔🔕🔈🔉🔊🔇🎵🎶]+')


def _clean_text(text: str) -> str:
    """TTS用にテキストをクリーニング（絵文字・特殊文字除去）"""
    # 絵文字を除去
    text = _EMOJI_PATTERN.sub('', text)
    text = _COMMON_EMOJI.sub('', text)
    # 特殊記号を除去
    text = re.sub(r'[★☆◆◇●○▲△▼▽■□□■♦♢♣♠♥]', '', text)
    # 空白を正規化
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _split_text(text: str, max_chars: int = _MAX_CHUNK_CHARS) -> list[str]:
    """テキストを文単位で分割し、音素上限を超えないチャンクにまとめる"""
    # テキストをクリーニング（絵文字除去）
    text = _clean_text(text)

    if not text:
        return []

    sentences = _split_sentences_for_tts(text)

    if not sentences:
        return [text] if text.strip() else []

    # 短い文は max_chars まで結合して、kokoro-onnx の呼び出し回数を抑える。
    chunks = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) > max_chars and current:
            chunks.append(current)
            current = sent
        else:
            current += sent

    if current:
        chunks.append(current)

    # 単一チャンクが長すぎる場合はさらに強制分割
    result = []
    for chunk in chunks:
        while len(chunk) > max_chars:
            result.append(chunk[:max_chars])
            chunk = chunk[max_chars:]
        if chunk:
            result.append(chunk)

    return result


def _split_sentences_for_tts(text: str) -> list[str]:
    """TTS向けに文末で分割する。読点では抑揚を切らない。"""
    sentences = _SENTENCE_SPLIT.split(text)
    return [s.strip() for s in sentences if s.strip()]


def _pause_samples(sample_rate: int, duration_sec: float = _CHUNK_PAUSE_SEC) -> np.ndarray:
    """チャンク間に入れる短い無音を生成する。"""
    return np.zeros(int(sample_rate * duration_sec), dtype=np.float32)


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
        self._ja_g2p = None

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
        providers = self._resolve_onnx_providers()
        if providers is None:
            self._kokoro = Kokoro(str(self._model_path), str(self._voices_path))
        else:
            import onnxruntime as ort
            if any(
                (p[0] if isinstance(p, tuple) else p) == "CUDAExecutionProvider"
                for p in providers
            ) and hasattr(ort, "preload_dlls"):
                try:
                    ort.preload_dlls(directory="")
                except Exception as e:
                    print(f"[TTS] CUDAライブラリのプリロードに失敗: {e}")
            session = ort.InferenceSession(str(self._model_path), providers=providers)
            self._kokoro = Kokoro.from_session(session, str(self._voices_path))
        elapsed = time.time() - start
        provider_text = ", ".join(self._kokoro.sess.get_providers())
        print(f"[TTS] モデルロード完了 ({elapsed:.1f}秒, providers={provider_text})")

        if self.lang == "ja":
            self._load_ja_g2p()

    @staticmethod
    def _resolve_onnx_providers() -> list | None:
        """Return explicit ONNX providers for TTS, or None for kokoro defaults."""
        provider = (
            os.environ.get("TTS_ONNX_PROVIDER", "").strip()
            or os.environ.get("ONNX_PROVIDER", "").strip()
        )
        if not provider:
            return None

        if provider != "CUDAExecutionProvider":
            return [provider]

        raw_device_id = os.environ.get("TTS_ONNX_DEVICE_ID", "").strip()
        try:
            device_id = int(raw_device_id) if raw_device_id else 0
        except ValueError:
            device_id = 0
        return [
            (
                "CUDAExecutionProvider",
                {"device_id": device_id},
            ),
            "CPUExecutionProvider",
        ]

    def _load_ja_g2p(self) -> None:
        """日本語G2P (misaki) をロード。

        kokoro-onnx 内蔵の espeak-ng は漢字を読めない（「今日」が
        "Chinese letter" と読まれる）ため、misaki + unidic で
        音素化してから is_phonemes=True で合成する。
        """
        if self._ja_g2p is not None:
            return
        try:
            from misaki import ja
            self._ja_g2p = ja.JAG2P()
            print("[TTS] 日本語G2P (misaki) ロード完了")
        except Exception as e:
            print(f"[TTS] ⚠️ 日本語G2Pロード失敗 (espeak-ngにフォールバック、漢字は読めません): {e}")
            print("[TTS]    修復方法: .venv/bin/python -m unidic download")

    def _create_chunk(self, chunk: str) -> tuple[np.ndarray, int]:
        """1チャンクを合成。日本語はmisakiで音素化してから渡す。"""
        if self.lang == "ja" and self._ja_g2p is not None:
            phonemes, _ = self._ja_g2p(chunk)
            return self._kokoro.create(
                phonemes,
                voice=self.voice,
                speed=self.speed,
                is_phonemes=True,
            )
        return self._kokoro.create(
            chunk,
            voice=self.voice,
            speed=self.speed,
            lang=self.lang,
        )

    def synthesize(
        self,
        text: str,
        style: str | None = None,
        style_weight: float | None = None,
    ) -> bytes:
        """
        テキストを音声に変換し、WAVバイトデータを返す。

        kokoro-onnx の音素上限 (510) を超えないよう、
        テキストを文単位でチャンク分割して合成・結合する。

        Args:
            text: 合成するテキスト
            style: 感情スタイル (kokoro は非対応のため無視。呼び出し側の分岐削減用)
            style_weight: スタイル強度 (kokoro は非対応のため無視)

        Returns:
            WAV形式のバイトデータ
        """
        self.load()

        start = time.time()
        chunks = _split_text(text)

        if not chunks:
            # 空テキスト → 無音WAV
            return self._empty_wav()

        all_samples = []
        sr = self.sample_rate

        for index, chunk in enumerate(chunks):
            try:
                samples, sr = self._create_chunk(chunk)
                all_samples.append(samples)
                if index < len(chunks) - 1:
                    all_samples.append(_pause_samples(sr))
            except (IndexError, Exception) as e:
                # 音素変換エラー時はスキップして続行
                print(f"[TTS] チャンク合成スキップ: {e} ({chunk[:20]}...)")
                continue

        if not all_samples:
            print("[TTS] 全チャンク失敗")
            return self._empty_wav()

        combined = np.concatenate(all_samples)
        self.sample_rate = sr
        elapsed = time.time() - start

        # float32 → int16 → WAV
        audio_int16 = (combined * 32767).astype(np.int16)
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(audio_int16.tobytes())

        wav_data = wav_buffer.getvalue()
        duration = len(combined) / sr
        print(f"[TTS] 合成完了 ({elapsed:.2f}秒, 音声{duration:.1f}秒, {len(chunks)}チャンク): {text[:30]}{'...' if len(text) > 30 else ''}")

        return wav_data

    def _empty_wav(self) -> bytes:
        """無音のWAVデータを返す"""
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(b"\x00\x00" * self.sample_rate)  # 1秒無音
        return wav_buffer.getvalue()

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
        samples, sr = self._create_chunk(text)
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
