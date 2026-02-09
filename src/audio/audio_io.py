"""
オーディオ入出力モジュール
Phase 3: マイク入力・スピーカー出力を担当
sounddevice を使用
"""
import wave
import io
import time
import numpy as np
from typing import Optional, Callable
import sounddevice as sd


class AudioRecorder:
    """マイクからの音声録音"""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        dtype: str = "float32",
        device: Optional[int] = None,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self.device = device

    def record(self, duration: float) -> np.ndarray:
        """
        指定秒数だけ録音する

        Args:
            duration: 録音時間 (秒)

        Returns:
            録音データ (float32, モノラル)
        """
        frames = int(duration * self.sample_rate)
        print(f"[Audio] 録音開始 ({duration}秒)...")
        audio = sd.rec(
            frames,
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=self.dtype,
            device=self.device,
        )
        sd.wait()
        print("[Audio] 録音完了")
        return audio.flatten()

    def open_stream(
        self,
        callback: Callable,
        frame_size: int = 480,
    ) -> sd.InputStream:
        """
        コールバック付きの入力ストリームを開く

        Args:
            callback: フレーム受信時のコールバック (indata, frames, time, status)
            frame_size: 1フレームのサンプル数

        Returns:
            入力ストリーム
        """
        return sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=self.dtype,
            blocksize=frame_size,
            device=self.device,
            callback=callback,
        )

    @staticmethod
    def list_devices() -> str:
        """利用可能なオーディオデバイスを一覧表示"""
        return str(sd.query_devices())

    @staticmethod
    def get_default_input() -> dict:
        """デフォルト入力デバイスの情報"""
        idx = sd.default.device[0]
        if idx is None or idx < 0:
            # デフォルトが設定されていない場合、入力対応デバイスを探す
            devices = sd.query_devices()
            for i, d in enumerate(devices):
                if d['max_input_channels'] > 0:
                    return dict(d)
            return {}
        return dict(sd.query_devices(idx))


class AudioPlayer:
    """スピーカーへの音声再生"""

    def __init__(
        self,
        sample_rate: int = 22050,
        device: Optional[int] = None,
    ):
        self.sample_rate = sample_rate
        self.device = device

    def play(self, audio_data: np.ndarray, sample_rate: Optional[int] = None, blocking: bool = True) -> None:
        """
        音声データを再生する

        Args:
            audio_data: 音声データ (float32)
            sample_rate: サンプルレート (Noneならself.sample_rate)
            blocking: 再生完了まで待機するか
        """
        sr = sample_rate or self.sample_rate
        sd.play(audio_data, sr, device=self.device)
        if blocking:
            sd.wait()

    def play_wav(self, wav_data: bytes, blocking: bool = True) -> None:
        """
        WAVバイトデータを再生する

        Args:
            wav_data: WAV形式のバイトデータ
            blocking: 再生完了まで待機するか
        """
        wav_buffer = io.BytesIO(wav_data)
        with wave.open(wav_buffer, "rb") as wf:
            sr = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

        sd.play(audio, sr, device=self.device)
        if blocking:
            sd.wait()

    def play_file(self, filepath: str, blocking: bool = True) -> None:
        """WAVファイルを再生する"""
        with open(filepath, "rb") as f:
            wav_data = f.read()
        self.play_wav(wav_data, blocking)

    def stop(self) -> None:
        """再生を停止"""
        sd.stop()

    @staticmethod
    def get_default_output() -> dict:
        """デフォルト出力デバイスの情報"""
        idx = sd.default.device[1]
        if idx is None or idx < 0:
            devices = sd.query_devices()
            for i, d in enumerate(devices):
                if d['max_output_channels'] > 0:
                    return dict(d)
            return {}
        return dict(sd.query_devices(idx))
