"""
音声区間検出 (VAD) モジュール
Phase 3: エネルギーベースのシンプルなVAD
将来的にはSilero VADにアップグレード可能
"""
import numpy as np
from collections import deque
from typing import Optional


class EnergyVAD:
    """
    エネルギーベースの音声区間検出

    音声のRMSエネルギーが閾値を超えた区間を「発話中」と判定する。
    発話終了の検出には、無音が一定フレーム続くことを条件とする。
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: int = 30,
        energy_threshold: float = 0.01,
        silence_duration_ms: int = 800,
        speech_pad_ms: int = 300,
        min_speech_duration_ms: int = 250,
    ):
        """
        Args:
            sample_rate: サンプルレート
            frame_duration_ms: 1フレームの長さ (ms)
            energy_threshold: 発話判定の閾値 (RMS)
            silence_duration_ms: この時間無音が続いたら発話終了と判定 (ms)
            speech_pad_ms: 発話の前後に追加するパディング (ms)
            min_speech_duration_ms: 最低発話長。これ未満は雑音として無視 (ms)
        """
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.energy_threshold = energy_threshold
        self.silence_duration_ms = silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        self.min_speech_duration_ms = min_speech_duration_ms

        # 1フレームのサンプル数
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        # 無音フレーム数の閾値
        self._silence_frames = int(silence_duration_ms / frame_duration_ms)
        # パディングフレーム数
        self._pad_frames = int(speech_pad_ms / frame_duration_ms)
        # 最低発話フレーム数
        self._min_speech_frames = int(min_speech_duration_ms / frame_duration_ms)

        # 状態管理
        self._is_speaking = False
        self._silence_count = 0
        self._speech_frames: list[np.ndarray] = []
        self._pre_buffer: deque = deque(maxlen=self._pad_frames)
        self._calibrated = False
        self._noise_floor = 0.0

    def calibrate(self, audio_data: np.ndarray, duration_sec: float = 1.0) -> float:
        """
        環境ノイズのキャリブレーション

        Args:
            audio_data: キャリブレーション用の音声データ
            duration_sec: 使用する長さ (秒)

        Returns:
            計測したノイズフロア (RMS)
        """
        samples = int(self.sample_rate * duration_sec)
        data = audio_data[:samples].astype(np.float32)
        if np.max(np.abs(data)) > 1.0:
            data = data / 32768.0

        # RMSを計算
        self._noise_floor = np.sqrt(np.mean(data ** 2))
        # 閾値をノイズフロアの3倍に設定
        self.energy_threshold = max(self._noise_floor * 3, 0.005)
        self._calibrated = True
        print(f"[VAD] キャリブレーション完了: noise_floor={self._noise_floor:.4f}, threshold={self.energy_threshold:.4f}")
        return self._noise_floor

    def process_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        1フレーム分の音声を処理する

        Args:
            frame: 1フレーム分の音声データ (float32, モノラル)

        Returns:
            発話が完了した場合は発話全体の音声データ (np.ndarray)
            まだ発話中または無音の場合は None
        """
        if frame.dtype != np.float32:
            frame = frame.astype(np.float32)
        if np.max(np.abs(frame)) > 1.0:
            frame = frame / 32768.0

        rms = np.sqrt(np.mean(frame ** 2))
        is_speech = rms > self.energy_threshold

        if not self._is_speaking:
            # 非発話中
            self._pre_buffer.append(frame.copy())
            if is_speech:
                self._is_speaking = True
                self._silence_count = 0
                # プリバッファの内容を発話に含める
                self._speech_frames = list(self._pre_buffer)
                self._speech_frames.append(frame.copy())
                self._pre_buffer.clear()
        else:
            # 発話中
            self._speech_frames.append(frame.copy())
            if not is_speech:
                self._silence_count += 1
                if self._silence_count >= self._silence_frames:
                    # 発話終了
                    self._is_speaking = False
                    self._silence_count = 0

                    # 最低発話長チェック
                    if len(self._speech_frames) >= self._min_speech_frames:
                        result = np.concatenate(self._speech_frames)
                        self._speech_frames = []
                        return result
                    else:
                        # 短すぎる → 雑音として破棄
                        self._speech_frames = []
                        return None
            else:
                self._silence_count = 0

        return None

    def reset(self) -> None:
        """状態をリセット"""
        self._is_speaking = False
        self._silence_count = 0
        self._speech_frames = []
        self._pre_buffer.clear()

    @property
    def is_speaking(self) -> bool:
        """現在発話中かどうか"""
        return self._is_speaking

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated
