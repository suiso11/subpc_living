"""
EnergyVAD の基本テスト
"""
import numpy as np

from src.audio.vad import EnergyVAD


def test_vad_silence():
    """無音フレームは発話として検出されないこと"""
    vad = EnergyVAD()
    frame = np.zeros(vad.frame_size, dtype=np.float32)
    assert vad.process_frame(frame) is None


def test_vad_speech_then_silence():
    """発話後に無音が続くと発話区間が返ること"""
    vad = EnergyVAD(
        energy_threshold=0.1,
        silence_duration_ms=60,
        min_speech_duration_ms=30,
        speech_pad_ms=0,
    )
    frame_size = vad.frame_size

    # 発話フレーム
    speech_frames = 5
    for _ in range(speech_frames):
        frame = np.ones(frame_size, dtype=np.float32) * 0.5
        result = vad.process_frame(frame)
        assert result is None

    # 無音フレーム（発話終了検出）
    silence_frames = vad._silence_frames
    result = None
    for _ in range(silence_frames):
        frame = np.zeros(frame_size, dtype=np.float32)
        result = vad.process_frame(frame)

    assert result is not None
    assert isinstance(result, np.ndarray)
