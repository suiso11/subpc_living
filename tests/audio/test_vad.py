"""
EnergyVAD の基本テスト
"""
import io
import sys
import unittest
from contextlib import redirect_stdout
from unittest import mock

import numpy as np

from src.audio.vad import EnergyVAD, SileroVAD, create_vad


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


class SileroFallbackCanaryTest(unittest.TestCase):
    """Silero VAD の構築・ロード失敗時のサニタイズ (canary) 検証。

    raw のパス・モデル・エラー内容を漏らさず、固定メッセージ (ASCII) と
    例外型名だけを出すことを保証する。auto は Energy VAD にフォールバックする。
    """

    _SILERO_SECRETS = ("/secret/silero", "silero-model-xyz", "http://silero-secret")

    class CanaryError(RuntimeError):
        def __str__(self) -> str:
            return " | ".join(SileroFallbackCanaryTest._SILERO_SECRETS)

    def test_auto_falls_back_to_energy_on_silero_failure(self):
        with mock.patch(
            "src.audio.vad.SileroVAD", side_effect=self.CanaryError()
        ):
            out = io.StringIO()
            with redirect_stdout(out):
                vad = create_vad("auto", sample_rate=16000)
        assert isinstance(vad, EnergyVAD)
        text = out.getvalue()
        assert self.CanaryError.__name__ in text
        for secret in self._SILERO_SECRETS:
            assert secret not in text
        for line in text.splitlines():
            line.encode("ascii")

    def test_explicit_silero_raises_fixed_error(self):
        with mock.patch(
            "src.audio.vad.SileroVAD", side_effect=self.CanaryError()
        ):
            try:
                create_vad("silero", sample_rate=16000)
                assert False, "expected RuntimeError"
            except RuntimeError as exc:
                assert str(exc) == "silero vad initialization failed"
                assert exc.__cause__ is None
                for secret in self._SILERO_SECRETS:
                    assert secret not in str(exc)

    def test_silero_load_error_is_fixed_and_nonsecret(self):
        fake_torch = mock.Mock()
        fake_torch.hub.load.side_effect = self.CanaryError()
        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            try:
                SileroVAD(sample_rate=16000)
                assert False, "expected RuntimeError"
            except RuntimeError as exc:
                assert str(exc) == "silero vad model load failed"
                assert exc.__cause__ is None
                for secret in self._SILERO_SECRETS:
                    assert secret not in str(exc)


if __name__ == "__main__":
    unittest.main()
