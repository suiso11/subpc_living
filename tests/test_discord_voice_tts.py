from __future__ import annotations

import io
import unittest
import wave

import numpy as np

from src.discord_bot.voice_tts import (
    DISCORD_CHANNELS,
    DISCORD_SAMPLE_RATE,
    VoiceTTSConfig,
    wav_to_discord_pcm,
)


def _make_wav(samples: np.ndarray, rate: int) -> bytes:
    pcm16 = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue()


class WavToDiscordPCMTest(unittest.TestCase):
    def test_resamples_24k_mono_to_48k_stereo(self) -> None:
        t = np.arange(2400) / 24000.0
        wav = _make_wav(np.sin(2 * np.pi * 440 * t) * 0.5, 24000)

        pcm = wav_to_discord_pcm(wav)

        frame_bytes = int(DISCORD_SAMPLE_RATE * 0.02) * DISCORD_CHANNELS * 2
        self.assertEqual(len(pcm) % frame_bytes, 0)
        samples = np.frombuffer(pcm, dtype="<i2").reshape((-1, 2))
        # 0.1秒の音声 → 48kHzで4800フレーム (+無音パディング)
        self.assertGreaterEqual(samples.shape[0], 4800)
        # 左右チャンネルは同一
        np.testing.assert_array_equal(samples[:, 0], samples[:, 1])
        # 波形の振幅が保たれている
        self.assertGreater(np.abs(samples[:4800, 0]).max(), 12000)

    def test_empty_wav_returns_empty(self) -> None:
        wav = _make_wav(np.empty(0, dtype=np.float32), 24000)
        self.assertEqual(wav_to_discord_pcm(wav), b"")

    def test_config_defaults(self) -> None:
        config = VoiceTTSConfig.from_env()
        self.assertEqual(config.voice, "jf_alpha")
        self.assertTrue(config.autoread)
        self.assertEqual(config.max_chars, 500)


if __name__ == "__main__":
    unittest.main()
