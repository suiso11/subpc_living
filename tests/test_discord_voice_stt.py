from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.discord_bot.voice_stt import (
    SpeechSegmenter,
    VoiceSTTConfig,
    _is_likely_hallucination,
    pcm48_stereo_to_16k_mono,
)


class DiscordVoiceSTTTest(unittest.TestCase):
    def test_converts_discord_pcm_to_16k_mono(self) -> None:
        frames_48k = 960
        stereo = np.full((frames_48k, 2), 3277, dtype=np.int16)

        audio = pcm48_stereo_to_16k_mono(stereo.tobytes())

        self.assertEqual(audio.dtype, np.float32)
        self.assertEqual(audio.shape, (320,))
        self.assertAlmostEqual(float(audio.mean()), 0.1, places=3)

    def test_segmenter_emits_after_silence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = VoiceSTTConfig(
                enabled=True,
                transcript_channel_id=None,
                transcript_dir=Path(tmp),
                energy_threshold=0.01,
                silence_duration_ms=60,
                speech_pad_ms=0,
                min_speech_duration_ms=60,
                max_segment_seconds=5.0,
            )
            segmenter = SpeechSegmenter(config)
            now = datetime.now(timezone.utc)
            speech_frame = np.full(segmenter.frame_size, 0.1, dtype=np.float32)
            silence_frame = np.zeros(segmenter.frame_size, dtype=np.float32)

            emitted = []
            for frame in [speech_frame, speech_frame, speech_frame, silence_frame, silence_frame]:
                emitted.extend(segmenter.add_audio(frame, now))

            self.assertEqual(len(emitted), 1)
            self.assertGreaterEqual(emitted[0].audio.size, segmenter.frame_size * 3)
            self.assertEqual(emitted[0].reason, "silence")

    def test_hallucination_filter_drops_known_junk(self) -> None:
        self.assertTrue(_is_likely_hallucination("ご視聴ありがとうございました"))
        self.assertTrue(_is_likely_hallucination("おつかれさまでした"))
        self.assertTrue(_is_likely_hallucination("すっ"))
        self.assertTrue(_is_likely_hallucination("ん"))

    def test_hallucination_filter_keeps_real_speech(self) -> None:
        self.assertFalse(_is_likely_hallucination("今日はいい天気ですね"))
        self.assertFalse(_is_likely_hallucination("うん、分かりました"))
        self.assertFalse(_is_likely_hallucination(""))
        self.assertFalse(_is_likely_hallucination("OK"))


if __name__ == "__main__":
    unittest.main()
