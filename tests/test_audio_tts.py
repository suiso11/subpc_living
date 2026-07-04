from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from src.audio.tts import KokoroTTS, _pause_samples, _split_sentences_for_tts


class KokoroTTSTest(unittest.TestCase):
    def test_resolve_onnx_providers_defaults_to_kokoro(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(KokoroTTS._resolve_onnx_providers())

    def test_resolve_cuda_provider_with_device_id(self) -> None:
        with patch.dict(
            os.environ,
            {
                "TTS_ONNX_PROVIDER": "CUDAExecutionProvider",
                "TTS_ONNX_DEVICE_ID": "1",
            },
            clear=True,
        ):
            self.assertEqual(
                KokoroTTS._resolve_onnx_providers(),
                [
                    ("CUDAExecutionProvider", {"device_id": 1}),
                    "CPUExecutionProvider",
                ],
            )

    def test_sentence_split_does_not_break_on_comma(self) -> None:
        self.assertEqual(
            _split_sentences_for_tts("これは、テストです。次です。"),
            ["これは、テストです。", "次です。"],
        )

    def test_pause_samples_uses_configured_duration(self) -> None:
        pause = _pause_samples(24000)

        self.assertEqual(len(pause), 4320)
        self.assertEqual(str(pause.dtype), "float32")
        self.assertEqual(float(pause.max()), 0.0)


if __name__ == "__main__":
    unittest.main()
