from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from src.audio.style_bert_vits2_tts import StyleBertVITS2TTS
from src.audio.tts import KokoroTTS
from src.audio.tts_factory import create_tts_backend, normalize_tts_backend


class TTSFactoryTest(unittest.TestCase):
    def test_normalize_style_bert_aliases(self) -> None:
        self.assertEqual(normalize_tts_backend("sbv2"), "style_bert_vits2")
        self.assertEqual(normalize_tts_backend("style-bert-vits2"), "style_bert_vits2")

    def test_create_style_bert_backend_uses_compatible_default_voice(self) -> None:
        tts = create_tts_backend(backend="style_bert_vits2", voice="jf_alpha")

        self.assertIsInstance(tts, StyleBertVITS2TTS)
        self.assertEqual(tts.voice, "jvnv-F1-jp")

    def test_create_style_bert_backend_keeps_tsukuyomi_voice(self) -> None:
        tts = create_tts_backend(backend="style_bert_vits2", voice="tsukuyomi-chan")

        self.assertIsInstance(tts, StyleBertVITS2TTS)
        self.assertEqual(tts.voice, "tsukuyomi-chan")

    def test_tsukuyomi_voice_is_registered(self) -> None:
        self.assertIn("tsukuyomi-chan", StyleBertVITS2TTS.JA_VOICES)

    def test_create_kokoro_backend_uses_compatible_default_voice(self) -> None:
        tts = create_tts_backend(backend="kokoro", voice="jvnv-F1-jp")

        self.assertIsInstance(tts, KokoroTTS)
        self.assertEqual(tts.voice, "jf_alpha")


class StyleBertVITS2SynthesizeTest(unittest.TestCase):
    def _capture_payload(self, tts: StyleBertVITS2TTS, **kwargs) -> dict:
        captured: dict = {}

        def fake_post(url, json=None):
            captured.update(json)
            resp = MagicMock()
            resp.content = b"wav"
            resp.raise_for_status = MagicMock()
            return resp

        client = MagicMock()
        client.post.side_effect = fake_post
        client.__enter__ = MagicMock(return_value=client)
        client.__exit__ = MagicMock(return_value=False)

        with patch(
            "src.audio.style_bert_vits2_tts.httpx.Client", return_value=client
        ):
            tts.synthesize("text", **kwargs)
        return captured

    def test_style_defaults_to_configured_style(self) -> None:
        tts = StyleBertVITS2TTS(voice="jvnv-F1-jp", style="Neutral")
        payload = self._capture_payload(tts)
        self.assertEqual(payload["style"], "Neutral")

    def test_explicit_style_is_used(self) -> None:
        tts = StyleBertVITS2TTS(voice="jvnv-F1-jp", style="Neutral")
        payload = self._capture_payload(tts, style="Happy", style_weight=2.0)
        self.assertEqual(payload["style"], "Happy")
        self.assertEqual(payload["style_weight"], 2.0)

    def test_tsukuyomi_forces_neutral(self) -> None:
        tts = StyleBertVITS2TTS(voice="tsukuyomi-chan", style="Neutral")
        payload = self._capture_payload(tts, style="Angry")
        self.assertEqual(payload["style"], "Neutral")


if __name__ == "__main__":
    unittest.main()
