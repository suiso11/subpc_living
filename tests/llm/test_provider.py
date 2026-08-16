from __future__ import annotations

import unittest

from src.chat.client import OllamaClient
from src.llm.provider import LLMProvider
from src.llm.providers.fake import FakeProvider


class FakeProviderTest(unittest.TestCase):
    def test_generate_returns_response_and_captures_a_copy_of_call(self) -> None:
        messages = [{"role": "user", "content": "hello"}]
        provider = FakeProvider(response="world", model="fake-chat")

        result = provider.generate(messages, temperature=0.2, timeout=4.5)
        messages[0]["content"] = "changed"

        self.assertEqual(result, "world")
        self.assertEqual(provider.model, "fake-chat")
        self.assertEqual(provider.calls[0]["kind"], "generate")
        self.assertEqual(provider.calls[0]["messages"][0]["content"], "hello")
        self.assertEqual(provider.calls[0]["options"]["temperature"], 0.2)
        self.assertEqual(provider.calls[0]["options"]["timeout"], 4.5)

    def test_generate_stream_returns_configured_chunks(self) -> None:
        provider = FakeProvider(stream_chunks=["a", "b", "c"])

        chunks = list(
            provider.generate_stream(
                [{"role": "user", "content": "hello"}], num_predict=12
            )
        )

        self.assertEqual(chunks, ["a", "b", "c"])
        self.assertEqual(provider.calls[0]["kind"], "generate_stream")
        self.assertEqual(provider.calls[0]["options"]["num_predict"], 12)

    def test_availability_stats_and_close_are_deterministic(self) -> None:
        provider = FakeProvider(stats={"eval_count": 8})

        self.assertTrue(provider.is_available())
        self.assertEqual(provider.last_stats, {"eval_count": 8})
        provider.close()
        self.assertFalse(provider.is_available())

    def test_fake_and_existing_ollama_client_satisfy_runtime_contract(self) -> None:
        fake = FakeProvider()
        ollama = OllamaClient()
        try:
            self.assertIsInstance(fake, LLMProvider)
            self.assertIsInstance(ollama, LLMProvider)
        finally:
            ollama.close()


if __name__ == "__main__":
    unittest.main()
