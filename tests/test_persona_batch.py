from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import date
from pathlib import Path

from src.llm.errors import ProviderRequestError
from src.llm.providers.fake import FakeProvider
from src.persona.daily_personalizer import DailyPersonalizer
from src.persona.summarizer import ConversationSummarizer


class ProfileDouble:
    def __init__(self) -> None:
        self.facts: list[str] = []

    def add_extracted_facts(self, facts: list[str]) -> int:
        self.facts.extend(facts)
        return len(facts)


class FailingProvider(FakeProvider):
    def generate(self, messages, **kwargs) -> str:
        raise ProviderRequestError("fake", "generate", "request failed")


class ConversationSummarizerTest(unittest.TestCase):
    messages = [
        {"role": "user", "content": "コーヒーが好きです"},
        {"role": "assistant", "content": "覚えておきます"},
    ]

    def test_summarize_session_uses_fake_provider_options(self) -> None:
        provider = FakeProvider(response="コーヒーの好みについて話した。")
        with tempfile.TemporaryDirectory() as tmp:
            summarizer = ConversationSummarizer(summaries_dir=tmp)
            result = summarizer.summarize_session(
                provider,
                self.messages,
                "session-1",
                temperature=0.45,
                num_ctx=2048,
            )

        self.assertEqual(result, "コーヒーの好みについて話した。")
        self.assertEqual(provider.calls[0]["options"]["temperature"], 0.45)
        self.assertEqual(provider.calls[0]["options"]["num_ctx"], 2048)

    def test_extract_facts_parses_json_array(self) -> None:
        provider = FakeProvider(response='["浅煎りのコーヒーが好き"]')
        profile = ProfileDouble()
        with tempfile.TemporaryDirectory() as tmp:
            summarizer = ConversationSummarizer(summaries_dir=tmp)
            result = summarizer.extract_facts(provider, self.messages, profile)

        self.assertEqual(result, ["浅煎りのコーヒーが好き"])
        self.assertEqual(profile.facts, result)

    def test_extract_facts_parses_fenced_json_array(self) -> None:
        provider = FakeProvider(response='```json\n["猫を2匹飼っている"]\n```')
        profile = ProfileDouble()
        with tempfile.TemporaryDirectory() as tmp:
            summarizer = ConversationSummarizer(summaries_dir=tmp)
            result = summarizer.extract_facts(provider, self.messages, profile)

        self.assertEqual(result, ["猫を2匹飼っている"])
        self.assertEqual(profile.facts, result)

    def test_extract_facts_returns_empty_list_for_invalid_json(self) -> None:
        provider = FakeProvider(response="not json")
        profile = ProfileDouble()
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            summarizer = ConversationSummarizer(summaries_dir=tmp)
            result = summarizer.extract_facts(provider, self.messages, profile)

        self.assertEqual(result, [])
        self.assertEqual(profile.facts, [])

    def test_summarize_session_returns_none_for_provider_error(self) -> None:
        provider = FailingProvider()
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            summarizer = ConversationSummarizer(summaries_dir=tmp)
            result = summarizer.summarize_session(
                provider,
                self.messages,
                "session-error",
            )

        self.assertIsNone(result)


class DailyPersonalizerTest(unittest.TestCase):
    def test_run_builds_candidates_with_fake_provider(self) -> None:
        candidates = {
            "preferences": [
                {
                    "key": "coffee",
                    "value": "浅煎りが好き",
                    "confidence": 0.9,
                    "reason": "日記に好みとして出ている",
                }
            ],
            "habits": [],
            "notes": [],
            "facts": [],
        }
        provider = FakeProvider(response=json.dumps(candidates, ensure_ascii=False))
        with tempfile.TemporaryDirectory() as tmp:
            personalizer = DailyPersonalizer(
                project_root=Path(tmp),
                llm=provider,
                temperature=0.25,
                num_ctx=4096,
            )
            result = personalizer.run(
                date(2026, 8, 18),
                diary_markdown="浅煎りのコーヒーが好き。",
                dry_run=True,
            )

        self.assertEqual(result.candidates, candidates)
        self.assertEqual(
            result.applied["preferences"],
            [{"key": "coffee", "value": "浅煎りが好き"}],
        )
        self.assertEqual(provider.calls[0]["options"]["temperature"], 0.25)
        self.assertEqual(provider.calls[0]["options"]["num_ctx"], 4096)


if __name__ == "__main__":
    unittest.main()
