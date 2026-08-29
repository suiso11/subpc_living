from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.chat.config import ChatConfig
from src.llm.errors import ProviderRequestError
from src.llm.providers.fake import FakeProvider
from src.persona.daily_personalizer import DailyPersonalizer
from src.persona.personalize_daily import main as personalize_daily_main
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


class PersonalizeDailyMainEntrypointTest(unittest.TestCase):
    """Offline wiring tests for the daily personalization entrypoint."""

    def test_main_uses_build_local_provider_ollama_and_closes(self) -> None:
        provider_cls, providers = _recording_provider_class()
        personalizer_cls, personalizers = _recording_personalizer_class(fail=False)
        config = ChatConfig(ollama_base_url="http://localhost:9999", model="qwen")
        with (
            patch("sys.argv", ["personalize-daily", "--date", "2026-08-18", "--dry-run"]),
            patch.object(ChatConfig, "load", return_value=config),
            patch("src.assistant.factory.OllamaProvider", provider_cls),
            patch("src.persona.personalize_daily.DailyPersonalizer", personalizer_cls),
            redirect_stdout(io.StringIO()),
        ):
            personalize_daily_main()

        self.assertEqual(len(providers), 1)
        self.assertEqual(
            providers[0].kwargs,
            {"base_url": "http://localhost:9999", "model": "qwen", "provider_id": "ollama"},
        )
        self.assertEqual(len(personalizers), 1)
        self.assertIs(personalizers[0].kwargs["llm"], providers[0])
        self.assertEqual(personalizers[0].kwargs["num_ctx"], config.num_ctx)
        self.assertTrue(providers[0].closed)

    def test_main_selects_openai_compatible_and_closes_on_failure(self) -> None:
        provider_cls, providers = _recording_provider_class()
        personalizer_cls, _personalizers = _recording_personalizer_class(fail=True)
        config = ChatConfig(
            local_provider_kind="openai_compatible",
            local_base_url="http://localhost:8000/v1",
            model="qwen",
        )
        with (
            patch("sys.argv", ["personalize-daily", "--date", "2026-08-18", "--dry-run"]),
            patch.object(ChatConfig, "load", return_value=config),
            patch("src.assistant.factory.LocalOpenAICompatibleProvider", provider_cls),
            patch("src.persona.personalize_daily.DailyPersonalizer", personalizer_cls),
            redirect_stdout(io.StringIO()),
        ):
            with self.assertRaises(RuntimeError):
                personalize_daily_main()

        self.assertEqual(len(providers), 1)
        self.assertEqual(
            providers[0].kwargs,
            {
                "model": "qwen",
                "base_url": "http://localhost:8000/v1",
                "provider_id": "local-openai",
                "api_key": None,
            },
        )
        self.assertTrue(providers[0].closed)

    def test_main_closes_provider_when_personalizer_constructor_fails(self) -> None:
        provider_cls, providers = _recording_provider_class()
        config = ChatConfig()

        class ExplodingPersonalizer:
            def __init__(self, **kwargs) -> None:
                raise RuntimeError("forced personalizer constructor failure")

        with (
            patch("sys.argv", ["personalize-daily", "--date", "2026-08-18", "--dry-run"]),
            patch.object(ChatConfig, "load", return_value=config),
            patch("src.assistant.factory.OllamaProvider", provider_cls),
            patch("src.persona.personalize_daily.DailyPersonalizer", ExplodingPersonalizer),
            redirect_stdout(io.StringIO()),
        ):
            with self.assertRaises(RuntimeError):
                personalize_daily_main()

        self.assertEqual(len(providers), 1)
        self.assertTrue(providers[0].closed)


def _recording_provider_class():
    instances = []

    class RecordingProvider:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.closed = False
            instances.append(self)

        def close(self) -> None:
            self.closed = True

    return RecordingProvider, instances


def _recording_personalizer_class(*, fail: bool):
    instances = []

    class RecordingPersonalizer:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            instances.append(self)

        def run(self, target_date, **kwargs) -> SimpleNamespace:
            if fail:
                raise RuntimeError("forced entrypoint failure")
            return SimpleNamespace(
                target_date=target_date.isoformat(),
                dry_run=kwargs.get("dry_run", False),
                applied_count=1,
                audit_path="",
                skipped=[],
            )

    return RecordingPersonalizer, instances


if __name__ == "__main__":
    unittest.main()
