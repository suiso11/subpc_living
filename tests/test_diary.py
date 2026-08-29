from __future__ import annotations

import io
import json
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

from src.chat.config import ChatConfig
from src.diary.collector import DiaryCollector, DiarySources
from src.diary.main import main as diary_main
from src.diary.service import DailyDiaryService
from src.integrations.google_calendar import GoogleCalendarMCPClient
from src.llm.errors import ProviderRequestError
from src.llm.providers.fake import FakeProvider
from src.persona.daily_personalizer import DailyPersonalizer


class FakeLLM:
    def __init__(self, text: str = "# 2026-07-01 の日記\n\n今日はログをまとめた。"):
        self.text = text

    def generate(self, messages, **kwargs):
        return self.text


class FailingProvider:
    def generate(self, messages, **kwargs):
        raise ProviderRequestError("test", "generate", "forced failure")


class FakeDiaryCollector:
    def collect(self, target_date, **kwargs):
        return DiarySources(
            target_date=target_date.isoformat(),
            timezone="Asia/Tokyo",
            generated_at="2026-07-02T00:00:00+09:00",
            calendar={"enabled": False, "events": [], "error": ""},
            manual_schedule=[],
            discord_turns=[],
            voice_transcripts=[],
            recent_summaries=[],
            metrics_summary={"available": False},
            profile={},
        )


class DiaryCollectorTest(unittest.TestCase):
    def test_collects_discord_turns_and_metrics_for_day(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_jsonl(
                root / "data" / "discord_training" / "conversations.jsonl",
                [
                    {
                        "created_at": "2026-07-01T09:30:00",
                        "channel_id": 1,
                        "user": "朝に予定を確認した",
                        "assistant": "予定を確認しました",
                    },
                    {
                        "created_at": "2026-07-02T00:10:00",
                        "channel_id": 1,
                        "user": "翌日",
                        "assistant": "翌日です",
                    },
                ],
            )
            self._write_jsonl(
                root / "data" / "discord_voice" / "transcripts" / "2026-07-01.jsonl",
                [
                    {
                        "created_at": "2026-07-01T21:15:00+09:00",
                        "voice_channel_id": 10,
                        "user_name": "haruka",
                        "text": "通話で今日の作業を振り返った",
                        "duration_sec": 3.2,
                    },
                    {
                        "created_at": "2026-07-02T00:10:00+09:00",
                        "voice_channel_id": 10,
                        "user_name": "haruka",
                        "text": "翌日の通話",
                        "duration_sec": 2.0,
                    },
                ],
            )
            self._write_json(
                root / "data" / "profile" / "user_profile.json",
                {
                    "name": "はるか",
                    "schedule": [
                        {"date": "2026-07-01", "time": "12:00", "title": "昼の予定"},
                        {"date": "2026-07-02", "time": "12:00", "title": "翌日の予定"},
                    ],
                },
            )
            self._write_metrics(root / "data" / "metrics" / "system_metrics.db")

            sources = DiaryCollector(root, timezone="Asia/Tokyo").collect(
                date(2026, 7, 1),
                include_calendar=False,
            )

            self.assertEqual(len(sources.discord_turns), 1)
            self.assertEqual(sources.discord_turns[0]["user"], "朝に予定を確認した")
            self.assertEqual(len(sources.voice_transcripts), 1)
            self.assertEqual(sources.voice_transcripts[0]["text"], "通話で今日の作業を振り返った")
            self.assertEqual(len(sources.manual_schedule), 1)
            self.assertEqual(sources.manual_schedule[0]["title"], "昼の予定")
            self.assertTrue(sources.metrics_summary["available"])
            self.assertEqual(sources.metrics_summary["sample_count"], 1)

    def test_google_calendar_missing_credentials_returns_error_without_process(self) -> None:
        client = GoogleCalendarMCPClient(credentials_path="/tmp/definitely-missing-google-oauth.json")
        result = client.list_events_for_day(date(2026, 7, 1))

        self.assertFalse(result.ok)
        self.assertEqual(result.events, [])
        self.assertIn("OAuth credentials not found", result.error)

    @staticmethod
    def _write_json(path: Path, data: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )

    @staticmethod
    def _write_metrics(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path))
        conn.execute(
            """
            CREATE TABLE metrics (
                timestamp REAL NOT NULL,
                cpu_percent REAL,
                mem_percent REAL,
                gpu_util_percent REAL,
                gpu_power_w REAL,
                gpu_temp_c REAL,
                cpu_temp_c REAL
            )
            """
        )
        ts = datetime(2026, 7, 1, 12, 0, tzinfo=ZoneInfo("Asia/Tokyo")).timestamp()
        conn.execute(
            """
            INSERT INTO metrics (
                timestamp, cpu_percent, mem_percent, gpu_util_percent,
                gpu_power_w, gpu_temp_c, cpu_temp_c
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (ts, 20.0, 40.0, 10.0, 80.0, 55.0, 50.0),
        )
        conn.commit()
        conn.close()


class DailyDiaryServiceTest(unittest.TestCase):
    def test_generate_accepts_provider_and_forwards_options(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            provider = FakeProvider(response="# 2026-07-01 の日記\n\nProviderで生成した。")
            service = DailyDiaryService(
                project_root=root,
                llm=provider,
                collector=FakeDiaryCollector(),
                timezone="Asia/Tokyo",
                temperature=0.25,
                num_ctx=4096,
            )

            result = service.generate(
                date(2026, 7, 1),
                save=False,
                include_calendar=False,
            )

            self.assertEqual(result.markdown, "# 2026-07-01 の日記\n\nProviderで生成した。\n")
            self.assertEqual(len(provider.calls), 1)
            self.assertEqual(provider.calls[0]["options"]["temperature"], 0.25)
            self.assertEqual(provider.calls[0]["options"]["num_ctx"], 4096)

    def test_provider_error_returns_fallback_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            service = DailyDiaryService(
                project_root=root,
                llm=FailingProvider(),
                collector=FakeDiaryCollector(),
                timezone="Asia/Tokyo",
            )

            result = service.generate(
                date(2026, 7, 1),
                save=False,
                include_calendar=False,
            )

            self.assertIn("LLMでの日記生成に失敗したため", result.markdown)
            self.assertIn("test.generate: forced failure", result.markdown)

    def test_generate_saves_markdown_and_reuses_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            collector = DiaryCollector(root, timezone="Asia/Tokyo")
            service = DailyDiaryService(
                project_root=root,
                llm=FakeLLM(),
                collector=collector,
                timezone="Asia/Tokyo",
            )

            first = service.generate(date(2026, 7, 1), include_calendar=False)
            second = service.generate(date(2026, 7, 1), include_calendar=False)

            self.assertTrue(first.generated)
            self.assertFalse(second.generated)
            self.assertTrue(Path(first.markdown_path).exists())
            self.assertEqual(first.markdown, second.markdown)
            self.assertFalse(service.was_posted(date(2026, 7, 1)))
            service.mark_posted(date(2026, 7, 1), channel_id=123)
            self.assertTrue(service.was_posted(date(2026, 7, 1)))


class DailyPersonalizerTest(unittest.TestCase):
    def test_dry_run_does_not_update_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_profile(root)
            self._write_diary(root)
            llm = FakeLLM(
                json.dumps(
                    {
                        "preferences": [
                            {
                                "key": "coffee",
                                "value": "ホンジュラスのコーヒーを好む",
                                "confidence": 0.9,
                            }
                        ],
                        "habits": [],
                        "notes": [],
                        "facts": [],
                    },
                    ensure_ascii=False,
                )
            )
            personalizer = DailyPersonalizer(project_root=root, llm=llm)

            result = personalizer.run(date(2026, 7, 1), dry_run=True)
            profile = json.loads((root / "data" / "profile" / "user_profile.json").read_text(encoding="utf-8"))

            self.assertEqual(result.applied_count, 1)
            self.assertEqual(profile["preferences"], {})
            self.assertTrue(Path(result.audit_path).exists())

    def test_apply_updates_profile_and_skips_low_confidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_profile(root)
            self._write_diary(root)
            llm = FakeLLM(
                "```json\n"
                + json.dumps(
                    {
                        "preferences": [
                            {
                                "key": "coffee",
                                "value": "ホンジュラスのコーヒーを好む",
                                "confidence": 0.9,
                            }
                        ],
                        "habits": [
                            {
                                "key": "sleep_pattern",
                                "value": "夜更かし気味",
                                "confidence": 0.6,
                            }
                        ],
                        "notes": [
                            {
                                "text": "制作や探索の文脈を優先する",
                                "confidence": 0.8,
                            }
                        ],
                        "facts": [
                            {
                                "text": "コーヒーイベントの記録を日記材料として残すと役立つ",
                                "confidence": 0.75,
                            }
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n```"
            )
            personalizer = DailyPersonalizer(project_root=root, llm=llm)

            result = personalizer.run(date(2026, 7, 1), dry_run=False)
            profile = json.loads((root / "data" / "profile" / "user_profile.json").read_text(encoding="utf-8"))

            self.assertEqual(result.applied_count, 3)
            self.assertEqual(profile["preferences"]["coffee"], "ホンジュラスのコーヒーを好む")
            self.assertNotIn("sleep_pattern", profile["habits"])
            self.assertIn("制作や探索の文脈を優先する", profile["notes"])
            self.assertIn("コーヒーイベントの記録を日記材料として残すと役立つ", profile["extracted_facts"])

    @staticmethod
    def _write_profile(root: Path) -> None:
        path = root / "data" / "profile" / "user_profile.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "name": "はるか",
                    "nickname": "",
                    "preferences": {},
                    "habits": {},
                    "schedule": [],
                    "notes": [],
                    "extracted_facts": [],
                    "updated_at": "",
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _write_diary(root: Path) -> None:
        path = root / "data" / "diary" / "2026-07-01.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# 2026-07-01 の日記\n\nコーヒーイベントの話があった。", encoding="utf-8")


class DiaryMainEntrypointTest(unittest.TestCase):
    """Offline wiring tests for the manual diary entrypoint (src.diary.main)."""

    def test_main_uses_build_local_provider_ollama_default_and_closes(self) -> None:
        provider_cls, providers = _recording_provider_class()
        service_cls, services = _recording_service_class(fail=False)
        config = ChatConfig(ollama_base_url="http://localhost:9999", model="qwen")
        with (
            patch("sys.argv", ["diary-main", "--date", "2026-07-01", "--no-calendar", "--no-save"]),
            patch.object(ChatConfig, "load", return_value=config),
            patch("src.assistant.factory.OllamaProvider", provider_cls),
            patch("src.diary.main.DiaryCollector", _FakeCollectorClass),
            patch("src.diary.main.DailyDiaryService", service_cls),
            redirect_stdout(io.StringIO()),
        ):
            diary_main()

        self.assertEqual(len(providers), 1)
        self.assertEqual(
            providers[0].kwargs,
            {"base_url": "http://localhost:9999", "model": "qwen", "provider_id": "ollama"},
        )
        self.assertEqual(len(services), 1)
        self.assertIs(services[0].kwargs["llm"], providers[0])
        self.assertEqual(services[0].kwargs["num_ctx"], config.num_ctx)
        self.assertTrue(providers[0].closed)

    def test_main_selects_openai_compatible_provider_and_closes(self) -> None:
        provider_cls, providers = _recording_provider_class()
        service_cls, services = _recording_service_class(fail=False)
        config = ChatConfig(
            local_provider_kind="openai_compatible",
            local_base_url="http://localhost:8000/v1",
            model="qwen",
        )
        with (
            patch("sys.argv", ["diary-main", "--date", "2026-07-01", "--no-calendar", "--no-save"]),
            patch.object(ChatConfig, "load", return_value=config),
            patch("src.assistant.factory.LocalOpenAICompatibleProvider", provider_cls),
            patch("src.diary.main.DiaryCollector", _FakeCollectorClass),
            patch("src.diary.main.DailyDiaryService", service_cls),
            redirect_stdout(io.StringIO()),
        ):
            diary_main()

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
        self.assertIs(services[0].kwargs["llm"], providers[0])
        self.assertTrue(providers[0].closed)

    def test_main_closes_provider_when_generation_fails(self) -> None:
        provider_cls, providers = _recording_provider_class()
        service_cls, _services = _recording_service_class(fail=True)
        config = ChatConfig()
        with (
            patch("sys.argv", ["diary-main", "--date", "2026-07-01", "--no-calendar", "--no-save"]),
            patch.object(ChatConfig, "load", return_value=config),
            patch("src.assistant.factory.OllamaProvider", provider_cls),
            patch("src.diary.main.DiaryCollector", _FakeCollectorClass),
            patch("src.diary.main.DailyDiaryService", service_cls),
            redirect_stdout(io.StringIO()),
        ):
            with self.assertRaises(RuntimeError):
                diary_main()

        self.assertEqual(len(providers), 1)
        self.assertTrue(providers[0].closed)

    def test_main_closes_provider_when_calendar_client_construction_fails(self) -> None:
        provider_cls, providers = _recording_provider_class()
        config = ChatConfig()

        class ExplodingCalendar:
            @classmethod
            def from_env(cls):
                raise RuntimeError("forced calendar client failure")

        with (
            patch("sys.argv", ["diary-main", "--date", "2026-07-01"]),
            patch.object(ChatConfig, "load", return_value=config),
            patch("src.assistant.factory.OllamaProvider", provider_cls),
            patch("src.diary.main.GoogleCalendarMCPClient", ExplodingCalendar),
            patch("src.diary.main.DiaryCollector", _FakeCollectorClass),
            redirect_stdout(io.StringIO()),
        ):
            with self.assertRaises(RuntimeError):
                diary_main()

        self.assertEqual(len(providers), 1)
        self.assertTrue(providers[0].closed)

    def test_main_closes_provider_when_service_constructor_fails(self) -> None:
        provider_cls, providers = _recording_provider_class()
        config = ChatConfig()

        class ExplodingService:
            def __init__(self, **kwargs) -> None:
                raise RuntimeError("forced service constructor failure")

        with (
            patch("sys.argv", ["diary-main", "--date", "2026-07-01", "--no-calendar", "--no-save"]),
            patch.object(ChatConfig, "load", return_value=config),
            patch("src.assistant.factory.OllamaProvider", provider_cls),
            patch("src.diary.main.DiaryCollector", _FakeCollectorClass),
            patch("src.diary.main.DailyDiaryService", ExplodingService),
            redirect_stdout(io.StringIO()),
        ):
            with self.assertRaises(RuntimeError):
                diary_main()

        self.assertEqual(len(providers), 1)
        self.assertTrue(providers[0].closed)


class _FakeCollectorClass:
    def __init__(self, *args, **kwargs) -> None:
        pass


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


def _recording_service_class(*, fail: bool):
    instances = []

    class RecordingService:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            instances.append(self)

        def generate(self, *args, **kwargs) -> SimpleNamespace:
            if fail:
                raise RuntimeError("forced entrypoint failure")
            return SimpleNamespace(markdown="# 2026-07-01 の日記\n\n生成された。", markdown_path="")

    return RecordingService, instances


if __name__ == "__main__":
    unittest.main()
