from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from src.diary.collector import DiaryCollector
from src.diary.service import DailyDiaryService
from src.integrations.google_calendar import GoogleCalendarMCPClient


class FakeLLM:
    def __init__(self, text: str = "# 2026-07-01 の日記\n\n今日はログをまとめた。"):
        self.text = text

    def generate(self, messages, **kwargs):
        return self.text


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


if __name__ == "__main__":
    unittest.main()
