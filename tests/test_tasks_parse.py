from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from src.discord_bot.task_ui import parse_due, parse_snooze, validate_extraction

UTC = timezone.utc
JST = ZoneInfo("Asia/Tokyo")


class ParseDueTest(unittest.TestCase):
    def setUp(self) -> None:
        # JST 2026-07-03 10:00
        self.now = datetime(2026, 7, 3, 1, 0, tzinfo=UTC)

    def test_ashita_is_date_2359(self) -> None:
        due, gran = parse_due("明日", self.now, JST)
        self.assertEqual(gran, "date")
        local = due.astimezone(JST)
        self.assertEqual((local.month, local.day), (7, 4))
        self.assertEqual((local.hour, local.minute), (23, 59))

    def test_kyou(self) -> None:
        due, gran = parse_due("今日中に", self.now, JST)
        self.assertEqual(gran, "date")
        self.assertEqual(due.astimezone(JST).day, 3)

    def test_asatte(self) -> None:
        due, gran = parse_due("明後日", self.now, JST)
        self.assertEqual(due.astimezone(JST).day, 5)

    def test_md_date(self) -> None:
        due, gran = parse_due("7/10", self.now, JST)
        self.assertEqual(gran, "date")
        local = due.astimezone(JST)
        self.assertEqual((local.month, local.day, local.hour), (7, 10, 23))

    def test_md_datetime(self) -> None:
        due, gran = parse_due("7/10 15:00", self.now, JST)
        self.assertEqual(gran, "datetime")
        local = due.astimezone(JST)
        self.assertEqual((local.month, local.day, local.hour, local.minute), (7, 10, 15, 0))

    def test_ashita_with_time(self) -> None:
        due, gran = parse_due("明日 9:30", self.now, JST)
        self.assertEqual(gran, "datetime")
        local = due.astimezone(JST)
        self.assertEqual((local.day, local.hour, local.minute), (4, 9, 30))

    def test_time_only_future(self) -> None:
        due, gran = parse_due("15:00", self.now, JST)
        self.assertEqual(gran, "datetime")
        local = due.astimezone(JST)
        self.assertEqual((local.day, local.hour), (3, 15))

    def test_time_only_past_rolls_to_tomorrow(self) -> None:
        due, gran = parse_due("8:00", self.now, JST)  # now JST 10:00 -> 翌日
        self.assertEqual(due.astimezone(JST).day, 4)

    def test_relative_minutes(self) -> None:
        due, gran = parse_due("30分後", self.now, JST)
        self.assertEqual(gran, "datetime")
        self.assertEqual(due, self.now + timedelta(minutes=30))

    def test_month_kanji(self) -> None:
        due, gran = parse_due("7月10日", self.now, JST)
        self.assertEqual((due.astimezone(JST).month, due.astimezone(JST).day), (7, 10))

    def test_unparseable(self) -> None:
        due, gran = parse_due("よろしくお願いします", self.now, JST)
        self.assertIsNone(due)
        self.assertIsNone(gran)


class ParseSnoozeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime(2026, 7, 3, 1, 0, tzinfo=UTC)

    def test_minutes(self) -> None:
        self.assertEqual(parse_snooze("30m", self.now, JST), self.now + timedelta(minutes=30))
        self.assertEqual(parse_snooze("30分", self.now, JST), self.now + timedelta(minutes=30))

    def test_hours(self) -> None:
        self.assertEqual(parse_snooze("2h", self.now, JST), self.now + timedelta(hours=2))
        self.assertEqual(parse_snooze("2時間", self.now, JST), self.now + timedelta(hours=2))

    def test_tomorrow(self) -> None:
        until = parse_snooze("明日", self.now, JST)
        local = until.astimezone(JST)
        self.assertEqual((local.day, local.hour), (4, 9))

    def test_invalid(self) -> None:
        self.assertIsNone(parse_snooze("いつか", self.now, JST))


class ValidateExtractionTest(unittest.TestCase):
    def test_valid_dict(self) -> None:
        out = validate_extraction({"is_task": True, "title": "買い物", "due": None, "priority": "high"})
        self.assertEqual(out["title"], "買い物")
        self.assertEqual(out["priority"], "high")
        self.assertIsNone(out["due_at"])

    def test_valid_json_string(self) -> None:
        out = validate_extraction('{"is_task": true, "title": "提出", "due": "2026-07-10T15:00:00+09:00", "priority": "normal"}')
        self.assertEqual(out["title"], "提出")
        self.assertEqual(out["due_at"].astimezone(JST).hour, 15)

    def test_code_fence_stripped(self) -> None:
        out = validate_extraction('```json\n{"is_task": true, "title": "x", "due": null, "priority": "low"}\n```')
        self.assertIsNotNone(out)
        self.assertEqual(out["title"], "x")

    def test_is_task_false(self) -> None:
        self.assertIsNone(validate_extraction({"is_task": False, "title": "x"}))

    def test_missing_title(self) -> None:
        self.assertIsNone(validate_extraction({"is_task": True, "title": "  "}))

    def test_bad_json(self) -> None:
        self.assertIsNone(validate_extraction("not json at all"))

    def test_priority_defaults_normal(self) -> None:
        out = validate_extraction({"is_task": True, "title": "x", "priority": "urgent"})
        self.assertEqual(out["priority"], "normal")

    def test_z_suffix_iso(self) -> None:
        out = validate_extraction({"is_task": True, "title": "x", "due": "2026-07-10T06:00:00Z"})
        self.assertEqual(out["due_at"].astimezone(UTC).hour, 6)

    def test_naive_iso_assumed_local(self) -> None:
        # tzなしのdueは抽出プロンプトの前提であるローカル時刻 (Asia/Tokyo) として解釈する
        out = validate_extraction({"is_task": True, "title": "x", "due": "2026-07-05T23:59:00"})
        self.assertEqual(out["due_at"].astimezone(JST).hour, 23)
        self.assertEqual(out["due_at"].astimezone(UTC).hour, 14)


if __name__ == "__main__":
    unittest.main()
