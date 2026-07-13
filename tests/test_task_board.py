from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from src.discord_bot.task_board import (
    BOARD_MAX_ITEMS,
    build_board_description,
    build_select_options,
    format_due_input,
    format_board_line,
    format_remaining,
    resolve_task_edit_input,
    resolve_task_input,
    select_option_label,
)

UTC = timezone.utc
JST = ZoneInfo("Asia/Tokyo")


def _task(
    task_id: int,
    title: str = "タスク",
    *,
    due_at=None,
    granularity=None,
    priority: str = "normal",
    status: str = "open",
) -> dict:
    return {
        "id": task_id,
        "title": title,
        "note": None,
        "action_hint": None,
        "due_at": due_at,
        "due_granularity": granularity,
        "priority": priority,
        "status": status,
        "source": "command",
        "created_at": None,
        "completed_at": None,
    }


class FormatRemainingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime(2026, 7, 3, 0, 0, tzinfo=UTC)

    def test_none_is_empty(self) -> None:
        self.assertEqual(format_remaining(None, self.now), "")

    def test_days(self) -> None:
        self.assertEqual(format_remaining(self.now + timedelta(days=2, hours=3), self.now), "あと2日")

    def test_hours(self) -> None:
        self.assertEqual(format_remaining(self.now + timedelta(hours=5), self.now), "あと5時間")

    def test_minutes_floor_at_one(self) -> None:
        self.assertEqual(format_remaining(self.now + timedelta(seconds=30), self.now), "あと1分")

    def test_overdue(self) -> None:
        self.assertEqual(format_remaining(self.now - timedelta(days=3), self.now), "超過3日")


class BoardDescriptionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime(2026, 7, 3, 0, 0, tzinfo=UTC)  # JST 09:00

    def test_empty(self) -> None:
        self.assertEqual(build_board_description([], JST, self.now), "タスクはありません ✨")

    def test_line_has_id_priority_and_remaining(self) -> None:
        due = self.now + timedelta(days=2)
        line = format_board_line(_task(12, "レポート", due_at=due, granularity="datetime", priority="high"), JST, self.now)
        self.assertIn("#12", line)
        self.assertIn("🔴", line)
        self.assertIn("レポート", line)
        self.assertIn("あと2日", line)
        self.assertFalse(line.startswith("⚠️"))

    def test_overdue_line_marked(self) -> None:
        due = self.now - timedelta(days=1)
        line = format_board_line(_task(7, "家賃", due_at=due, granularity="date"), JST, self.now)
        self.assertTrue(line.startswith("⚠️"))
        self.assertIn("超過1日", line)

    def test_no_due_line(self) -> None:
        line = format_board_line(_task(3, "買い物"), JST, self.now)
        self.assertIn("期限なし", line)
        self.assertNotIn("あと", line)

    def test_date_granularity_shows_date_only(self) -> None:
        due = datetime(2026, 7, 10, 14, 59, tzinfo=UTC)  # JST 7/10 23:59
        line = format_board_line(_task(1, "x", due_at=due, granularity="date"), JST, self.now)
        self.assertIn("7/10", line)
        self.assertNotIn("23:59", line)

    def test_overflow_shows_hidden_count(self) -> None:
        tasks = [_task(i) for i in range(BOARD_MAX_ITEMS + 5)]
        desc = build_board_description(tasks, JST, self.now)
        self.assertIn(f"…ほか {5} 件", desc)
        self.assertEqual(desc.count("\n"), BOARD_MAX_ITEMS)  # 15 lines + overflow line => 15 newlines

    def test_long_title_truncated(self) -> None:
        line = format_board_line(_task(1, "あ" * 100), JST, self.now, title_limit=10)
        self.assertIn("…", line)


class SelectOptionsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime(2026, 7, 3, 0, 0, tzinfo=UTC)

    def test_label_and_description_within_limits(self) -> None:
        task = _task(99, "あ" * 200, due_at=self.now + timedelta(hours=3), granularity="datetime", priority="high")
        label, desc = select_option_label(task, JST, self.now)
        self.assertLessEqual(len(label), 100)
        self.assertLessEqual(len(desc), 100)
        self.assertTrue(label.startswith("#99"))

    def test_empty_gives_no_options(self) -> None:
        self.assertEqual(build_select_options([], JST, self.now), [])

    def test_caps_at_25(self) -> None:
        tasks = [_task(i) for i in range(40)]
        options = build_select_options(tasks, JST, self.now)
        self.assertEqual(len(options), 25)
        self.assertEqual(options[0].value, "0")


class ResolveTaskInputTest(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime(2026, 7, 3, 1, 0, tzinfo=UTC)  # JST 10:00

    def test_success_no_due(self) -> None:
        kwargs, error = resolve_task_input(
            title="  買い物  ", due_raw="", priority_raw="", note_raw="", now=self.now, tz=JST
        )
        self.assertIsNone(error)
        self.assertEqual(kwargs["title"], "買い物")
        self.assertIsNone(kwargs["due_at"])
        self.assertIsNone(kwargs["due_granularity"])
        self.assertEqual(kwargs["priority"], "normal")
        self.assertIsNone(kwargs["note"])

    def test_success_with_due_and_priority(self) -> None:
        kwargs, error = resolve_task_input(
            title="レポート", due_raw="明日", priority_raw="HIGH", note_raw="序論だけ", now=self.now, tz=JST
        )
        self.assertIsNone(error)
        self.assertEqual(kwargs["priority"], "high")
        self.assertEqual(kwargs["due_granularity"], "date")
        self.assertEqual(kwargs["due_at"].astimezone(JST).day, 4)
        self.assertEqual(kwargs["note"], "序論だけ")

    def test_empty_title_errors(self) -> None:
        kwargs, error = resolve_task_input(
            title="   ", due_raw="", priority_raw="", note_raw="", now=self.now, tz=JST
        )
        self.assertIsNone(kwargs)
        self.assertIsNotNone(error)

    def test_unparseable_due_errors_and_does_not_register(self) -> None:
        kwargs, error = resolve_task_input(
            title="x", due_raw="いつか", priority_raw="", note_raw="", now=self.now, tz=JST
        )
        self.assertIsNone(kwargs)
        self.assertIn("いつか", error)

    def test_invalid_priority_falls_back_to_normal(self) -> None:
        kwargs, error = resolve_task_input(
            title="x", due_raw="", priority_raw="urgent", note_raw="", now=self.now, tz=JST
        )
        self.assertIsNone(error)
        self.assertEqual(kwargs["priority"], "normal")


class ResolveTaskEditInputTest(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime(2026, 7, 3, 1, 0, tzinfo=UTC)  # JST 10:00

    def test_due_prefill_formats_for_modal(self) -> None:
        due = datetime(2026, 7, 10, 14, 59, tzinfo=UTC)  # JST 7/10 23:59
        self.assertEqual(format_due_input(due, "date", JST), "7/10")
        self.assertEqual(format_due_input(due, "datetime", JST), "7/10 23:59")
        self.assertEqual(format_due_input(None, None, JST), "")

    def test_unchanged_overdue_due_is_preserved(self) -> None:
        due = datetime(2026, 7, 1, 14, 59, tzinfo=UTC)  # JST 7/1 23:59, already past
        task = _task(1, "古い", due_at=due, granularity="date")
        kwargs, error = resolve_task_edit_input(
            current_task=task,
            title="古い 修正",
            due_raw="7/1",
            default_due_raw="7/1",
            priority_raw="normal",
            note_raw="",
            now=self.now,
            tz=JST,
        )
        self.assertIsNone(error)
        self.assertEqual(kwargs["due_at"], due)
        self.assertEqual(kwargs["due_granularity"], "date")
        self.assertFalse(kwargs["clear_due"])

    def test_blank_due_clears_existing_due(self) -> None:
        due = self.now + timedelta(days=1)
        task = _task(1, "x", due_at=due, granularity="datetime")
        kwargs, error = resolve_task_edit_input(
            current_task=task,
            title="x",
            due_raw="",
            default_due_raw="7/4 10:00",
            priority_raw="high",
            note_raw="",
            now=self.now,
            tz=JST,
        )
        self.assertIsNone(error)
        self.assertTrue(kwargs["clear_due"])
        self.assertIsNone(kwargs["due_at"])

    def test_changed_due_is_parsed(self) -> None:
        task = _task(1, "x")
        kwargs, error = resolve_task_edit_input(
            current_task=task,
            title="x",
            due_raw="明日",
            default_due_raw="",
            priority_raw="",
            note_raw="",
            now=self.now,
            tz=JST,
        )
        self.assertIsNone(error)
        self.assertEqual(kwargs["due_granularity"], "date")
        self.assertEqual(kwargs["due_at"].astimezone(JST).day, 4)


if __name__ == "__main__":
    unittest.main()
