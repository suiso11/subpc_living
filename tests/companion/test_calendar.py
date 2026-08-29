from __future__ import annotations

import unittest
from datetime import datetime

from src.companion.calendar import CalendarSource, NextEvent


def ts(date_str: str, time_str: str = "") -> float:
    if time_str:
        return datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M").timestamp()
    return datetime.strptime(date_str, "%Y-%m-%d").timestamp()


class FakeProfile:
    def __init__(self, schedule: list[dict], *, raise_on_call: bool = False) -> None:
        self._schedule = schedule
        self._raise = raise_on_call

    def get_upcoming_schedule(self, days: int = 7) -> list[dict]:
        if self._raise:
            raise RuntimeError("boom")
        return self._schedule


class CalendarSourceTest(unittest.TestCase):
    def test_returns_single_future_event_with_min_start(self) -> None:
        now = ts("2026-08-20", "10:00")
        profile = FakeProfile(
            [
                {"date": "2026-08-20", "time": "12:00", "title": "昼食"},
                {"date": "2026-08-20", "time": "11:00", "title": "会議"},
                {"date": "2026-08-21", "time": "09:00", "title": "朝礼"},
            ]
        )
        event = CalendarSource(profile).next_event(now)
        self.assertEqual(event.start_at, ts("2026-08-20", "11:00"))
        self.assertEqual(event.title, "会議")

    def test_past_events_are_ignored(self) -> None:
        now = ts("2026-08-20", "10:00")
        profile = FakeProfile(
            [
                {"date": "2026-08-20", "time": "09:00", "title": "過去の会議"},
                {"date": "2026-08-20", "time": "11:00", "title": "未来の会議"},
            ]
        )
        event = CalendarSource(profile).next_event(now)
        self.assertEqual(event.start_at, ts("2026-08-20", "11:00"))
        self.assertEqual(event.title, "未来の会議")

    def test_event_without_time_is_midnight_of_date(self) -> None:
        now = ts("2026-08-19", "23:00")
        profile = FakeProfile(
            [{"date": "2026-08-20", "time": "", "title": "終日予定"}]
        )
        event = CalendarSource(profile).next_event(now)
        self.assertEqual(event.start_at, ts("2026-08-20"))
        self.assertEqual(event.title, "終日予定")

    def test_no_events_returns_none(self) -> None:
        now = ts("2026-08-20", "10:00")
        event = CalendarSource(FakeProfile([])).next_event(now)
        self.assertEqual(event, NextEvent(None, None))

    def test_invalid_date_or_time_is_skipped(self) -> None:
        now = ts("2026-08-20", "10:00")
        profile = FakeProfile(
            [
                {"date": "not-a-date", "time": "12:00", "title": "不正日付"},
                {"date": "2026-08-20", "time": "99:99", "title": "不正時刻"},
                {"date": "2026-08-20", "time": "11:00", "title": "有効な予定"},
            ]
        )
        event = CalendarSource(profile).next_event(now)
        self.assertEqual(event.start_at, ts("2026-08-20", "11:00"))
        self.assertEqual(event.title, "有効な予定")

    def test_profile_exception_returns_none(self) -> None:
        event = CalendarSource(FakeProfile([], raise_on_call=True)).next_event(1000.0)
        self.assertEqual(event, NextEvent(None, None))

    def test_is_pure(self) -> None:
        now = ts("2026-08-20", "10:00")
        profile = FakeProfile(
            [
                {"date": "2026-08-20", "time": "12:00", "title": "昼食"},
                {"date": "2026-08-21", "time": "09:00", "title": "朝礼"},
            ]
        )
        src = CalendarSource(profile)
        self.assertEqual(src.next_event(now), src.next_event(now))


if __name__ == "__main__":
    unittest.main()
