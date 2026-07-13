"""EventReminderEngine と format_event_reminder のテスト。"""
import json
import os
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from src.tasks.event_reminder import EventReminderEngine, format_event_reminder

UTC = timezone.utc


class FormatEventReminderTest(unittest.TestCase):
    def test_includes_title_time_and_minutes(self):
        """format_event_reminder が「まもなく予定」「HH:MM」「あとN分」を含む文を返す。"""
        start = datetime(2026, 7, 11, 15, 30, tzinfo=UTC)
        now = datetime(2026, 7, 11, 15, 20, tzinfo=UTC)
        title = "歯医者"
        location = "〇〇歯科"

        msg = format_event_reminder(title, start, now, location)

        self.assertIn("まもなく予定", msg)
        self.assertIn("15:30", msg)
        self.assertIn("あと", msg)
        self.assertIn("分", msg)
        self.assertIn(title, msg)
        self.assertIn(location, msg)

    def test_without_location(self):
        """location 無しの場合も正しくフォーマットされる。"""
        start = datetime(2026, 7, 11, 15, 30, tzinfo=UTC)
        now = datetime(2026, 7, 11, 15, 20, tzinfo=UTC)
        title = "会議"

        msg = format_event_reminder(title, start, now)

        self.assertIn("まもなく予定", msg)
        self.assertIn("15:30", msg)
        self.assertIn("あと", msg)
        self.assertIn("分", msg)
        self.assertIn(title, msg)


class EventReminderEngineTest(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.upcoming_path = os.path.join(self.tmp_dir, "upcoming.json")
        self.db_path = os.path.join(self.tmp_dir, "tasks.db")
        self.call_count = 0
        self.last_call_kwargs = None

    def _callback(self, **kwargs):
        """テスト用 callback。呼び出しをカウント。"""
        self.call_count += 1
        self.last_call_kwargs = kwargs

    def _make_upcoming_json(self, events):
        """upcoming.json を作成。"""
        os.makedirs(os.path.dirname(self.upcoming_path), exist_ok=True)
        with open(self.upcoming_path, "w", encoding="utf-8") as f:
            json.dump({
                "generated_at": datetime(2026, 7, 11, 6, 0, tzinfo=UTC).isoformat(),
                "timezone": "Asia/Tokyo",
                "events": events,
            }, f)

    def test_event_fires_in_lead_window(self):
        """開始 lead_min(15) 分前の窓に入ったイベントで callback が trigger_type="event_remind" で1回発火。"""
        # 15:00 に開始するイベント。lead_min=15 なので 14:45-15:00 が発火窓。
        # now=14:50(UTC) = JST 23:50 (14:50+9=23:50) -> イベント start は JST 15:00 = UTC 06:00
        events = [{
            "title": "歯医者",
            "start": "2026-07-11T15:00:00+09:00",
            "end": "2026-07-11T16:00:00+09:00",
            "location": "",
            "event_id": "ev1",
        }]
        self._make_upcoming_json(events)

        # now = 14:50 UTC = 23:50 JST (前日)
        # start = 15:00 JST = 06:00 UTC (翌日)
        # 差分: 06:00 - 14:50 = 今日の06:00 - 昨日の14:50
        # -> 実は14:50 UTC の時点では 06:00 UTC が未来
        # 正確に: now = 2026-07-11T14:50 UTC, start = 2026-07-11T15:00+09:00 = 2026-07-11T06:00 UTC
        # つまり start - lead = 06:00 - 15min = 05:45 UTC
        # 14:50 UTC は 05:45 UTC より後なので、窓に入っていない。

        # 正しくセット: now = 2026-07-11T05:50 UTC = 14:50 JST
        now = datetime(2026, 7, 11, 5, 50, tzinfo=UTC)

        engine = EventReminderEngine(
            callback=self._callback,
            upcoming_path=self.upcoming_path,
            db_path=self.db_path,
            lead_min=15.0,
            timezone_name="Asia/Tokyo",
            quiet_hours=(0, 0),  # quiet 無効
            now_fn=lambda: now,
        )
        engine.initialize()
        fired = engine.run_once(now)

        self.assertEqual(fired, 1)
        self.assertEqual(self.call_count, 1)
        self.assertEqual(self.last_call_kwargs["trigger_type"], "event_remind")
        self.assertIn("event_id", self.last_call_kwargs)
        self.assertEqual(self.last_call_kwargs["event_id"], "ev1")
        engine.close()

    def test_event_dedup_same_engine(self):
        """同じイベントで run_once を2回呼んでも2回目は発火しない (SQLite dedup)。"""
        events = [{
            "title": "歯医者",
            "start": "2026-07-11T15:00:00+09:00",
            "end": "2026-07-11T16:00:00+09:00",
            "location": "",
            "event_id": "ev1",
        }]
        self._make_upcoming_json(events)
        now = datetime(2026, 7, 11, 5, 50, tzinfo=UTC)

        engine = EventReminderEngine(
            callback=self._callback,
            upcoming_path=self.upcoming_path,
            db_path=self.db_path,
            lead_min=15.0,
            timezone_name="Asia/Tokyo",
            quiet_hours=(0, 0),
            now_fn=lambda: now,
        )
        engine.initialize()

        # 1回目
        fired1 = engine.run_once(now)
        self.assertEqual(fired1, 1)
        self.assertEqual(self.call_count, 1)

        # 2回目（同じ now、同じイベント）
        fired2 = engine.run_once(now)
        self.assertEqual(fired2, 0)  # 発火しない
        self.assertEqual(self.call_count, 1)  # カウントは増えない

        engine.close()

    def test_event_dedup_new_engine_same_db(self):
        """エンジンを作り直して同じ db_path でも発火しない (永続化確認)。"""
        events = [{
            "title": "歯医者",
            "start": "2026-07-11T15:00:00+09:00",
            "end": "2026-07-11T16:00:00+09:00",
            "location": "",
            "event_id": "ev1",
        }]
        self._make_upcoming_json(events)
        now = datetime(2026, 7, 11, 5, 50, tzinfo=UTC)

        # 1つめのエンジン
        engine1 = EventReminderEngine(
            callback=self._callback,
            upcoming_path=self.upcoming_path,
            db_path=self.db_path,
            lead_min=15.0,
            timezone_name="Asia/Tokyo",
            quiet_hours=(0, 0),
            now_fn=lambda: now,
        )
        engine1.initialize()
        fired1 = engine1.run_once(now)
        self.assertEqual(fired1, 1)
        self.assertEqual(self.call_count, 1)
        engine1.close()

        # 2つめのエンジン（同じ db_path、別の callback）
        self.call_count = 0
        engine2 = EventReminderEngine(
            callback=self._callback,
            upcoming_path=self.upcoming_path,
            db_path=self.db_path,
            lead_min=15.0,
            timezone_name="Asia/Tokyo",
            quiet_hours=(0, 0),
            now_fn=lambda: now,
        )
        engine2.initialize()
        fired2 = engine2.run_once(now)
        self.assertEqual(fired2, 0)  # 同じ db なので発火しない
        self.assertEqual(self.call_count, 0)
        engine2.close()

    def test_event_ignores_before_lead_window(self):
        """開始16分前では発火しない。"""
        events = [{
            "title": "歯医者",
            "start": "2026-07-11T15:00:00+09:00",
            "end": "2026-07-11T16:00:00+09:00",
            "location": "",
            "event_id": "ev1",
        }]
        self._make_upcoming_json(events)

        # now = 14:44 JST = 05:44 UTC (start - 16 分 = 14:44)
        now = datetime(2026, 7, 11, 5, 44, tzinfo=UTC)

        engine = EventReminderEngine(
            callback=self._callback,
            upcoming_path=self.upcoming_path,
            db_path=self.db_path,
            lead_min=15.0,
            timezone_name="Asia/Tokyo",
            quiet_hours=(0, 0),
            now_fn=lambda: now,
        )
        engine.initialize()
        fired = engine.run_once(now)

        self.assertEqual(fired, 0)
        self.assertEqual(self.call_count, 0)
        engine.close()

    def test_event_ignores_after_start(self):
        """開始後は発火しない。"""
        events = [{
            "title": "歯医者",
            "start": "2026-07-11T15:00:00+09:00",
            "end": "2026-07-11T16:00:00+09:00",
            "location": "",
            "event_id": "ev1",
        }]
        self._make_upcoming_json(events)

        # now = 15:01 JST = 06:01 UTC (start 後)
        now = datetime(2026, 7, 11, 6, 1, tzinfo=UTC)

        engine = EventReminderEngine(
            callback=self._callback,
            upcoming_path=self.upcoming_path,
            db_path=self.db_path,
            lead_min=15.0,
            timezone_name="Asia/Tokyo",
            quiet_hours=(0, 0),
            now_fn=lambda: now,
        )
        engine.initialize()
        fired = engine.run_once(now)

        self.assertEqual(fired, 0)
        self.assertEqual(self.call_count, 0)
        engine.close()

    def test_event_ignores_all_day_events(self):
        """終日イベント (start に時刻が無い) は発火しない。"""
        events = [{
            "title": "終日イベント",
            "start": "2026-07-11",  # T なし = 終日
            "end": "2026-07-12",
            "location": "",
            "event_id": "ev1",
        }]
        self._make_upcoming_json(events)

        now = datetime(2026, 7, 11, 5, 50, tzinfo=UTC)

        engine = EventReminderEngine(
            callback=self._callback,
            upcoming_path=self.upcoming_path,
            db_path=self.db_path,
            lead_min=15.0,
            timezone_name="Asia/Tokyo",
            quiet_hours=(0, 0),
            now_fn=lambda: now,
        )
        engine.initialize()
        fired = engine.run_once(now)

        self.assertEqual(fired, 0)
        self.assertEqual(self.call_count, 0)
        engine.close()

    def test_quiet_hours_suppression(self):
        """quiet hours 中は発火しない。"""
        events = [{
            "title": "歯医者",
            "start": "2026-07-11T15:00:00+09:00",
            "end": "2026-07-11T16:00:00+09:00",
            "location": "",
            "event_id": "ev1",
        }]
        self._make_upcoming_json(events)

        # now = 02:50 JST = 02:50 JST = UTC + 9 時間後ろ
        # JST 02:50 = UTC 2026-07-10T17:50
        now = datetime(2026, 7, 10, 17, 50, tzinfo=UTC)

        engine = EventReminderEngine(
            callback=self._callback,
            upcoming_path=self.upcoming_path,
            db_path=self.db_path,
            lead_min=15.0,
            timezone_name="Asia/Tokyo",
            quiet_hours=(1, 8),  # 1:00-8:00 JST は quiet
            now_fn=lambda: now,
        )
        engine.initialize()
        fired = engine.run_once(now)

        self.assertEqual(fired, 0)  # quiet 中なので発火しない
        self.assertEqual(self.call_count, 0)
        engine.close()


if __name__ == "__main__":
    unittest.main()
