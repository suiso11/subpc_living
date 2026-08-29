"""タスク ⇔ Google Calendar 同期のテスト (全てモック、実 npx/ネットワーク禁止)。"""
import json
import os
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from src.integrations.google_calendar import CalendarEvent, CalendarMutationResult
from src.tasks.calendar_sync import (
    CalendarContext,
    CalendarPullWorker,
    TaskCalendarSync,
    TASK_MARKER_PREFIX,
)
from src.tasks.store import TaskStore

UTC = timezone.utc


class FakeCalendarClient:
    """GoogleCalendarMCPClient のインメモリ・モック。"""

    def __init__(self):
        self.calls = []
        self.next_event_id = "evt-1"
        self.create_fail_times = 0  # 最初の N 回 create を失敗させる
        self.range_events = []
        self.delete_raise_ids = set()  # 指定 event_id の delete を例外で失敗させる
        self.update_raise_ids = set()  # 指定 event_id の update を例外で失敗させる

    def create_event(self, **kwargs):
        self.calls.append(("create", kwargs))
        if self.create_fail_times > 0:
            self.create_fail_times -= 1
            return CalendarMutationResult(ok=False, error="boom")
        eid = self.next_event_id
        return CalendarMutationResult(ok=True, event_id=eid)

    def update_event(self, event_id, **kwargs):
        self.calls.append(("update", event_id, kwargs))
        if event_id in self.update_raise_ids:
            self.update_raise_ids.discard(event_id)
            raise RuntimeError("boom update")
        return CalendarMutationResult(ok=True, event_id=event_id)

    def delete_event(self, event_id, **kwargs):
        self.calls.append(("delete", event_id, kwargs))
        if event_id in self.delete_raise_ids:
            self.delete_raise_ids.discard(event_id)
            raise RuntimeError("boom delete")
        return CalendarMutationResult(ok=True, event_id=event_id)

    def list_events_range(self, start_date, end_date, **kwargs):
        self.calls.append(("list_range", start_date, end_date, kwargs))
        from src.integrations.google_calendar import CalendarFetchResult
        return CalendarFetchResult(events=list(self.range_events))

    def kinds(self):
        return [c[0] for c in self.calls]


def make_store():
    d = tempfile.mkdtemp()
    return TaskStore(db_path=os.path.join(d, "tasks.db")).initialize()


class MigrationTest(unittest.TestCase):
    def test_alter_table_on_existing_db_idempotent(self):
        d = tempfile.mkdtemp()
        db = os.path.join(d, "t.db")
        conn = sqlite3.connect(db)
        conn.execute(
            "CREATE TABLE tasks (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL, "
            "note TEXT, action_hint TEXT, due_at TEXT, due_granularity TEXT, "
            "priority TEXT NOT NULL DEFAULT 'normal', status TEXT NOT NULL DEFAULT 'open', "
            "source TEXT NOT NULL DEFAULT 'command', created_at TEXT NOT NULL, completed_at TEXT)"
        )
        conn.execute("INSERT INTO tasks (title, created_at) VALUES ('old', '2020-01-01T00:00:00+00:00')")
        conn.commit()
        conn.close()

        store = TaskStore(db_path=db).initialize()
        row = store.get(1)
        self.assertIn("calendar_event_id", row)
        self.assertIsNone(row["calendar_event_id"])
        store.close()

        # 再初期化しても壊れない (冪等)。
        store = TaskStore(db_path=db).initialize()
        cols = {r[1] for r in store._require().execute("PRAGMA table_info(tasks)").fetchall()}
        self.assertIn("calendar_event_id", cols)
        self.assertIn("calendar_synced_at", cols)
        store.close()


class TaskCalendarSyncTest(unittest.TestCase):
    def setUp(self):
        self.store = make_store()
        self.client = FakeCalendarClient()
        self.sync = TaskCalendarSync(
            store=self.store,
            calendar_client=self.client,
            calendar_id="cal-x",
            enabled=True,
            timezone="Asia/Tokyo",
            max_retries=3,
            retry_base_delay=0.0,  # テストは待たない
        )

    def _add(self, **kw):
        return self.store.add("買い物", **kw)

    def test_add_dated_task_creates_event_and_stores_id(self):
        tid = self._add(due_at=datetime.now(UTC) + timedelta(hours=3), due_granularity="datetime")
        self.sync._sync_task(tid, "add")
        self.assertEqual(self.client.kinds(), ["create"])
        # summary は 📋、description にマーカー
        _, kwargs = self.client.calls[0]
        self.assertTrue(kwargs["summary"].startswith("📋"))
        self.assertIn(f"{TASK_MARKER_PREFIX}{tid}", kwargs["description"])
        self.assertEqual(kwargs["calendar_id"], "cal-x")
        self.assertEqual(self.store.get(tid)["calendar_event_id"], "evt-1")

    def test_task_without_due_is_skipped(self):
        tid = self._add()  # 期限なし
        ok, _ = self.sync._sync_task(tid, "add")
        self.assertTrue(ok)
        self.assertEqual(self.client.calls, [])

    def test_date_granularity_makes_all_day_window(self):
        due = datetime.now(UTC) + timedelta(days=2)
        tid = self._add(due_at=due, due_granularity="date")
        self.sync._sync_task(tid, "add")
        _, kwargs = self.client.calls[0]
        # 終日: start/end は YYYY-MM-DD で、end は start の翌日
        self.assertRegex(kwargs["start"], r"^\d{4}-\d{2}-\d{2}$")
        self.assertRegex(kwargs["end"], r"^\d{4}-\d{2}-\d{2}$")
        self.assertNotEqual(kwargs["start"], kwargs["end"])

    def test_update_existing_event(self):
        tid = self._add(due_at=datetime.now(UTC) + timedelta(hours=3), due_granularity="datetime")
        self.store.set_calendar_event(tid, "evt-existing")
        self.sync._sync_task(tid, "update")
        self.assertEqual(self.client.kinds(), ["update"])
        self.assertEqual(self.client.calls[0][1], "evt-existing")

    def test_done_updates_summary_to_check(self):
        tid = self._add(due_at=datetime.now(UTC) + timedelta(hours=3), due_granularity="datetime")
        self.store.set_calendar_event(tid, "evt-existing")
        self.store.done(tid)
        self.sync._sync_task(tid, "done")
        self.assertEqual(self.client.kinds(), ["update"])
        self.assertTrue(self.client.calls[0][2]["summary"].startswith("✅"))

    def test_done_without_event_is_noop(self):
        tid = self._add(due_at=datetime.now(UTC) + timedelta(hours=3), due_granularity="datetime")
        self.store.done(tid)
        self.sync._sync_task(tid, "done")
        self.assertEqual(self.client.calls, [])

    def test_drop_deletes_event_and_clears_mapping(self):
        tid = self._add(due_at=datetime.now(UTC) + timedelta(hours=3), due_granularity="datetime")
        self.store.set_calendar_event(tid, "evt-existing")
        self.sync._sync_task(tid, "drop")
        self.assertEqual(self.client.kinds(), ["delete"])
        self.assertIsNone(self.store.get(tid)["calendar_event_id"])

    def test_stale_add_after_done_updates_summary_not_creates(self):
        # done 済みタスクへの遅延 add: open イベントは作らず、既存イベントを完了サマリへ
        tid = self._add(due_at=datetime.now(UTC) + timedelta(hours=3), due_granularity="datetime")
        self.store.set_calendar_event(tid, "evt-existing")
        self.store.done(tid)
        self.sync._sync_task(tid, "add")
        self.assertEqual(self.client.kinds(), ["update"])
        self.assertTrue(self.client.calls[0][2]["summary"].startswith("✅"))

    def test_stale_add_after_done_without_event_is_noop(self):
        # done 済みで mapping 無し: open イベントを絶対に作らない
        tid = self._add(due_at=datetime.now(UTC) + timedelta(hours=3), due_granularity="datetime")
        self.store.done(tid)
        self.sync._sync_task(tid, "add")
        self.assertEqual(self.client.calls, [])

    def test_stale_add_after_drop_deletes_event(self):
        # dropped 済みタスクへの遅延 add: 現在状態 (dropped) に従いイベントを削除
        tid = self._add(due_at=datetime.now(UTC) + timedelta(hours=3), due_granularity="datetime")
        self.store.set_calendar_event(tid, "evt-existing")
        self.store.drop(tid)
        self.sync._sync_task(tid, "add")
        self.assertEqual(self.client.kinds(), ["delete"])
        self.assertIsNone(self.store.get(tid)["calendar_event_id"])

    def test_stale_done_label_on_open_task_reconciles_open(self):
        # open タスクへの遅延 done: 状態に従い通常の open イベントを作成
        tid = self._add(due_at=datetime.now(UTC) + timedelta(hours=3), due_granularity="datetime")
        self.sync._sync_task(tid, "done")
        self.assertEqual(self.client.kinds(), ["create"])

    def test_clearing_due_deletes_event(self):
        tid = self._add(due_at=datetime.now(UTC) + timedelta(hours=3), due_granularity="datetime")
        self.store.set_calendar_event(tid, "evt-existing")
        # 期限を消した後 update イベント
        self.sync._sync_task(tid, "update")  # まだ due あり -> update
        self.client.calls.clear()
        self.store.update(tid, clear_due=True)
        self.sync._sync_task(tid, "update")
        self.assertEqual(self.client.kinds(), ["delete"])

    def test_retry_then_success(self):
        self.client.create_fail_times = 2  # 2回失敗、3回目成功
        tid = self._add(due_at=datetime.now(UTC) + timedelta(hours=3), due_granularity="datetime")
        ok = self.sync._process_with_retry(tid, "add")
        self.assertTrue(ok)
        self.assertEqual(self.client.kinds(), ["create", "create", "create"])
        self.assertEqual(self.store.get(tid)["calendar_event_id"], "evt-1")

    def test_retry_exhausted_does_not_raise(self):
        self.client.create_fail_times = 99
        tid = self._add(due_at=datetime.now(UTC) + timedelta(hours=3), due_granularity="datetime")
        ok = self.sync._process_with_retry(tid, "add")
        self.assertFalse(ok)
        self.assertEqual(len(self.client.calls), 3)  # max_retries
        self.assertIsNone(self.store.get(tid)["calendar_event_id"])

    def test_disabled_enqueue_is_noop(self):
        self.sync.enabled = False
        self.sync.enqueue(1, "add")
        self.assertTrue(self.sync._queue.empty())

    def test_store_hook_end_to_end_via_worker(self):
        # on_change -> enqueue -> worker thread -> create
        self.store.on_change = self.sync.enqueue
        self.sync.start()
        try:
            tid = self.store.add(
                "会議", due_at=datetime.now(UTC) + timedelta(hours=1), due_granularity="datetime"
            )
            # ワーカーが処理するのを待つ
            self.sync._queue.join()
        finally:
            self.sync.stop()
        self.assertEqual(self.store.get(tid)["calendar_event_id"], "evt-1")


class ProfileStub:
    """UserProfile の最小スタブ (load/save/schedule/data)。"""

    def __init__(self, schedule=None):
        self._data = {"schedule": list(schedule or [])}
        self.saved = 0

    def load(self):
        return self._data

    def save(self):
        self.saved += 1

    @property
    def schedule(self):
        return self._data.get("schedule", [])

    @property
    def data(self):
        return self._data


class StoreStub:
    def __init__(self, tasks):
        self._tasks = tasks
        self.set_calls = []

    def get(self, tid):
        return self._tasks.get(tid)

    def set_calendar_event(self, tid, eid):
        self.set_calls.append((tid, eid))
        self._tasks[tid]["calendar_event_id"] = eid


class CalendarPullWorkerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.upcoming = os.path.join(self.tmp, "calendar", "upcoming.json")
        self.client = FakeCalendarClient()
        self.tz = "Asia/Tokyo"
        self.now = datetime(2026, 7, 4, 9, 0, tzinfo=timezone(timedelta(hours=9)))

    def _ev(self, title, start, end="", desc="", eid="e1", loc=""):
        return CalendarEvent(
            title=title, start=start, end=end, description=desc, event_id=eid, location=loc
        )

    def test_marker_events_excluded_from_upcoming(self):
        self.client.range_events = [
            self._ev("外部予定", "2026-07-04T14:00:00+09:00", eid="ext1"),
            self._ev("自作", "2026-07-04T15:00:00+09:00", desc="subpc-task:5", eid="own1"),
        ]
        worker = CalendarPullWorker(
            calendar_client=self.client,
            profile=None,
            store=None,
            timezone=self.tz,
            upcoming_path=self.upcoming,
        )
        worker.run_once(now=self.now)
        with open(self.upcoming, encoding="utf-8") as f:
            data = json.load(f)
        titles = [e["title"] for e in data["events"]]
        self.assertIn("外部予定", titles)
        self.assertNotIn("自作", titles)

    def test_profile_schedule_washes_gcal_keeps_manual(self):
        profile = ProfileStub(
            schedule=[
                {"date": "2026-07-04", "time": "10:00", "title": "手動", "note": "手で追加"},
                {"date": "2026-07-04", "time": "08:00", "title": "古いgcal", "note": "gcal:old"},
            ]
        )
        self.client.range_events = [
            self._ev("今日の会議", "2026-07-04T14:00:00+09:00", eid="g1"),
            self._ev("明日の予定", "2026-07-05T09:00:00+09:00", eid="g2"),
            self._ev("来週", "2026-07-10T09:00:00+09:00", eid="g3"),  # 対象外 (今日/明日でない)
        ]
        worker = CalendarPullWorker(
            calendar_client=self.client,
            profile=profile,
            store=None,
            timezone=self.tz,
            upcoming_path=self.upcoming,
        )
        worker.run_once(now=self.now)
        sched = profile.schedule
        notes = [s["note"] for s in sched]
        titles = [s["title"] for s in sched]
        # 手動エントリは残る
        self.assertIn("手動", titles)
        # 古い gcal は消える
        self.assertNotIn("gcal:old", notes)
        # 今日・明日の gcal が入る、来週は入らない
        self.assertIn("今日の会議", titles)
        self.assertIn("明日の予定", titles)
        self.assertNotIn("来週", titles)
        # gcal エントリは note に event_id
        g1 = next(s for s in sched if s["title"] == "今日の会議")
        self.assertEqual(g1["note"], "gcal:g1")
        self.assertEqual(g1["time"], "14:00")

    def test_reconcile_backfills_missing_mapping(self):
        store = StoreStub({7: {"id": 7, "status": "open", "calendar_event_id": None}})
        self.client.range_events = [
            self._ev("自作", "2026-07-04T15:00:00+09:00", desc="subpc-task:7\nメモ", eid="own7"),
        ]
        worker = CalendarPullWorker(
            calendar_client=self.client,
            profile=None,
            store=store,
            timezone=self.tz,
            upcoming_path=self.upcoming,
        )
        worker.run_once(now=self.now)
        self.assertEqual(store.set_calls, [(7, "own7")])

    def _reconcile_worker(self, store):
        return CalendarPullWorker(
            calendar_client=self.client,
            profile=None,
            store=store,
            timezone=self.tz,
            upcoming_path=self.upcoming,
        )

    def test_lost_terminal_queue_recovery_for_done_task(self):
        # done 済みだが mapping が失われ、open サマリのマーカーが残っている
        store = make_store()
        tid = store.add(
            "掃除",
            due_at=datetime(2026, 7, 4, 10, 0, tzinfo=UTC),
            due_granularity="datetime",
        )
        store.done(tid)
        self.client.range_events = [
            self._ev(
                "📋 掃除",
                "2026-07-04T09:30:00+09:00",
                desc=f"subpc-task:{tid}",
                eid="lost1",
            ),
        ]
        self._reconcile_worker(store).run_once(now=self.now)
        # マーカーは保持され、完了サマリへ更新、対応付けが復元される
        updates = [c for c in self.client.calls if c[0] == "update"]
        self.assertEqual(len(updates), 1)
        self.assertTrue(updates[0][2]["summary"].startswith("✅"))
        self.assertEqual(store.get(tid)["calendar_event_id"], "lost1")
        # マーカーは upcoming へ取り込まれない
        with open(self.upcoming, encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["events"], [])

    def test_missing_task_orphan_marker_deleted(self):
        store = make_store()
        self.client.range_events = [
            self._ev("残骸", "2026-07-04T10:00:00+09:00", desc="subpc-task:99999", eid="orphan1"),
        ]
        self._reconcile_worker(store).run_once(now=self.now)
        self.assertEqual([c[1] for c in self.client.calls if c[0] == "delete"], ["orphan1"])

    def test_dropped_task_orphan_marker_deleted_and_cleared(self):
        store = make_store()
        tid = store.add(
            "破棄",
            due_at=datetime(2026, 7, 4, 10, 0, tzinfo=UTC),
            due_granularity="datetime",
        )
        store.set_calendar_event(tid, "drop-orphan")
        store.drop(tid)
        self.client.range_events = [
            self._ev("📋 破棄", "2026-07-04T09:30:00+09:00", desc=f"subpc-task:{tid}", eid="drop-orphan"),
        ]
        self._reconcile_worker(store).run_once(now=self.now)
        self.assertEqual([c[1] for c in self.client.calls if c[0] == "delete"], ["drop-orphan"])
        self.assertIsNone(store.get(tid)["calendar_event_id"])

    def test_duplicate_markers_converge_to_stored_mapping(self):
        # 二つの同期インスタンスが二重作成した想定。対応付け済み id を正準に残す
        store = make_store()
        tid = store.add(
            "二重",
            due_at=datetime(2026, 7, 4, 10, 0, tzinfo=UTC),
            due_granularity="datetime",
        )
        store.set_calendar_event(tid, "evtA")
        self.client.range_events = [
            self._ev("📋 二重", "2026-07-04T09:30:00+09:00", desc=f"subpc-task:{tid}", eid="evtB"),
            self._ev("📋 二重", "2026-07-04T10:00:00+09:00", desc=f"subpc-task:{tid}", eid="evtA"),
        ]
        self._reconcile_worker(store).run_once(now=self.now)
        self.assertEqual([c[1] for c in self.client.calls if c[0] == "delete"], ["evtB"])
        self.assertEqual(store.get(tid)["calendar_event_id"], "evtA")

    def test_duplicate_markers_backfill_deterministic_when_no_mapping(self):
        # mapping が無い場合も (start, event_id) 辞書順最小へ決定的に収束する
        store = make_store()
        tid = store.add(
            "二重2",
            due_at=datetime(2026, 7, 4, 10, 0, tzinfo=UTC),
            due_granularity="datetime",
        )
        self.client.range_events = [
            self._ev("📋 二重2", "2026-07-04T10:00:00+09:00", desc=f"subpc-task:{tid}", eid="z-2"),
            self._ev("📋 二重2", "2026-07-04T09:00:00+09:00", desc=f"subpc-task:{tid}", eid="a-1"),
        ]
        self._reconcile_worker(store).run_once(now=self.now)
        self.assertEqual([c[1] for c in self.client.calls if c[0] == "delete"], ["z-2"])
        self.assertEqual(store.get(tid)["calendar_event_id"], "a-1")

    def test_reconcile_error_does_not_abort_other_markers(self):
        # 片方の削除が失敗しても、残りのマーカー整理は続行される (best-effort)
        store = make_store()
        self.client.delete_raise_ids = {"orph-x"}
        self.client.range_events = [
            self._ev("残骸1", "2026-07-04T10:00:00+09:00", desc="subpc-task:998", eid="orph-x"),
            self._ev("残骸2", "2026-07-04T11:00:00+09:00", desc="subpc-task:999", eid="orph-y"),
        ]
        ok = self._reconcile_worker(store).run_once(now=self.now)
        self.assertTrue(ok)
        deletes = [c[1] for c in self.client.calls if c[0] == "delete"]
        self.assertIn("orph-y", deletes)  # 失敗を挟んでも他は処理される

    def test_write_upcoming_atomic_and_shaped(self):
        self.client.range_events = [self._ev("A", "2026-07-04T14:00:00+09:00", eid="a")]
        worker = CalendarPullWorker(
            calendar_client=self.client, profile=None, store=None,
            timezone=self.tz, upcoming_path=self.upcoming,
        )
        worker.run_once(now=self.now)
        self.assertTrue(os.path.exists(self.upcoming))
        self.assertFalse(os.path.exists(self.upcoming + ".tmp"))
        with open(self.upcoming, encoding="utf-8") as f:
            data = json.load(f)
        self.assertIn("generated_at", data)
        self.assertEqual(data["events"][0]["event_id"], "a")


class CalendarContextTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp, "upcoming.json")
        self.tz = "Asia/Tokyo"
        self.now = datetime(2026, 7, 4, 9, 0, tzinfo=timezone(timedelta(hours=9)))

    def _write(self, generated_at, events):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"generated_at": generated_at, "events": events}, f)

    def test_missing_file_returns_empty(self):
        ctx = CalendarContext(upcoming_path=os.path.join(self.tmp, "nope.json"), timezone=self.tz)
        self.assertEqual(ctx.get_context_text(now=self.now), "")

    def test_formats_today_and_tomorrow(self):
        self._write(
            self.now.isoformat(),
            [
                {"title": "会議", "start": "2026-07-04T14:00:00+09:00"},
                {"title": "歯医者", "start": "2026-07-05T10:30:00+09:00"},
                {"title": "終日イベント", "start": "2026-07-04"},
                {"title": "来週", "start": "2026-07-12T10:00:00+09:00"},
            ],
        )
        ctx = CalendarContext(upcoming_path=self.path, timezone=self.tz)
        text = ctx.get_context_text(now=self.now)
        self.assertIn("--- 予定 (Google Calendar) ---", text)
        self.assertIn("今日 14:00 会議", text)
        self.assertIn("明日 10:30 歯医者", text)
        self.assertIn("今日 終日 終日イベント", text)
        self.assertNotIn("来週", text)

    def test_stale_over_24h_returns_empty(self):
        old = (self.now - timedelta(hours=30)).isoformat()
        self._write(old, [{"title": "会議", "start": "2026-07-04T14:00:00+09:00"}])
        ctx = CalendarContext(upcoming_path=self.path, timezone=self.tz)
        self.assertEqual(ctx.get_context_text(now=self.now), "")

    def test_older_than_10min_but_within_24h_still_shows(self):
        recent = (self.now - timedelta(minutes=45)).isoformat()
        self._write(recent, [{"title": "会議", "start": "2026-07-04T14:00:00+09:00"}])
        ctx = CalendarContext(upcoming_path=self.path, timezone=self.tz)
        self.assertIn("会議", ctx.get_context_text(now=self.now))

    def test_max_items_capped(self):
        evs = [
            {"title": f"予定{i}", "start": f"2026-07-04T{10 + i:02d}:00:00+09:00"}
            for i in range(10)
        ]
        self._write(self.now.isoformat(), evs)
        ctx = CalendarContext(upcoming_path=self.path, timezone=self.tz, max_items=3)
        text = ctx.get_context_text(now=self.now)
        # ヘッダ + 3 行
        self.assertEqual(len(text.strip().splitlines()), 4)


if __name__ == "__main__":
    unittest.main()
