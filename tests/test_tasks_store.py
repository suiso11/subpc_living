from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.tasks.store import TaskStore, build_task_context

UTC = timezone.utc


class TaskStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self._tmp.name) / "tasks.db")
        self.store = TaskStore(db_path=self.db_path, timezone_name="Asia/Tokyo").initialize()
        self.now = datetime(2026, 7, 3, 3, 0, tzinfo=UTC)  # JST 12:00

    def tearDown(self) -> None:
        self.store.close()
        self._tmp.cleanup()

    def test_add_and_get(self) -> None:
        tid = self.store.add("レポート", note="n", action_hint="序論", priority="high", now=self.now)
        t = self.store.get(tid)
        self.assertEqual(t["title"], "レポート")
        self.assertEqual(t["priority"], "high")
        self.assertEqual(t["status"], "open")
        self.assertEqual(t["action_hint"], "序論")

    def test_list_orders_by_due(self) -> None:
        far = self.store.add("far", due_at=self.now + timedelta(days=5), due_granularity="datetime", now=self.now)
        near = self.store.add("near", due_at=self.now + timedelta(hours=2), due_granularity="datetime", now=self.now)
        nodue = self.store.add("nodue", now=self.now)
        ids = [t["id"] for t in self.store.list("open")]
        # 期限あり(近い順)が先、期限なしが最後
        self.assertEqual(ids, [near, far, nodue])

    def test_done_and_drop(self) -> None:
        tid = self.store.add("t", now=self.now)
        self.assertTrue(self.store.done(tid, now=self.now))
        self.assertFalse(self.store.done(tid, now=self.now))  # already done
        self.assertEqual(self.store.get(tid)["status"], "done")
        self.assertIsNotNone(self.store.get(tid)["completed_at"])

        tid2 = self.store.add("t2", now=self.now)
        self.assertTrue(self.store.drop(tid2, now=self.now))
        self.assertEqual(self.store.get(tid2)["status"], "dropped")

    def test_snooze_blocks_claim(self) -> None:
        tid = self.store.add("t", due_at=self.now + timedelta(hours=2), due_granularity="datetime", now=self.now)
        until = self.now + timedelta(hours=1)
        self.assertTrue(self.store.snooze(tid, until, now=self.now))
        # snooze 中は claim されない
        claimed = self.store.claim_due_notifications("o1", self.now)
        self.assertEqual(claimed, [])
        # snooze 明けは claim される
        claimed2 = self.store.claim_due_notifications("o1", until + timedelta(minutes=1))
        self.assertEqual([c["id"] for c in claimed2], [tid])

    def test_get_context_tasks_priority_order(self) -> None:
        overdue = self.store.add("overdue", due_at=self.now - timedelta(hours=1), due_granularity="datetime", now=self.now)
        today = self.store.add("today", due_at=self.now + timedelta(hours=5), due_granularity="datetime", now=self.now)
        soon = self.store.add("soon", due_at=self.now + timedelta(days=2), due_granularity="datetime", now=self.now)
        highp = self.store.add("high", priority="high", now=self.now)
        self.store.add("low", priority="low", now=self.now)  # 除外される

        ctx = self.store.get_context_tasks(limit=8, now=self.now)
        ids = [t["id"] for t in ctx]
        self.assertEqual(ids, [overdue, today, soon, highp])

    def test_get_context_tasks_excludes_done(self) -> None:
        tid = self.store.add("done-task", due_at=self.now - timedelta(hours=1), due_granularity="datetime", now=self.now)
        self.store.done(tid, now=self.now)
        ctx = self.store.get_context_tasks(now=self.now)
        self.assertEqual(ctx, [])

    def test_build_task_context_text(self) -> None:
        self.store.add("レポート", due_at=self.now - timedelta(hours=1), due_granularity="datetime",
                       action_hint="序論を書く", now=self.now)
        text = build_task_context(self.store, now=self.now)
        self.assertIn("--- 未完了タスク ---", text)
        self.assertIn("レポート", text)
        self.assertIn("期限超過", text)
        self.assertIn("次の一手: 序論を書く", text)

    def test_build_task_context_empty(self) -> None:
        self.assertEqual(build_task_context(self.store, now=self.now), "")

    def test_claim_lease_is_exclusive(self) -> None:
        """2接続で同時にclaimしても片方だけが取れる (lease排他)。"""
        store2 = TaskStore(db_path=self.db_path, timezone_name="Asia/Tokyo").initialize()
        try:
            tid = self.store.add("t", due_at=self.now + timedelta(hours=2),
                                 due_granularity="datetime", now=self.now)
            c1 = self.store.claim_due_notifications("owner1", self.now)
            c2 = store2.claim_due_notifications("owner2", self.now)
            self.assertEqual([c["id"] for c in c1], [tid])
            self.assertEqual(c2, [])  # lease を owner1 が保持

            # lease 期限切れ後は owner2 も取れる
            later = self.now + timedelta(seconds=200)
            c3 = store2.claim_due_notifications("owner2", later)
            self.assertEqual([c["id"] for c in c3], [tid])
        finally:
            store2.close()

    def test_claim_skips_tasks_without_due(self) -> None:
        self.store.add("no-due", now=self.now)
        self.assertEqual(self.store.claim_due_notifications("o", self.now), [])


if __name__ == "__main__":
    unittest.main()
