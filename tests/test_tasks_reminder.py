from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.tasks.reminder import (
    TaskReminderEngine,
    compute_action,
    in_quiet_hours,
    parse_quiet_hours,
)
from src.tasks.store import TaskStore

UTC = timezone.utc


def _engine(store, fired, *, quiet=(0, 0), now_fn=None):
    return TaskReminderEngine(
        store,
        lambda **k: fired.append(k),
        owner="test",
        timezone_name="UTC",
        quiet_hours=quiet,
        now_fn=now_fn or (lambda: datetime.now(UTC)),
    )


class QuietHoursTest(unittest.TestCase):
    def test_parse(self) -> None:
        self.assertEqual(parse_quiet_hours("1-8"), (1, 8))
        self.assertEqual(parse_quiet_hours("22-6"), (22, 6))
        self.assertEqual(parse_quiet_hours(""), (1, 8))
        self.assertEqual(parse_quiet_hours("garbage"), (1, 8))

    def test_in_quiet(self) -> None:
        base = datetime(2026, 1, 1, tzinfo=UTC)
        self.assertTrue(in_quiet_hours(base.replace(hour=3), (1, 8)))
        self.assertFalse(in_quiet_hours(base.replace(hour=9), (1, 8)))
        # 折り返し
        self.assertTrue(in_quiet_hours(base.replace(hour=23), (22, 6)))
        self.assertTrue(in_quiet_hours(base.replace(hour=2), (22, 6)))
        self.assertFalse(in_quiet_hours(base.replace(hour=12), (22, 6)))


class ComputeActionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)

    def test_before_24h_no_fire(self) -> None:
        due = self.now + timedelta(hours=30)
        fire, stage, nxt, _ = compute_action(due, self.now, None, 0)
        self.assertFalse(fire)
        self.assertEqual(nxt, due - timedelta(hours=24))

    def test_24h_fires_once(self) -> None:
        due = self.now + timedelta(hours=10)
        fire, stage, _, _ = compute_action(due, self.now, None, 0)
        self.assertTrue(fire)
        self.assertEqual(stage, "24h")
        # 既に24h済みなら再発火しない
        fire2, _, _, _ = compute_action(due, self.now, "24h", 1)
        self.assertFalse(fire2)

    def test_3h_skips_missed_24h(self) -> None:
        due = self.now + timedelta(hours=2)
        fire, stage, _, _ = compute_action(due, self.now, None, 0)
        self.assertTrue(fire)
        self.assertEqual(stage, "3h")

    def test_overdue_repeats(self) -> None:
        due = self.now - timedelta(hours=1)
        fire, stage, nxt, count = compute_action(due, self.now, "overdue", 2)
        self.assertTrue(fire)
        self.assertEqual(stage, "overdue")
        self.assertEqual(count, 3)
        self.assertEqual(nxt, self.now + timedelta(hours=2))


class ReminderEngineTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self._tmp.name) / "tasks.db")
        self.store = TaskStore(db_path=self.db_path, timezone_name="UTC").initialize()

    def tearDown(self) -> None:
        self.store.close()
        self._tmp.cleanup()

    def test_stage_progression(self) -> None:
        now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        due = now + timedelta(hours=2)  # 3h window
        tid = self.store.add("t", due_at=due, due_granularity="datetime", now=now)
        fired: list = []
        eng = _engine(self.store, fired)

        self.assertEqual(eng.run_once(now), 1)
        self.assertEqual(fired[-1]["stage"], "3h")
        self.assertEqual(fired[-1]["task_id"], tid)
        # 直後の再評価では発火しない
        self.assertEqual(eng.run_once(now), 0)

        # 1h窓
        now2 = due - timedelta(minutes=30)
        self.assertEqual(eng.run_once(now2), 1)
        self.assertEqual(fired[-1]["stage"], "1h")
        self.assertEqual(eng.run_once(now2), 0)  # 30分経っていない

        # 超過
        now3 = due + timedelta(minutes=1)
        self.assertEqual(eng.run_once(now3), 1)
        self.assertEqual(fired[-1]["stage"], "overdue")

    def test_message_contains_title_and_hint(self) -> None:
        now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        self.store.add("レポート提出", due_at=now + timedelta(hours=2),
                       due_granularity="datetime", action_hint="序論を書く", now=now)
        fired: list = []
        eng = _engine(self.store, fired)
        eng.run_once(now)
        msg = fired[-1]["message"]
        self.assertIn("レポート提出", msg)
        self.assertIn("あと", msg)
        self.assertIn("序論を書く", msg)

    def test_quiet_hours_defer_and_carry(self) -> None:
        now = datetime(2026, 1, 1, 2, 0, tzinfo=UTC)  # quiet 1-8
        due = now + timedelta(hours=8)  # 24h window (>3h)
        self.store.add("t", due_at=due, due_granularity="datetime", now=now)
        fired: list = []
        eng = _engine(self.store, fired, quiet=(1, 8))

        # quiet 中は発火しない
        self.assertEqual(eng.run_once(now), 0)
        self.assertEqual(fired, [])
        # まだ quiet 中 (3:00)
        self.assertEqual(eng.run_once(now + timedelta(hours=1)), 0)
        # quiet 明け (10:00) に繰り越し発火。この時点 remaining=... due=10:00 -> ちょうど。少し前に。
        out = datetime(2026, 1, 1, 8, 30, tzinfo=UTC)  # remaining 1.5h -> 3h window
        self.assertEqual(eng.run_once(out), 1)
        self.assertIn(fired[-1]["stage"], ("3h", "24h"))

    def test_overdue_fires_during_quiet(self) -> None:
        now = datetime(2026, 1, 1, 3, 0, tzinfo=UTC)  # quiet
        due = now - timedelta(hours=1)  # overdue
        self.store.add("t", due_at=due, due_granularity="datetime", now=now)
        fired: list = []
        eng = _engine(self.store, fired, quiet=(1, 8))
        self.assertEqual(eng.run_once(now), 1)
        self.assertEqual(fired[-1]["stage"], "overdue")

    def test_restart_does_not_duplicate(self) -> None:
        now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        due = now + timedelta(hours=2)
        self.store.add("t", due_at=due, due_granularity="datetime", now=now)
        fired: list = []
        eng1 = _engine(self.store, fired)
        self.assertEqual(eng1.run_once(now), 1)

        # 別インスタンス(=再起動相当)。永続化された状態を読むので再発火しない。
        fired2: list = []
        eng2 = _engine(self.store, fired2)
        self.assertEqual(eng2.run_once(now), 0)
        self.assertEqual(fired2, [])


if __name__ == "__main__":
    unittest.main()
