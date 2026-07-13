from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.tasks.prioritizer import (
    PriorityController,
    build_priority_context,
    rank_tasks,
)
from src.tasks.store import TaskStore, build_task_context

UTC = timezone.utc


class PriorityControllerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.store = TaskStore(str(root / "tasks.db")).initialize()
        self.state_path = root / "priority.json"
        self.upcoming_path = root / "upcoming.json"
        self.now = datetime(2026, 7, 14, 3, 0, tzinfo=UTC)  # JST 12:00
        self.controller = PriorityController(
            self.store,
            state_path=self.state_path,
            upcoming_path=self.upcoming_path,
            skip_hours=2,
            calendar_buffer_min=10,
        )

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_overdue_beats_high_priority_without_due(self) -> None:
        high = self.store.add("重要だが期限なし", priority="high", now=self.now)
        overdue = self.store.add(
            "期限超過",
            due_at=self.now - timedelta(hours=1),
            due_granularity="datetime",
            now=self.now,
        )
        ranked = rank_tasks(self.store.list("open"), now=self.now)
        self.assertEqual(ranked[0].task["id"], overdue)
        self.assertNotEqual(ranked[0].task["id"], high)
        self.assertTrue(any("超過" in reason for reason in ranked[0].reasons))

    def test_recommend_is_fixed_until_done_or_next(self) -> None:
        first = self.store.add(
            "近い", due_at=self.now + timedelta(hours=1),
            due_granularity="datetime", now=self.now,
        )
        self.store.add("別件", priority="high", now=self.now)
        chosen = self.controller.recommend(now=self.now)
        self.assertEqual(chosen.task["id"], first)

        # より緊急なタスクが後から来ても、現在の決定は勝手に切り替えない。
        self.store.add(
            "さらに緊急", due_at=self.now - timedelta(minutes=5),
            due_granularity="datetime", now=self.now,
        )
        same = self.controller.recommend(now=self.now + timedelta(minutes=1))
        self.assertEqual(same.task["id"], first)
        self.assertTrue(same.current)

    def test_next_defers_and_persists_feedback(self) -> None:
        first = self.store.add("一件目", priority="high", now=self.now)
        second = self.store.add("二件目", priority="normal", now=self.now)
        self.assertEqual(self.controller.recommend(now=self.now).task["id"], first)
        next_decision = self.controller.next(now=self.now)
        self.assertEqual(next_decision.task["id"], second)

        persisted = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.assertEqual(persisted["feedback"][str(first)]["skip_count"], 1)
        self.assertIn(str(first), persisted["deferred_until"])

        restored = PriorityController(
            self.store, state_path=self.state_path, upcoming_path=self.upcoming_path
        )
        self.assertEqual(restored.recommend(now=self.now).task["id"], second)

    def test_start_complete_selects_next_and_updates_momentum(self) -> None:
        first = self.store.add("一件目", priority="high", now=self.now)
        second = self.store.add("二件目", now=self.now)
        started = self.controller.start(now=self.now)
        self.assertTrue(started.started)
        self.assertEqual(started.task["id"], first)

        completed, following = self.controller.complete(now=self.now + timedelta(minutes=20))
        self.assertTrue(completed)
        self.assertEqual(self.store.get(first)["status"], "done")
        self.assertEqual(following.task["id"], second)
        self.assertEqual(following.completed_today, 1)
        self.assertEqual(following.streak_days, 1)

    def test_pick_overrides_score(self) -> None:
        urgent = self.store.add(
            "緊急", due_at=self.now + timedelta(minutes=30),
            due_granularity="datetime", now=self.now,
        )
        manual = self.store.add("今日はこれ", priority="low", now=self.now)
        self.assertEqual(self.controller.recommend(now=self.now).task["id"], urgent)
        picked = self.controller.pick(manual, now=self.now)
        self.assertEqual(picked.task["id"], manual)
        self.assertEqual(self.controller.recommend(now=self.now).task["id"], manual)

    def test_focus_window_uses_next_non_task_calendar_event(self) -> None:
        self.store.add("作業", now=self.now)
        self.upcoming_path.write_text(
            json.dumps({
                "events": [
                    {
                        "title": "タスク期限",
                        "start": (self.now + timedelta(minutes=20)).isoformat(),
                        "description": "subpc-task:1",
                    },
                    {
                        "title": "会議",
                        "start": (self.now + timedelta(minutes=35)).isoformat(),
                        "description": "",
                    },
                ]
            }),
            encoding="utf-8",
        )
        decision = self.controller.recommend(now=self.now)
        self.assertEqual(decision.focus_minutes, 25)
        self.assertEqual(decision.next_event_title, "会議")

    def test_corrupt_state_falls_back_without_crashing(self) -> None:
        self.store.add("作業", now=self.now)
        self.state_path.write_text("{broken", encoding="utf-8")
        controller = PriorityController(self.store, state_path=self.state_path)
        self.assertIsNotNone(controller.recommend(now=self.now, select=False))
        self.assertIsNotNone(controller.last_error)

    def test_all_chat_context_contains_same_recommendation(self) -> None:
        task_id = self.store.add("通常・期限なし", now=self.now)
        text = build_priority_context(
            self.store, now=self.now, state_path=self.state_path,
            upcoming_path=self.upcoming_path,
        )
        self.assertIn(f"#{task_id} 通常・期限なし", text)
        # 従来は context 対象外だった通常・期限なしだけでも、推奨は注入される。
        context = build_task_context(self.store, now=self.now)
        self.assertIn("優先順位オーケストレーター", context)
        self.assertIn("通常・期限なし", context)


if __name__ == "__main__":
    unittest.main()
