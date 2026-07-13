from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.growth.tracker import GrowthTracker

UTC = timezone.utc


class GrowthTrackerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "growth.db"
        self.tracker = GrowthTracker(self.db)
        self.now = datetime(2026, 7, 14, 3, 0, tzinfo=UTC)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_conversation_points_and_memory_bonus(self) -> None:
        self.tracker.record_conversation(
            source="web", session_id="s1", user_chars=10,
            assistant_chars=20, memory_saved=True, now=self.now,
        )
        self.tracker.record_conversation(
            source="discord", session_id="s2", user_chars=5,
            assistant_chars=8, memory_saved=False, now=self.now,
        )
        summary = self.tracker.summary(now=self.now)
        self.assertEqual(summary["growth_points"], 30)
        self.assertEqual(summary["total_turns"], 2)
        self.assertEqual(summary["memory_turns"], 1)

    def test_signals_and_duplicate_event_key(self) -> None:
        self.assertTrue(self.tracker.record_signal(
            kind="feedback", source="discord", event_key="feedback:1", now=self.now
        ))
        self.assertFalse(self.tracker.record_signal(
            kind="feedback", source="discord", event_key="feedback:1", now=self.now
        ))
        self.tracker.record_signal(
            kind="correction", source="discord", event_key="correction:1", now=self.now
        )
        summary = self.tracker.summary(now=self.now)
        self.assertEqual(summary["growth_points"], 50)
        self.assertEqual(summary["signals"]["feedback"], 1)
        self.assertEqual(summary["signals"]["correction"], 1)

    def test_daily_streak_and_empty_days(self) -> None:
        for days_ago in (2, 1, 0):
            self.tracker.record_conversation(
                source="web", session_id=str(days_ago), user_chars=1,
                assistant_chars=1, memory_saved=False,
                now=self.now - timedelta(days=days_ago),
            )
        summary = self.tracker.summary(now=self.now, days=5)
        self.assertEqual(summary["streak_days"], 3)
        self.assertEqual(len(summary["daily"]), 5)
        self.assertEqual(sum(day["turns"] for day in summary["daily"]), 3)

    def test_level_boundaries(self) -> None:
        for i in range(10):
            self.tracker.record_conversation(
                source="web", session_id="s", user_chars=1, assistant_chars=1,
                memory_saved=False, event_key=f"turn:{i}", now=self.now,
            )
        summary = self.tracker.summary(now=self.now)
        self.assertEqual(summary["growth_points"], 100)
        self.assertEqual(summary["level"], 2)
        self.assertEqual(summary["level_progress"], 0)
        self.assertEqual(summary["next_level_points"], 400)

    def test_conversation_text_is_not_stored(self) -> None:
        secret = "本文を保存しない"
        self.tracker.record_conversation(
            source="web", session_id="s", user_chars=len(secret),
            assistant_chars=3, memory_saved=True, now=self.now,
        )
        raw = self.db.read_bytes()
        self.assertNotIn(secret.encode("utf-8"), raw)

    def test_two_instances_share_wal_database(self) -> None:
        other = GrowthTracker(self.db)
        self.tracker.record_conversation(
            source="web", session_id="a", user_chars=1,
            assistant_chars=1, memory_saved=False, now=self.now,
        )
        other.record_signal(
            kind="training_turn", source="discord", event_key="train:1", now=self.now
        )
        self.assertEqual(other.summary(now=self.now)["growth_points"], 15)

    def test_existing_assets_contribute_with_visible_breakdown(self) -> None:
        summary = self.tracker.summary(
            now=self.now,
            asset_counts={"retrievable_memories": 10, "profile_facts": 2},
        )
        self.assertEqual(summary["tracked_points"], 0)
        self.assertEqual(summary["asset_points"], 40)
        self.assertEqual(summary["growth_points"], 40)
        self.assertEqual(summary["asset_counts"]["retrievable_memories"], 10)

    def test_daily_game_metrics_and_existing_event_keys(self) -> None:
        self.tracker.record_conversation(
            source="web", session_id="game", user_chars=240,
            assistant_chars=380, memory_saved=True,
            event_key="conversation:game", now=self.now,
        )
        self.tracker.record_signal(
            kind="quest_reward", source="web_game",
            event_key="quest:2026-07-14:first", points=10, now=self.now,
        )
        summary = self.tracker.summary(now=self.now)
        self.assertEqual(summary["today_turns"], 1)
        self.assertEqual(summary["today_memory_turns"], 1)
        self.assertEqual(summary["today_chars"], 620)
        self.assertEqual(summary["signals"]["quest_reward"], 1)
        self.assertEqual(
            self.tracker.existing_event_keys([
                "quest:2026-07-14:first", "quest:2026-07-14:missing"
            ]),
            {"quest:2026-07-14:first"},
        )

    def test_unsupported_signal_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.tracker.record_signal(
                kind="fake", source="x", event_key="x", now=self.now
            )


if __name__ == "__main__":
    unittest.main()
