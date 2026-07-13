from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from src.growth.tracker import GrowthTracker
from src.web import server


class WebGameTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.tracker = GrowthTracker(Path(self.tmp.name) / "growth.db")
        self.now = datetime(2026, 7, 14, 12, 0, tzinfo=ZoneInfo("Asia/Tokyo"))
        self.original_tracker = server.growth_tracker
        self.original_timezone = server.tasks_timezone
        server.growth_tracker = self.tracker
        server.tasks_timezone = "Asia/Tokyo"

    def tearDown(self) -> None:
        server.growth_tracker = self.original_tracker
        server.tasks_timezone = self.original_timezone
        self.tmp.cleanup()

    def test_missions_are_fixed_and_derived_from_todays_conversation(self) -> None:
        self.tracker.record_conversation(
            source="web",
            session_id="game",
            user_chars=240,
            assistant_chars=380,
            memory_saved=True,
            event_key="conversation:game",
            now=self.now,
        )

        state = server._game_state(now=self.now, asset_counts={})
        missions = {mission["id"]: mission for mission in state["missions"]}
        self.assertTrue(state["enabled"])
        self.assertTrue(missions["first_turn"]["complete"])
        self.assertTrue(missions["deep_talk"]["complete"])
        self.assertFalse(missions["three_turns"]["complete"])
        self.assertEqual(state["claimable_missions"], 2)
        self.assertEqual(len(state["starters"]), 5)
        self.assertIn("tasks", {starter["id"] for starter in state["starters"]})
        talk_badge = next(badge for badge in state["badges"] if badge["id"] == "talk10")
        self.assertEqual(talk_badge["current"], 1)
        self.assertEqual(talk_badge["target"], 10)
        self.assertEqual(talk_badge["unit"], "往復")

    def test_reward_can_only_be_claimed_once(self) -> None:
        self.tracker.record_conversation(
            source="web",
            session_id="game",
            user_chars=10,
            assistant_chars=20,
            memory_saved=False,
            event_key="conversation:one",
            now=self.now,
        )

        first = server._claim_game_mission(
            "first_turn", now=self.now, asset_counts={}
        )
        second = server._claim_game_mission(
            "first_turn", now=self.now, asset_counts={}
        )

        self.assertTrue(first["ok"])
        self.assertTrue(first["claimed_now"])
        self.assertEqual(first["reward"], 10)
        self.assertTrue(second["ok"])
        self.assertFalse(second["claimed_now"])
        self.assertEqual(second["reward"], 0)
        summary = self.tracker.summary(now=self.now)
        self.assertEqual(summary["tracked_points"], 20)
        self.assertEqual(summary["signals"]["quest_reward"], 1)

    def test_incomplete_and_unknown_missions_are_rejected(self) -> None:
        incomplete = server._claim_game_mission(
            "three_turns", now=self.now, asset_counts={}
        )
        unknown = server._claim_game_mission(
            "does-not-exist", now=self.now, asset_counts={}
        )
        self.assertEqual(incomplete["status"], 409)
        self.assertEqual(unknown["status"], 404)

    def test_static_ui_exposes_game_routes_and_controls(self) -> None:
        root = Path(__file__).resolve().parents[1]
        html = (root / "src/web/static/index.html").read_text(encoding="utf-8")
        js = (root / "src/web/static/app.js").read_text(encoding="utf-8")
        self.assertIn('id="game-hub"', html)
        self.assertIn('id="game-mission-list"', html)
        self.assertIn('id="game-starter-list"', html)
        self.assertIn("fetch('/api/game'", js)
        self.assertIn("fetch('/api/game/claim'", js)
        achievements_html = (root / "src/web/static/achievements.html").read_text(encoding="utf-8")
        achievements_js = (root / "src/web/static/achievements.js").read_text(encoding="utf-8")
        self.assertIn('href="/achievements"', html)
        self.assertIn('id="achievement-grid"', achievements_html)
        self.assertIn("fetch('/api/game'", achievements_js)


if __name__ == "__main__":
    unittest.main()
