from __future__ import annotations

import asyncio
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from src.growth.tracker import GrowthTracker
from src.web import server


class WebGrowthTest(unittest.TestCase):
    def test_growth_api_exposes_honest_metric_and_asset_breakdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            original = server.growth_tracker
            try:
                tracker = GrowthTracker(Path(tmp) / "growth.db")
                tracker.record_conversation(
                    source="web", session_id="s", user_chars=2,
                    assistant_chars=3, memory_saved=True,
                    now=datetime(2026, 7, 14, tzinfo=timezone.utc),
                )
                server.growth_tracker = tracker
                assets = {
                    "retrievable_memories": 3,
                    "knowledge_items": 0,
                    "training_turns": 0,
                    "feedback_items": 0,
                    "correction_candidates": 1,
                    "profile_facts": 0,
                    "conversation_summaries": 0,
                }
                with patch.object(server, "_growth_asset_counts", return_value=assets):
                    result = asyncio.run(server.growth_summary(days=7))
            finally:
                server.growth_tracker = original

        self.assertTrue(result["enabled"])
        self.assertIn("モデル重み", result["metric_note"])
        self.assertEqual(result["tracked_points"], 20)
        self.assertEqual(result["asset_points"], 31)
        self.assertEqual(result["growth_points"], 51)
        self.assertEqual(len(result["daily"]), 7)

    def test_chat_page_contains_live_growth_panel(self) -> None:
        root = Path(server.PROJECT_ROOT)
        html = (root / "src/web/static/index.html").read_text(encoding="utf-8")
        js = (root / "src/web/static/app.js").read_text(encoding="utf-8")
        self.assertIn('id="growth-panel"', html)
        self.assertIn("/api/growth?days=14", js)
        self.assertIn("loadGrowth({ animate: true })", js)


if __name__ == "__main__":
    unittest.main()
