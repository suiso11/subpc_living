from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.chat.session import ChatSession


class FakeRAG:
    def store_turn(self, **kwargs):
        return "memory-1"


class FakeGrowthTracker:
    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.calls = []

    def record_conversation(self, **kwargs):
        if self.fail:
            raise RuntimeError("measurement failed")
        self.calls.append(kwargs)
        return True


class ChatGrowthTest(unittest.TestCase):
    def test_successful_turn_records_source_sizes_and_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tracker = FakeGrowthTracker()
            session = ChatSession(
                history_dir=tmp,
                rag=FakeRAG(),
                growth_tracker=tracker,
                conversation_source="web",
            )
            session.session_id = "web_keep"
            session.add_user_message("こんにちは")
            session.add_assistant_message("やあ")

            self.assertEqual(len(tracker.calls), 1)
            event = tracker.calls[0]
            self.assertEqual(event["source"], "web")
            self.assertEqual(event["session_id"], "web_keep")
            self.assertEqual(event["user_chars"], 5)
            self.assertEqual(event["assistant_chars"], 2)
            self.assertTrue(event["memory_saved"])

    def test_measurement_failure_does_not_break_conversation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            session = ChatSession(
                history_dir=tmp,
                growth_tracker=FakeGrowthTracker(fail=True),
            )
            session.add_user_message("続ける")
            session.add_assistant_message("続けられます")
            self.assertEqual(session.messages[-1]["content"], "続けられます")

    def test_assistant_without_user_is_not_counted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tracker = FakeGrowthTracker()
            session = ChatSession(history_dir=tmp, growth_tracker=tracker)
            session.add_assistant_message("先行メッセージ")
            self.assertEqual(tracker.calls, [])


if __name__ == "__main__":
    unittest.main()
