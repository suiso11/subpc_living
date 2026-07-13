from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.persona.conversation_loop import ConversationLoopStore


class ConversationLoopStoreTest(unittest.TestCase):
    def _store(self, path=None, **kwargs) -> ConversationLoopStore:
        return ConversationLoopStore(
            path,
            base_interval_sec=kwargs.get("base_interval_sec", 3600),
            reply_timeout_sec=kwargs.get("reply_timeout_sec", 600),
            daily_limit=kwargs.get("daily_limit", 1),
            max_backoff_sec=kwargs.get("max_backoff_sec", 8 * 3600),
        )

    def test_pending_prompt_persists_and_is_consumed_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            first = self._store(path)
            first.record_prompt(10, "好きな音楽は？", now=1000)

            second = self._store(path)
            prompt = second.consume_reply(10, now=1100)

            self.assertEqual(prompt, "好きな音楽は？")
            self.assertIsNone(second.consume_reply(10, now=1101))

    def test_daily_limit_blocks_second_prompt(self) -> None:
        store = self._store(daily_limit=1)
        store.record_prompt(10, "q1", now=1000)
        store.consume_reply(10, now=1010)

        self.assertFalse(store.can_prompt(10, now=1000 + 2 * 3600))
        self.assertTrue(store.can_prompt(10, now=1000 + 25 * 3600))

    def test_ignored_prompt_doubles_interval(self) -> None:
        store = self._store(reply_timeout_sec=600)
        store.record_prompt(10, "q1", now=1000)

        self.assertFalse(store.can_prompt(10, now=1601))
        status = store.status(10)
        self.assertEqual(status["ignored_streak"], 1)
        self.assertEqual(status["next_interval_hours"], 2.0)
        self.assertTrue(store.can_prompt(10, now=1000 + 25 * 3600))

    def test_reply_resets_ignored_streak_without_storing_reply_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            store = self._store(path)
            store.record_prompt(10, "q1", now=1000)
            store.consume_reply(10, now=1100)

            raw = path.read_text(encoding="utf-8")
            state = store.status(10)

            self.assertEqual(state["ignored_streak"], 0)
            self.assertEqual(state["reply_count"], 1)
            self.assertNotIn("user_reply", raw)

    def test_pause_survives_reload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            store = self._store(path)
            store.pause(10)

            loaded = self._store(path)

            self.assertFalse(loaded.can_prompt(10, now=100000))
            loaded.resume(10)
            self.assertTrue(loaded.can_prompt(10, now=100000))

    def test_corrupt_state_does_not_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            path.write_text("{broken", encoding="utf-8")

            store = self._store(path)

            self.assertTrue(store.last_error)
            self.assertTrue(store.can_prompt(10, now=1000))


if __name__ == "__main__":
    unittest.main()
