from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from fastapi.responses import JSONResponse

from src.web import server


class WebChatResumeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.history_dir = Path(self.tmp.name)
        self.original_config = server.config
        self.original_sessions = server.sessions
        server.config = SimpleNamespace(
            system_prompt="現在のプロンプト",
            max_history_turns=20,
            history_dir=str(self.history_dir),
            emotion_tag_enabled=False,
        )
        server.sessions = {}

    def tearDown(self) -> None:
        server.sessions = self.original_sessions
        server.config = self.original_config
        self.tmp.cleanup()

    def test_session_is_saved_under_browser_id_and_restored_after_memory_clear(self) -> None:
        session = server.get_or_create_session("web_keep")
        session.add_user_message("前の話")
        session.add_assistant_message("覚えています")
        saved = session.save()
        self.assertEqual(saved.name, "session_web_keep.json")

        server.sessions.clear()  # Webサービス再起動相当
        restored = server.get_or_create_session("web_keep")
        self.assertEqual(restored.messages[-2]["content"], "前の話")
        self.assertEqual(restored.messages[-1]["content"], "覚えています")
        self.assertEqual(restored.system_prompt, "現在のプロンプト")

    def test_resume_with_explicit_new_id_does_not_fall_back_to_latest(self) -> None:
        old = server.get_or_create_session("web_old")
        old.add_user_message("古い会話")
        old.add_assistant_message("古い返答")
        old.save()
        server.sessions.clear()

        result = asyncio.run(server.chat_resume("web_new"))
        self.assertEqual(result["session_id"], "web_new")
        self.assertEqual(result["messages"], [])

    def test_resume_without_id_uses_latest_valid_history(self) -> None:
        old = server.get_or_create_session("web_old")
        old.add_user_message("続けたい話")
        old.add_assistant_message("続き")
        old.save()
        server.sessions.clear()

        result = asyncio.run(server.chat_resume())
        self.assertEqual(result["session_id"], "web_old")
        self.assertEqual([m["content"] for m in result["messages"]], ["続けたい話", "続き"])

    def test_resume_rejects_unsafe_id(self) -> None:
        result = asyncio.run(server.chat_resume("../secret"))
        self.assertIsInstance(result, JSONResponse)
        self.assertEqual(result.status_code, 400)
        self.assertEqual(json.loads(result.body)["error"], "invalid session_id")


if __name__ == "__main__":
    unittest.main()
