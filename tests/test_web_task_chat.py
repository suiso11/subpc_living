from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.tasks.chat_editor import TaskChatEditor
from src.tasks.store import TaskStore
from src.web import server


class WebTaskChatTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = TaskStore(
            str(Path(self.tmp.name) / "tasks.db"),
            timezone_name="Asia/Tokyo",
        ).initialize()
        self.original_store = server.task_store
        self.original_editor = server.task_chat_editor
        server.task_store = self.store
        server.task_chat_editor = TaskChatEditor()

    def tearDown(self) -> None:
        server.task_store = self.original_store
        server.task_chat_editor = self.original_editor
        self.store.close()
        self.tmp.cleanup()

    def test_web_wrapper_uses_task_store_without_llm(self) -> None:
        task_id = self.store.add("統合テスト")
        reply = server._try_edit_task_text("タスクを見せて", "web-session")
        self.assertIn(f"#{task_id} 統合テスト", reply)

    def test_task_branch_runs_before_calendar_and_llm_branches(self) -> None:
        source = Path(server.__file__).read_text(encoding="utf-8")
        websocket_start = source.index("async def websocket_chat")
        task_branch = source.index("task_reply = await asyncio.to_thread", websocket_start)
        event_branch = source.index("event_reply = await asyncio.to_thread", websocket_start)
        llm_branch = source.index("token_queue = llm.generate_stream_queue", websocket_start)
        self.assertLess(task_branch, event_branch)
        self.assertLess(task_branch, llm_branch)


if __name__ == "__main__":
    unittest.main()
