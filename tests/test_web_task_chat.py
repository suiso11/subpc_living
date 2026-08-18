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
        llm_branch = source.index("assistant_service.generate_stream(", websocket_start)
        candidate_branch = source.index("_launch_task_candidate_offer(", llm_branch)
        self.assertLess(task_branch, event_branch)
        self.assertLess(task_branch, llm_branch)
        self.assertLess(event_branch, llm_branch)
        self.assertLess(llm_branch, candidate_branch)

    def test_task_reply_branch_passes_store_memory_false(self) -> None:
        """task_reply ブランチだけ _send_direct_chat_reply に store_memory=False を渡す。"""
        source = Path(server.__file__).read_text(encoding="utf-8")
        websocket_start = source.index("async def websocket_chat")
        task_branch = source.index("task_reply = await asyncio.to_thread", websocket_start)
        event_branch = source.index("event_reply = await asyncio.to_thread", websocket_start)

        # task_reply 分岐内の _send_direct_chat_reply 呼び出し
        task_call = source.index("_send_direct_chat_reply", task_branch)
        task_call_block = source[source.index("await _send_direct_chat_reply(", task_branch)
                                 :source.index("continue", task_call)]
        self.assertIn("store_memory=False", task_call_block)

        # event_reply 分岐内の _send_direct_chat_reply 呼び出しには
        # store_memory=False が無い (既定の True を維持)
        event_call = source.index("_send_direct_chat_reply", event_branch)
        event_call_block = source[source.index("await _send_direct_chat_reply(", event_branch)
                                  :source.index("continue", event_call)]
        self.assertNotIn("store_memory=False", event_call_block)

    def test_send_direct_chat_reply_passes_store_memory_to_session(self) -> None:
        """_send_direct_chat_reply は store_memory を add_assistant_message に伝達する。"""
        source = Path(server.__file__).read_text(encoding="utf-8")
        func_start = source.index("async def _send_direct_chat_reply")
        func_body = source[func_start:source.index("def _effective_system_prompt")]
        # シグネチャに store_memory がある
        self.assertIn("store_memory: bool = True", func_body)
        # session.add_assistant_message に渡している
        self.assertIn("session.add_assistant_message(reply, store_memory=store_memory)", func_body)
        # save() は常に呼ばれる
        self.assertIn("session.save()", func_body)


if __name__ == "__main__":
    unittest.main()
