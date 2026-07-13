from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.tasks.chat_editor import TaskChatEditor
from src.tasks.store import TaskStore

UTC = timezone.utc


class TaskChatEditorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = TaskStore(
            str(Path(self.tmp.name) / "tasks.db"),
            timezone_name="Asia/Tokyo",
        ).initialize()
        self.editor = TaskChatEditor()
        self.now = datetime(2026, 7, 14, 3, 0, tzinfo=UTC)

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def handle(self, text: str, session: str = "s1", now=None):
        return self.editor.handle(
            text,
            store=self.store,
            session_id=session,
            now=now or self.now,
        )

    def test_regular_conversation_is_not_intercepted(self) -> None:
        self.assertIsNone(self.handle("今日はいい天気だね"))
        self.assertIsNone(self.handle("タスク管理って難しいね"))

    def test_add_and_list_tasks_in_conversation(self) -> None:
        reply = self.handle("タスク: 明日 牛乳を買う")
        self.assertIn("タスクを追加しました", reply)
        task = self.store.list("open")[0]
        self.assertEqual(task["title"], "牛乳を買う")
        self.assertEqual(task["due_at"].astimezone(self.store.tz).date().isoformat(), "2026-07-15")

        listing = self.handle("タスクを見せて")
        self.assertIn(f"#{task['id']} 牛乳を買う", listing)
        self.assertIn("番号で話せます", listing)
        self.assertIn("まず:", listing)

    def test_breakdown_can_be_shown_and_regenerated_in_conversation(self) -> None:
        task_id = self.store.add("実験レポートを書く", now=self.now)
        shown = self.handle(f"タスク{task_id}の最初の一歩を教えて")
        self.assertIn("まず5分", shown)
        self.assertIn("1.", shown)

        regenerated = self.handle(f"タスク{task_id}を細分化して")
        self.assertIn("ここから始められます", regenerated)
        task = self.store.get(task_id)
        self.assertTrue(task["action_hint"])
        self.assertGreaterEqual(len(task["steps"]), 1)
        self.assertLessEqual(len(task["steps"]), 3)

    def test_update_due_priority_title_and_note_by_id(self) -> None:
        task_id = self.store.add("レポート", now=self.now)

        due_reply = self.handle(f"タスク{task_id}を明日に変更")
        self.assertIn("期限を「7/15」", due_reply)
        self.assertEqual(
            self.store.get(task_id)["due_at"].astimezone(self.store.tz).date().isoformat(),
            "2026-07-15",
        )

        priority_reply = self.handle(f"タスク{task_id}をだいじにして")
        self.assertIn("優先度を「だいじ」", priority_reply)
        self.assertEqual(self.store.get(task_id)["priority"], "high")

        rename_reply = self.handle(f"タスク{task_id}の名前を「実験レポート」に変更")
        self.assertIn("名前を「実験レポート」", rename_reply)
        self.assertEqual(self.store.get(task_id)["title"], "実験レポート")

        note_reply = self.handle(f"タスク{task_id}のメモを「3章から書く」に変更")
        self.assertIn("メモ", note_reply)
        self.assertEqual(self.store.get(task_id)["note"], "3章から書く")

    def test_field_clear_does_not_trigger_task_deletion(self) -> None:
        task_id = self.store.add(
            "会議",
            note="資料あり",
            due_at=self.now + timedelta(days=1),
            due_granularity="datetime",
            now=self.now,
        )
        due_reply = self.handle(f"タスク{task_id}の期限を消して")
        self.assertIn("期限を「なし」", due_reply)
        self.assertEqual(self.store.get(task_id)["status"], "open")
        self.assertIsNone(self.store.get(task_id)["due_at"])

        note_reply = self.handle(f"タスク{task_id}のメモを消して")
        self.assertIn("メモを「なし」", note_reply)
        self.assertEqual(self.store.get(task_id)["status"], "open")
        self.assertEqual(self.store.get(task_id)["note"], "")

    def test_date_in_new_title_does_not_change_due(self) -> None:
        task_id = self.store.add("レポート", now=self.now)
        reply = self.handle(f"タスク{task_id}の名前を「7/15レポート」に変更")
        self.assertIn("名前を「7/15レポート」", reply)
        task = self.store.get(task_id)
        self.assertEqual(task["title"], "7/15レポート")
        self.assertIsNone(task["due_at"])

    def test_complete_unique_title(self) -> None:
        task_id = self.store.add("部屋の掃除", now=self.now)
        reply = self.handle("部屋の掃除のタスクを完了にして")
        self.assertIn("完了にしました", reply)
        self.assertEqual(self.store.get(task_id)["status"], "done")

    def test_ambiguous_title_returns_ids_without_editing(self) -> None:
        first = self.store.add("買い物 スーパー", now=self.now)
        second = self.store.add("買い物 薬局", now=self.now)
        reply = self.handle("買い物のタスクを完了にして")
        self.assertIn("複数のタスクが一致", reply)
        self.assertIn(f"#{first}", reply)
        self.assertIn(f"#{second}", reply)
        self.assertEqual(self.store.get(first)["status"], "open")
        self.assertEqual(self.store.get(second)["status"], "open")

    def test_delete_requires_confirmation_and_is_session_scoped(self) -> None:
        task_id = self.store.add("古いタスク", now=self.now)
        prompt = self.handle(f"タスク{task_id}を削除", session="a")
        self.assertIn("削除しますか", prompt)
        self.assertEqual(self.store.get(task_id)["status"], "open")

        self.assertIsNone(self.handle("削除する", session="b"))
        self.assertEqual(self.store.get(task_id)["status"], "open")

        reply = self.handle("削除する", session="a")
        self.assertIn("削除しました", reply)
        self.assertEqual(self.store.get(task_id)["status"], "dropped")

    def test_delete_can_be_cancelled_or_expire(self) -> None:
        task_id = self.store.add("残すタスク", now=self.now)
        self.handle(f"タスク{task_id}を削除")
        self.assertIn("キャンセル", self.handle("キャンセル"))
        self.assertEqual(self.store.get(task_id)["status"], "open")

        self.handle(f"タスク{task_id}を削除")
        expired = self.handle("削除する", now=self.now + timedelta(minutes=6))
        self.assertIn("時間切れ", expired)
        self.assertEqual(self.store.get(task_id)["status"], "open")

    def test_help_explains_supported_dialogue(self) -> None:
        reply = self.handle("タスクを編集したい")
        self.assertIn("話すだけで", reply)
        self.assertIn("タスク13を明日に変更", reply)


if __name__ == "__main__":
    unittest.main()
