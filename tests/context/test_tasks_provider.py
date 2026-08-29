"""Tasks Context Provider / ChatSession互換wiringのテスト。"""
from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from src.chat.session import ChatSession
from src.context import TasksContextProvider, TasksSource
from src.context.builder import ContextBuilder
from src.context.contracts import TASKS_SOURCE, ContextBlock
from src.context.providers.tasks import TasksContextProvider as ProviderImpl
from src.context.providers.tasks import TasksSource as SourceImpl
from src.tasks.store import TaskStore


class _FakeStore:
    """get_context_tasks / tz を提供する最小の TasksSource 適合ダミー。"""

    def __init__(self, text: str = "\n--- 未完了タスク ---\n- テスト (期限なし)") -> None:
        self._text = text
        self.calls: int = 0

    def get_context_tasks(self, limit: int = 8, *, now=None) -> list[dict]:
        self.calls += 1
        return []

    @property
    def tz(self):
        return __import__("zoneinfo").ZoneInfo("Asia/Tokyo")


class _BrokenStore:
    def get_context_tasks(self, limit: int = 8, *, now=None) -> list[dict]:
        return [{"id": 1, "due_at": None, "due_granularity": "day", "priority": "normal", "title": "x", "action_hint": ""}]

    @property
    def tz(self):
        raise RuntimeError("secret task body")


class TasksSourceProtocolTest(unittest.TestCase):
    def test_fake_store_conforms_to_tasks_source(self) -> None:
        self.assertIsInstance(_FakeStore(), TasksSource)
        self.assertIsInstance(_BrokenStore(), TasksSource)

    def test_unrelated_object_does_not_conform(self) -> None:
        self.assertNotIsInstance(object(), TasksSource)


class TasksContextProviderTest(unittest.TestCase):
    def test_returns_block_with_expected_metadata(self) -> None:
        source = _FakeStore()
        result = TasksContextProvider.collect(source)
        self.assertIsNotNone(result)
        self.assertEqual(result.source, TASKS_SOURCE)
        self.assertEqual(result.sensitivity, "personal")
        self.assertIs(result.local_only, True)
        self.assertIsInstance(result.content, str)
        self.assertEqual(source.calls, 1)

    def test_exception_returns_none(self) -> None:
        self.assertIsNone(TasksContextProvider.collect(_BrokenStore()))

    def test_exception_logs_type_only_not_body(self) -> None:
        with self.assertLogs(
            "src.context.providers.tasks", level="WARNING"
        ) as captured:
            result = TasksContextProvider.collect(_BrokenStore())
        self.assertIsNone(result)
        self.assertEqual(len(captured.output), 1)
        self.assertIn("RuntimeError", captured.output[0])
        self.assertNotIn("secret task body", captured.output[0])


class TasksCloudFilterTest(unittest.TestCase):
    def test_cloud_target_excludes_personal_local_only_tasks(self) -> None:
        builder = ContextBuilder("base")
        tasks_block = TasksContextProvider.collect(_FakeStore())
        public_block = ContextBlock(
            source="tasks", content="公開", sensitivity="public"
        )
        result = builder.build_system_content(
            [tasks_block, public_block], privacy="cloud_allowed", target_local=False
        )
        self.assertEqual(result, "base公開")


class TasksAuthorityViaBuilderTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = TaskStore(
            db_path=str(Path(self._tmp.name) / "tasks.db"),
            timezone_name="Asia/Tokyo",
        ).initialize()

    def tearDown(self) -> None:
        self.store.close()

    def test_zero_task_store_still_produces_authority_block(self) -> None:
        result = TasksContextProvider.collect(self.store)
        self.assertIsNotNone(result)
        self.assertEqual(result.source, TASKS_SOURCE)
        self.assertIsInstance(result.content, str)
        self.assertIn("タスク状態 (権威)", result.content)
        self.assertIn("1件もない", result.content)

    def test_tasks_block_is_system_final_before_history_role_messages(self) -> None:
        session = ChatSession(
            system_prompt="sys",
            task_store=self.store,
            emotion_tags=True,
        )
        session.add_user_message("hi")
        session.add_assistant_message("a")
        messages = session.build_messages()
        self.assertEqual(messages[0]["role"], "system")
        content = messages[0]["content"]
        authority_idx = content.find("--- タスク状態 (権威) ---")
        self.assertNotEqual(authority_idx, -1)
        emotion_idx = content.find("感情")
        self.assertNotEqual(emotion_idx, -1)
        self.assertLess(emotion_idx, authority_idx)
        self.assertTrue(content.endswith("最終的に無効化する。"))
        # History は system より後ろの role messages として配置される。
        self.assertEqual(messages[1], {"role": "user", "content": "hi"})
        self.assertEqual(messages[2], {"role": "assistant", "content": "a"})

    def test_exact_calendar_emotion_tasks_history_order(self) -> None:
        session = ChatSession(
            system_prompt="sys",
            calendar_context=_FakeCalendar(),
            task_store=self.store,
            emotion_tags=True,
        )
        session.add_user_message("hi")
        session.add_assistant_message("a")
        messages = session.build_messages()
        self.assertEqual(messages[0]["role"], "system")
        content = messages[0]["content"]
        cal_idx = content.index("--- 予定 (Google Calendar) ---")
        emotion_idx = content.find("感情")
        authority_idx = content.find("--- タスク状態 (権威) ---")
        self.assertLess(cal_idx, emotion_idx)
        self.assertLess(emotion_idx, authority_idx)
        # History は system message の後ろの role messages。
        self.assertEqual(messages[1], {"role": "user", "content": "hi"})
        self.assertEqual(messages[2], {"role": "assistant", "content": "a"})


class _FakeCalendar:
    def get_context_text(self) -> str:
        return "\n--- 予定 (Google Calendar) ---\n今日の会議: 14:00"


class RootContextPublicAPITest(unittest.TestCase):
    def test_tasks_provider_exported_from_root(self) -> None:
        import src.context

        self.assertIn("TasksContextProvider", src.context.__all__)
        self.assertIs(src.context.TasksContextProvider, ProviderImpl)

    def test_tasks_source_exported_from_root(self) -> None:
        import src.context

        self.assertIn("TasksSource", src.context.__all__)
        self.assertIs(src.context.TasksSource, SourceImpl)
        self.assertIsInstance(_FakeStore(), src.context.TasksSource)

    def test_tasks_exported_from_providers(self) -> None:
        import src.context.providers

        self.assertIn("TasksContextProvider", src.context.providers.__all__)
        self.assertIn("TasksSource", src.context.providers.__all__)


if __name__ == "__main__":
    unittest.main()
