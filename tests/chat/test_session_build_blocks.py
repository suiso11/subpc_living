import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.chat.emotion import EMOTION_TAG_INSTRUCTION
from src.chat.session import ChatSession
from src.context.builder import ContextBuilder
from src.tasks.store import TaskStore


class _FakePreloader:
    def build_preload_context(self) -> str:
        return "\n--- 現在の状況 ---\n- 日時: テスト"


class _FakeRAG:
    def build_context_prompt(self, query: str) -> str:
        return "\n[RAG] 記憶"

    def store_turn(self, user_message, assistant_message, session_id):
        return None


class _FakeWebSearch:
    def build_context_prompt(self, query: str) -> str:
        return "\n\n[Web検索結果]\n検索日時: 2026-01-01 00:00:00\n"


class _FakeVision:
    def get_context_text(self) -> str:
        return "\n[Vision] 現在の視界"


class _FakeMonitor:
    def get_context_text(self) -> str:
        return "\n--- サブPCの現在の状態 ---\n- CPU: 10% (低負荷)"


class _FakeScreen:
    def get_context_text(self) -> str:
        return "\n[Screen] 画面の内容"


class _FakeCalendar:
    def get_context_text(self) -> str:
        return "\n--- 予定 (Google Calendar) ---\n今日の会議: 14:00"


class _BrokenRAG:
    def build_context_prompt(self, query: str) -> str:
        raise RuntimeError("secret rag body")

    def store_turn(self, user_message, assistant_message, session_id):
        return None


class _EmptyWebSearch:
    def build_context_prompt(self, query: str) -> str:
        return ""


class _BrokenCalendar:
    def get_context_text(self) -> str:
        raise RuntimeError("secret calendar body")


class _EmptyPreloader:
    def build_preload_context(self) -> str:
        return ""


class _EnvironmentIsolatedStore:
    """priority state を一時パスへ向けつつ空 TaskStore を生成するヘルパー。"""

    def __init__(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._state_path = Path(self._tmp.name) / "priority_state.json"
        self._upcoming_path = Path(self._tmp.name) / "upcoming.json"
        self._env_patcher = mock.patch.dict(
            os.environ,
            {
                "PRIORITY_STATE_PATH": str(self._state_path),
                "PRIORITY_UPCOMING_PATH": str(self._upcoming_path),
            },
            clear=False,
        )
        self._env_patcher.start()
        self.store = TaskStore(
            db_path=str(Path(self._tmp.name) / "tasks.db"),
            timezone_name="Asia/Tokyo",
        ).initialize()

    def close(self) -> None:
        self._env_patcher.stop()
        self.store.close()
        self._tmp.cleanup()


def _all_provider_session(store, *, emotion: bool) -> ChatSession:
    session = ChatSession(
        system_prompt="sys",
        preloader=_FakePreloader(),
        rag=_FakeRAG(),
        web_search=_FakeWebSearch(),
        vision_context=_FakeVision(),
        monitor_context=_FakeMonitor(),
        screen_context=_FakeScreen(),
        calendar_context=_FakeCalendar(),
        task_store=store,
        emotion_tags=emotion,
    )
    session.add_user_message("hello")
    session.add_assistant_message("hi")
    return session


class ChatSessionBuildBlocksTest(unittest.TestCase):
    def test_build_blocks_returns_tuple(self):
        session = ChatSession(system_prompt="test prompt")
        blocks = session.build_blocks()
        self.assertIsInstance(blocks, tuple)

    def test_build_blocks_includes_history(self):
        """build_blocks should include a history ContextBlock after messages exist."""
        session = ChatSession(system_prompt="test prompt")
        session.add_user_message("hello")
        session.add_assistant_message("hi", store_memory=False)
        blocks = session.build_blocks()
        sources = [b.source for b in blocks]
        self.assertIn("history", sources)

    def test_build_blocks_includes_tasks(self):
        """build_blocks should include tasks block when task_store is configured."""
        from unittest.mock import MagicMock

        session = ChatSession(system_prompt="test prompt", task_store=MagicMock())
        blocks = session.build_blocks()
        sources = [b.source for b in blocks]
        self.assertIn("tasks", sources)

    def test_build_blocks_empty_without_providers(self):
        """build_blocks with no providers returns only history (if messages) or empty."""
        session = ChatSession(system_prompt="")
        blocks = session.build_blocks()
        # No providers, no messages -> empty tuple
        self.assertEqual(len(blocks), 0)


class ChatSessionEmotionBlockTest(unittest.TestCase):
    def test_emotion_block_present_when_enabled(self):
        session = ChatSession(system_prompt="sys", emotion_tags=True)
        session.add_user_message("hello")
        blocks = session.build_blocks()
        emotion_blocks = [b for b in blocks if b.source == "emotion"]
        self.assertEqual(len(emotion_blocks), 1)
        block = emotion_blocks[0]
        self.assertEqual(block.content, EMOTION_TAG_INSTRUCTION)
        self.assertEqual(block.sensitivity, "public")
        self.assertIs(block.local_only, False)

    def test_no_emotion_block_when_disabled(self):
        session = ChatSession(system_prompt="sys", emotion_tags=False)
        session.add_user_message("hello")
        blocks = session.build_blocks()
        self.assertNotIn("emotion", [b.source for b in blocks])

    def test_emotion_only_content_when_nothing_else(self):
        session = ChatSession(system_prompt="", emotion_tags=True)
        self.assertEqual(
            session.build_messages(),
            [{"role": "system", "content": EMOTION_TAG_INSTRUCTION}],
        )


class ChatSessionAllProviderOrderTest(unittest.TestCase):
    """全プロバイダを同時に有効化したときの marker 順と build_blocks 順を固定する。"""

    def setUp(self):
        self._store_env = _EnvironmentIsolatedStore()
        self.addCleanup(self._store_env.close)

    def test_exact_all_provider_marker_order_with_emotion(self):
        session = _all_provider_session(self._store_env.store, emotion=True)
        messages = session.build_messages()
        content = messages[0]["content"]
        markers = [
            "--- 現在の状況 ---",
            "[RAG] 記憶",
            "[Web検索結果]",
            "[Vision] 現在の視界",
            "--- サブPCの現在の状態 ---",
            "[Screen] 画面の内容",
            "--- 予定 (Google Calendar) ---",
            "感情",
            "--- タスク状態 (権威) ---",
        ]
        positions = [content.find(marker) for marker in markers]
        self.assertNotIn(-1, positions)
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(messages[1], {"role": "user", "content": "hello"})
        self.assertEqual(messages[2], {"role": "assistant", "content": "hi"})

    def test_exact_all_provider_marker_order_without_emotion(self):
        session = _all_provider_session(self._store_env.store, emotion=False)
        messages = session.build_messages()
        content = messages[0]["content"]
        markers = [
            "--- 現在の状況 ---",
            "[RAG] 記憶",
            "[Web検索結果]",
            "[Vision] 現在の視界",
            "--- サブPCの現在の状態 ---",
            "[Screen] 画面の内容",
            "--- 予定 (Google Calendar) ---",
            "--- タスク状態 (権威) ---",
        ]
        positions = [content.find(marker) for marker in markers]
        self.assertNotIn(-1, positions)
        self.assertEqual(positions, sorted(positions))
        self.assertNotIn("感情", content)
        self.assertEqual(messages[1], {"role": "user", "content": "hello"})
        self.assertEqual(messages[2], {"role": "assistant", "content": "hi"})

    def test_build_blocks_source_order_with_emotion(self):
        session = _all_provider_session(self._store_env.store, emotion=True)
        sources = [b.source for b in session.build_blocks()]
        self.assertEqual(
            sources,
            [
                "preload",
                "rag",
                "web_search",
                "vision",
                "monitor",
                "screen",
                "calendar",
                "emotion",
                "tasks",
                "history",
            ],
        )

    def test_build_blocks_source_order_without_emotion(self):
        session = _all_provider_session(self._store_env.store, emotion=False)
        sources = [b.source for b in session.build_blocks()]
        self.assertEqual(
            sources,
            [
                "preload",
                "rag",
                "web_search",
                "vision",
                "monitor",
                "screen",
                "calendar",
                "tasks",
                "history",
            ],
        )

    def test_broken_and_empty_providers_skipped_but_emotion_tasks_history_render(self):
        session = ChatSession(
            system_prompt="sys",
            preloader=_EmptyPreloader(),
            rag=_BrokenRAG(),
            web_search=_EmptyWebSearch(),
            calendar_context=_BrokenCalendar(),
            task_store=self._store_env.store,
            emotion_tags=True,
        )
        session.add_user_message("hello")
        session.add_assistant_message("hi")
        with self.assertLogs("src.context.providers", level="WARNING"):
            messages = session.build_messages()
        blocks = session.build_blocks()
        self.assertEqual([b.source for b in blocks], ["emotion", "tasks", "history"])
        content = messages[0]["content"]
        self.assertNotIn("[RAG]", content)
        self.assertNotIn("[Web検索結果]", content)
        self.assertNotIn("予定 (Google Calendar)", content)
        self.assertTrue(content.startswith("sys" + EMOTION_TAG_INSTRUCTION))
        self.assertIn("--- タスク状態 (権威) ---", content)
        self.assertEqual(messages[1], {"role": "user", "content": "hello"})
        self.assertEqual(messages[2], {"role": "assistant", "content": "hi"})


class ChatSessionCompositionParityTest(unittest.TestCase):
    """build_blocks + ContextBuilder の出力が build_messages と完全一致するか検証する。"""

    def setUp(self):
        self._store_env = _EnvironmentIsolatedStore()
        self.addCleanup(self._store_env.close)

    def test_parity_with_emotion_on(self):
        session = _all_provider_session(self._store_env.store, emotion=True)
        messages_direct = session.build_messages()
        blocks = session.build_blocks()
        messages_from_blocks = ContextBuilder("sys").build_messages(
            blocks, privacy="local_only", target_local=True
        )
        self.assertEqual(messages_from_blocks, messages_direct)

    def test_parity_with_emotion_off(self):
        session = _all_provider_session(self._store_env.store, emotion=False)
        messages_direct = session.build_messages()
        blocks = session.build_blocks()
        messages_from_blocks = ContextBuilder("sys").build_messages(
            blocks, privacy="local_only", target_local=True
        )
        self.assertEqual(messages_from_blocks, messages_direct)

    def test_build_messages_delegates_to_build_blocks(self):
        session = _all_provider_session(self._store_env.store, emotion=True)
        blocks = session.build_blocks()
        with mock.patch.object(
            ChatSession, "build_blocks", return_value=blocks
        ) as build_blocks_mock:
            messages = session.build_messages()
        build_blocks_mock.assert_called_once_with()
        self.assertTrue(messages)
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("感情", messages[0]["content"])
        self.assertIn("--- タスク状態 (権威) ---", messages[0]["content"])
        self.assertEqual(messages[1], {"role": "user", "content": "hello"})
        self.assertEqual(messages[2], {"role": "assistant", "content": "hi"})


if __name__ == "__main__":
    unittest.main()