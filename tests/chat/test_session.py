"""
ChatSession の基本テスト
"""
import os
import tempfile
import unittest
import unittest.mock
from datetime import datetime, timezone
from pathlib import Path

from src.chat.session import ChatSession
from src.tasks.store import TaskStore


class _FakeRAG:
    """固定テキストを返す RAG のテスト用スタブ。"""

    def __init__(self, text: str) -> None:
        self._text = text

    def build_context_prompt(self, query: str) -> str:
        return self._text


def test_add_messages():
    """メッセージ追加とビルド"""
    session = ChatSession(system_prompt="sys")
    session.add_user_message("hello")
    session.add_assistant_message("hi")

    messages = session.build_messages()
    assert messages[0] == {"role": "system", "content": "sys"}
    assert messages[1] == {"role": "user", "content": "hello"}
    assert messages[2] == {"role": "assistant", "content": "hi"}


class TaskAuthorityTests(unittest.TestCase):
    """タスク権威ブロックの回帰テストを unittest 形式で実行する。"""

    def setUp(self) -> None:
        # build_task_context が優先順位状態を読む環境変数を一時パスへ向ける。
        # 実data/ 配下を読まないようにするための隔離。
        self._env_tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._env_tmp.cleanup)
        self._state_path = Path(self._env_tmp.name) / "priority_state.json"
        self._upcoming_path = Path(self._env_tmp.name) / "upcoming.json"
        env = {
            "PRIORITY_STATE_PATH": str(self._state_path),
            "PRIORITY_UPCOMING_PATH": str(self._upcoming_path),
        }
        patcher = unittest.mock.patch.dict(os.environ, env, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_task_authority_block_final_even_when_no_tasks(self):
        """task_store が空でも権威ブロックがシステムプロンプト末尾に置かれる。"""
        with tempfile.TemporaryDirectory() as d:
            store = TaskStore(db_path=str(Path(d) / "tasks.db"), timezone_name="Asia/Tokyo").initialize()
            try:
                session = ChatSession(system_prompt="sys", task_store=store)
                session.add_user_message("hi")
                messages = session.build_messages()
                system_content = messages[0]["content"]
                authority_idx = system_content.find("--- タスク状態 (権威) ---")
                self.assertNotEqual(authority_idx, -1)
                # セマンティック: 未完了タスクが_NONE_ であることを示す句と、
                # 提案禁止のトークンが権威ブロック見出しより後に現れること。
                self.assertIn("未完了", system_content)
                self.assertIn("1件もない", system_content)
                proposal_idx = system_content.find("提案も禁止", authority_idx)
                self.assertNotEqual(proposal_idx, -1)
                self.assertGreater(proposal_idx, authority_idx)
            finally:
                store.close()

    def test_emotion_tags_come_before_task_authority(self):
        """感情タグ指示はタスク状態の権威ブロックより前に配置される。"""
        with tempfile.TemporaryDirectory() as d:
            store = TaskStore(db_path=str(Path(d) / "tasks.db"), timezone_name="Asia/Tokyo").initialize()
            try:
                session = ChatSession(system_prompt="sys", task_store=store, emotion_tags=True)
                session.add_user_message("hi")
                messages = session.build_messages()
                system_content = messages[0]["content"]
                emotion_idx = system_content.rfind("[emo:")
                if emotion_idx == -1:
                    emotion_idx = system_content.find("感情")
                authority_idx = system_content.find("--- タスク状態 (権威) ---")
                self.assertNotEqual(emotion_idx, -1)
                self.assertNotEqual(authority_idx, -1)
                self.assertLess(emotion_idx, authority_idx)
                # 感情指示より権威ブロックが後ろであれば、正確な末尾文句に依存しない。
                self.assertGreater(authority_idx, emotion_idx)
            finally:
                store.close()

    def test_authority_overrides_stale_rag_about_completed_task(self):
        """RAG が完了済みタスクを未完了のように語っても、末尾の権威ブロックが
        それを完了/破棄として強制的に無視させる指示を出す。"""
        stale_rag = (
            "\n\n[RAG] 過去の記憶: 「古いバグ修正をまだ行う必要がある」と話題に上がる。"
            "リマインドしてね。"
        )
        with tempfile.TemporaryDirectory() as d:
            store = TaskStore(db_path=str(Path(d) / "tasks.db"), timezone_name="Asia/Tokyo").initialize()
            # もともと存在したタスクを完了済みにしておく
            tid = store.add("古いバグ修正", now=datetime(2026, 7, 3, tzinfo=timezone.utc))
            store.done(tid)
            try:
                rag = _FakeRAG(stale_rag)
                session = ChatSession(system_prompt="sys", task_store=store, rag=rag)
                session.add_user_message("リマインドして")
                messages = session.build_messages()
                system_content = messages[0]["content"]
                # RAG 由来の文言が残る位置は権威ブロックより前でなければならない
                rag_idx = system_content.find("過去の記憶")
                authority_idx = system_content.find("--- タスク状態 (権威) ---")
                self.assertNotEqual(rag_idx, -1)
                self.assertNotEqual(authority_idx, -1)
                self.assertLess(rag_idx, authority_idx)
                # 権威ブロックは完了上書きと未完了0件を明示する
                self.assertIn("未完了", system_content)
                self.assertIn("1件もない", system_content)
                self.assertIn("done", system_content)
                self.assertIn("リマインドを提案してはならない", system_content)
            finally:
                store.close()


def test_history_trimming():
    """履歴ターン数が上限を超えたら古いものから削除されること"""
    session = ChatSession(max_history_turns=2)
    for i in range(5):
        session.add_user_message(f"user{i}")
        session.add_assistant_message(f"assistant{i}")

    # user+assistant で 2 ターン分 = 4 メッセージのみ保持
    assert len(session.messages) == 4
    assert session.messages[0]["content"] == "user3"
    assert session.messages[-1]["content"] == "assistant4"


def test_save_and_load(tmp_path):
    """セッションの保存・読み込み"""
    session = ChatSession(system_prompt="test", history_dir=str(tmp_path))
    session.add_user_message("foo")
    session.add_assistant_message("bar")

    saved = session.save()
    assert saved.exists()

    loaded = ChatSession.load(saved, history_dir=str(tmp_path))
    assert loaded.system_prompt == "test"
    assert len(loaded.messages) == 2


class _RecordingRAG:
    """store_turn 呼び出しを記録する RAG スタブ。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    def build_context_prompt(self, query: str) -> str:
        return ""

    def store_turn(self, user_message, assistant_message, session_id):
        self.calls.append((user_message, assistant_message, session_id))
        return f"mem-{len(self.calls)}"


class _RecordingGrowth:
    """record_conversation 呼び出しを記録する GrowthTracker スタブ。"""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def record_conversation(self, **kwargs):
        self.calls.append(kwargs)


class StoreMemoryTests(unittest.TestCase):
    """store_memory 回帰テストを unittest 形式で実行する。

    各テストは独立した一時ディレクトリで履歴を保存する。
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.history_dir = self._tmp.name

    def _make_session(self, **kwargs):
        kwargs.setdefault("system_prompt", "sys")
        kwargs.setdefault("history_dir", self.history_dir)
        return ChatSession(**kwargs)

    def test_default_persists_to_rag(self):
        """既定 (store_memory=True) では RAG に保存される。"""
        rag = _RecordingRAG()
        growth = _RecordingGrowth()
        session = self._make_session(rag=rag, growth_tracker=growth)
        session.add_user_message("hi")
        session.add_assistant_message("hello")

        self.assertEqual(len(rag.calls), 1)
        self.assertEqual(rag.calls[0][0], "hi")
        self.assertEqual(rag.calls[0][1], "hello")
        self.assertEqual(len(growth.calls), 1)
        self.assertEqual(session.messages[-1]["content"], "hello")
        # save() も正常動作
        self.assertTrue(session.save().exists())

    def test_false_skips_rag_but_keeps_history_and_save(self):
        """store_memory=False でも履歴追加・save() は行い、
        成長台帳には memory_saved=False で記録する。"""
        rag = _RecordingRAG()
        growth = _RecordingGrowth()
        session = self._make_session(rag=rag, growth_tracker=growth)
        session.add_user_message("タスクを見せて")
        session.add_assistant_message("#1 完了済み", store_memory=False)

        # RAG には保存しない
        self.assertEqual(rag.calls, [])
        # 履歴には残る
        self.assertEqual(len(session.messages), 2)
        self.assertEqual(session.messages[-1]["content"], "#1 完了済み")
        # save() は正常
        saved = session.save()
        self.assertTrue(saved.exists())
        loaded = ChatSession.load(saved, history_dir=self.history_dir)
        self.assertEqual(len(loaded.messages), 2)
        # 成長台帳は memory_saved=False で記録
        self.assertEqual(len(growth.calls), 1)
        self.assertIs(growth.calls[0]["memory_saved"], False)

    def test_both_false_keeps_pair_without_rag_or_growth(self):
        """両方 False でもインメモリ履歴への追加だけは行う。"""
        rag = _RecordingRAG()
        growth = _RecordingGrowth()
        session = self._make_session(rag=rag, growth_tracker=growth)
        session.add_user_message("覚えておいて")
        session.add_assistant_message(
            "了解しました", store_memory=False, record_growth=False
        )

        self.assertEqual(
            session.messages,
            [
                {"role": "user", "content": "覚えておいて"},
                {"role": "assistant", "content": "了解しました"},
            ],
        )
        self.assertEqual(rag.calls, [])
        self.assertEqual(growth.calls, [])

    def test_false_does_not_classify_by_text(self):
        """store_memory=False はテキスト内容に依存しない。
        通常会話文と似た内容でも明示的に False なら保存しない。"""
        rag = _RecordingRAG()
        growth = _RecordingGrowth()
        session = self._make_session(rag=rag, growth_tracker=growth)
        session.add_user_message("今日どう?")
        # 通常会話のような内容でも store_memory=False なら RAG へ保存しない
        session.add_assistant_message("元気だよ、ありがとう", store_memory=False)

        self.assertEqual(rag.calls, [])
        self.assertIs(growth.calls[0]["memory_saved"], False)

    def test_default_with_no_rag_records_growth_false(self):
        """RAG がない構成でも既定呼び出しは成長台帳に memory_saved=False で記録する。"""
        growth = _RecordingGrowth()
        session = self._make_session(growth_tracker=growth)
        session.add_user_message("hi")
        session.add_assistant_message("hello")

        self.assertEqual(len(growth.calls), 1)
        self.assertIs(growth.calls[0]["memory_saved"], False)


class RollbackLastUserMessageTests(unittest.TestCase):
    """rollback_last_user_message の回帰テスト (CLI / Voice / Web / Discord の巻き戻し用)。"""

    def _make_session(self):
        return ChatSession(system_prompt="sys", max_history_turns=20)

    def test_rolls_back_final_user_message_and_returns_true(self):
        """最終メッセージが user のときだけ原子的に除去して True を返す。"""
        session = self._make_session()
        session.add_user_message("保留中の質問")
        session.add_user_message("最後の質問")

        self.assertIs(session.rollback_last_user_message(), True)
        self.assertEqual(
            session.messages,
            [{"role": "user", "content": "保留中の質問"}],
        )

    def test_returns_false_when_final_message_is_assistant(self):
        """最終メッセージが assistant のときは変更せず False を返す。"""
        session = self._make_session()
        session.add_user_message("質問")
        session.add_assistant_message("回答")

        self.assertIs(session.rollback_last_user_message(), False)
        self.assertEqual(len(session.messages), 2)
        self.assertEqual(session.messages[-1]["role"], "assistant")

    def test_returns_false_when_history_is_empty(self):
        """履歴が空のときは変更せず False を返す。"""
        session = self._make_session()
        self.assertIs(session.rollback_last_user_message(), False)
        self.assertEqual(session.messages, [])

    def test_returns_boolean_no_content_leak(self):
        """戻り値は bool のみで、メッセージ内容や診断文字列を漏らさない。"""
        session = self._make_session()
        session.add_user_message("機密内容")
        result = session.rollback_last_user_message()
        self.assertIsInstance(result, bool)
        self.assertNotIn("機密内容", str(result))
        self.assertEqual(result, True)

    def test_repeated_rollback_is_idempotent_until_non_user(self):
        """繰り返し呼んでも、user が残る間は真、assistant に出会ったら偽で停止する。"""
        session = self._make_session()
        session.add_user_message("質問")
        session.add_assistant_message("回答")
        session.add_user_message("追加の質問")

        self.assertIs(session.rollback_last_user_message(), True)
        self.assertEqual(session.messages[-1]["role"], "assistant")
        self.assertIs(session.rollback_last_user_message(), False)
        # 2回目で何も除去されていない
        self.assertEqual(session.messages[-1]["role"], "assistant")
