from __future__ import annotations

import os
import tempfile
import sqlite3
import unittest
import unittest.mock
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.tasks.store import TaskStore, build_task_authority_block, build_task_context

UTC = timezone.utc


class TaskStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db_path = str(Path(self._tmp.name) / "tasks.db")
        self.store = TaskStore(db_path=self.db_path, timezone_name="Asia/Tokyo").initialize()
        self.addCleanup(self.store.close)
        self.now = datetime(2026, 7, 3, 3, 0, tzinfo=UTC)  # JST 12:00
        # build_task_context が優先順位状態を読む環境変数を一時パスへ向ける。
        # 実data/ 配下を読まないようにするための隔離。
        self._priority_state_path = str(Path(self._tmp.name) / "priority_state.json")
        self._priority_upcoming_path = str(Path(self._tmp.name) / "upcoming.json")
        env = {
            "PRIORITY_STATE_PATH": self._priority_state_path,
            "PRIORITY_UPCOMING_PATH": self._priority_upcoming_path,
        }
        patcher = unittest.mock.patch.dict(os.environ, env, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_add_and_get(self) -> None:
        tid = self.store.add("レポート", note="n", action_hint="序論", priority="high", now=self.now)
        t = self.store.get(tid)
        self.assertEqual(t["title"], "レポート")
        self.assertEqual(t["priority"], "high")
        self.assertEqual(t["status"], "open")
        self.assertEqual(t["action_hint"], "序論")
        self.assertEqual(t["steps"][0], "序論")
        self.assertEqual(t["step_done"], [False] * len(t["steps"]))

    def test_add_automatically_creates_first_step_and_three_or_fewer_steps(self) -> None:
        tid = self.store.add("実験レポートを書く", now=self.now)
        task = self.store.get(tid)
        self.assertTrue(task["action_hint"])
        self.assertEqual(task["steps"][0], task["action_hint"])
        self.assertGreaterEqual(len(task["steps"]), 1)
        self.assertLessEqual(len(task["steps"]), 3)

    def test_regenerate_breakdown_uses_current_title(self) -> None:
        tid = self.store.add("部屋の掃除", now=self.now)
        self.store.update(tid, title="友人へメールを送る", now=self.now)
        self.assertTrue(self.store.regenerate_breakdown(tid, now=self.now))
        task = self.store.get(tid)
        self.assertIn("要点", task["action_hint"])

    def test_step_done_can_be_checked_and_unchecked(self) -> None:
        tid = self.store.add("実験レポートを書く", now=self.now)
        self.assertTrue(self.store.set_step_done(tid, 0, True, now=self.now))
        self.assertTrue(self.store.get(tid)["step_done"][0])

        self.assertTrue(self.store.set_step_done(tid, 0, False, now=self.now))
        self.assertFalse(self.store.get(tid)["step_done"][0])

    def test_step_done_rejects_invalid_or_closed_task(self) -> None:
        tid = self.store.add("実験レポートを書く", now=self.now)
        self.assertFalse(self.store.set_step_done(tid, -1, True, now=self.now))
        self.assertFalse(self.store.set_step_done(tid, 99, True, now=self.now))
        self.store.done(tid, now=self.now)
        self.assertFalse(self.store.set_step_done(tid, 0, True, now=self.now))

    def test_legacy_string_steps_are_read_as_unchecked(self) -> None:
        tid = self.store.add("旧形式", now=self.now)
        with self.store._tx(immediate=True) as conn:
            conn.execute(
                "UPDATE tasks SET breakdown_json = ? WHERE id = ?",
                ('["手順A", "手順B"]', tid),
            )
        task = self.store.get(tid)
        self.assertEqual(task["steps"], ["手順A", "手順B"])
        self.assertEqual(task["step_done"], [False, False])

    def test_regenerate_breakdown_resets_step_checks(self) -> None:
        tid = self.store.add("実験レポートを書く", now=self.now)
        self.store.set_step_done(tid, 0, True, now=self.now)
        self.assertTrue(self.store.regenerate_breakdown(tid, now=self.now))
        task = self.store.get(tid)
        self.assertEqual(task["step_done"], [False] * len(task["steps"]))

    def test_initialize_backfills_existing_open_task_idempotently(self) -> None:
        legacy_path = str(Path(self._tmp.name) / "legacy.db")
        conn = sqlite3.connect(legacy_path)
        conn.execute(
            """CREATE TABLE tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL, note TEXT,
                action_hint TEXT, due_at TEXT, due_granularity TEXT,
                priority TEXT NOT NULL DEFAULT 'normal', status TEXT NOT NULL DEFAULT 'open',
                source TEXT NOT NULL DEFAULT 'command', created_at TEXT NOT NULL,
                completed_at TEXT
            )"""
        )
        conn.execute(
            "INSERT INTO tasks (title, status, source, created_at) VALUES (?, 'open', 'command', ?)",
            ("部屋を掃除する", self.now.isoformat()),
        )
        conn.commit()
        conn.close()

        legacy = TaskStore(legacy_path, timezone_name="Asia/Tokyo").initialize()
        try:
            first = legacy.get(1)
            self.assertTrue(first["action_hint"])
            self.assertGreaterEqual(len(first["steps"]), 1)
            before = list(first["steps"])
        finally:
            legacy.close()

        legacy = TaskStore(legacy_path, timezone_name="Asia/Tokyo").initialize()
        try:
            self.assertEqual(legacy.get(1)["steps"], before)
        finally:
            legacy.close()

    def test_list_orders_by_due(self) -> None:
        far = self.store.add("far", due_at=self.now + timedelta(days=5), due_granularity="datetime", now=self.now)
        near = self.store.add("near", due_at=self.now + timedelta(hours=2), due_granularity="datetime", now=self.now)
        nodue = self.store.add("nodue", now=self.now)
        ids = [t["id"] for t in self.store.list("open")]
        # 期限あり(近い順)が先、期限なしが最後
        self.assertEqual(ids, [near, far, nodue])

    def test_done_and_drop(self) -> None:
        tid = self.store.add("t", now=self.now)
        self.assertTrue(self.store.done(tid, now=self.now))
        self.assertFalse(self.store.done(tid, now=self.now))  # already done
        self.assertEqual(self.store.get(tid)["status"], "done")
        self.assertIsNotNone(self.store.get(tid)["completed_at"])

        tid2 = self.store.add("t2", now=self.now)
        self.assertTrue(self.store.drop(tid2, now=self.now))
        self.assertEqual(self.store.get(tid2)["status"], "dropped")

    def test_snooze_blocks_claim(self) -> None:
        tid = self.store.add("t", due_at=self.now + timedelta(hours=2), due_granularity="datetime", now=self.now)
        until = self.now + timedelta(hours=1)
        self.assertTrue(self.store.snooze(tid, until, now=self.now))
        # snooze 中は claim されない
        claimed = self.store.claim_due_notifications("o1", self.now)
        self.assertEqual(claimed, [])
        # snooze 明けは claim される
        claimed2 = self.store.claim_due_notifications("o1", until + timedelta(minutes=1))
        self.assertEqual([c["id"] for c in claimed2], [tid])

    def test_get_context_tasks_priority_order(self) -> None:
        overdue = self.store.add("overdue", due_at=self.now - timedelta(hours=1), due_granularity="datetime", now=self.now)
        today = self.store.add("today", due_at=self.now + timedelta(hours=5), due_granularity="datetime", now=self.now)
        soon = self.store.add("soon", due_at=self.now + timedelta(days=2), due_granularity="datetime", now=self.now)
        highp = self.store.add("high", priority="high", now=self.now)
        self.store.add("low", priority="low", now=self.now)  # 除外される

        ctx = self.store.get_context_tasks(limit=8, now=self.now)
        ids = [t["id"] for t in ctx]
        self.assertEqual(ids, [overdue, today, soon, highp])

    def test_get_context_tasks_excludes_done(self) -> None:
        tid = self.store.add("done-task", due_at=self.now - timedelta(hours=1), due_granularity="datetime", now=self.now)
        self.store.done(tid, now=self.now)
        ctx = self.store.get_context_tasks(now=self.now)
        self.assertEqual(ctx, [])

    def test_build_task_context_text(self) -> None:
        self.store.add("レポート", due_at=self.now - timedelta(hours=1), due_granularity="datetime",
                       action_hint="序論を書く", now=self.now)
        text = build_task_context(self.store, now=self.now)
        self.assertIn("--- 未完了タスク ---", text)
        self.assertIn("レポート", text)
        self.assertIn("期限超過", text)
        self.assertIn("次の一手: 序論を書く", text)

    def test_build_task_context_empty(self) -> None:
        # 優先順位推奨も未完了タスクも無いとき。権威ブロックは必ず返し、
        # 「リマインド候補は1件もない・提示禁止」と done/dropped の無効化を含むこと。
        text = build_task_context(self.store, now=self.now)
        self.assertIn("--- タスク状態 (権威) ---", text)
        self.assertIn("1件もない", text)
        self.assertIn("リマインドを提案してはならない", text)
        self.assertIn("done", text)
        self.assertIn("dropped", text)
        # 候補0件時は未完了タスクリストを出さない
        self.assertNotIn("--- 未完了タスク ---", text)
        # 優先順位ブロックも無い (open タスクが無く推奨対象が無いため。
        # 隔離した状態ファイルの有無ではなく、未完了タスク0件によるもの)
        self.assertNotIn("--- 優先順位オーケストレーター ---", text)

    def test_build_task_context_authority_overrides_completed(self) -> None:
        # 完了済みの古いタスク。RAG等に残っていても未完了扱いさせてはいけない。
        stale_title = "期限切れレポート提出"
        tid = self.store.add(
            stale_title,
            due_at=self.now - timedelta(hours=1),
            due_granularity="datetime",
            now=self.now,
        )
        self.assertTrue(self.store.done(tid, now=self.now))
        text = build_task_context(self.store, now=self.now)
        # 完了タスクのタイトルは候補リストに出してはならない
        self.assertNotIn("--- 未完了タスク ---", text)
        self.assertNotIn(stale_title, text)
        # 権威ブロックは完了/破棄の無効化と「1件もない」を明示する
        auth_idx = text.find("--- タスク状態 (権威) ---")
        self.assertGreater(auth_idx, -1)
        self.assertIn("1件もない", text[auth_idx:])
        self.assertIn("done", text[auth_idx:])
        self.assertIn("リマインドを提案してはならない", text[auth_idx:])

    def test_build_task_context_authority_with_open_tasks(self) -> None:
        self.store.add("進行中タスク", due_at=self.now + timedelta(hours=2),
                       due_granularity="datetime", action_hint="着手", now=self.now)
        text = build_task_context(self.store, now=self.now)
        list_idx = text.find("--- 未完了タスク ---")
        auth_idx = text.find("--- タスク状態 (権威) ---")
        self.assertGreater(list_idx, -1)
        self.assertGreater(auth_idx, -1)
        # リストは権威ブロックより前
        self.assertLess(list_idx, auth_idx)
        # タスク本体はリスト部に、候補数は権威ブロック部に出る
        self.assertIn("進行中タスク", text[:auth_idx])
        self.assertIn("1 件", text[auth_idx:])
        self.assertIn("完全集合", text[auth_idx:])
        # 権威ブロックは完了/破棄の無効化で終わる (末尾の最終無効化文言を含む)
        self.assertIn("最終的に無効化", text[auth_idx:])
        self.assertTrue(text.rstrip().endswith("最終的に無効化する。"))

    def test_build_task_context_authority_with_dropped_task(self) -> None:
        dropped_title = "破棄されたタスク"
        tid = self.store.add(dropped_title, now=self.now)
        self.assertTrue(self.store.drop(tid, now=self.now))
        text = build_task_context(self.store, now=self.now)
        self.assertNotIn(dropped_title, text)
        auth_idx = text.find("--- タスク状態 (権威) ---")
        self.assertGreater(auth_idx, -1)
        self.assertIn("dropped", text[auth_idx:])
        self.assertIn("1件もない", text[auth_idx:])

    def test_build_task_context_authority_with_priority_no_list(self) -> None:
        # 優先順位推奨だけ存在し未完了リスト候補が 0 件のケース:
        # 期日も高優先度も無い open タスクは get_context_tasks に載らないが、
        # 優先順位オーケストレーターは推奨として出せる。このとき権威は
        # 推奨1件を「ユーザーから求められていない場面で自発的に催促してよい完全集合」とし、
        # 「1件もない」の禁止文言にはしないこと。
        self.store.add("推奨専用タスク", now=self.now)  # 期限なし・通常優先度
        text = build_task_context(self.store, now=self.now)
        prio_idx = text.find("--- 優先順位オーケストレーター ---")
        auth_idx = text.find("--- タスク状態 (権威) ---")
        self.assertGreater(prio_idx, -1)
        self.assertGreater(auth_idx, -1)
        self.assertLess(prio_idx, auth_idx)
        # 期日/高優先度が無いので未完了リストは出ない
        self.assertNotIn("--- 未完了タスク ---", text)
        # 権威は推奨1件を完全集合とし、禁止文言にはしない
        self.assertIn("推奨1件", text[auth_idx:])
        self.assertIn("完全集合", text[auth_idx:])
        self.assertNotIn("1件もない", text[auth_idx:])

    def test_build_task_context_state_unavailable_returns_empty(self) -> None:
        # get_context_tasks が例外を送ったときは状態不明とみなし、空文字列を返す。
        # このとき権威ブロック (0 件の「1件もない」を含む) を置いてはならない。
        # 設定の no-reminder ガードは権威ブロック不在を前提にした状態不明時の非催促を前提とする。
        with unittest.mock.patch.object(
            self.store, "get_context_tasks", side_effect=RuntimeError("store unavailable")
        ):
            text = build_task_context(self.store, now=self.now)
        self.assertEqual(text, "")
        self.assertNotIn("--- タスク状態 (権威) ---", text)
        self.assertNotIn("1件もない", text)
        self.assertNotIn("--- 未完了タスク ---", text)

    def test_build_task_authority_block_does_not_include_titles(self) -> None:
        # 候補数表示だけであり、個別タイトル等を含めない純粋な関数。
        # 完了/破棄の無効化指示は契約文言として含む。
        b0 = build_task_authority_block(0)
        self.assertIn("1件もない", b0)
        self.assertIn("done", b0)  # 契約文言の一部
        self.assertIn("dropped", b0)
        b3 = build_task_authority_block(3)
        self.assertIn("3 件", b3)
        self.assertIn("完全集合", b3)
        # has_priority=True でも候補0件なら「推奨1件」が完全集合
        b_p = build_task_authority_block(0, has_priority=True)
        self.assertIn("推奨1件", b_p)
        self.assertIn("完全集合", b_p)
        self.assertNotIn("1件もない", b_p)
        # 両方ある場合は両者を併記
        b_both = build_task_authority_block(2, has_priority=True)
        self.assertIn("推奨1件", b_both)
        self.assertIn("2 件", b_both)
        self.assertIn("完全集合", b_both)

    def test_build_task_authority_block_clause_separator(self) -> None:
        # 許容リマインド条項と done/dropped 無効化条項は別行でなければならない。
        # `)。- status` のような続行文にならないよう改行で区切られていること。
        for block in (
            build_task_authority_block(0),
            build_task_authority_block(0, has_priority=True),
            build_task_authority_block(3),
            build_task_authority_block(3, has_priority=True),
        ):
            self.assertNotIn(")。-", block)
            self.assertIn(")。", block)
            self.assertIn("\n- status='done'", block)

    def test_claim_lease_is_exclusive(self) -> None:
        """2接続で同時にclaimしても片方だけが取れる (lease排他)。"""
        store2 = TaskStore(db_path=self.db_path, timezone_name="Asia/Tokyo").initialize()
        try:
            tid = self.store.add("t", due_at=self.now + timedelta(hours=2),
                                 due_granularity="datetime", now=self.now)
            c1 = self.store.claim_due_notifications("owner1", self.now)
            c2 = store2.claim_due_notifications("owner2", self.now)
            self.assertEqual([c["id"] for c in c1], [tid])
            self.assertEqual(c2, [])  # lease を owner1 が保持

            # lease 期限切れ後は owner2 も取れる
            later = self.now + timedelta(seconds=200)
            c3 = store2.claim_due_notifications("owner2", later)
            self.assertEqual([c["id"] for c in c3], [tid])
        finally:
            store2.close()

    def test_claim_skips_tasks_without_due(self) -> None:
        self.store.add("no-due", now=self.now)
        self.assertEqual(self.store.claim_due_notifications("o", self.now), [])


if __name__ == "__main__":
    unittest.main()
