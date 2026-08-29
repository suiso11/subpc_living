from __future__ import annotations

import os
import tempfile
import sqlite3
import unittest
import unittest.mock
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.tasks.store import TaskStore, build_task_authority_block, build_task_context, from_iso

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


class TaskRevTest(unittest.TestCase):
    """楽観的リビジョン (rev) と lease 再検証のテスト。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db_path = str(Path(self._tmp.name) / "tasks.db")
        self.store = TaskStore(db_path=self.db_path, timezone_name="Asia/Tokyo").initialize()
        self.addCleanup(self.store.close)
        self.now = datetime(2026, 7, 3, 3, 0, tzinfo=UTC)  # JST 12:00

    def _add_due(self) -> int:
        return self.store.add(
            "t", due_at=self.now + timedelta(hours=2),
            due_granularity="datetime", now=self.now,
        )

    def test_rev_starts_zero_and_increments_on_mutations(self) -> None:
        tid = self.store.add("t", now=self.now)
        self.assertEqual(self.store.get(tid)["rev"], 0)
        self.store.update(tid, title="u", now=self.now)
        self.assertEqual(self.store.get(tid)["rev"], 1)
        self.store.snooze(tid, self.now + timedelta(hours=1), now=self.now)
        self.assertEqual(self.store.get(tid)["rev"], 2)
        self.store.regenerate_breakdown(tid, now=self.now)
        self.assertEqual(self.store.get(tid)["rev"], 3)
        self.store.done(tid, now=self.now)
        self.assertEqual(self.store.get(tid)["rev"], 4)

    def test_drop_and_due_update_increment_rev(self) -> None:
        tid = self.store.add("t", now=self.now)
        self.store.drop(tid, now=self.now)
        self.assertEqual(self.store.get(tid)["rev"], 1)
        tid2 = self.store.add("t2", now=self.now)
        self.store.update(tid2, due_at=self.now + timedelta(days=1),
                          due_granularity="datetime", now=self.now)
        self.assertEqual(self.store.get(tid2)["rev"], 1)

    def test_claim_payload_includes_rev(self) -> None:
        tid = self._add_due()
        claimed = self.store.claim_due_notifications("o", self.now)
        self.assertEqual([c["id"] for c in claimed], [tid])
        self.assertEqual(claimed[0]["rev"], 0)
        self.store.update(tid, title="変更", now=self.now)
        claimed2 = self.store.claim_due_notifications("o", self.now)
        self.assertEqual(claimed2[0]["rev"], 1)

    def test_rev_migration_idempotent(self) -> None:
        legacy_path = str(Path(self._tmp.name) / "legacy_rev.db")
        conn = sqlite3.connect(legacy_path)
        conn.execute(
            """CREATE TABLE tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL, note TEXT,
                action_hint TEXT, due_at TEXT, due_granularity TEXT,
                priority TEXT NOT NULL DEFAULT 'normal', status TEXT NOT NULL DEFAULT 'open',
                source TEXT NOT NULL DEFAULT 'command', created_at TEXT NOT NULL,
                completed_at TEXT, breakdown_json TEXT NOT NULL DEFAULT '[]'
            )"""
        )
        conn.execute(
            "INSERT INTO tasks (title, status, source, created_at) VALUES (?, 'open', 'command', ?)",
            ("旧タスク", self.now.isoformat()),
        )
        conn.commit()
        conn.close()

        legacy = TaskStore(legacy_path, timezone_name="Asia/Tokyo").initialize()
        try:
            self.assertEqual(legacy.get(1)["rev"], 0)  # 旧行は 0 から
            legacy.update(1, title="変更", now=self.now)
            self.assertEqual(legacy.get(1)["rev"], 1)
            # 再初期化しても冪等 (ALTER 失敗せず、rev は維持される)
            legacy.close()
            legacy = TaskStore(legacy_path, timezone_name="Asia/Tokyo").initialize()
            self.assertEqual(legacy.get(1)["rev"], 1)
            self.assertEqual(legacy.get(1)["title"], "変更")
        finally:
            legacy.close()

    def test_revalidate_lease_valid_and_invalid(self) -> None:
        tid = self._add_due()
        self.store.claim_due_notifications("o1", self.now, lease_seconds=120)
        # rev 一致・自 owner → lease 延長して True
        self.assertTrue(self.store.revalidate_notification_lease(
            tid, "o1", 0, lease_seconds=120, now=self.now))
        # rev 不一致 → False + 自 owner の lease がクリアされる
        self.store.claim_due_notifications("o1", self.now, lease_seconds=120)
        self.assertFalse(self.store.revalidate_notification_lease(
            tid, "o1", 999, lease_seconds=120, now=self.now))
        # クリアされたので別 owner が取れる
        self.assertEqual(
            [c["id"] for c in self.store.claim_due_notifications("o2", self.now)], [tid])

    def test_revalidate_lease_requires_owner(self) -> None:
        tid = self._add_due()
        self.store.claim_due_notifications("o2", self.now)
        # 別 owner の lease は触らず False
        self.assertFalse(self.store.revalidate_notification_lease(
            tid, "o1", 0, now=self.now))
        # 他 owner が持ったまま
        self.assertEqual(
            [c["id"] for c in self.store.claim_due_notifications("o1", self.now)], [])

    def test_revalidate_lease_requires_open(self) -> None:
        tid = self._add_due()
        self.store.claim_due_notifications("o1", self.now)
        self.store.done(tid, now=self.now)
        self.assertFalse(self.store.revalidate_notification_lease(tid, "o1", 0, now=self.now))
        # done 済みは claim されない
        self.assertEqual(self.store.claim_due_notifications("o1", self.now), [])

    def test_revalidate_lease_unknown_task(self) -> None:
        self.assertFalse(self.store.revalidate_notification_lease(9999, "o1", 0, now=self.now))

    def test_record_notification_expected_rev_stale_rejected(self) -> None:
        tid = self._add_due()
        self.store.claim_due_notifications("o1", self.now)
        self.store.update(tid, title="変わった", now=self.now)  # rev 0→1
        # 古い rev での記録は拒否され、通知状態を上書きしない
        ok = self.store.record_notification(
            tid, "o1", stage="3h", next_notify_at=self.now + timedelta(minutes=30),
            repeat_count=1, fired=True, now=self.now, expected_rev=0,
        )
        self.assertFalse(ok)
        with self.store._tx() as conn:
            n = conn.execute(
                "SELECT next_notify_at FROM task_notifications WHERE task_id = ?", (tid,)
            ).fetchone()
        self.assertIsNone(n["next_notify_at"])
        # 現在の rev なら成功 (lease を取り直してから)
        self.store.claim_due_notifications("o1", self.now)
        ok2 = self.store.record_notification(
            tid, "o1", stage="3h", next_notify_at=self.now + timedelta(minutes=30),
            repeat_count=1, fired=True, now=self.now,
            expected_rev=self.store.get(tid)["rev"],
        )
        self.assertTrue(ok2)

    def test_record_notification_stale_rejects_when_closed(self) -> None:
        tid = self._add_due()
        self.store.claim_due_notifications("o1", self.now)
        self.store.done(tid, now=self.now)
        self.assertFalse(self.store.record_notification(
            tid, "o1", stage="3h", next_notify_at=self.now + timedelta(hours=1),
            repeat_count=1, fired=True, now=self.now, expected_rev=0,
        ))

    def test_record_notification_without_expected_rev_backward_compat(self) -> None:
        tid = self._add_due()
        # 旧呼び出し形 (expected_rev 省略) は無条件更新して True
        self.assertTrue(self.store.record_notification(
            tid, "o1", stage="3h", next_notify_at=self.now + timedelta(hours=1),
            repeat_count=1, fired=False, now=self.now,
        ))
        with self.store._tx() as conn:
            n = conn.execute(
                "SELECT next_notify_at FROM task_notifications WHERE task_id = ?", (tid,)
            ).fetchone()
        self.assertEqual(from_iso(n["next_notify_at"]), self.now + timedelta(hours=1))

    def test_revalidate_and_record_cross_connection(self) -> None:
        tid = self._add_due()
        store2 = TaskStore(db_path=self.db_path, timezone_name="Asia/Tokyo").initialize()
        try:
            claimed2 = store2.claim_due_notifications("o2", self.now)
            self.assertEqual([c["id"] for c in claimed2], [tid])
            # 別接続から正しい rev / 自 owner で revalidate → True
            self.assertTrue(store2.revalidate_notification_lease(tid, "o2", 0, now=self.now))
            # 別接続の他 owner は revalidate できない
            self.assertFalse(self.store.revalidate_notification_lease(tid, "o1", 0, now=self.now))
            # 別接続の done で rev が進むと、古い rev の record は拒否される
            self.store.done(tid, now=self.now)
            self.assertFalse(store2.record_notification(
                tid, "o2", stage="3h", next_notify_at=self.now + timedelta(hours=1),
                repeat_count=1, fired=True, now=self.now, expected_rev=0,
            ))
            self.assertEqual(self.store.get(tid)["status"], "done")
        finally:
            store2.close()


class TaskCandidateStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db_path = str(Path(self._tmp.name) / "tasks.db")
        self.store = TaskStore(db_path=self.db_path, timezone_name="Asia/Tokyo").initialize()
        self.addCleanup(self.store.close)
        self.now = datetime(2026, 7, 3, 3, 0, tzinfo=UTC)  # JST 12:00

    # --- create / list / get ---

    def test_create_returns_id_and_lists_pending(self) -> None:
        cid = self.store.create_candidate(
            title="レポートを書く",
            due_at=self.now + timedelta(days=1),
            priority="high",
            source="chat",
            now=self.now,
        )
        self.assertIsNotNone(cid)
        pending = self.store.list_candidates("pending")
        self.assertEqual([c["id"] for c in pending], [cid])
        got = self.store.get_candidate(cid)
        self.assertEqual(got["title"], "レポートを書く")
        self.assertEqual(got["status"], "pending")
        self.assertEqual(got["priority"], "high")
        self.assertIsNone(got["task_id"])
        self.assertIsNone(got["decided_at"])
        # registered candidate may follow the normal task_events not being polluted
        self.assertEqual(self.store.events(limit=1000), [])

    def test_create_validates_title(self) -> None:
        with self.assertRaises(ValueError):
            self.store.create_candidate(title="  ", now=self.now)

    def test_create_rejects_sensitive_title_at_store_boundary(self) -> None:
        with self.assertRaises(ValueError):
            self.store.create_candidate(title="password=short", now=self.now)

    def test_create_normalizes_invalid_priority_and_source(self) -> None:
        cid = self.store.create_candidate(
            title="x", priority="xxx", source="unknown", now=self.now
        )
        c = self.store.get_candidate(cid)
        self.assertEqual(c["priority"], "normal")
        self.assertEqual(c["source"], "chat")

    def test_create_nfkc_equivalence_dedups(self) -> None:
        # 半角/全角・結合文字は NFKC で同じ指紋になるはず。
        cid1 = self.store.create_candidate(title="ﾚﾎﾟｰﾄ", now=self.now)
        cid2 = self.store.create_candidate(title="レポート", now=self.now)
        self.assertEqual(cid1, cid2)
        self.assertEqual(len(self.store.list_candidates("pending")), 1)

    # --- dedup ---

    def test_create_dedups_pending_same_source(self) -> None:
        cid1 = self.store.create_candidate(
            title="掃除", due_at=self.now + timedelta(days=1), priority="normal", source="voice",
            now=self.now,
        )
        cid2 = self.store.create_candidate(
            title="掃除", due_at=self.now + timedelta(days=1), priority="normal", source="voice",
            now=self.now,
        )
        self.assertEqual(cid1, cid2)
        self.assertEqual(len(self.store.list_candidates("pending")), 1)

    def test_create_dedups_pending_different_source(self) -> None:
        cid1 = self.store.create_candidate(
            title="掃除", due_at=self.now + timedelta(days=1), priority="normal", source="chat",
            now=self.now,
        )
        cid2 = self.store.create_candidate(
            title="掃除", due_at=self.now + timedelta(days=1), priority="normal", source="voice",
            now=self.now,
        )
        self.assertEqual(cid1, cid2)

    def test_create_does_not_dedup_equivalent_different_due_only(self) -> None:
        c1 = self.store.create_candidate(title="掃除", due_at=self.now + timedelta(days=1), now=self.now)
        c2 = self.store.create_candidate(title="掃除", due_at=self.now + timedelta(days=2), now=self.now)
        self.assertNotEqual(c1, c2)
        self.assertEqual(len(self.store.list_candidates("pending")), 2)

    def test_create_does_not_dedup_different_granularity_only(self) -> None:
        # 同 title/due/priority でも granularity が違う候補は別の指紋。
        due = self.now + timedelta(days=1)
        c1 = self.store.create_candidate(
            title="掃除", due_at=due, due_granularity="datetime", priority="normal",
            source="chat", now=self.now,
        )
        c2 = self.store.create_candidate(
            title="掃除", due_at=due, due_granularity="date", priority="normal",
            source="chat", now=self.now,
        )
        self.assertNotEqual(c1, c2)
        self.assertEqual(len(self.store.list_candidates("pending")), 2)
        self.assertEqual(self.store.get_candidate(c1)["due_granularity"], "datetime")
        self.assertEqual(self.store.get_candidate(c2)["due_granularity"], "date")

    # --- suppression ---

    def test_create_suppressed_within_30_days_after_accept(self) -> None:
        # 候補受入直後に同一指紋 (granularity 含む) で送り直しても抑制される。
        cid = self.store.create_candidate(
            title="掃除", due_at=self.now + timedelta(days=1),
            due_granularity="datetime", priority="normal", now=self.now,
        )
        tid, created = self.store.accept_candidate(cid, now=self.now)
        self.assertTrue(created)
        result = self.store.create_candidate(
            title="掃除", due_at=self.now + timedelta(days=1),
            due_granularity="datetime", priority="normal", now=self.now,
        )
        self.assertIsNone(result)  # 抑制
        self.assertEqual(len(self.store.list_candidates("pending")), 0)
        self.assertEqual(len(self.store.list_candidates("accepted")), 1)

    def test_create_suppressed_within_30_days_after_dismiss(self) -> None:
        cid = self.store.create_candidate(
            title="掃除", due_at=self.now + timedelta(days=1),
            due_granularity="datetime", priority="normal", now=self.now,
        )
        self.assertTrue(self.store.dismiss_candidate(cid, now=self.now))
        result = self.store.create_candidate(
            title="掃除", due_at=self.now + timedelta(days=1),
            due_granularity="datetime", priority="normal",
            now=self.now + timedelta(days=29),
        )
        self.assertIsNone(result)  # 29日後は抑制
        self.assertEqual(len(self.store.list_candidates("pending")), 0)

    def test_create_suppressed_at_exactly_30_days_after_accept(self) -> None:
        # ちょうど30日後でも、cutoff (now - 30日) と decided_at が同じ時刻なので
        # decided_at >= cutoff が成立して抑制される。
        cid = self.store.create_candidate(
            title="掃除", due_at=self.now + timedelta(days=1),
            due_granularity="datetime", priority="normal", now=self.now,
        )
        self.store.accept_candidate(cid, now=self.now)
        later = self.now + timedelta(days=30)
        result = self.store.create_candidate(
            title="掃除", due_at=self.now + timedelta(days=1),
            due_granularity="datetime", priority="normal", now=later,
        )
        self.assertIsNone(result)  # ちょうど30日 = 抑制
        self.assertEqual(len(self.store.list_candidates("pending")), 0)

    def test_create_allowed_just_beyond_30_days_after_accept(self) -> None:
        # 30日 + 1秒 では cutoff が decided_at を超えるので抑制されない。
        cid = self.store.create_candidate(
            title="掃除", due_at=self.now + timedelta(days=1),
            due_granularity="datetime", priority="normal", now=self.now,
        )
        self.store.accept_candidate(cid, now=self.now)
        later = self.now + timedelta(days=30, seconds=1)
        result = self.store.create_candidate(
            title="掃除", due_at=self.now + timedelta(days=1),
            due_granularity="datetime", priority="normal", now=later,
        )
        self.assertIsNotNone(result)
        self.assertEqual(len(self.store.list_candidates("pending")), 1)
        self.assertEqual(len(self.store.list_candidates("accepted")), 1)

    def test_create_suppressed_at_exactly_30_days_after_dismiss(self) -> None:
        cid = self.store.create_candidate(
            title="掃除", due_at=self.now + timedelta(days=1),
            due_granularity="datetime", priority="normal", now=self.now,
        )
        self.assertTrue(self.store.dismiss_candidate(cid, now=self.now))
        later = self.now + timedelta(days=30)
        result = self.store.create_candidate(
            title="掃除", due_at=self.now + timedelta(days=1),
            due_granularity="datetime", priority="normal", now=later,
        )
        self.assertIsNone(result)
        self.assertEqual(len(self.store.list_candidates("pending")), 0)

    def test_create_allowed_just_beyond_30_days_after_dismiss(self) -> None:
        cid = self.store.create_candidate(
            title="掃除", due_at=self.now + timedelta(days=1),
            due_granularity="datetime", priority="normal", now=self.now,
        )
        self.assertTrue(self.store.dismiss_candidate(cid, now=self.now))
        later = self.now + timedelta(days=30, seconds=1)
        result = self.store.create_candidate(
            title="掃除", due_at=self.now + timedelta(days=1),
            due_granularity="datetime", priority="normal", now=later,
        )
        self.assertIsNotNone(result)
        self.assertEqual(len(self.store.list_candidates("pending")), 1)
        self.assertEqual(len(self.store.list_candidates("dismissed")), 1)

    # --- accept ---

    def test_accept_creates_regular_task_and_updates_candidate(self) -> None:
        cid = self.store.create_candidate(
            title="レポート", due_at=self.now + timedelta(days=1), priority="high",
            source="chat", now=self.now,
        )
        fired: list[tuple[int, str]] = []
        self.store.on_change = lambda i, e: fired.append((i, e))
        tid, created = self.store.accept_candidate(cid, now=self.now)
        self.assertTrue(created)
        self.assertGreater(tid, 0)
        # regular task
        task = self.store.get(tid)
        self.assertIsNotNone(task)
        self.assertEqual(task["title"], "レポート")
        self.assertEqual(task["priority"], "high")
        self.assertEqual(task["status"], "open")
        self.assertEqual(task["source"], "chat")
        self.assertEqual(task["due_granularity"], "datetime")
        self.assertTrue(task["action_hint"])
        self.assertGreaterEqual(len(task["steps"]), 1)
        # candidate updated
        cand = self.store.get_candidate(cid)
        self.assertEqual(cand["status"], "accepted")
        self.assertEqual(cand["task_id"], tid)
        self.assertIsNotNone(cand["decided_at"])
        # exactly one on_change fire, exactly one event log entry
        self.assertEqual(fired, [(tid, "accept")])
        evs = self.store.events(task_id=tid)
        self.assertEqual(len(evs), 1)
        self.assertEqual(evs[0]["event"], "accept")

    def test_accept_idempotent_repeat(self) -> None:
        cid = self.store.create_candidate(title="x", now=self.now)
        tid, created1 = self.store.accept_candidate(cid, now=self.now)
        self.assertTrue(created1)
        tid2, created2 = self.store.accept_candidate(cid, now=self.now)
        self.assertFalse(created2)
        self.assertEqual(tid, tid2)
        # no second regular task
        self.assertEqual(len(self.store.list("open", limit=1000)), 1)
        self.assertEqual(len(self.store.list_candidates("accepted")), 1)

    def test_accept_idempotent_under_concurrency(self) -> None:
        cid = self.store.create_candidate(title="x", now=self.now)
        store2 = TaskStore(db_path=self.db_path, timezone_name="Asia/Tokyo").initialize()
        try:
            fired1: list[tuple[int, str]] = []
            fired2: list[tuple[int, str]] = []
            self.store.on_change = lambda i, e: fired1.append((i, e))
            store2.on_change = lambda i, e: fired2.append((i, e))
            t1, c1 = self.store.accept_candidate(cid, now=self.now)
            t2, c2 = store2.accept_candidate(cid, now=self.now)
            # exactly one created; second is no-op returning same task id
            self.assertNotEqual(c1, c2)
            self.assertEqual(t1, t2)
            self.assertEqual(len(self.store.list("open", limit=1000)), 1)
            self.assertEqual(fired1 + fired2, [(t1, "accept")])
        finally:
            store2.close()

    def test_accept_dismissed_conflicts(self) -> None:
        cid = self.store.create_candidate(title="x", now=self.now)
        self.assertTrue(self.store.dismiss_candidate(cid, now=self.now))
        with self.assertRaises(ValueError):
            self.store.accept_candidate(cid, now=self.now)

    def test_accept_unknown_candidate_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.store.accept_candidate(9999, now=self.now)

    def test_accept_task_inherits_candidate_source(self) -> None:
        cid = self.store.create_candidate(title="x", source="voice", now=self.now)
        tid, _ = self.store.accept_candidate(cid, now=self.now)
        self.assertEqual(self.store.get(tid)["source"], "voice")

    def test_accept_preserves_date_granularity(self) -> None:
        # date 粒度の候補を受け入れた場合、生成タスクも date 粒度を保持する。
        due = self.now + timedelta(days=2)
        cid = self.store.create_candidate(
            title="定例会議", due_at=due, due_granularity="date",
            priority="normal", source="chat", now=self.now,
        )
        cand = self.store.get_candidate(cid)
        self.assertEqual(cand["due_granularity"], "date")
        tid, created = self.store.accept_candidate(cid, now=self.now)
        self.assertTrue(created)
        task = self.store.get(tid)
        self.assertEqual(task["due_granularity"], "date")
        self.assertEqual(task["due_at"], due)

    def test_accept_without_due_has_null_granularity(self) -> None:
        # due_at が無い候補を受け入れた場合、タスクの granularity は NULL となる。
        cid = self.store.create_candidate(title="期限なし", now=self.now)
        tid, _ = self.store.accept_candidate(cid, now=self.now)
        task = self.store.get(tid)
        self.assertIsNone(task["due_at"])
        self.assertIsNone(task["due_granularity"])

    def test_create_candidate_infers_datetime_when_omitted(self) -> None:
        # 旧呼び出し側との後方互換: due_at を渡して granularity 省略時に datetime。
        cid = self.store.create_candidate(
            title="掃除", due_at=self.now + timedelta(days=1), now=self.now
        )
        cand = self.store.get_candidate(cid)
        self.assertEqual(cand["due_granularity"], "datetime")
        # 暗黙 datetime と明示 datetime は同一指紋 → dedup
        cid2 = self.store.create_candidate(
            title="掃除", due_at=self.now + timedelta(days=1),
            due_granularity="datetime", now=self.now,
        )
        self.assertEqual(cid, cid2)

    def test_accept_candidate_does_not_create_notification_storm(self) -> None:
        cid = self.store.create_candidate(
            title="x", due_at=self.now + timedelta(days=1), now=self.now
        )
        tid, _ = self.store.accept_candidate(cid, now=self.now)
        with self.store._tx() as conn:
            n = conn.execute(
                "SELECT COUNT(*) AS c FROM task_notifications WHERE task_id = ?", (tid,)
            ).fetchone()["c"]
        self.assertEqual(n, 1)

    # --- dismiss ---

    def test_dismiss_pending_returns_true_then_idempotent_false(self) -> None:
        cid = self.store.create_candidate(title="x", now=self.now)
        self.assertTrue(self.store.dismiss_candidate(cid, now=self.now))
        self.assertFalse(self.store.dismiss_candidate(cid, now=self.now))  # already dismissed
        cand = self.store.get_candidate(cid)
        self.assertEqual(cand["status"], "dismissed")
        self.assertIsNotNone(cand["decided_at"])

    def test_dismiss_does_not_fire_on_change(self) -> None:
        cid = self.store.create_candidate(title="x", now=self.now)
        fired: list[tuple[int, str]] = []
        self.store.on_change = lambda i, e: fired.append((i, e))
        self.assertTrue(self.store.dismiss_candidate(cid, now=self.now))
        self.assertFalse(self.store.dismiss_candidate(cid, now=self.now))
        self.assertEqual(fired, [])

    def test_dismiss_conflicts_with_accepted(self) -> None:
        cid = self.store.create_candidate(title="x", now=self.now)
        self.store.accept_candidate(cid, now=self.now)
        with self.assertRaises(ValueError):
            self.store.dismiss_candidate(cid, now=self.now)

    def test_dismiss_unknown_candidate_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.store.dismiss_candidate(9999, now=self.now)

    # --- migration / legacy compatibility ---

    def test_candidate_table_created_on_legacy_db(self) -> None:
        """候補テーブルが無い古いDBを初期化しても壊さず候補テーブルが作られる。"""
        legacy_path = str(Path(self._tmp.name) / "legacy.db")
        conn = sqlite3.connect(legacy_path)
        conn.execute(
            """CREATE TABLE tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL, note TEXT,
                action_hint TEXT, due_at TEXT, due_granularity TEXT,
                priority TEXT NOT NULL DEFAULT 'normal', status TEXT NOT NULL DEFAULT 'open',
                source TEXT NOT NULL DEFAULT 'command', created_at TEXT NOT NULL,
                completed_at TEXT, breakdown_json TEXT NOT NULL DEFAULT '[]'
            )"""
        )
        conn.execute(
            "INSERT INTO tasks (title, status, source, created_at) VALUES (?, 'open', 'command', ?)",
            ("旧タスク", self.now.isoformat()),
        )
        conn.commit()
        conn.close()

        legacy = TaskStore(legacy_path, timezone_name="Asia/Tokyo").initialize()
        try:
            # existing tasks still readable
            self.assertEqual(legacy.get(1)["title"], "旧タスク")
            # candidate Inbox available
            cid = legacy.create_candidate(title="新候補", now=self.now)
            self.assertIsNotNone(cid)
            self.assertEqual(len(legacy.list_candidates("pending")), 1)
        finally:
            legacy.close()

    def test_candidate_table_migrates_legacy_due_granularity(self) -> None:
        """due_granularity カラム無しの task_candidates 旧DBを壊さずに ALTER する。"""
        legacy_path = str(Path(self._tmp.name) / "legacy_cand.db")
        conn = sqlite3.connect(legacy_path)
        conn.execute(
            """CREATE TABLE tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL, note TEXT,
                action_hint TEXT, due_at TEXT, due_granularity TEXT,
                priority TEXT NOT NULL DEFAULT 'normal', status TEXT NOT NULL DEFAULT 'open',
                source TEXT NOT NULL DEFAULT 'command', created_at TEXT NOT NULL,
                completed_at TEXT, breakdown_json TEXT NOT NULL DEFAULT '[]'
            )"""
        )
        # due_granularity カラム無しの候補テーブル (本変更より前のスキーマ)
        conn.execute(
            """CREATE TABLE task_candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                title TEXT NOT NULL,
                due_at TEXT,
                priority TEXT NOT NULL DEFAULT 'normal',
                status TEXT NOT NULL DEFAULT 'pending',
                task_id INTEGER,
                created_at TEXT NOT NULL,
                decided_at TEXT
            )"""
        )
        conn.execute(
            "INSERT INTO task_candidates (source, fingerprint, title, due_at, priority, "
            "status, created_at) VALUES (?, ?, ?, ?, ?, 'pending', ?)",
            ("chat", "fp0", "旧候補", "2026-08-01T00:00:00+00:00",
             "normal", "2026-07-01T00:00:00+00:00"),
        )
        conn.commit()
        conn.close()

        legacy = TaskStore(legacy_path, timezone_name="Asia/Tokyo").initialize()
        try:
            # 旧行は datetime 粒度と新fingerprintへ移行される。
            cand = legacy.list_candidates("pending")[0]
            self.assertEqual(cand["title"], "旧候補")
            self.assertEqual(cand["due_granularity"], "datetime")
            duplicate = legacy.create_candidate(
                title="旧候補", due_at=datetime(2026, 8, 1, tzinfo=UTC),
                due_granularity="datetime", priority="normal", now=self.now,
            )
            self.assertEqual(duplicate, cand["id"])
            # 受け入れでも datetime を維持する
            tid, created = legacy.accept_candidate(cand["id"], now=self.now)
            self.assertTrue(created)
            self.assertEqual(legacy.get(tid)["due_granularity"], "datetime")
            # 新規候補は due_granularity を保存する
            cid = legacy.create_candidate(
                title="新候補", due_at=self.now + timedelta(days=1),
                due_granularity="date", now=self.now,
            )
            self.assertEqual(legacy.get_candidate(cid)["due_granularity"], "date")
        finally:
            legacy.close()


class TaskNotificationMigrationTest(unittest.TestCase):
    """legacy な task_notifications (lease_owner/lease_until 欠落) のマイグレーション。

    - 旧DBで lease カラムが無くても (片方だけでも両方無くても) initialize で
      NULL 許容カラムが追加され、既存の通知状態・rev・task 行を保持したまま
      claim / revalidate / record が使えることを確認する。
    - 再 initialize しても冪等 (ALTER 失敗せず、状態・rev を維持) であること。
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.now = datetime(2026, 7, 3, 3, 0, tzinfo=UTC)  # JST 12:00

    @staticmethod
    def _legacy_notif_create(has_lease_owner: bool, has_lease_until: bool) -> str:
        cols = [
            "task_id INTEGER PRIMARY KEY",
            "last_stage TEXT",
            "last_notified_at TEXT",
            "next_notify_at TEXT",
            "snoozed_until TEXT",
            "repeat_count INTEGER NOT NULL DEFAULT 0",
        ]
        if has_lease_owner:
            cols.append("lease_owner TEXT")
        if has_lease_until:
            cols.append("lease_until TEXT")
        return "CREATE TABLE task_notifications (" + ", ".join(cols) + ")"

    def _make_legacy_db(self, name: str, *, has_lease_owner: bool, has_lease_until: bool) -> str:
        path = str(Path(self._tmp.name) / name)
        conn = sqlite3.connect(path)
        conn.execute(
            """CREATE TABLE tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL, note TEXT,
                action_hint TEXT, due_at TEXT, due_granularity TEXT,
                priority TEXT NOT NULL DEFAULT 'normal', status TEXT NOT NULL DEFAULT 'open',
                source TEXT NOT NULL DEFAULT 'command', created_at TEXT NOT NULL,
                completed_at TEXT, breakdown_json TEXT NOT NULL DEFAULT '[]'
            )"""
        )
        conn.execute(self._legacy_notif_create(has_lease_owner, has_lease_until))
        conn.execute(
            "INSERT INTO tasks (title, status, source, created_at, due_at, due_granularity) "
            "VALUES (?, 'open', 'command', ?, ?, 'datetime')",
            ("旧タスク", self.now.isoformat(),
             (self.now + timedelta(hours=2)).isoformat()),
        )
        conn.execute(
            "INSERT INTO task_notifications "
            "(task_id, last_stage, last_notified_at, next_notify_at, snoozed_until, repeat_count) "
            "VALUES (1, '3h', ?, ?, NULL, 3)",
            (self.now.isoformat(), self.now.isoformat()),
        )
        conn.commit()
        conn.close()
        return path

    def test_lease_columns_migrated_and_claim_revalidate_record(self) -> None:
        cases = [
            ("neither", False, False),
            ("only_owner", True, False),
            ("only_until", False, True),
            ("both", True, True),  # 最新スキーマ (既存挙動を壊さない確認)
        ]
        for label, has_owner, has_until in cases:
            with self.subTest(label=label):
                path = self._make_legacy_db(
                    f"legacy_{label}.db", has_lease_owner=has_owner, has_lease_until=has_until
                )
                store = TaskStore(path, timezone_name="Asia/Tokyo").initialize()
                try:
                    conn = store._require()
                    cols = {
                        r["name"] for r in conn.execute(
                            "PRAGMA table_info(task_notifications)").fetchall()
                    }
                    self.assertIn("lease_owner", cols)
                    self.assertIn("lease_until", cols)
                    # 既存の通知状態・rev・task 行が保持されている
                    n = conn.execute(
                        "SELECT * FROM task_notifications WHERE task_id = 1").fetchone()
                    self.assertEqual(n["last_stage"], "3h")
                    self.assertEqual(n["repeat_count"], 3)
                    self.assertEqual(from_iso(n["next_notify_at"]), self.now)
                    self.assertIsNone(n["lease_owner"])
                    self.assertIsNone(n["lease_until"])
                    self.assertEqual(store.get(1)["title"], "旧タスク")
                    self.assertEqual(store.get(1)["rev"], 0)
                    # claim → revalidate → record が一通り動く
                    claimed = store.claim_due_notifications("o1", self.now)
                    self.assertEqual([c["id"] for c in claimed], [1])
                    self.assertTrue(store.revalidate_notification_lease(
                        1, "o1", 0, lease_seconds=120, now=self.now))
                    ok = store.record_notification(
                        1, "o1", stage="1h",
                        next_notify_at=self.now + timedelta(hours=1),
                        repeat_count=4, fired=True, now=self.now, expected_rev=0,
                    )
                    self.assertTrue(ok)
                    with store._tx() as conn2:
                        n2 = conn2.execute(
                            "SELECT last_stage, repeat_count, lease_owner "
                            "FROM task_notifications WHERE task_id = 1").fetchone()
                    self.assertEqual(n2["last_stage"], "1h")
                    self.assertEqual(n2["repeat_count"], 4)
                    self.assertIsNone(n2["lease_owner"])  # record で lease 解放
                finally:
                    store.close()

    def test_reinitialize_idempotent_after_lease_migration(self) -> None:
        path = self._make_legacy_db(
            "legacy_reinit.db", has_lease_owner=False, has_lease_until=False
        )
        store = TaskStore(path, timezone_name="Asia/Tokyo").initialize()
        try:
            self.assertTrue(store.record_notification(
                1, "o1", stage="3h", next_notify_at=self.now + timedelta(hours=3),
                repeat_count=2, fired=True, now=self.now,
            ))
        finally:
            store.close()
        # 再 initialize しても ALTER 失敗せず、状態と rev が維持される
        store = TaskStore(path, timezone_name="Asia/Tokyo").initialize()
        try:
            with store._tx() as conn:
                n = conn.execute(
                    "SELECT last_stage, repeat_count FROM task_notifications WHERE task_id = 1"
                ).fetchone()
            self.assertEqual(n["last_stage"], "3h")
            self.assertEqual(n["repeat_count"], 2)
            self.assertEqual(store.get(1)["rev"], 0)
            self.assertEqual(store.get(1)["title"], "旧タスク")
            # 再初期化後も claim できる (next_notify_at を過ぎた時点で)
            self.assertEqual(
                [c["id"] for c in store.claim_due_notifications(
                    "o1", self.now + timedelta(hours=4))], [1])
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main()
