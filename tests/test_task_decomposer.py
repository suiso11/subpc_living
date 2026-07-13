"""src.tasks.decomposer の決定論的分解のテスト。"""
import unittest

from src.tasks.decomposer import TaskBreakdown, MAX_TITLE_LEN, decompose_task


class DecomposeTaskTest(unittest.TestCase):
    def test_empty_title_raises(self):
        with self.assertRaises(ValueError):
            decompose_task("")

    def test_whitespace_only_raises(self):
        with self.assertRaises(ValueError):
            decompose_task("   \n\t  ")

    def test_none_title_raises(self):
        with self.assertRaises(ValueError):
            decompose_task(None)  # type: ignore[arg-type]

    def test_buy_category(self):
        r = decompose_task("新しいマウスを買う")
        self.assertEqual(r.category, "買い物")
        self.assertTrue(r.first_step)
        self.assertEqual(len(r.steps), 2)
        self.assertLessEqual(len(r.steps), 3)

    def test_write_category(self):
        r = decompose_task("週報レポートを書く")
        self.assertEqual(r.category, "執筆")
        self.assertIn("週報レポートを書く", r.first_step)

    def test_submission_takes_precedence_over_report_writing(self):
        r = decompose_task("実験レポートの提出")
        self.assertEqual(r.category, "提出")
        self.assertIn("提出先", r.first_step)

    def test_going_out_category(self):
        r = decompose_task("駿との外出")
        self.assertEqual(r.category, "外出")
        self.assertIn("待ち合わせ", r.first_step)

    def test_research_category(self):
        r = decompose_task("SSDの比較調査")
        self.assertEqual(r.category, "調査")

    def test_contact_category(self):
        r = decompose_task("大家に電話で連絡")
        self.assertEqual(r.category, "連絡")

    def test_booking_category(self):
        r = decompose_task("歯医者の予約")
        self.assertEqual(r.category, "予約")

    def test_tidy_category(self):
        r = decompose_task("机の上を片付ける")
        self.assertEqual(r.category, "片付け")

    def test_study_category(self):
        r = decompose_task("Pythonの勉強")
        self.assertEqual(r.category, "学習")

    def test_coding_category(self):
        r = decompose_task("バグを直す")
        self.assertEqual(r.category, "コーディング")

    def test_generic_fallback(self):
        r = decompose_task("ふしぎなたまご")
        self.assertEqual(r.category, "汎用")
        self.assertIn("ふしぎなたまご", r.first_step)
        self.assertGreaterEqual(len(r.steps), 1)
        self.assertLessEqual(len(r.steps), 3)

    def test_steps_count_within_bounds(self):
        for title in ("書く", "調べる", "買う", "片付ける", "何か"):
            r = decompose_task(title)
            self.assertGreaterEqual(len(r.steps), 1, title)
            self.assertLessEqual(len(r.steps), 3, title)

    def test_first_step_is_nonempty(self):
        for title in ("書く", "調べる", "買う", "片付ける", "何か"):
            r = decompose_task(title)
            self.assertTrue(r.first_step.strip(), title)

    def test_long_title_truncated(self):
        long = "あ" * (MAX_TITLE_LEN + 500)
        # 極端に長くても例外を投げず、テンプレートに埋め込まれる
        r = decompose_task(long)
        self.assertEqual(r.category, "汎用")
        self.assertLessEqual(len(r.first_step), MAX_TITLE_LEN * 2)
        self.assertLessEqual(len(r.first_step), 1000)

    def test_note_and_action_hint_ignored_safely(self):
        r_no = decompose_task("日記を書く")
        r_with = decompose_task("日記を書く", note="夜にやる", action_hint="<done>")
        self.assertEqual(r_no.category, r_with.category)
        self.assertEqual(r_no.first_step, r_with.first_step)
        self.assertEqual(r_no.steps, r_with.steps)

    def test_as_dict_shape(self):
        r = decompose_task("報告メールを送る")
        d = r.as_dict()
        self.assertEqual(set(d.keys()), {"category", "first_step", "steps"})
        self.assertIsInstance(d["steps"], list)
        self.assertEqual(d["steps"], list(r.steps))

    def test_frozen_dataclass(self):
        r = decompose_task("テスト")
        with self.assertRaises(Exception):
            r.first_step = "x"  # type: ignore[misc]

    def test_deterministic(self):
        a = decompose_task("データを比較する")
        b = decompose_task("データを比較する")
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
