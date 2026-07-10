from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from src.discord_bot.task_ui import split_quick_input

UTC = timezone.utc
JST = ZoneInfo("Asia/Tokyo")


class SplitQuickInputTest(unittest.TestCase):
    def setUp(self) -> None:
        # JST 2026-07-10 10:00 (金曜)
        self.now = datetime(2026, 7, 10, 1, 0, tzinfo=UTC)

    def test_simple_title_no_due(self) -> None:
        """due 表現なしの普通の文"""
        result = split_quick_input("買い物をする", self.now, JST)
        self.assertEqual(result["title"], "買い物をする")
        self.assertIsNone(result["due_at"])
        self.assertIsNone(result["due_granularity"])
        self.assertEqual(result["priority"], "normal")

    def test_birthday_not_confused_with_date(self) -> None:
        """「誕生日」を日付と誤認しない"""
        result = split_quick_input("誕生日プレゼントを買う", self.now, JST)
        self.assertEqual(result["title"], "誕生日プレゼントを買う")
        self.assertIsNone(result["due_at"])

    def test_five_days_not_confused_with_date(self) -> None:
        """「5日分」を日付にしない"""
        result = split_quick_input("5日分の資料をまとめる", self.now, JST)
        self.assertEqual(result["title"], "5日分の資料をまとめる")
        self.assertIsNone(result["due_at"])

    def test_report_with_friday_due(self) -> None:
        """「レポート提出 金曜」→ title「レポート提出」, due=次の金曜 23:59, granularity=date"""
        result = split_quick_input("レポート提出 金曜", self.now, JST)
        self.assertEqual(result["title"], "レポート提出")
        self.assertEqual(result["due_granularity"], "date")
        local = result["due_at"].astimezone(JST)
        # 今日が金曜なので、「金曜」は今日 (7/10)
        self.assertEqual((local.month, local.day, local.hour, local.minute), (7, 10, 23, 59))

    def test_30min_later_laundry(self) -> None:
        """「30分後に洗濯物を取り込む」→ title「洗濯物を取り込む」, due=now+30分, datetime"""
        result = split_quick_input("30分後に洗濯物を取り込む", self.now, JST)
        self.assertEqual(result["title"], "洗濯物を取り込む")
        self.assertEqual(result["due_granularity"], "datetime")
        expected_due = self.now + timedelta(minutes=30)
        self.assertEqual(result["due_at"], expected_due)

    def test_tomorrow_morning_garbage(self) -> None:
        """「明日の朝ゴミ出し」→ title「ゴミ出し」, due=明日9:00, datetime"""
        result = split_quick_input("明日の朝ゴミ出し", self.now, JST)
        self.assertEqual(result["title"], "ゴミ出し")
        self.assertEqual(result["due_granularity"], "datetime")
        local = result["due_at"].astimezone(JST)
        self.assertEqual((local.month, local.day, local.hour), (7, 11, 9))

    def test_ymd_with_time_dentist(self) -> None:
        """「7/15 15:00 歯医者」→ title「歯医者」"""
        result = split_quick_input("7/15 15:00 歯医者", self.now, JST)
        self.assertEqual(result["title"], "歯医者")
        self.assertEqual(result["due_granularity"], "datetime")
        local = result["due_at"].astimezone(JST)
        self.assertEqual((local.month, local.day, local.hour, local.minute), (7, 15, 15, 0))

    def test_high_priority_urgent(self) -> None:
        """「至急 サーバ再起動！」→ priority=high, title「サーバ再起動」"""
        result = split_quick_input("至急 サーバ再起動！", self.now, JST)
        self.assertEqual(result["priority"], "high")
        self.assertEqual(result["title"], "サーバ再起動")

    def test_high_priority_prefix_exclamation(self) -> None:
        """先頭の ! マーク"""
        result = split_quick_input("! 緊急修正", self.now, JST)
        self.assertEqual(result["priority"], "high")
        self.assertEqual(result["title"], "緊急修正")

    def test_high_priority_suffix_exclamation(self) -> None:
        """末尾の ! マーク"""
        result = split_quick_input("緊急対応！！", self.now, JST)
        self.assertEqual(result["priority"], "high")
        self.assertEqual(result["title"], "緊急対応")

    def test_low_priority_after(self) -> None:
        """「あとで」→ priority=low"""
        result = split_quick_input("あとで調べる", self.now, JST)
        self.assertEqual(result["priority"], "low")
        self.assertEqual(result["title"], "調べる")

    def test_tomorrow_with_particle_ni(self) -> None:
        """「明日に提出」— 「に」も一緒に除去"""
        result = split_quick_input("報告書を明日に提出", self.now, JST)
        # 「明日に」が除去されて「報告書を提出」が残る
        self.assertEqual(result["title"], "報告書を提出")

    def test_ymd_with_particles_made(self) -> None:
        """「7/20までに完成」— 「までに」も一緒に除去"""
        result = split_quick_input("プロジェクト完成 7/20までに", self.now, JST)
        # 「7/20までに」が除去される
        self.assertEqual(result["title"], "プロジェクト完成")

    def test_next_week_wednesday(self) -> None:
        """「来週水曜」の期限"""
        result = split_quick_input("来週水曜までに資料提出", self.now, JST)
        self.assertEqual(result["due_granularity"], "date")
        local = result["due_at"].astimezone(JST)
        # 2026-07-10 は金曜。来週水曜 = 7/15
        self.assertEqual((local.month, local.day), (7, 15))

    def test_whitespace_normalization(self) -> None:
        """連続空白を1つに畳む"""
        result = split_quick_input("買い物  する  です", self.now, JST)
        self.assertEqual(result["title"], "買い物 する です")

    def test_empty_text(self) -> None:
        """空のテキスト"""
        result = split_quick_input("", self.now, JST)
        self.assertEqual(result["title"], "")
        self.assertIsNone(result["due_at"])
        self.assertEqual(result["priority"], "normal")

    def test_only_due_expression_returns_due(self) -> None:
        """期限表現のみの入力の場合、title は正規化済み入力全体を使用"""
        result = split_quick_input("明日 18時", self.now, JST)
        # 期限表現のみなので、title は正規化済みテキスト全体
        self.assertEqual(result["title"], "明日 18時")
        self.assertEqual(result["due_granularity"], "datetime")

    def test_zenkaku_digits(self) -> None:
        """全角数字の正規化"""
        result = split_quick_input("７月１５日 買い物", self.now, JST)
        self.assertEqual(result["title"], "買い物")
        local = result["due_at"].astimezone(JST)
        self.assertEqual((local.month, local.day), (7, 15))

    def test_hiragana_alias(self) -> None:
        """ひらがな日付表現の置換"""
        result = split_quick_input("あしたの９時に会議", self.now, JST)
        self.assertEqual(result["title"], "会議")
        local = result["due_at"].astimezone(JST)
        self.assertEqual((local.day, local.hour), (11, 9))


if __name__ == "__main__":
    unittest.main()
