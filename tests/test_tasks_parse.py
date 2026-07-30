from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from src.discord_bot.task_ui import parse_due, parse_snooze, validate_extraction
from src.tasks.extractor import (
    build_extraction_prompt,
    build_multi_extraction_prompt,
    is_sensitive_text,
    validate_extraction as extractor_validate_extraction,
    validate_multi_extraction,
)

UTC = timezone.utc
JST = ZoneInfo("Asia/Tokyo")


class ParseDueTest(unittest.TestCase):
    def setUp(self) -> None:
        # JST 2026-07-03 10:00
        self.now = datetime(2026, 7, 3, 1, 0, tzinfo=UTC)

    def test_ashita_is_date_2359(self) -> None:
        due, gran = parse_due("明日", self.now, JST)
        self.assertEqual(gran, "date")
        local = due.astimezone(JST)
        self.assertEqual((local.month, local.day), (7, 4))
        self.assertEqual((local.hour, local.minute), (23, 59))

    def test_kyou(self) -> None:
        due, gran = parse_due("今日中に", self.now, JST)
        self.assertEqual(gran, "date")
        self.assertEqual(due.astimezone(JST).day, 3)

    def test_asatte(self) -> None:
        due, gran = parse_due("明後日", self.now, JST)
        self.assertEqual(due.astimezone(JST).day, 5)

    def test_md_date(self) -> None:
        due, gran = parse_due("7/10", self.now, JST)
        self.assertEqual(gran, "date")
        local = due.astimezone(JST)
        self.assertEqual((local.month, local.day, local.hour), (7, 10, 23))

    def test_md_datetime(self) -> None:
        due, gran = parse_due("7/10 15:00", self.now, JST)
        self.assertEqual(gran, "datetime")
        local = due.astimezone(JST)
        self.assertEqual((local.month, local.day, local.hour, local.minute), (7, 10, 15, 0))

    def test_ashita_with_time(self) -> None:
        due, gran = parse_due("明日 9:30", self.now, JST)
        self.assertEqual(gran, "datetime")
        local = due.astimezone(JST)
        self.assertEqual((local.day, local.hour, local.minute), (4, 9, 30))

    def test_time_only_future(self) -> None:
        due, gran = parse_due("15:00", self.now, JST)
        self.assertEqual(gran, "datetime")
        local = due.astimezone(JST)
        self.assertEqual((local.day, local.hour), (3, 15))

    def test_time_only_past_rolls_to_tomorrow(self) -> None:
        due, gran = parse_due("8:00", self.now, JST)  # now JST 10:00 -> 翌日
        self.assertEqual(due.astimezone(JST).day, 4)

    def test_relative_minutes(self) -> None:
        due, gran = parse_due("30分後", self.now, JST)
        self.assertEqual(gran, "datetime")
        self.assertEqual(due, self.now + timedelta(minutes=30))

    def test_month_kanji(self) -> None:
        due, gran = parse_due("7月10日", self.now, JST)
        self.assertEqual((due.astimezone(JST).month, due.astimezone(JST).day), (7, 10))

    def test_unparseable(self) -> None:
        due, gran = parse_due("よろしくお願いします", self.now, JST)
        self.assertIsNone(due)
        self.assertIsNone(gran)

    # --- 拡張表現 (now = 2026-07-03 金曜 JST 10:00) ---

    def test_weekday_plain_is_next_occurrence(self) -> None:
        due, gran = parse_due("月曜", self.now, JST)
        self.assertEqual(gran, "date")
        local = due.astimezone(JST)
        self.assertEqual((local.month, local.day), (7, 6))

    def test_weekday_today_counts(self) -> None:
        # 今日は金曜なので「金曜」は今日
        due, _ = parse_due("金曜日", self.now, JST)
        self.assertEqual(due.astimezone(JST).day, 3)

    def test_next_week_weekday(self) -> None:
        due, _ = parse_due("来週の水曜", self.now, JST)
        local = due.astimezone(JST)
        self.assertEqual((local.month, local.day), (7, 8))

    def test_this_week_weekday(self) -> None:
        due, _ = parse_due("今週土曜", self.now, JST)
        self.assertEqual(due.astimezone(JST).day, 4)

    def test_weekday_with_time(self) -> None:
        due, gran = parse_due("来週月曜 15:00", self.now, JST)
        self.assertEqual(gran, "datetime")
        local = due.astimezone(JST)
        self.assertEqual((local.day, local.hour), (6, 15))

    def test_next_week_alone_is_end_of_next_week(self) -> None:
        due, gran = parse_due("来週", self.now, JST)
        self.assertEqual(gran, "date")
        self.assertEqual(due.astimezone(JST).day, 12)  # 来週の日曜

    def test_weekend(self) -> None:
        due, _ = parse_due("週末", self.now, JST)
        self.assertEqual(due.astimezone(JST).day, 4)  # 次の土曜

    def test_pm_kanji_time(self) -> None:
        due, gran = parse_due("明日の午後3時", self.now, JST)
        self.assertEqual(gran, "datetime")
        local = due.astimezone(JST)
        self.assertEqual((local.day, local.hour), (4, 15))

    def test_hour_han(self) -> None:
        due, _ = parse_due("明日 18時半", self.now, JST)
        local = due.astimezone(JST)
        self.assertEqual((local.hour, local.minute), (18, 30))

    def test_hour_kanji_minute(self) -> None:
        due, _ = parse_due("明日 18時15分", self.now, JST)
        local = due.astimezone(JST)
        self.assertEqual((local.hour, local.minute), (18, 15))

    def test_time_word_morning(self) -> None:
        due, gran = parse_due("明日の朝", self.now, JST)
        self.assertEqual(gran, "datetime")
        local = due.astimezone(JST)
        self.assertEqual((local.day, local.hour), (4, 9))

    def test_konya(self) -> None:
        due, gran = parse_due("今夜", self.now, JST)
        self.assertEqual(gran, "datetime")
        local = due.astimezone(JST)
        self.assertEqual((local.day, local.hour), (3, 20))

    def test_day_only_this_month(self) -> None:
        due, gran = parse_due("10日", self.now, JST)
        self.assertEqual(gran, "date")
        local = due.astimezone(JST)
        self.assertEqual((local.month, local.day), (7, 10))

    def test_day_only_past_rolls_to_next_month(self) -> None:
        due, _ = parse_due("1日", self.now, JST)
        local = due.astimezone(JST)
        self.assertEqual((local.month, local.day), (8, 1))

    def test_ymd_hyphen(self) -> None:
        due, _ = parse_due("2026-09-01", self.now, JST)
        local = due.astimezone(JST)
        self.assertEqual((local.year, local.month, local.day), (2026, 9, 1))

    def test_ymd_kanji(self) -> None:
        due, _ = parse_due("2026年9月1日 10:00", self.now, JST)
        local = due.astimezone(JST)
        self.assertEqual((local.month, local.day, local.hour), (9, 1, 10))

    def test_relative_weeks(self) -> None:
        due, gran = parse_due("2週間後", self.now, JST)
        self.assertEqual(gran, "date")
        self.assertEqual(due.astimezone(JST).day, 17)

    def test_hiragana_alias(self) -> None:
        due, _ = parse_due("あさって", self.now, JST)
        self.assertEqual(due.astimezone(JST).day, 5)
        due, _ = parse_due("あしたの15時", self.now, JST)
        local = due.astimezone(JST)
        self.assertEqual((local.day, local.hour), (4, 15))

    def test_zenkaku_digits(self) -> None:
        due, _ = parse_due("７/１０ １５:００", self.now, JST)
        local = due.astimezone(JST)
        self.assertEqual((local.month, local.day, local.hour), (7, 10, 15))


class ParseSnoozeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime(2026, 7, 3, 1, 0, tzinfo=UTC)

    def test_minutes(self) -> None:
        self.assertEqual(parse_snooze("30m", self.now, JST), self.now + timedelta(minutes=30))
        self.assertEqual(parse_snooze("30分", self.now, JST), self.now + timedelta(minutes=30))

    def test_hours(self) -> None:
        self.assertEqual(parse_snooze("2h", self.now, JST), self.now + timedelta(hours=2))
        self.assertEqual(parse_snooze("2時間", self.now, JST), self.now + timedelta(hours=2))

    def test_tomorrow(self) -> None:
        until = parse_snooze("明日", self.now, JST)
        local = until.astimezone(JST)
        self.assertEqual((local.day, local.hour), (4, 9))

    def test_invalid(self) -> None:
        self.assertIsNone(parse_snooze("いつか", self.now, JST))


class ValidateExtractionTest(unittest.TestCase):
    def test_valid_dict(self) -> None:
        out = validate_extraction({"is_task": True, "title": "買い物", "due": None, "priority": "high"})
        self.assertEqual(out["title"], "買い物")
        self.assertEqual(out["priority"], "high")
        self.assertIsNone(out["due_at"])

    def test_valid_json_string(self) -> None:
        out = validate_extraction('{"is_task": true, "title": "提出", "due": "2026-07-10T15:00:00+09:00", "priority": "normal"}')
        self.assertEqual(out["title"], "提出")
        self.assertEqual(out["due_at"].astimezone(JST).hour, 15)

    def test_code_fence_stripped(self) -> None:
        out = validate_extraction('```json\n{"is_task": true, "title": "x", "due": null, "priority": "low"}\n```')
        self.assertIsNotNone(out)
        self.assertEqual(out["title"], "x")

    def test_is_task_false(self) -> None:
        self.assertIsNone(validate_extraction({"is_task": False, "title": "x"}))

    def test_missing_title(self) -> None:
        self.assertIsNone(validate_extraction({"is_task": True, "title": "  "}))

    def test_bad_json(self) -> None:
        self.assertIsNone(validate_extraction("not json at all"))

    def test_priority_defaults_normal(self) -> None:
        out = validate_extraction({"is_task": True, "title": "x", "priority": "urgent"})
        self.assertEqual(out["priority"], "normal")

    def test_z_suffix_iso(self) -> None:
        out = validate_extraction({"is_task": True, "title": "x", "due": "2026-07-10T06:00:00Z"})
        self.assertEqual(out["due_at"].astimezone(UTC).hour, 6)

    def test_naive_iso_assumed_local(self) -> None:
        # tzなしのdueは抽出プロンプトの前提であるローカル時刻 (Asia/Tokyo) として解釈する
        out = validate_extraction({"is_task": True, "title": "x", "due": "2026-07-05T23:59:00"})
        self.assertEqual(out["due_at"].astimezone(JST).hour, 23)
        self.assertEqual(out["due_at"].astimezone(UTC).hour, 14)

    def test_task_ui_reexport_matches_extractor(self) -> None:
        # task_ui.validate_extraction は src.tasks.extractor からの再エクスポート
        self.assertIs(validate_extraction, extractor_validate_extraction)
        self.assertIsNotNone(build_extraction_prompt(datetime(2026, 7, 3, 10, 0, tzinfo=JST)))

    def test_discord_offer_preserves_extracted_due_granularity(self) -> None:
        source = Path("src/discord_bot/bot.py").read_text(encoding="utf-8")
        start = source.index("async def maybe_offer_task_from_chat")
        end = source.index("async def handle_voice_reply", start)
        block = source[start:end]
        self.assertIn('granularity = extracted.get("due_granularity")', block)
        self.assertIn("due_granularity=granularity", block)


class ValidateMultiExtractionTest(unittest.TestCase):
    def _cand(self, title="x", due=None, priority="normal", is_task=True):
        return {"is_task": is_task, "title": title, "due": due, "priority": priority}

    def test_one_valid(self) -> None:
        out = validate_multi_extraction({"tasks": [self._cand("買い物", priority="high")]})
        self.assertIsNotNone(out)
        self.assertEqual(len(out["tasks"]), 1)
        self.assertEqual(out["tasks"][0]["title"], "買い物")
        self.assertEqual(out["tasks"][0]["priority"], "high")
        self.assertIsNone(out["tasks"][0]["due_at"])

    def test_three_valid(self) -> None:
        raw = {"tasks": [
            self._cand("A", due="2026-07-10T15:00:00+09:00"),
            self._cand("B", priority="low"),
            self._cand("C"),
        ]}
        out = validate_multi_extraction(raw)
        self.assertIsNotNone(out)
        self.assertEqual([t["title"] for t in out["tasks"]], ["A", "B", "C"])
        self.assertEqual(out["tasks"][0]["due_at"].astimezone(JST).hour, 15)
        self.assertEqual(out["tasks"][1]["priority"], "low")

    def test_json_string(self) -> None:
        out = validate_multi_extraction(
            '{"tasks": [{"is_task": true, "title": "x", "due": null, "priority": "normal"}]}'
        )
        self.assertIsNotNone(out)
        self.assertEqual(out["tasks"][0]["title"], "x")

    def test_code_fence_stripped(self) -> None:
        out = validate_multi_extraction(
            '```json\n{"tasks": [{"is_task": true, "title": "x", "due": null, "priority": "low"}]}\n```'
        )
        self.assertIsNotNone(out)
        self.assertEqual(out["tasks"][0]["title"], "x")

    def test_empty_array_is_valid_no_candidates(self) -> None:
        self.assertEqual(validate_multi_extraction({"tasks": []}), {"tasks": []})
        self.assertEqual(validate_multi_extraction('{"tasks": []}'), {"tasks": []})

    def test_four_candidates_fails_closed(self) -> None:
        raw = {"tasks": [self._cand("a"), self._cand("b"), self._cand("c"), self._cand("d")]}
        self.assertIsNone(validate_multi_extraction(raw))

    def test_is_task_false_candidate_fails_closed(self) -> None:
        raw = {"tasks": [self._cand("ok"), self._cand("chat", is_task=False)]}
        self.assertIsNone(validate_multi_extraction(raw))

    def test_empty_title_candidate_fails_closed(self) -> None:
        raw = {"tasks": [self._cand("ok"), self._cand("   ")]}
        self.assertIsNone(validate_multi_extraction(raw))

    def test_tasks_not_list_fails_closed(self) -> None:
        self.assertIsNone(validate_multi_extraction({"tasks": {"is_task": True}}))
        self.assertIsNone(validate_multi_extraction({"tasks": "x"}))

    def test_missing_tasks_key_fails_closed(self) -> None:
        self.assertIsNone(validate_multi_extraction({"is_task": True, "title": "x"}))

    def test_root_extra_key_fails_closed(self) -> None:
        raw = {"tasks": [self._cand("x")], "unexpected": True}
        self.assertIsNone(validate_multi_extraction(raw))

    def test_bad_json_fails_closed(self) -> None:
        self.assertIsNone(validate_multi_extraction("not json at all"))
        self.assertIsNone(validate_multi_extraction(123))
        self.assertIsNone(validate_multi_extraction(None))

    def test_unsafe_priority_fails_closed(self) -> None:
        self.assertIsNone(
            validate_multi_extraction({"tasks": [self._cand("x", priority="urgent")]})
        )
        self.assertIsNone(
            validate_multi_extraction({"tasks": [self._cand("x", priority=None)]})
        )

    def test_missing_due_key_fails_closed(self) -> None:
        raw = {"tasks": [{"is_task": True, "title": "x", "priority": "normal"}]}
        self.assertIsNone(validate_multi_extraction(raw))

    def test_extra_key_fails_closed(self) -> None:
        candidate = self._cand("x")
        candidate["note"] = "unexpected"
        self.assertIsNone(validate_multi_extraction({"tasks": [candidate]}))

    def test_invalid_non_null_due_fails_closed(self) -> None:
        self.assertIsNone(
            validate_multi_extraction({"tasks": [self._cand("x", due="not-a-date")]})
        )
        self.assertIsNone(
            validate_multi_extraction({"tasks": [self._cand("x", due="2026-13-99")]})
        )

    def test_null_due_still_accepted(self) -> None:
        out = validate_multi_extraction({"tasks": [self._cand("x", due=None)]})
        self.assertIsNotNone(out)
        self.assertIsNone(out["tasks"][0]["due_at"])
        self.assertIsNone(out["tasks"][0]["due_granularity"])

    def test_naive_due_assumed_local(self) -> None:
        out = validate_multi_extraction({"tasks": [self._cand("x", due="2026-07-05T23:59:00")]})
        self.assertEqual(out["tasks"][0]["due_at"].astimezone(JST).hour, 23)
        self.assertEqual(out["tasks"][0]["due_at"].astimezone(UTC).hour, 14)

    def test_z_suffix_due(self) -> None:
        out = validate_multi_extraction({"tasks": [self._cand("x", due="2026-07-10T06:00:00Z")]})
        self.assertEqual(out["tasks"][0]["due_at"].astimezone(UTC).hour, 6)

    def test_title_truncated_to_200(self) -> None:
        long_title = "あ" * 250
        out = validate_multi_extraction({"tasks": [self._cand(long_title)]})
        self.assertEqual(len(out["tasks"][0]["title"]), 200)

    def test_returns_tasks_key(self) -> None:
        out = validate_multi_extraction({"tasks": [self._cand("x")]})
        self.assertEqual(list(out.keys()), ["tasks"])
        self.assertIsInstance(out["tasks"], list)

    def test_multi_prompt_has_tasks_shape(self) -> None:
        prompt = build_multi_extraction_prompt(datetime(2026, 7, 3, 10, 0, tzinfo=JST))
        self.assertIn('"tasks":', prompt)
        self.assertIn("Asia/Tokyo", prompt)
        self.assertIn("1〜3件", prompt)
        self.assertIn("4件以上は禁止", prompt)
        self.assertIn("YYYY-MM-DD の日付だけ", prompt)


class IsSensitiveTextTest(unittest.TestCase):
    """高信頼度クレデンシャル検出器。実際の秘密値は一切含めない (サンプル・構造化形式のみ)。"""

    def test_non_string_is_not_sensitive(self) -> None:
        self.assertFalse(is_sensitive_text(None))
        self.assertFalse(is_sensitive_text(123))
        self.assertFalse(is_sensitive_text(""))

    def test_benign_text_not_sensitive(self) -> None:
        self.assertFalse(is_sensitive_text("APIキーの再発行をお願い"))
        self.assertFalse(is_sensitive_text("パスワードを忘れた"))
        self.assertFalse(is_sensitive_text("明日の会議の準備"))

    def test_github_token_structured_sample_is_sensitive(self) -> None:
        self.assertTrue(is_sensitive_text("ghp_" + "a" * 36))
        self.assertTrue(is_sensitive_text("token: ghp_" + "a" * 36))

    def test_aws_access_key_structured_sample_is_sensitive(self) -> None:
        self.assertTrue(is_sensitive_text("AKIA" + "A" * 16))

    def test_google_api_key_structured_sample_is_sensitive(self) -> None:
        self.assertTrue(is_sensitive_text("AIza" + "a" * 35))

    def test_stripe_key_structured_sample_is_sensitive(self) -> None:
        self.assertTrue(is_sensitive_text("sk_live_" + "a" * 24))

    def test_openai_style_token_structured_sample_is_sensitive(self) -> None:
        self.assertTrue(is_sensitive_text("sk-" + "a" * 24))

    def test_private_key_block_is_sensitive(self) -> None:
        self.assertTrue(
            is_sensitive_text("-----BEGIN RSA PRIVATE KEY-----\nMIIE...")
        )

    def test_key_value_assignment_is_sensitive(self) -> None:
        self.assertTrue(is_sensitive_text("password=AbCdEfGh12345678XyZ"))
        self.assertTrue(is_sensitive_text("api_key: " + "a" * 20))

    def test_explicit_short_secret_assignment_is_sensitive(self) -> None:
        self.assertTrue(is_sensitive_text("password=short"))

    def test_japanese_secret_assignment_is_sensitive(self) -> None:
        self.assertTrue(is_sensitive_text("パスワードは hunter2"))


class DueGranularityTest(unittest.TestCase):
    """シングル/マルチ検証結果の due_granularity (date vs datetime vs None)。"""

    def test_single_date_only_due_is_date(self) -> None:
        out = validate_extraction({"is_task": True, "title": "x", "due": "2026-07-10"})
        self.assertIsNotNone(out)
        self.assertEqual(out["due_granularity"], "date")
        local = out["due_at"].astimezone(JST)
        self.assertEqual((local.month, local.hour, local.minute), (7, 23, 59))

    def test_single_datetime_due_is_datetime(self) -> None:
        out = validate_extraction(
            {"is_task": True, "title": "x", "due": "2026-07-10T15:00:00+09:00"}
        )
        self.assertIsNotNone(out)
        self.assertEqual(out["due_granularity"], "datetime")

    def test_single_null_due_is_none(self) -> None:
        out = validate_extraction({"is_task": True, "title": "x", "due": None})
        self.assertIsNotNone(out)
        self.assertIsNone(out["due_granularity"])
        self.assertIsNone(out["due_at"])

    def test_multi_date_only_due_is_date(self) -> None:
        out = validate_multi_extraction(
            {"tasks": [{"is_task": True, "title": "x", "due": "2026-07-10", "priority": "normal"}]}
        )
        self.assertIsNotNone(out)
        self.assertEqual(out["tasks"][0]["due_granularity"], "date")
        local = out["tasks"][0]["due_at"].astimezone(JST)
        self.assertEqual((local.hour, local.minute), (23, 59))

    def test_multi_datetime_due_is_datetime(self) -> None:
        out = validate_multi_extraction(
            {"tasks": [{"is_task": True, "title": "x", "due": "2026-07-10T15:00:00", "priority": "normal"}]}
        )
        self.assertIsNotNone(out)
        self.assertEqual(out["tasks"][0]["due_granularity"], "datetime")


class SensitiveTitleRejectionTest(unittest.TestCase):
    """抽出済み title にクレデンシャルが含まれる場合は棄却 (シングルもマルチも fail closed)。"""

    def _secret_title(self) -> str:
        return "token: ghp_" + "a" * 36

    def test_single_sensitive_title_rejected(self) -> None:
        out = validate_extraction(
            {"is_task": True, "title": self._secret_title(), "due": None, "priority": "normal"}
        )
        self.assertIsNone(out)

    def test_single_benign_title_kept(self) -> None:
        out = validate_extraction({"is_task": True, "title": "買い物", "due": None, "priority": "normal"})
        self.assertIsNotNone(out)
        self.assertEqual(out["title"], "買い物")

    def test_multi_sensitive_title_fails_closed(self) -> None:
        raw = {"tasks": [
            {"is_task": True, "title": "OK task", "due": None, "priority": "normal"},
            {"is_task": True, "title": self._secret_title(), "due": None, "priority": "normal"},
        ]}
        self.assertIsNone(validate_multi_extraction(raw))

    def test_multi_benign_title_kept(self) -> None:
        raw = {"tasks": [
            {"is_task": True, "title": "OK task", "due": None, "priority": "normal"},
        ]}
        out = validate_multi_extraction(raw)
        self.assertIsNotNone(out)
        self.assertEqual(out["tasks"][0]["title"], "OK task")


class PromptNoEchoCredentialsTest(unittest.TestCase):
    """プロンプトにクレデンシャル非エコー指示が含まれる。"""

    def test_single_prompt_instructs_no_echo(self) -> None:
        prompt = build_extraction_prompt(datetime(2026, 7, 3, 10, 0, tzinfo=JST))
        self.assertIn("クレデンシャル", prompt)
        self.assertIn("転写・引用", prompt)

    def test_multi_prompt_instructs_no_echo(self) -> None:
        prompt = build_multi_extraction_prompt(datetime(2026, 7, 3, 10, 0, tzinfo=JST))
        self.assertIn("クレデンシャル", prompt)
        self.assertIn("転写・引用", prompt)


if __name__ == "__main__":
    unittest.main()
