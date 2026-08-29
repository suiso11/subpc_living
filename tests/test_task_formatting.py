from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from src.tasks.formatting import format_short_due
from src.tasks.store import format_local_due, from_iso, to_iso

UTC = timezone.utc
JST = ZoneInfo("Asia/Tokyo")


class FormatShortDueTest(unittest.TestCase):
    """format_short_due はゼロ埋めなし M/D とゼロ埋め HH:MM を組み立てる。

    クロスプラットフォーム固定: `strftime("%-m/%-d")` は Windows 非対応のため、
    datetime 属性から組み立てる。M/D のゼロ埋めなし・時刻のゼロ埋めありを
    プラットフォームに依存せず固定する。
    """

    def test_single_digit_month_and_day(self) -> None:
        self.assertEqual(format_short_due(datetime(2026, 1, 2)), "1/2")

    def test_double_digit_month_and_day(self) -> None:
        self.assertEqual(format_short_due(datetime(2026, 12, 31)), "12/31")

    def test_mixed_digit_month_day_no_zero_padding(self) -> None:
        self.assertEqual(format_short_due(datetime(2026, 11, 5)), "11/5")
        self.assertEqual(format_short_due(datetime(2026, 9, 30)), "9/30")
        self.assertNotIn("09/05", format_short_due(datetime(2026, 9, 5)))

    def test_with_time_zero_pads_hour_and_minute(self) -> None:
        self.assertEqual(
            format_short_due(datetime(2026, 3, 5, 9, 7), with_time=True), "3/5 09:07"
        )

    def test_with_time_midnight_zero_pads(self) -> None:
        self.assertEqual(
            format_short_due(datetime(2026, 11, 23, 0, 0), with_time=True), "11/23 00:00"
        )

    def test_exact_m_d_output_with_time(self) -> None:
        self.assertEqual(
            format_short_due(datetime(2026, 12, 31, 23, 59), with_time=True), "12/31 23:59"
        )

    def test_timezone_aware_input_clock_is_preserved(self) -> None:
        # format_short_due は時刻を変換せず、入力のローカル時計をそのまま使う。
        aware = datetime(2026, 7, 3, 4, 5, tzinfo=timezone(timedelta(hours=5, minutes=30)))
        self.assertEqual(format_short_due(aware), "7/3")
        self.assertEqual(format_short_due(aware, with_time=True), "7/3 04:05")


class FormatLocalDueTest(unittest.TestCase):
    """store の date/datetime 粒度に応じたローカル表示を固定する。"""

    def test_no_due_returns_none_text(self) -> None:
        self.assertEqual(format_local_due(None, None, JST, datetime(2026, 7, 3, tzinfo=UTC)), "期限なし")

    def test_date_granularity_omits_time(self) -> None:
        due = datetime(2026, 7, 3, tzinfo=UTC)
        self.assertEqual(format_local_due(due, "date", JST, due), "7/3")

    def test_datetime_granularity_includes_time(self) -> None:
        due = datetime(2026, 7, 3, 3, 30, tzinfo=UTC)
        self.assertEqual(format_local_due(due, "datetime", JST, due), "7/3 12:30")

    def test_datetime_granularity_converts_to_local_tz(self) -> None:
        # UTC 15:00 -> JST 翌日 00:00。日付繰り上がりを固定する。
        due = datetime(2026, 1, 1, 15, 0, tzinfo=UTC)
        self.assertEqual(format_local_due(due, "datetime", JST, due), "1/2 00:00")

    def test_date_granularity_converts_to_local_tz(self) -> None:
        # date 粒度でも表示はローカル日付に変換する。
        due = datetime(2026, 1, 1, 15, 0, tzinfo=UTC)
        self.assertEqual(format_local_due(due, "date", JST, due), "1/2")


class StoreIsoRoundTripTest(unittest.TestCase):
    """store の UTC ISO 保存が aware datetime の瞬間を保持することを固定する。"""

    def test_to_iso_converts_aware_input_to_utc(self) -> None:
        aware = datetime(2026, 7, 3, 12, 30, tzinfo=timezone(timedelta(hours=9)))
        self.assertEqual(to_iso(aware), "2026-07-03T03:30:00+00:00")

    def test_to_iso_naive_treated_as_utc(self) -> None:
        self.assertEqual(to_iso(datetime(2026, 7, 3, 3, 30)), "2026-07-03T03:30:00+00:00")

    def test_from_iso_returns_utc_aware(self) -> None:
        dt = from_iso("2026-07-03T12:30:00+09:00")
        self.assertEqual(dt, datetime(2026, 7, 3, 3, 30, tzinfo=UTC))
        self.assertIsNotNone(dt.tzinfo)

    def test_round_trip_preserves_instant(self) -> None:
        aware = datetime(2026, 1, 1, 15, 0, tzinfo=UTC)
        self.assertEqual(from_iso(to_iso(aware)), aware)

    def test_from_iso_empty_returns_none(self) -> None:
        self.assertIsNone(from_iso(None))
        self.assertIsNone(from_iso(""))


if __name__ == "__main__":
    unittest.main()