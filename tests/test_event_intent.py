"""自然言語からの Google Calendar 予定登録のテスト。"""
import json
import os
import tempfile
import types
import unittest
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from src.tasks.event_intent import (
    detect_event_intent,
    parse_event_request,
    try_register_event,
)

UTC = timezone.utc
JST = ZoneInfo("Asia/Tokyo")


class DetectEventIntentTest(unittest.TestCase):
    def test_explicit_prefix_colon(self):
        """「予定: ...」で intent を検出。"""
        text = "予定: 明日15時 歯医者"
        result = detect_event_intent(text)
        self.assertIsNotNone(result)
        self.assertIn("明日", result)
        self.assertIn("歯医者", result)

    def test_explicit_prefix_ascii_colon(self):
        """「よてい: ...」（ASCII コロン）で intent を検出。"""
        text = "よてい: 明日15時 歯医者"
        result = detect_event_intent(text)
        self.assertIsNotNone(result)
        self.assertIn("明日", result)

    def test_natural_language_event_trigger(self):
        """「明日15時に歯医者の予定入れて」で intent を検出。"""
        text = "明日15時に歯医者の予定入れて"
        result = detect_event_intent(text)
        self.assertIsNotNone(result)
        self.assertIn("歯医者", result)

    def test_calendar_trigger(self):
        """「カレンダーに金曜の飲み会入れて」で intent を検出。"""
        text = "カレンダーに金曜の飲み会入れて"
        result = detect_event_intent(text)
        self.assertIsNotNone(result)
        self.assertIn("飲み会", result)

    def test_no_intent_casual_chat(self):
        """「今日は暑いね」では intent なし。"""
        text = "今日は暑いね"
        result = detect_event_intent(text)
        self.assertIsNone(result)

    def test_no_intent_task_prefix(self):
        """「タスク: 明日 買い物」では intent なし。"""
        text = "タスク: 明日 買い物"
        result = detect_event_intent(text)
        self.assertIsNone(result)

    def test_no_intent_empty(self):
        """空文字列では intent なし。"""
        result = detect_event_intent("")
        self.assertIsNone(result)


class ParseEventRequestTest(unittest.TestCase):
    def test_parse_with_time(self):
        """「明日15時 歯医者」→ title、start、all_day=False を取得。"""
        now = datetime(2026, 7, 10, 3, 0, tzinfo=UTC)
        tz = JST
        body = "明日15時 歯医者"

        req = parse_event_request(body, now, tz)

        self.assertIsNotNone(req)
        self.assertEqual(req["title"], "歯医者")
        self.assertEqual(req["start"], "2026-07-11T15:00:00")
        self.assertFalse(req["all_day"])

    def test_parse_all_day(self):
        """「明日 歯医者」→ all_day=True, start=\"YYYY-MM-DD\"。"""
        now = datetime(2026, 7, 10, 3, 0, tzinfo=UTC)
        tz = JST
        body = "明日 歯医者"

        req = parse_event_request(body, now, tz)

        self.assertIsNotNone(req)
        self.assertEqual(req["title"], "歯医者")
        self.assertEqual(req["start"], "2026-07-11")
        self.assertTrue(req["all_day"])

    def test_parse_no_date(self):
        """「歯医者」（日時なし）→ None。"""
        now = datetime(2026, 7, 10, 3, 0, tzinfo=UTC)
        tz = JST
        body = "歯医者"

        req = parse_event_request(body, now, tz)

        self.assertIsNone(req)


class TryRegisterEventTest(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.upcoming_path = os.path.join(self.tmp_dir, "upcoming.json")

    def _make_fake_client(self):
        """create_event が ok=True, event_id="ev1" を返す fake client。"""
        client = types.SimpleNamespace()
        client.create_event = lambda **kwargs: types.SimpleNamespace(
            ok=True, event_id="ev1"
        )
        return client

    def test_register_with_intent_and_date(self):
        """予定意図あり+日時あり → 登録成功メッセージ。"""
        now = datetime(2026, 7, 10, 3, 0, tzinfo=UTC)
        client = self._make_fake_client()
        text = "予定: 明日15時 歯医者"

        result = try_register_event(
            text,
            client=client,
            calendar_id="primary",
            timezone_name="Asia/Tokyo",
            now=now,
        )

        self.assertIsNotNone(result)
        self.assertIn("予定を登録しました", result)
        self.assertIn("歯医者", result)

    def test_register_appends_to_upcoming(self):
        """upcoming_path を渡した場合、upcoming.json に append される。"""
        now = datetime(2026, 7, 10, 3, 0, tzinfo=UTC)
        client = self._make_fake_client()
        text = "予定: 明日15時 歯医者"

        result = try_register_event(
            text,
            client=client,
            calendar_id="primary",
            timezone_name="Asia/Tokyo",
            upcoming_path=self.upcoming_path,
            now=now,
        )

        self.assertIsNotNone(result)
        self.assertIn("予定を登録しました", result)

        # upcoming.json が作成されていることを確認
        self.assertTrue(os.path.exists(self.upcoming_path))
        with open(self.upcoming_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertIn("events", data)
        self.assertEqual(len(data["events"]), 1)
        self.assertEqual(data["events"][0]["title"], "歯医者")

    def test_no_intent_returns_none(self):
        """意図なしテキスト → None。"""
        now = datetime(2026, 7, 10, 3, 0, tzinfo=UTC)
        client = self._make_fake_client()
        text = "今日は暑いね"

        result = try_register_event(
            text,
            client=client,
            calendar_id="primary",
            timezone_name="Asia/Tokyo",
            now=now,
        )

        self.assertIsNone(result)

    def test_intent_without_date_returns_guidance(self):
        """意図あり日時なし → 「日時がわかりません」を含む案内文。"""
        now = datetime(2026, 7, 10, 3, 0, tzinfo=UTC)
        client = self._make_fake_client()
        text = "予定: 歯医者"

        result = try_register_event(
            text,
            client=client,
            calendar_id="primary",
            timezone_name="Asia/Tokyo",
            now=now,
        )

        self.assertIsNotNone(result)
        self.assertIn("日時がわかりません", result)

    def test_client_none_returns_guidance(self):
        """client=None で意図+日時あり → 「登録できません」系の案内文。"""
        now = datetime(2026, 7, 10, 3, 0, tzinfo=UTC)
        text = "予定: 明日15時 歯医者"

        result = try_register_event(
            text,
            client=None,
            calendar_id="primary",
            timezone_name="Asia/Tokyo",
            now=now,
        )

        self.assertIsNotNone(result)
        self.assertIn("登録できません", result)

    def test_create_event_called_with_correct_args(self):
        """client.create_event が calendar_id、summary、start、end で呼ばれる。"""
        now = datetime(2026, 7, 10, 3, 0, tzinfo=UTC)
        call_log = []

        client = types.SimpleNamespace()
        def mock_create_event(**kwargs):
            call_log.append(kwargs)
            return types.SimpleNamespace(ok=True, event_id="ev1")
        client.create_event = mock_create_event

        text = "予定: 明日15時 歯医者"
        result = try_register_event(
            text,
            client=client,
            calendar_id="primary",
            timezone_name="Asia/Tokyo",
            now=now,
        )

        self.assertIsNotNone(result)
        self.assertEqual(len(call_log), 1)
        kwargs = call_log[0]
        self.assertEqual(kwargs["calendar_id"], "primary")
        self.assertEqual(kwargs["summary"], "歯医者")
        self.assertIn("2026-07-11", kwargs["start"])
        self.assertIn("15:00", kwargs["start"])


if __name__ == "__main__":
    unittest.main()
