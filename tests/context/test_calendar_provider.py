"""Calendar Context Provider / ChatSession互換wiringのテスト。"""
from __future__ import annotations

import unittest

from src.chat.session import ChatSession
from src.context import CalendarContextProvider, CalendarSource
from src.context.builder import ContextBuilder
from src.context.contracts import ContextBlock
from src.context.providers.calendar import CalendarContextProvider as ProviderImpl
from src.context.providers.calendar import CalendarSource as SourceImpl


class _FakeCalendar:
    def __init__(
        self,
        text: str = "\n--- 予定 (Google Calendar) ---\n今日の会議: 14:00",
    ) -> None:
        self._text = text
        self.calls: int = 0

    def get_context_text(self) -> str:
        self.calls += 1
        return self._text


class _EmptyCalendar:
    def get_context_text(self) -> str:
        return ""


class _NonStrCalendar:
    def get_context_text(self) -> str:
        return None  # type: ignore[return-value]


class _BrokenCalendar:
    def get_context_text(self) -> str:
        raise RuntimeError("secret calendar body")


class _FakeVision:
    def get_context_text(self) -> str:
        return "\n[Vision] 現在の視界"


class _FakeMonitor:
    def get_context_text(self) -> str:
        return "\n--- サブPCの現在の状態 ---\n- CPU: 10% (低負荷)"


class _FakeScreen:
    def get_context_text(self) -> str:
        return "\n[Screen] 画面の内容"


class CalendarSourceProtocolTest(unittest.TestCase):
    def test_calendars_conform_to_calendar_source(self) -> None:
        self.assertIsInstance(_FakeCalendar(), CalendarSource)
        self.assertIsInstance(_BrokenCalendar(), CalendarSource)

    def test_unrelated_object_does_not_conform(self) -> None:
        self.assertNotIsInstance(object(), CalendarSource)


class CalendarContextProviderTest(unittest.TestCase):
    def test_returns_block_with_expected_metadata(self) -> None:
        calendar = _FakeCalendar()
        result = CalendarContextProvider.collect(calendar)
        self.assertIsNotNone(result)
        self.assertEqual(result.source, "calendar")
        self.assertEqual(result.sensitivity, "personal")
        self.assertIs(result.local_only, True)
        self.assertIsInstance(result.content, str)
        self.assertEqual(result.content, calendar._text)
        self.assertEqual(calendar.calls, 1)

    def test_empty_text_returns_none(self) -> None:
        self.assertIsNone(CalendarContextProvider.collect(_EmptyCalendar()))

    def test_whitespace_text_returns_none(self) -> None:
        self.assertIsNone(CalendarContextProvider.collect(_FakeCalendar("   \n  ")))

    def test_non_str_result_returns_none(self) -> None:
        self.assertIsNone(CalendarContextProvider.collect(_NonStrCalendar()))

    def test_exception_returns_none(self) -> None:
        self.assertIsNone(CalendarContextProvider.collect(_BrokenCalendar()))

    def test_exception_logs_type_only_not_body(self) -> None:
        with self.assertLogs(
            "src.context.providers.calendar", level="WARNING"
        ) as captured:
            result = CalendarContextProvider.collect(_BrokenCalendar())
        self.assertIsNone(result)
        self.assertEqual(len(captured.output), 1)
        self.assertIn("RuntimeError", captured.output[0])
        self.assertNotIn("secret calendar body", captured.output[0])


class CalendarCloudFilterTest(unittest.TestCase):
    def test_cloud_target_excludes_personal_local_only_calendar(self) -> None:
        builder = ContextBuilder("base")
        calendar_block = CalendarContextProvider.collect(_FakeCalendar())
        public_block = ContextBlock(
            source="tasks", content="公開", sensitivity="public"
        )
        result = builder.build_system_content(
            [calendar_block, public_block], privacy="cloud_allowed", target_local=False
        )
        self.assertEqual(result, "base公開")


class ChatSessionCalendarWiringTest(unittest.TestCase):
    def test_exact_payload_with_vision_monitor_screen_calendar_order(self) -> None:
        calendar = _FakeCalendar()
        session = ChatSession(
            system_prompt="sys",
            vision_context=_FakeVision(),
            monitor_context=_FakeMonitor(),
            screen_context=_FakeScreen(),
            calendar_context=calendar,
        )
        session.add_user_message("hi")
        session.add_assistant_message("a")
        messages = session.build_messages()
        self.assertEqual(calendar.calls, 1)
        self.assertEqual(
            messages,
            [
                {
                    "role": "system",
                    "content": (
                        "sys\n[Vision] 現在の視界"
                        "\n--- サブPCの現在の状態 ---\n- CPU: 10% (低負荷)"
                        "\n[Screen] 画面の内容"
                        "\n--- 予定 (Google Calendar) ---\n今日の会議: 14:00"
                    ),
                },
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "a"},
            ],
        )

    def test_empty_calendar_payload_matches_no_calendar(self) -> None:
        session = ChatSession(
            system_prompt="sys",
            vision_context=_FakeVision(),
            monitor_context=_FakeMonitor(),
            screen_context=_FakeScreen(),
            calendar_context=_EmptyCalendar(),
        )
        session.add_user_message("hi")
        session.add_assistant_message("a")
        self.assertEqual(
            session.build_messages(),
            [
                {
                    "role": "system",
                    "content": (
                        "sys\n[Vision] 現在の視界"
                        "\n--- サブPCの現在の状態 ---\n- CPU: 10% (低負荷)"
                        "\n[Screen] 画面の内容"
                    ),
                },
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "a"},
            ],
        )

    def test_broken_calendar_does_not_stop_conversation(self) -> None:
        session = ChatSession(
            system_prompt="sys",
            vision_context=_FakeVision(),
            monitor_context=_FakeMonitor(),
            screen_context=_FakeScreen(),
            calendar_context=_BrokenCalendar(),
        )
        session.add_user_message("hi")
        session.add_assistant_message("a")
        with self.assertLogs("src.context.providers.calendar", level="WARNING"):
            messages = session.build_messages()
        self.assertEqual(
            messages,
            [
                {
                    "role": "system",
                    "content": (
                        "sys\n[Vision] 現在の視界"
                        "\n--- サブPCの現在の状態 ---\n- CPU: 10% (低負荷)"
                        "\n[Screen] 画面の内容"
                    ),
                },
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "a"},
            ],
        )

    def test_no_calendar_unchanged(self) -> None:
        session = ChatSession(
            system_prompt="sys",
            vision_context=_FakeVision(),
            monitor_context=_FakeMonitor(),
            screen_context=_FakeScreen(),
        )
        session.add_user_message("hi")
        session.add_assistant_message("a")
        self.assertEqual(
            session.build_messages(),
            [
                {
                    "role": "system",
                    "content": (
                        "sys\n[Vision] 現在の視界"
                        "\n--- サブPCの現在の状態 ---\n- CPU: 10% (低負荷)"
                        "\n[Screen] 画面の内容"
                    ),
                },
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "a"},
            ],
        )

    def test_calendar_before_emotion_tags(self) -> None:
        session = ChatSession(
            system_prompt="sys",
            screen_context=_FakeScreen(),
            calendar_context=_FakeCalendar(),
            emotion_tags=True,
        )
        session.add_user_message("hi")
        session.add_assistant_message("a")
        messages = session.build_messages()
        self.assertEqual(messages[0]["role"], "system")
        content = messages[0]["content"]
        cal_idx = content.index("--- 予定 (Google Calendar) ---")
        screen_idx = content.index("[Screen] 画面の内容")
        self.assertLess(screen_idx, cal_idx)
        self.assertTrue(content.endswith("タグは応答の先頭以外に書かないでください。"))


class RootContextPublicAPITest(unittest.TestCase):
    def test_calendar_provider_exported_from_root(self) -> None:
        import src.context

        self.assertIn("CalendarContextProvider", src.context.__all__)
        self.assertIs(src.context.CalendarContextProvider, ProviderImpl)

    def test_calendar_source_exported_from_root(self) -> None:
        import src.context

        self.assertIn("CalendarSource", src.context.__all__)
        self.assertIs(src.context.CalendarSource, SourceImpl)
        self.assertIsInstance(_FakeCalendar(), src.context.CalendarSource)

    def test_calendar_exported_from_providers(self) -> None:
        import src.context.providers

        self.assertIn("CalendarContextProvider", src.context.providers.__all__)
        self.assertIn("CalendarSource", src.context.providers.__all__)


if __name__ == "__main__":
    unittest.main()
