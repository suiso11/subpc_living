"""Screen Context Provider / ChatSession互換wiringのテスト。"""
from __future__ import annotations

import unittest

from src.chat.session import ChatSession
from src.context import ScreenContextProvider, ScreenSource
from src.context.builder import ContextBuilder
from src.context.contracts import ContextBlock
from src.context.providers.screen import ScreenContextProvider as ProviderImpl
from src.context.providers.screen import ScreenSource as SourceImpl


class _FakeScreen:
    def __init__(self, text: str = "\n[Screen] 画面の内容") -> None:
        self._text = text
        self.calls: int = 0

    def get_context_text(self) -> str:
        self.calls += 1
        return self._text


class _EmptyScreen:
    def get_context_text(self) -> str:
        return ""


class _NonStrScreen:
    def get_context_text(self) -> str:
        return None  # type: ignore[return-value]


class _BrokenScreen:
    def get_context_text(self) -> str:
        raise RuntimeError("secret screen body")


class _FakeVision:
    def get_context_text(self) -> str:
        return "\n[Vision] 現在の視界"


class _FakeMonitor:
    def get_context_text(self) -> str:
        return "\n--- サブPCの現在の状態 ---\n- CPU: 10% (低負荷)"


class _FakeCalendar:
    def get_context_text(self) -> str:
        return "\n--- 予定 ---\n今日の会議: 14:00"


class ScreenSourceProtocolTest(unittest.TestCase):
    def test_screens_conform_to_screen_source(self) -> None:
        self.assertIsInstance(_FakeScreen(), ScreenSource)
        self.assertIsInstance(_BrokenScreen(), ScreenSource)

    def test_unrelated_object_does_not_conform(self) -> None:
        self.assertNotIsInstance(object(), ScreenSource)


class ScreenContextProviderTest(unittest.TestCase):
    def test_returns_block_with_expected_metadata(self) -> None:
        screen = _FakeScreen()
        result = ScreenContextProvider.collect(screen)
        self.assertIsNotNone(result)
        self.assertEqual(result.source, "screen")
        self.assertEqual(result.sensitivity, "secret")
        self.assertIs(result.local_only, True)
        self.assertIsInstance(result.content, str)
        self.assertEqual(result.content, screen._text)
        self.assertEqual(screen.calls, 1)

    def test_empty_text_returns_none(self) -> None:
        self.assertIsNone(ScreenContextProvider.collect(_EmptyScreen()))

    def test_whitespace_text_returns_none(self) -> None:
        self.assertIsNone(ScreenContextProvider.collect(_FakeScreen("   \n  ")))

    def test_non_str_result_returns_none(self) -> None:
        self.assertIsNone(ScreenContextProvider.collect(_NonStrScreen()))

    def test_exception_returns_none(self) -> None:
        self.assertIsNone(ScreenContextProvider.collect(_BrokenScreen()))

    def test_exception_logs_type_only_not_body(self) -> None:
        with self.assertLogs(
            "src.context.providers.screen", level="WARNING"
        ) as captured:
            result = ScreenContextProvider.collect(_BrokenScreen())
        self.assertIsNone(result)
        self.assertEqual(len(captured.output), 1)
        self.assertIn("RuntimeError", captured.output[0])
        self.assertNotIn("secret screen body", captured.output[0])


class ScreenCloudFilterTest(unittest.TestCase):
    def test_cloud_target_excludes_secret_local_only_screen(self) -> None:
        builder = ContextBuilder("base")
        screen_block = ScreenContextProvider.collect(_FakeScreen())
        public_block = ContextBlock(
            source="tasks", content="公開", sensitivity="public"
        )
        result = builder.build_system_content(
            [screen_block, public_block], privacy="cloud_allowed", target_local=False
        )
        self.assertEqual(result, "base公開")


class ChatSessionScreenWiringTest(unittest.TestCase):
    def test_exact_payload_with_vision_monitor_screen_calendar_order(self) -> None:
        screen = _FakeScreen()
        session = ChatSession(
            system_prompt="sys",
            vision_context=_FakeVision(),
            monitor_context=_FakeMonitor(),
            screen_context=screen,
            calendar_context=_FakeCalendar(),
        )
        session.add_user_message("hi")
        session.add_assistant_message("a")
        messages = session.build_messages()
        self.assertEqual(screen.calls, 1)
        self.assertEqual(
            messages,
            [
                {
                    "role": "system",
                    "content": (
                        "sys\n[Vision] 現在の視界"
                        "\n--- サブPCの現在の状態 ---\n- CPU: 10% (低負荷)"
                        "\n[Screen] 画面の内容"
                        "\n--- 予定 ---\n今日の会議: 14:00"
                    ),
                },
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "a"},
            ],
        )

    def test_exact_payload_with_monitor_screen_order_no_vision(self) -> None:
        session = ChatSession(
            system_prompt="sys",
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
                        "sys"
                        "\n--- サブPCの現在の状態 ---\n- CPU: 10% (低負荷)"
                        "\n[Screen] 画面の内容"
                    ),
                },
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "a"},
            ],
        )

    def test_empty_screen_payload_matches_no_screen(self) -> None:
        session = ChatSession(
            system_prompt="sys",
            vision_context=_FakeVision(),
            monitor_context=_FakeMonitor(),
            screen_context=_EmptyScreen(),
            calendar_context=_FakeCalendar(),
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
                        "\n--- 予定 ---\n今日の会議: 14:00"
                    ),
                },
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "a"},
            ],
        )

    def test_broken_screen_does_not_stop_conversation(self) -> None:
        session = ChatSession(
            system_prompt="sys",
            vision_context=_FakeVision(),
            monitor_context=_FakeMonitor(),
            screen_context=_BrokenScreen(),
            calendar_context=_FakeCalendar(),
        )
        session.add_user_message("hi")
        session.add_assistant_message("a")
        with self.assertLogs("src.context.providers.screen", level="WARNING"):
            messages = session.build_messages()
        self.assertEqual(
            messages,
            [
                {
                    "role": "system",
                    "content": (
                        "sys\n[Vision] 現在の視界"
                        "\n--- サブPCの現在の状態 ---\n- CPU: 10% (低負荷)"
                        "\n--- 予定 ---\n今日の会議: 14:00"
                    ),
                },
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "a"},
            ],
        )

    def test_no_screen_unchanged(self) -> None:
        session = ChatSession(
            system_prompt="sys",
            vision_context=_FakeVision(),
            monitor_context=_FakeMonitor(),
            calendar_context=_FakeCalendar(),
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
                        "\n--- 予定 ---\n今日の会議: 14:00"
                    ),
                },
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "a"},
            ],
        )


class RootContextPublicAPITest(unittest.TestCase):
    def test_screen_provider_exported_from_root(self) -> None:
        import src.context

        self.assertIn("ScreenContextProvider", src.context.__all__)
        self.assertIs(src.context.ScreenContextProvider, ProviderImpl)

    def test_screen_source_exported_from_root(self) -> None:
        import src.context

        self.assertIn("ScreenSource", src.context.__all__)
        self.assertIs(src.context.ScreenSource, SourceImpl)
        self.assertIsInstance(_FakeScreen(), src.context.ScreenSource)

    def test_screen_exported_from_providers(self) -> None:
        import src.context.providers

        self.assertIn("ScreenContextProvider", src.context.providers.__all__)
        self.assertIn("ScreenSource", src.context.providers.__all__)


if __name__ == "__main__":
    unittest.main()
