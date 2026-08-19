"""Monitor Context Provider / ChatSession互換wiringのテスト。"""
from __future__ import annotations

import unittest

from src.chat.session import ChatSession
from src.context import MonitorContextProvider, MonitorSource
from src.context.builder import ContextBuilder
from src.context.contracts import ContextBlock
from src.context.providers.monitor import MonitorContextProvider as ProviderImpl
from src.context.providers.monitor import MonitorSource as SourceImpl


class _FakeMonitor:
    def __init__(
        self,
        text: str = "\n--- サブPCの現在の状態 ---\n- CPU: 10% (低負荷)",
    ) -> None:
        self._text = text
        self.calls: int = 0

    def get_context_text(self) -> str:
        self.calls += 1
        return self._text


class _EmptyMonitor:
    def get_context_text(self) -> str:
        return ""


class _NonStrMonitor:
    def get_context_text(self) -> str:
        return None  # type: ignore[return-value]


class _BrokenMonitor:
    def get_context_text(self) -> str:
        raise RuntimeError("secret monitor body")


class _FakeVision:
    def get_context_text(self) -> str:
        return "\n[Vision] 現在の視界"


class _FakeScreen:
    def get_context_text(self) -> str:
        return "\n[Screen] 画面の内容"


class MonitorSourceProtocolTest(unittest.TestCase):
    def test_monitors_conform_to_monitor_source(self) -> None:
        self.assertIsInstance(_FakeMonitor(), MonitorSource)
        self.assertIsInstance(_BrokenMonitor(), MonitorSource)

    def test_unrelated_object_does_not_conform(self) -> None:
        self.assertNotIsInstance(object(), MonitorSource)


class MonitorContextProviderTest(unittest.TestCase):
    def test_returns_block_with_expected_metadata(self) -> None:
        monitor = _FakeMonitor()
        result = MonitorContextProvider.collect(monitor)
        self.assertIsNotNone(result)
        self.assertEqual(result.source, "monitor")
        self.assertEqual(result.sensitivity, "personal")
        self.assertIs(result.local_only, True)
        self.assertIsInstance(result.content, str)
        self.assertEqual(result.content, monitor._text)
        self.assertEqual(monitor.calls, 1)

    def test_empty_text_returns_none(self) -> None:
        self.assertIsNone(MonitorContextProvider.collect(_EmptyMonitor()))

    def test_whitespace_text_returns_none(self) -> None:
        self.assertIsNone(MonitorContextProvider.collect(_FakeMonitor("   \n  ")))

    def test_non_str_result_returns_none(self) -> None:
        self.assertIsNone(MonitorContextProvider.collect(_NonStrMonitor()))

    def test_exception_returns_none(self) -> None:
        self.assertIsNone(MonitorContextProvider.collect(_BrokenMonitor()))

    def test_exception_logs_type_only_not_body(self) -> None:
        with self.assertLogs(
            "src.context.providers.monitor", level="WARNING"
        ) as captured:
            result = MonitorContextProvider.collect(_BrokenMonitor())
        self.assertIsNone(result)
        self.assertEqual(len(captured.output), 1)
        self.assertIn("RuntimeError", captured.output[0])
        self.assertNotIn("secret monitor body", captured.output[0])


class MonitorCloudFilterTest(unittest.TestCase):
    def test_cloud_target_excludes_personal_local_only_monitor(self) -> None:
        builder = ContextBuilder("base")
        monitor_block = MonitorContextProvider.collect(_FakeMonitor())
        public_block = ContextBlock(
            source="screen", content="公開", sensitivity="public"
        )
        result = builder.build_system_content(
            [monitor_block, public_block], privacy="cloud_allowed", target_local=False
        )
        self.assertEqual(result, "base公開")


class ChatSessionMonitorWiringTest(unittest.TestCase):
    def test_exact_payload_with_vision_monitor_screen_order(self) -> None:
        monitor = _FakeMonitor()
        session = ChatSession(
            system_prompt="sys",
            vision_context=_FakeVision(),
            monitor_context=monitor,
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

    def test_empty_monitor_payload_matches_no_monitor(self) -> None:
        session = ChatSession(
            system_prompt="sys",
            vision_context=_FakeVision(),
            monitor_context=_EmptyMonitor(),
            screen_context=_FakeScreen(),
        )
        session.add_user_message("hi")
        session.add_assistant_message("a")
        self.assertEqual(
            session.build_messages(),
            [
                {
                    "role": "system",
                    "content": "sys\n[Vision] 現在の視界\n[Screen] 画面の内容",
                },
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "a"},
            ],
        )

    def test_broken_monitor_does_not_stop_conversation(self) -> None:
        session = ChatSession(
            system_prompt="sys",
            vision_context=_FakeVision(),
            monitor_context=_BrokenMonitor(),
            screen_context=_FakeScreen(),
        )
        session.add_user_message("hi")
        session.add_assistant_message("a")
        with self.assertLogs("src.context.providers.monitor", level="WARNING"):
            messages = session.build_messages()
        self.assertEqual(
            messages,
            [
                {
                    "role": "system",
                    "content": "sys\n[Vision] 現在の視界\n[Screen] 画面の内容",
                },
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "a"},
            ],
        )

    def test_no_monitor_unchanged(self) -> None:
        session = ChatSession(
            system_prompt="sys",
            vision_context=_FakeVision(),
            screen_context=_FakeScreen(),
        )
        session.add_user_message("hi")
        session.add_assistant_message("a")
        self.assertEqual(
            session.build_messages(),
            [
                {
                    "role": "system",
                    "content": "sys\n[Vision] 現在の視界\n[Screen] 画面の内容",
                },
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "a"},
            ],
        )


class RootContextPublicAPITest(unittest.TestCase):
    def test_monitor_provider_exported_from_root(self) -> None:
        import src.context

        self.assertIn("MonitorContextProvider", src.context.__all__)
        self.assertIs(src.context.MonitorContextProvider, ProviderImpl)

    def test_monitor_source_exported_from_root(self) -> None:
        import src.context

        self.assertIn("MonitorSource", src.context.__all__)
        self.assertIs(src.context.MonitorSource, SourceImpl)
        self.assertIsInstance(_FakeMonitor(), src.context.MonitorSource)

    def test_monitor_exported_from_providers(self) -> None:
        import src.context.providers

        self.assertIn("MonitorContextProvider", src.context.providers.__all__)
        self.assertIn("MonitorSource", src.context.providers.__all__)


if __name__ == "__main__":
    unittest.main()
