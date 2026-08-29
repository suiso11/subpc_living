"""Vision Context Provider / ChatSession互換wiringのテスト。"""
from __future__ import annotations

import unittest

from src.chat.session import ChatSession
from src.context import VisionContextProvider, VisionSource
from src.context.builder import ContextBuilder
from src.context.contracts import ContextBlock
from src.context.providers.vision import VisionContextProvider as ProviderImpl
from src.context.providers.vision import VisionSource as SourceImpl


class _FakeVision:
    def __init__(self, text: str = "\n[Vision] 現在の視界") -> None:
        self._text = text
        self.calls: int = 0

    def get_context_text(self) -> str:
        self.calls += 1
        return self._text


class _EmptyVision:
    def get_context_text(self) -> str:
        return ""


class _NonStrVision:
    def get_context_text(self) -> str:
        return None  # type: ignore[return-value]


class _BrokenVision:
    def get_context_text(self) -> str:
        raise RuntimeError("secret vision body")


class _FakeMonitor:
    def get_context_text(self) -> str:
        return "\n--- サブPCの現在の状態 ---\n- CPU: 10% (低負荷)"


class _FakeScreen:
    def get_context_text(self) -> str:
        return "\n[Screen] 画面の内容"


class _FakeWebSearch:
    def build_context_prompt(self, query: str) -> str:
        return "\n\n[Web検索結果]\n検索日時: 2026-01-01 00:00:00\n"


class VisionSourceProtocolTest(unittest.TestCase):
    def test_visions_conform_to_vision_source(self) -> None:
        self.assertIsInstance(_FakeVision(), VisionSource)
        self.assertIsInstance(_BrokenVision(), VisionSource)

    def test_unrelated_object_does_not_conform(self) -> None:
        self.assertNotIsInstance(object(), VisionSource)


class VisionContextProviderTest(unittest.TestCase):
    def test_returns_block_with_expected_metadata(self) -> None:
        vision = _FakeVision()
        result = VisionContextProvider.collect(vision)
        self.assertIsNotNone(result)
        self.assertEqual(result.source, "vision")
        self.assertEqual(result.sensitivity, "secret")
        self.assertIs(result.local_only, True)
        self.assertIsInstance(result.content, str)
        self.assertEqual(result.content, vision._text)
        self.assertEqual(vision.calls, 1)

    def test_empty_text_returns_none(self) -> None:
        self.assertIsNone(VisionContextProvider.collect(_EmptyVision()))

    def test_whitespace_text_returns_none(self) -> None:
        self.assertIsNone(VisionContextProvider.collect(_FakeVision("   \n  ")))

    def test_non_str_result_returns_none(self) -> None:
        self.assertIsNone(VisionContextProvider.collect(_NonStrVision()))

    def test_exception_returns_none(self) -> None:
        self.assertIsNone(VisionContextProvider.collect(_BrokenVision()))

    def test_exception_logs_type_only_not_body(self) -> None:
        with self.assertLogs(
            "src.context.providers.vision", level="WARNING"
        ) as captured:
            result = VisionContextProvider.collect(_BrokenVision())
        self.assertIsNone(result)
        self.assertEqual(len(captured.output), 1)
        self.assertIn("RuntimeError", captured.output[0])
        self.assertNotIn("secret vision body", captured.output[0])


class VisionCloudFilterTest(unittest.TestCase):
    def test_cloud_target_excludes_secret_local_only_vision(self) -> None:
        builder = ContextBuilder("base")
        vision_block = VisionContextProvider.collect(_FakeVision())
        public_block = ContextBlock(
            source="screen", content="公開", sensitivity="public"
        )
        result = builder.build_system_content(
            [vision_block, public_block], privacy="cloud_allowed", target_local=False
        )
        self.assertEqual(result, "base公開")


class ChatSessionVisionWiringTest(unittest.TestCase):
    def test_exact_payload_with_web_search_vision_monitor_screen_order(self) -> None:
        web = _FakeWebSearch()
        vision = _FakeVision()
        monitor = _FakeMonitor()
        session = ChatSession(
            system_prompt="sys",
            web_search=web,
            vision_context=vision,
            monitor_context=monitor,
            screen_context=_FakeScreen(),
        )
        session.add_user_message("検索して")
        session.add_assistant_message("a")
        self.assertEqual(
            session.build_messages(),
            [
                {
                    "role": "system",
                    "content": (
                        "sys"
                        "\n\n[Web検索結果]\n検索日時: 2026-01-01 00:00:00\n"
                        "\n[Vision] 現在の視界"
                        "\n--- サブPCの現在の状態 ---\n- CPU: 10% (低負荷)"
                        "\n[Screen] 画面の内容"
                    ),
                },
                {"role": "user", "content": "検索して"},
                {"role": "assistant", "content": "a"},
            ],
        )

    def test_exact_payload_vision_monitor_screen_order(self) -> None:
        vision = _FakeVision()
        monitor = _FakeMonitor()
        session = ChatSession(
            system_prompt="sys",
            vision_context=vision,
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

    def test_empty_vision_payload_matches_no_vision(self) -> None:
        session = ChatSession(
            system_prompt="sys",
            vision_context=_EmptyVision(),
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

    def test_broken_vision_does_not_stop_conversation(self) -> None:
        session = ChatSession(
            system_prompt="sys",
            vision_context=_BrokenVision(),
            monitor_context=_FakeMonitor(),
            screen_context=_FakeScreen(),
        )
        session.add_user_message("hi")
        session.add_assistant_message("a")
        with self.assertLogs("src.context.providers.vision", level="WARNING"):
            messages = session.build_messages()
        self.assertEqual(
            messages,
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

    def test_no_vision_unchanged(self) -> None:
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


class RootContextPublicAPITest(unittest.TestCase):
    def test_vision_provider_exported_from_root(self) -> None:
        import src.context

        self.assertIn("VisionContextProvider", src.context.__all__)
        self.assertIs(src.context.VisionContextProvider, ProviderImpl)

    def test_vision_source_exported_from_root(self) -> None:
        import src.context

        self.assertIn("VisionSource", src.context.__all__)
        self.assertIs(src.context.VisionSource, SourceImpl)
        self.assertIsInstance(_FakeVision(), src.context.VisionSource)

    def test_vision_exported_from_providers(self) -> None:
        import src.context.providers

        self.assertIn("VisionContextProvider", src.context.providers.__all__)
        self.assertIn("VisionSource", src.context.providers.__all__)


if __name__ == "__main__":
    unittest.main()
