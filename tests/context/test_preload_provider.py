"""Preload Context Provider / build_system_content / ChatSession互換wiringのテスト。"""
from __future__ import annotations

import unittest

from src.chat.session import ChatSession
from src.context import (
    PreloadContextProvider,
    StructuredBlockNotAllowedError,
)
from src.context.builder import ContextBuilder
from src.context.contracts import ContextBlock, ContextMessage
from src.context.providers.preload import PreloadSource


def message(role: str, content: str) -> ContextMessage:
    return ContextMessage(role=role, content=content)


def block(
    source: str,
    content: str | tuple[ContextMessage, ...],
    sensitivity: str = "public",
    local_only: bool = False,
    priority: int = 0,
) -> ContextBlock:
    return ContextBlock(
        source=source,
        content=content,
        sensitivity=sensitivity,
        local_only=local_only,
        priority=priority,
    )


class _FakePreloader:
    def __init__(self, text: str = "\n--- 現在の状況 ---\n- 日時: テスト") -> None:
        self._text = text

    def build_preload_context(self) -> str:
        return self._text


class _BrokenPreloader:
    def build_preload_context(self) -> str:
        raise RuntimeError("secret context body")


class _FakeRAG:
    def build_context_prompt(self, query: str) -> str:
        return "\n[RAG] 記憶"

    def store_turn(self, user_message, assistant_message, session_id):
        return "mem-1"


class PreloadSourceProtocolTest(unittest.TestCase):
    def test_preloaders_conform_to_preload_source(self) -> None:
        self.assertIsInstance(_FakePreloader(), PreloadSource)
        self.assertIsInstance(_BrokenPreloader(), PreloadSource)

    def test_unrelated_object_does_not_conform(self) -> None:
        self.assertNotIsInstance(object(), PreloadSource)


class PreloadContextProviderTest(unittest.TestCase):
    def test_returns_block_with_expected_metadata(self) -> None:
        result = PreloadContextProvider.collect(_FakePreloader())
        self.assertIsNotNone(result)
        self.assertEqual(result.source, "preload")
        self.assertEqual(result.sensitivity, "personal")
        self.assertIs(result.local_only, True)
        self.assertIsInstance(result.content, str)
        self.assertEqual(result.content, "\n--- 現在の状況 ---\n- 日時: テスト")

    def test_empty_text_returns_none(self) -> None:
        self.assertIsNone(PreloadContextProvider.collect(_FakePreloader("")))

    def test_whitespace_text_returns_none(self) -> None:
        self.assertIsNone(PreloadContextProvider.collect(_FakePreloader("   \n  ")))

    def test_exception_returns_none(self) -> None:
        self.assertIsNone(PreloadContextProvider.collect(_BrokenPreloader()))

    def test_exception_logs_type_only_not_body(self) -> None:
        with self.assertLogs("src.context.providers.preload", level="WARNING") as captured:
            result = PreloadContextProvider.collect(_BrokenPreloader())
        self.assertIsNone(result)
        self.assertEqual(len(captured.output), 1)
        self.assertIn("RuntimeError", captured.output[0])
        self.assertNotIn("secret context body", captured.output[0])


class ContextBuilderSystemContentTest(unittest.TestCase):
    def test_str_blocks_concatenated_to_base_in_priority_order(self) -> None:
        builder = ContextBuilder("base")
        blocks = [block("a", "A", priority=2), block("b", "B", priority=1)]
        self.assertEqual(builder.build_system_content(blocks), "baseBA")

    def test_empty_base_and_no_blocks_returns_empty(self) -> None:
        self.assertEqual(ContextBuilder("").build_system_content([]), "")

    def test_base_preserved_when_no_blocks(self) -> None:
        self.assertEqual(ContextBuilder("sys").build_system_content([]), "sys")

    def test_structured_block_raises_explicit_error(self) -> None:
        builder = ContextBuilder("sys")
        structured = block("history", (message("user", "hi"),))
        with self.assertRaises(StructuredBlockNotAllowedError):
            builder.build_system_content([structured])

    def test_cloud_filter_excludes_personal_local_only(self) -> None:
        builder = ContextBuilder("base")
        blocks = [
            block("preload", "個人文脈", sensitivity="personal", local_only=True),
            block("screen", "公開", sensitivity="public"),
        ]
        result = builder.build_system_content(
            blocks, privacy="cloud_allowed", target_local=False
        )
        self.assertEqual(result, "base公開")


class ChatSessionPreloadWiringTest(unittest.TestCase):
    def test_exact_payload_with_preload_then_rag(self) -> None:
        session = ChatSession(
            system_prompt="sys",
            preloader=_FakePreloader(),
            rag=_FakeRAG(),
        )
        session.add_user_message("q")
        session.add_assistant_message("a")
        self.assertEqual(
            session.build_messages(),
            [
                {
                    "role": "system",
                    "content": "sys\n--- 現在の状況 ---\n- 日時: テスト\n[RAG] 記憶",
                },
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
            ],
        )

    def test_empty_preload_payload_matches_no_preloader(self) -> None:
        session = ChatSession(
            system_prompt="sys",
            preloader=_FakePreloader(""),
            rag=_FakeRAG(),
        )
        session.add_user_message("q")
        session.add_assistant_message("a")
        self.assertEqual(
            session.build_messages(),
            [
                {"role": "system", "content": "sys\n[RAG] 記憶"},
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
            ],
        )

    def test_broken_preloader_does_not_stop_conversation(self) -> None:
        session = ChatSession(
            system_prompt="sys",
            preloader=_BrokenPreloader(),
            rag=_FakeRAG(),
        )
        session.add_user_message("q")
        session.add_assistant_message("a")
        messages = session.build_messages()
        self.assertEqual(
            messages,
            [
                {"role": "system", "content": "sys\n[RAG] 記憶"},
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
            ],
        )

    def test_no_preloader_unchanged(self) -> None:
        session = ChatSession(system_prompt="sys", rag=_FakeRAG())
        session.add_user_message("q")
        session.add_assistant_message("a")
        self.assertEqual(
            session.build_messages(),
            [
                {"role": "system", "content": "sys\n[RAG] 記憶"},
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
            ],
        )


class RootContextPublicAPITest(unittest.TestCase):
    def test_preload_provider_exported_from_root(self) -> None:
        import src.context
        from src.context.providers.preload import PreloadContextProvider as Impl

        self.assertIn("PreloadContextProvider", src.context.__all__)
        self.assertIs(src.context.PreloadContextProvider, Impl)

    def test_structured_block_error_exported_from_root(self) -> None:
        import src.context
        from src.context.builder import StructuredBlockNotAllowedError as Impl

        self.assertIn("StructuredBlockNotAllowedError", src.context.__all__)
        self.assertIs(src.context.StructuredBlockNotAllowedError, Impl)


if __name__ == "__main__":
    unittest.main()
