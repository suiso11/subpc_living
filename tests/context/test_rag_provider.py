"""RAG Context Provider / ChatSession互換wiringのテスト。"""
from __future__ import annotations

import unittest

from src.chat.session import ChatSession
from src.context import RAGContextProvider, RAGSource
from src.context.builder import ContextBuilder
from src.context.contracts import ContextBlock
from src.context.providers.rag import RAGContextProvider as ProviderImpl
from src.context.providers.rag import RAGSource as SourceImpl


class _FakeRAG:
    def __init__(self, text: str = "\n\n--- 関連する過去の記憶 ---\nfoo\n--- 記憶ここまで ---\n") -> None:
        self._text = text

    def build_context_prompt(self, query: str) -> str:
        return self._text

    def store_turn(self, user_message, assistant_message, session_id):
        return "mem-1"


class _EmptyRAG:
    def build_context_prompt(self, query: str) -> str:
        return ""

    def store_turn(self, user_message, assistant_message, session_id):
        return None


class _NonStrRAG:
    def build_context_prompt(self, query: str) -> str:
        return None  # type: ignore[return-value]


class _BrokenRAG:
    def build_context_prompt(self, query: str) -> str:
        raise RuntimeError("secret context body")

    def store_turn(self, user_message, assistant_message, session_id):
        return None


class RAGSourceProtocolTest(unittest.TestCase):
    def test_retrievers_conform_to_rag_source(self) -> None:
        self.assertIsInstance(_FakeRAG(), RAGSource)
        self.assertIsInstance(_BrokenRAG(), RAGSource)

    def test_unrelated_object_does_not_conform(self) -> None:
        self.assertNotIsInstance(object(), RAGSource)


class RAGContextProviderTest(unittest.TestCase):
    def test_returns_block_with_expected_metadata(self) -> None:
        rag = _FakeRAG()
        result = RAGContextProvider.collect(rag, "query")
        self.assertIsNotNone(result)
        self.assertEqual(result.source, "rag")
        self.assertEqual(result.sensitivity, "personal")
        self.assertIs(result.local_only, True)
        self.assertIsInstance(result.content, str)
        self.assertEqual(result.content, rag._text)

    def test_empty_text_returns_none(self) -> None:
        self.assertIsNone(RAGContextProvider.collect(_EmptyRAG(), "q"))

    def test_whitespace_text_returns_none(self) -> None:
        self.assertIsNone(RAGContextProvider.collect(_FakeRAG("   \n  "), "q"))

    def test_non_str_result_returns_none(self) -> None:
        self.assertIsNone(RAGContextProvider.collect(_NonStrRAG(), "q"))

    def test_exception_returns_none(self) -> None:
        self.assertIsNone(RAGContextProvider.collect(_BrokenRAG(), "q"))

    def test_exception_logs_type_only_not_body_or_query(self) -> None:
        with self.assertLogs("src.context.providers.rag", level="WARNING") as captured:
            result = RAGContextProvider.collect(_BrokenRAG(), "secret query")
        self.assertIsNone(result)
        self.assertEqual(len(captured.output), 1)
        self.assertIn("RuntimeError", captured.output[0])
        self.assertNotIn("secret query", captured.output[0])
        self.assertNotIn("secret context body", captured.output[0])


class RAGCloudFilterTest(unittest.TestCase):
    def test_cloud_target_excludes_personal_local_only_rag(self) -> None:
        builder = ContextBuilder("base")
        rag_block = RAGContextProvider.collect(_FakeRAG(), "q")
        public_block = ContextBlock(
            source="screen", content="公開", sensitivity="public"
        )
        result = builder.build_system_content(
            [rag_block, public_block], privacy="cloud_allowed", target_local=False
        )
        self.assertEqual(result, "base公開")


class ChatSessionRAGWiringTest(unittest.TestCase):
    def test_exact_payload_with_rag(self) -> None:
        rag = _FakeRAG()
        session = ChatSession(system_prompt="sys", rag=rag)
        session.add_user_message("q")
        session.add_assistant_message("a")
        self.assertEqual(
            session.build_messages(),
            [
                {"role": "system", "content": "sys" + rag._text},
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
            ],
        )

    def test_exact_payload_preload_then_rag(self) -> None:
        class _FakePreloader:
            def build_preload_context(self) -> str:
                return "\n--- 現在の状況 ---\n- 日時: テスト"

        session = ChatSession(
            system_prompt="sys",
            preloader=_FakePreloader(),
            rag=_FakeRAG("\n[RAG] 記憶"),
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

    def test_empty_rag_payload_matches_no_rag(self) -> None:
        session = ChatSession(system_prompt="sys", rag=_EmptyRAG())
        session.add_user_message("q")
        session.add_assistant_message("a")
        self.assertEqual(
            session.build_messages(),
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
            ],
        )

    def test_broken_rag_does_not_stop_conversation(self) -> None:
        session = ChatSession(system_prompt="sys", rag=_BrokenRAG())
        session.add_user_message("q")
        session.add_assistant_message("a")
        with self.assertLogs("src.context.providers.rag", level="WARNING"):
            messages = session.build_messages()
        self.assertEqual(
            messages,
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
            ],
        )

    def test_no_rag_unchanged(self) -> None:
        session = ChatSession(system_prompt="sys")
        session.add_user_message("q")
        session.add_assistant_message("a")
        self.assertEqual(
            session.build_messages(),
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
            ],
        )

    def test_store_turn_still_recorded(self) -> None:
        calls: list[tuple[str, str, str]] = []

        class _RecordingRAG(_FakeRAG):
            def store_turn(self, user_message, assistant_message, session_id):
                calls.append((user_message, assistant_message, session_id))
                return "mem-1"

        session = ChatSession(system_prompt="sys", rag=_RecordingRAG())
        session.add_user_message("hi")
        session.add_assistant_message("hello")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "hi")
        self.assertEqual(calls[0][1], "hello")


class RootContextPublicAPITest(unittest.TestCase):
    def test_rag_provider_exported_from_root(self) -> None:
        import src.context

        self.assertIn("RAGContextProvider", src.context.__all__)
        self.assertIs(src.context.RAGContextProvider, ProviderImpl)

    def test_rag_source_exported_from_root(self) -> None:
        import src.context

        self.assertIn("RAGSource", src.context.__all__)
        self.assertIs(src.context.RAGSource, SourceImpl)
        self.assertIsInstance(_FakeRAG(), src.context.RAGSource)

    def test_rag_exported_from_providers(self) -> None:
        import src.context.providers

        self.assertIn("RAGContextProvider", src.context.providers.__all__)
        self.assertIn("RAGSource", src.context.providers.__all__)


if __name__ == "__main__":
    unittest.main()
