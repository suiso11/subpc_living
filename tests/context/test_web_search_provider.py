"""WebSearch Context Provider / ChatSession互換wiringのテスト。"""
from __future__ import annotations

import unittest

from src.chat.session import ChatSession
from src.context import WebSearchContextProvider, WebSearchSource
from src.context.builder import ContextBuilder
from src.context.contracts import ContextBlock
from src.context.providers.web_search import WebSearchContextProvider as ProviderImpl
from src.context.providers.web_search import WebSearchSource as SourceImpl


class _FakeWebSearch:
    def __init__(self, text: str = "\n\n[Web検索結果]\n検索日時: 2026-01-01 00:00:00\n") -> None:
        self._text = text
        self.calls: list[str] = []

    def build_context_prompt(self, query: str) -> str:
        self.calls.append(query)
        return self._text


class _EmptyWebSearch:
    def build_context_prompt(self, query: str) -> str:
        return ""


class _NonStrWebSearch:
    def build_context_prompt(self, query: str) -> str:
        return None  # type: ignore[return-value]


class _BrokenWebSearch:
    def build_context_prompt(self, query: str) -> str:
        raise RuntimeError("secret search body")


class WebSearchSourceProtocolTest(unittest.TestCase):
    def test_web_searches_conform_to_web_search_source(self) -> None:
        self.assertIsInstance(_FakeWebSearch(), WebSearchSource)
        self.assertIsInstance(_BrokenWebSearch(), WebSearchSource)

    def test_unrelated_object_does_not_conform(self) -> None:
        self.assertNotIsInstance(object(), WebSearchSource)


class WebSearchContextProviderTest(unittest.TestCase):
    def test_returns_block_with_expected_metadata(self) -> None:
        web = _FakeWebSearch()
        result = WebSearchContextProvider.collect(web, "query")
        self.assertIsNotNone(result)
        self.assertEqual(result.source, "web_search")
        self.assertEqual(result.sensitivity, "personal")
        self.assertIs(result.local_only, True)
        self.assertIsInstance(result.content, str)
        self.assertEqual(result.content, web._text)

    def test_passes_last_query_through(self) -> None:
        web = _FakeWebSearch()
        WebSearchContextProvider.collect(web, "検索して")
        self.assertEqual(web.calls, ["検索して"])

    def test_empty_text_returns_none(self) -> None:
        self.assertIsNone(WebSearchContextProvider.collect(_EmptyWebSearch(), "q"))

    def test_whitespace_text_returns_none(self) -> None:
        self.assertIsNone(WebSearchContextProvider.collect(_FakeWebSearch("   \n  "), "q"))

    def test_non_str_result_returns_none(self) -> None:
        self.assertIsNone(WebSearchContextProvider.collect(_NonStrWebSearch(), "q"))

    def test_exception_returns_none(self) -> None:
        self.assertIsNone(WebSearchContextProvider.collect(_BrokenWebSearch(), "q"))

    def test_exception_logs_type_only_not_body_or_query(self) -> None:
        with self.assertLogs(
            "src.context.providers.web_search", level="WARNING"
        ) as captured:
            result = WebSearchContextProvider.collect(_BrokenWebSearch(), "secret query")
        self.assertIsNone(result)
        self.assertEqual(len(captured.output), 1)
        self.assertIn("RuntimeError", captured.output[0])
        self.assertNotIn("secret query", captured.output[0])
        self.assertNotIn("secret search body", captured.output[0])


class WebSearchCloudFilterTest(unittest.TestCase):
    def test_cloud_target_excludes_personal_local_only_web_search(self) -> None:
        builder = ContextBuilder("base")
        web_block = WebSearchContextProvider.collect(_FakeWebSearch(), "q")
        public_block = ContextBlock(
            source="screen", content="公開", sensitivity="public"
        )
        result = builder.build_system_content(
            [web_block, public_block], privacy="cloud_allowed", target_local=False
        )
        self.assertEqual(result, "base公開")


class ChatSessionWebSearchWiringTest(unittest.TestCase):
    def test_exact_payload_with_web_search(self) -> None:
        web = _FakeWebSearch()
        session = ChatSession(system_prompt="sys", web_search=web)
        session.add_user_message("検索して")
        session.add_assistant_message("a")
        self.assertEqual(
            session.build_messages(),
            [
                {"role": "system", "content": "sys" + web._text},
                {"role": "user", "content": "検索して"},
                {"role": "assistant", "content": "a"},
            ],
        )

    def test_uses_last_user_message_as_query(self) -> None:
        web = _FakeWebSearch()
        session = ChatSession(system_prompt="sys", web_search=web)
        session.add_user_message("古い質問")
        session.add_user_message("最新の質問")
        session.build_messages()
        self.assertEqual(web.calls, ["最新の質問"])

    def test_empty_web_search_payload_matches_no_web_search(self) -> None:
        session = ChatSession(system_prompt="sys", web_search=_EmptyWebSearch())
        session.add_user_message("検索して")
        session.add_assistant_message("a")
        self.assertEqual(
            session.build_messages(),
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "検索して"},
                {"role": "assistant", "content": "a"},
            ],
        )

    def test_broken_web_search_does_not_stop_conversation(self) -> None:
        session = ChatSession(system_prompt="sys", web_search=_BrokenWebSearch())
        session.add_user_message("検索して")
        session.add_assistant_message("a")
        with self.assertLogs("src.context.providers.web_search", level="WARNING"):
            messages = session.build_messages()
        self.assertEqual(
            messages,
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "検索して"},
                {"role": "assistant", "content": "a"},
            ],
        )

    def test_no_web_search_unchanged(self) -> None:
        session = ChatSession(system_prompt="sys")
        session.add_user_message("検索して")
        session.add_assistant_message("a")
        self.assertEqual(
            session.build_messages(),
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "検索して"},
                {"role": "assistant", "content": "a"},
            ],
        )


class RootContextPublicAPITest(unittest.TestCase):
    def test_web_search_provider_exported_from_root(self) -> None:
        import src.context

        self.assertIn("WebSearchContextProvider", src.context.__all__)
        self.assertIs(src.context.WebSearchContextProvider, ProviderImpl)

    def test_web_search_source_exported_from_root(self) -> None:
        import src.context

        self.assertIn("WebSearchSource", src.context.__all__)
        self.assertIs(src.context.WebSearchSource, SourceImpl)
        self.assertIsInstance(_FakeWebSearch(), src.context.WebSearchSource)

    def test_web_search_exported_from_providers(self) -> None:
        import src.context.providers

        self.assertIn("WebSearchContextProvider", src.context.providers.__all__)
        self.assertIn("WebSearchSource", src.context.providers.__all__)


if __name__ == "__main__":
    unittest.main()
