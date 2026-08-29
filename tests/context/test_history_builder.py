"""History Context Provider / ContextBuilder / ChatSession互換wiringのテスト。"""
from __future__ import annotations

import unittest

from src.chat.session import ChatSession
from src.context.builder import ContextBuilder
from src.context.contracts import ContextBlock, ContextMessage
from src.context.policy import ContextPolicyError
from src.context.providers.history import HistoryContextProvider


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


class HistoryContextProviderTest(unittest.TestCase):
    def test_empty_history_returns_none(self) -> None:
        self.assertIsNone(HistoryContextProvider.collect([]))

    def test_returns_block_with_expected_metadata(self) -> None:
        result = HistoryContextProvider.collect(
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ]
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.source, "history")
        self.assertEqual(result.sensitivity, "personal")
        self.assertIs(result.local_only, True)
        self.assertEqual(
            result.content,
            (
                ContextMessage(role="user", content="hello"),
                ContextMessage(role="assistant", content="hi"),
            ),
        )

    def test_does_not_mutate_input(self) -> None:
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        before = [dict(m) for m in history]
        HistoryContextProvider.collect(history)
        self.assertEqual(history, before)

    def test_rejects_unknown_role(self) -> None:
        with self.assertRaises(ValueError):
            HistoryContextProvider.collect([{"role": "admin", "content": "x"}])


class ContextBuilderTest(unittest.TestCase):
    def test_str_content_concatenated_to_base_in_priority_order(self) -> None:
        builder = ContextBuilder("base")
        blocks = [block("a", "A", priority=2), block("b", "B", priority=1)]
        messages = builder.build_messages(blocks, privacy="local_only", target_local=True)
        self.assertEqual(messages, [{"role": "system", "content": "baseBA"}])

    def test_structured_content_rendered_after_system(self) -> None:
        builder = ContextBuilder("sys")
        history = (message("user", "hello"), message("assistant", "hi"))
        messages = builder.build_messages([block("history", history)])
        self.assertEqual(
            messages,
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ],
        )

    def test_str_and_structured_mix(self) -> None:
        builder = ContextBuilder("base")
        history = (message("user", "hi"),)
        blocks = [
            block("screen", "SCREEN", priority=1),
            block("history", history, sensitivity="personal", local_only=True, priority=0),
        ]
        messages = builder.build_messages(blocks)
        self.assertEqual(
            messages,
            [
                {"role": "system", "content": "baseSCREEN"},
                {"role": "user", "content": "hi"},
            ],
        )

    def test_heterogeneous_priority_never_places_structured_before_system(self) -> None:
        builder = ContextBuilder("base")
        history = (message("user", "hi"),)
        blocks = [
            block("history", history, sensitivity="personal", local_only=True, priority=-10),
            block("screen", "SCREEN", priority=100),
        ]
        messages = builder.build_messages(
            blocks, privacy="local_only", target_local=True
        )
        self.assertEqual(
            messages,
            [
                {"role": "system", "content": "baseSCREEN"},
                {"role": "user", "content": "hi"},
            ],
        )
        self.assertEqual(messages[0]["role"], "system")

    def test_heterogeneous_priority_does_not_interleave_message_positions(self) -> None:
        builder = ContextBuilder("sys")
        structured_hi = (message("user", "hi"),)
        structured_yo = (message("user", "yo"),)
        blocks = [
            block("a", structured_hi, priority=1),
            block("text", "T", priority=0),
            block("b", structured_yo, priority=-1),
        ]
        messages = builder.build_messages(blocks)
        self.assertEqual(
            messages,
            [
                {"role": "system", "content": "sysT"},
                {"role": "user", "content": "yo"},
                {"role": "user", "content": "hi"},
            ],
        )

    def test_non_local_filters_personal_local_only(self) -> None:
        builder = ContextBuilder("base")
        blocks = [
            block("screen", "公開", sensitivity="public"),
            block(
                "history",
                (message("user", "secret"),),
                sensitivity="personal",
                local_only=True,
            ),
        ]
        messages = builder.build_messages(
            blocks, privacy="cloud_allowed", target_local=False
        )
        self.assertEqual(messages, [{"role": "system", "content": "base公開"}])

    def test_non_local_without_cloud_allowed_raises(self) -> None:
        builder = ContextBuilder("base")
        with self.assertRaises(ContextPolicyError):
            builder.build_messages(
                [block("a", "x")], privacy="local_only", target_local=False
            )

    def test_empty_base_and_no_blocks_returns_empty(self) -> None:
        self.assertEqual(ContextBuilder("").build_messages([]), [])

    def test_empty_base_with_only_str_block(self) -> None:
        messages = ContextBuilder("").build_messages([block("a", "x")])
        self.assertEqual(messages, [{"role": "system", "content": "x"}])

    def test_does_not_mutate_input_blocks(self) -> None:
        blocks = [block("a", "x", priority=1), block("tasks", "y", priority=0)]
        before = tuple((b.source, b.content, b.priority) for b in blocks)
        ContextBuilder("base").build_messages(blocks)
        self.assertEqual(tuple((b.source, b.content, b.priority) for b in blocks), before)


class ChatSessionHistoryWiringTest(unittest.TestCase):
    def test_exact_payload_preserved(self) -> None:
        session = ChatSession(system_prompt="sys")
        session.add_user_message("hello")
        session.add_assistant_message("hi")
        self.assertEqual(
            session.build_messages(),
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ],
        )
        self.assertEqual(
            session.messages,
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ],
        )

    def test_empty_history_returns_only_system(self) -> None:
        self.assertEqual(
            ChatSession(system_prompt="sys").build_messages(),
            [{"role": "system", "content": "sys"}],
        )

    def test_existing_system_context_concatenation_before_history(self) -> None:
        class _FakeRAG:
            def build_context_prompt(self, query: str) -> str:
                return "\n[RAG] 記憶"

            def store_turn(self, user_message, assistant_message, session_id):
                return "mem-1"

        session = ChatSession(system_prompt="sys", rag=_FakeRAG())
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

    def test_history_after_emotion_instruction_and_before_nothing(self) -> None:
        session = ChatSession(system_prompt="sys", emotion_tags=True)
        session.add_user_message("q")
        session.add_assistant_message("a")
        messages = session.build_messages()
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("感情", messages[0]["content"])
        self.assertEqual(
            [m["role"] for m in messages],
            ["system", "user", "assistant"],
        )


if __name__ == "__main__":
    unittest.main()
