from __future__ import annotations

import unittest

from src.context.contracts import TASKS_SOURCE, ContextBlock
from src.context.policy import ContextPolicy, ContextPolicyError


def block(
    source: str,
    content: str,
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


class ContextBlockContractTest(unittest.TestCase):
    def test_rejects_empty_source(self) -> None:
        with self.assertRaises(ValueError):
            ContextBlock(source="", content="hello")

    def test_rejects_empty_content(self) -> None:
        with self.assertRaises(ValueError):
            ContextBlock(source="tasks", content="   ")

    def test_rejects_unknown_sensitivity(self) -> None:
        with self.assertRaises(ValueError):
            ContextBlock(source="tasks", content="x", sensitivity="topsecret")

    def test_rejects_non_str_source(self) -> None:
        with self.assertRaises(TypeError):
            ContextBlock(source=123, content="x")

    def test_rejects_non_str_content(self) -> None:
        with self.assertRaises(TypeError):
            ContextBlock(source="tasks", content=123)

    def test_rejects_non_bool_local_only(self) -> None:
        with self.assertRaises(TypeError):
            ContextBlock(source="tasks", content="x", local_only=1)

    def test_rejects_non_int_priority(self) -> None:
        with self.assertRaises(TypeError):
            ContextBlock(source="tasks", content="x", priority="1")

    def test_rejects_bool_priority(self) -> None:
        with self.assertRaises(TypeError):
            ContextBlock(source="tasks", content="x", priority=True)

    def test_is_frozen(self) -> None:
        b = block("tasks", "x")
        with self.assertRaises(AttributeError):
            b.content = "y"

    def test_history_is_out_of_scope(self) -> None:
        import src.context.contracts as contracts
        import src.context.policy as policy

        self.assertIn("history", contracts.__doc__.lower())
        self.assertIn("history", policy.__doc__.lower())


class ContextPolicySelectTest(unittest.TestCase):
    def test_returns_tuple_and_keeps_input(self) -> None:
        blocks = [block("a", "x", priority=5), block("b", "y", priority=1)]
        result = ContextPolicy.select(blocks, privacy="local_only", target_local=True)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(blocks), 2)
        self.assertEqual([b.source for b in result], ["b", "a"])

    def test_local_target_allows_all_sensitivities(self) -> None:
        blocks = [
            block("screen", "公開", sensitivity="public"),
            block("calendar", "予定", sensitivity="personal"),
            block("secret_src", "秘密", sensitivity="secret"),
            block("local", "ローカル", local_only=True),
        ]
        result = ContextPolicy.select(blocks, privacy="local_only", target_local=True)
        self.assertEqual(len(result), 4)

    def test_non_local_requires_cloud_allowed(self) -> None:
        blocks = [block("a", "x")]
        for privacy in ("local_only", "local_preferred"):
            with self.assertRaises(ContextPolicyError):
                ContextPolicy.select(blocks, privacy=privacy, target_local=False)

    def test_non_local_allows_only_public_and_non_local_only(self) -> None:
        blocks = [
            block("screen", "公開画面", sensitivity="public"),
            block("calendar", "予定", sensitivity="personal"),
            block("secret_src", "秘密", sensitivity="secret"),
            block("local_public", "ローカル限定公開", sensitivity="public", local_only=True),
            block("tasks", "タスク", sensitivity="personal", local_only=True),
        ]
        result = ContextPolicy.select(blocks, privacy="cloud_allowed", target_local=False)
        self.assertEqual([b.source for b in result], ["screen"])

    def test_order_priority_ascending_and_stable(self) -> None:
        blocks = [
            block("calendar", "p3", priority=3),
            block("screen", "p1a", priority=1),
            block("tasks", "p2", priority=2),
            block("profile", "p1b", priority=1),
        ]
        result = ContextPolicy.select(blocks, privacy="local_only", target_local=True)
        self.assertEqual(
            [b.source for b in result],
            ["screen", "profile", "calendar", "tasks"],
        )

    def test_tasks_always_last_even_with_lowest_priority(self) -> None:
        blocks = [
            block("tasks", "low", priority=0),
            block("profile", "high", priority=100),
            block("calendar", "mid", priority=50),
        ]
        result = ContextPolicy.select(blocks, privacy="local_only", target_local=True)
        self.assertEqual([b.source for b in result], ["calendar", "profile", "tasks"])

    def test_policy_ignores_content_text(self) -> None:
        blocks = [
            block("screen", "password=hunter2", sensitivity="public"),
            block("calendar", "何の変哲もない予定", sensitivity="personal"),
        ]
        result = ContextPolicy.select(blocks, privacy="cloud_allowed", target_local=False)
        self.assertEqual([b.source for b in result], ["screen"])

    def test_does_not_mutate_input_blocks(self) -> None:
        blocks = [block("a", "x", priority=1), block("tasks", "y", priority=0)]
        before = tuple((b.source, b.priority) for b in blocks)
        ContextPolicy.select(blocks, privacy="cloud_allowed", target_local=False)
        self.assertEqual(tuple((b.source, b.priority) for b in blocks), before)


class ContextPolicyPrivacyContractTest(unittest.TestCase):
    def test_unknown_privacy_rejected_even_for_local_target(self) -> None:
        blocks = [block("a", "x")]
        for target_local in (True, False):
            with self.assertRaises(ContextPolicyError):
                ContextPolicy.select(blocks, privacy="vague", target_local=target_local)

    def test_local_preferred_rejected_for_non_local(self) -> None:
        blocks = [block("a", "x")]
        with self.assertRaises(ContextPolicyError):
            ContextPolicy.select(blocks, privacy="local_preferred", target_local=False)

    def test_tasks_source_shared_with_contracts(self) -> None:
        import src.context.contracts as contracts
        import src.context.policy as policy

        self.assertIs(policy.TASKS_SOURCE, contracts.TASKS_SOURCE)
        self.assertEqual(TASKS_SOURCE, "tasks")

    def test_tasks_source_exported_from_context(self) -> None:
        import src.context as context

        self.assertIs(context.TASKS_SOURCE, TASKS_SOURCE)

    def test_privacy_typed_against_routing_privacy_mode(self) -> None:
        from src.context.contracts import VALID_PRIVACY_MODES
        from src.llm.routing.contracts import PrivacyMode

        self.assertEqual(frozenset(PrivacyMode.__args__), VALID_PRIVACY_MODES)

    def test_docstring_clarifies_cloud_approval_owner(self) -> None:
        import src.context.policy as policy

        docs = (policy.__doc__ or "") + (ContextPolicy.select.__doc__ or "")
        self.assertIn("allow_cloud", docs)
        self.assertIn("Router", docs)
        self.assertIn("AssistantService", docs)
        self.assertIn("最終承認", docs)

    def test_docstring_not_final_gate_for_cloud(self) -> None:
        import src.context.policy as policy

        docs = (policy.__doc__ or "") + (ContextPolicy.select.__doc__ or "")
        self.assertIn("最終ゲート", docs)

    def test_docstring_no_static_router_same_condition_claim(self) -> None:
        import src.context.policy as policy

        docs = ((policy.__doc__ or "") + (ContextPolicy.select.__doc__ or "")).lower()
        self.assertNotIn("staticrouter", docs)


if __name__ == "__main__":
    unittest.main()