import unittest

from src.context.contracts import ContextBlock, ContextMessage
from src.llm.approval import (
    ApprovalDeniedError,
    ApprovalGate,
    ApprovalRequiredError,
    CloudPayloadBuilder,
    CloudPreview,
)


def _blocks():
    return [
        ContextBlock(
            source="calendar", content="calendar secret",
            sensitivity="personal", local_only=True,
        ),
        ContextBlock(
            source="screen", content="screen secret",
            sensitivity="secret", local_only=True,
        ),
        ContextBlock(
            source="tasks", content="task list secret",
            sensitivity="personal", local_only=True,
        ),
        ContextBlock(
            source="public_note", content="public info",
            sensitivity="public", local_only=False,
        ),
    ]


class ApprovalGateTest(unittest.TestCase):
    def test_preview_returns_exact_payload(self):
        gate = ApprovalGate()
        msgs = [{"role": "user", "content": "hi"}]
        preview = gate.preview("r1", msgs)
        self.assertIsInstance(preview, CloudPreview)
        self.assertEqual(preview.request_id, "r1")
        self.assertEqual(preview.messages, msgs)

    def test_require_raises_without_approval(self):
        gate = ApprovalGate()
        with self.assertRaises(ApprovalRequiredError):
            gate.require("r1")

    def test_approve_then_require_ok(self):
        gate = ApprovalGate()
        gate.approve("r1")
        self.assertTrue(gate.is_approved("r1"))
        gate.require("r1")  # no raise

    def test_deny_blocks(self):
        gate = ApprovalGate()
        gate.deny("r1")
        with self.assertRaises(ApprovalDeniedError):
            gate.require("r1")

    def test_revoke(self):
        gate = ApprovalGate()
        gate.approve("r1")
        gate.revoke("r1")
        self.assertFalse(gate.is_approved("r1"))

    def test_max_entries_evicts_oldest(self):
        gate = ApprovalGate(max_entries=3)
        gate.approve("r1")
        gate.approve("r2")
        gate.approve("r3")
        gate.approve("r4")  # evicts r1
        self.assertFalse(gate.is_approved("r1"))
        self.assertTrue(gate.is_approved("r2"))
        self.assertTrue(gate.is_approved("r3"))
        self.assertTrue(gate.is_approved("r4"))

    def test_max_entries_deny_evicts_oldest(self):
        gate = ApprovalGate(max_entries=2)
        gate.deny("d1")
        gate.deny("d2")
        gate.deny("d3")  # evicts d1
        self.assertFalse(gate.is_denied("d1"))
        self.assertTrue(gate.is_denied("d2"))
        self.assertTrue(gate.is_denied("d3"))

    def test_default_max_entries_no_crash(self):
        gate = ApprovalGate()
        for i in range(2048):
            gate.approve(f"r{i}")
        self.assertTrue(gate.is_approved("r2047"))


class CloudPayloadBuilderTest(unittest.TestCase):
    def test_anonymizes_personal_secret(self):
        builder = CloudPayloadBuilder()
        msgs = builder.build(_blocks())
        joined = str(msgs)
        self.assertNotIn("calendar", joined)
        self.assertNotIn("screen", joined)
        self.assertNotIn("task list", joined)
        self.assertIn("public info", joined)

    def test_only_public_blocks_present(self):
        builder = CloudPayloadBuilder()
        msgs = builder.build(_blocks())
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[0]["content"], "public info")

    def test_history_filtered_in_cloud(self):
        """History passed as a ContextBlock with personal+local_only should NOT appear in cloud payload."""
        builder = CloudPayloadBuilder()
        history_block = ContextBlock(
            source="history",
            content=tuple([ContextMessage(role="user", content="secret chat")]),
            sensitivity="personal",
            local_only=True,
        )
        blocks = _blocks() + [history_block]
        msgs = builder.build(blocks)
        joined = str(msgs)
        self.assertNotIn("secret chat", joined)
        self.assertIn("public info", joined)

    def test_history_not_bypassing_policy(self):
        """History in blocks is filtered by ContextPolicy.select, not appended raw."""
        builder = CloudPayloadBuilder()
        history_block = ContextBlock(
            source="history",
            content=tuple([
                ContextMessage(role="user", content="hello"),
                ContextMessage(role="assistant", content="hi there"),
            ]),
            sensitivity="personal",
            local_only=True,
        )
        blocks = _blocks() + [history_block]
        msgs = builder.build(blocks)
        # personal+local_only history should be excluded for cloud target
        for msg in msgs:
            self.assertNotIn("hello", msg.get("content", ""))
            self.assertNotIn("hi there", msg.get("content", ""))


if __name__ == "__main__":
    unittest.main()
