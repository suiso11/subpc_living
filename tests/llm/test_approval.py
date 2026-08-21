import unittest

from src.context.contracts import ContextBlock
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

    def test_history_appended(self):
        builder = CloudPayloadBuilder()
        msgs = builder.build(_blocks(), history=[{"role": "user", "content": "hi"}])
        self.assertEqual(msgs[-1], {"role": "user", "content": "hi"})


if __name__ == "__main__":
    unittest.main()
