from __future__ import annotations
import unittest
from training.tokenize_check import scan_tokens

class FakeTokenizer:
    def apply_chat_template(self,messages,**kwargs):
        # Include role/template overhead so the test verifies the whole conversation is measured.
        return list(range(2 + sum(len(m["content"]) for m in messages)))

class TokenizeCheckTest(unittest.TestCase):
    def test_sft_measures_whole_chat_and_reports_only_index(self):
        rows=[{"messages":[{"role":"user","content":"1234"},{"role":"assistant","content":"5678"}]}]
        report=scan_tokens(rows,"sft",FakeTokenizer(),max_tokens=9)
        self.assertFalse(report.ok)
        self.assertEqual(report.issues[0].field,"messages")
        self.assertEqual(report.issues[0].token_len,10)
        self.assertNotIn("1234",str(report.as_dict()))

    def test_dpo_measures_chosen_and_rejected_sequences(self):
        rows=[{"prompt":"12","chosen":"345","rejected":"6"}]
        report=scan_tokens(rows,"dpo",FakeTokenizer(),max_tokens=6)
        self.assertEqual(report.stats.total,2)
        self.assertEqual(report.stats.max_tokens,7)
        self.assertEqual(report.stats.p99_tokens,7)
        self.assertEqual([x.field for x in report.issues],["chosen"])

if __name__ == "__main__": unittest.main()
