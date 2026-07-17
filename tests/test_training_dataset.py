from __future__ import annotations
import json, tempfile, unittest
from pathlib import Path
from training.dataset import detect_duplicates, validate_schema
from training.validate_dataset import run_preflight, safe_rows, main

GOOD = {"messages":[{"role":"user","content":"課題が多い"},{"role":"assistant","content":"一つずつ片づけなさい。"}]}

class DatasetValidationTest(unittest.TestCase):
    def test_schema_accepts_optional_system_and_rejects_empty(self):
        row={"messages":[{"role":"system","content":"短い契約"},*GOOD["messages"]]}
        self.assertTrue(validate_schema([row],"sft").ok)
        self.assertFalse(validate_schema([],"sft").ok)
        self.assertFalse(validate_schema([{"messages":[{"role":"user","content":""}]}],"sft").ok)

    def test_duplicate_and_secret_are_reported_without_value(self):
        secret="api_key='SUPER_SECRET_VALUE_1234567890'"
        rows=[GOOD,GOOD,{"messages":[{"role":"user","content":secret},{"role":"assistant","content":"削除"}]}]
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"in.jsonl"
            p.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in rows),encoding="utf-8")
            report=run_preflight(p,"sft")
            rendered=json.dumps(report.as_dict(),ensure_ascii=False)
        self.assertFalse(report.ok)
        self.assertEqual(len(report.duplicates.duplicates),1)
        self.assertTrue(report.pii.issues)
        self.assertNotIn("SUPER_SECRET_VALUE",rendered)

    def test_clean_output_drops_unsafe_duplicates_and_metadata(self):
        unsafe={"messages":[{"role":"user","content":"mail me x@example.com"},{"role":"assistant","content":"no"}]}
        rows=[{**GOOD,"metadata":{"channel_id":123}},GOOD,unsafe]
        with tempfile.TemporaryDirectory() as td:
            src=Path(td)/"in.jsonl"; dst=Path(td)/"clean.jsonl"
            src.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in rows),encoding="utf-8")
            rc=main(["--input",str(src),"--format","sft","--clean-output",str(dst),"--json"])
            cleaned=[json.loads(x) for x in dst.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rc,1)
            self.assertEqual(cleaned,[GOOD])
            self.assertTrue(run_preflight(dst,"sft").ok)

    def test_malformed_json_is_not_silently_skipped(self):
        with tempfile.TemporaryDirectory() as td:
            src=Path(td)/"bad.jsonl"
            src.write_text(json.dumps(GOOD,ensure_ascii=False)+"\n{broken\n",encoding="utf-8")
            report=run_preflight(src,"sft")
        self.assertFalse(report.schema.ok)
        self.assertEqual(report.total,2)

    def test_dpo_chosen_and_rejected_must_differ(self):
        report=validate_schema([{"prompt":"x","chosen":"same","rejected":"same"}],"dpo")
        self.assertFalse(report.ok)

if __name__ == "__main__": unittest.main()
