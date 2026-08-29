"""Web instrument backend label derives from /api/status provider_kind.

The frontend instrument label must reflect the selected local backend instead of
always rendering "OLLAMA", but only through an allowlisted display mapping so
arbitrary provider input is never rendered. Reachability continues to come from
the selected-provider ``status.ollama`` alias.
"""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APPJS = ROOT / "src/web/static/app.js"


def _extract_js_block(source: str, start_marker: str, func_name: str) -> str:
    """Return source from start_marker through the closing brace of func_name."""
    start = source.index(start_marker)
    func_idx = source.index(func_name)
    open_idx = source.index("{", func_idx)
    depth = 0
    for i in range(open_idx, len(source)):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                return source[start : i + 1]
    raise AssertionError(f"unbalanced braces in {func_name}")


class WebBackendLabelStaticTest(unittest.TestCase):
    def setUp(self) -> None:
        self.js = APPJS.read_text(encoding="utf-8")

    def test_label_derives_from_provider_kind_not_hardcoded_ollama(self) -> None:
        self.assertIn("provider_kind", self.js)
        self.assertIn("providerKind: status.provider_kind", self.js)
        self.assertIn("backendDisplayLabel(runtimeStatus.providerKind)", self.js)
        self.assertIn("'instrument-ollama'", self.js)
        self.assertNotIn("LOCAL · OLLAMA · ${statusWord", self.js)

    def test_allowlisted_display_mapping_is_present(self) -> None:
        self.assertIn("const BACKEND_DISPLAY_LABELS", self.js)
        self.assertIn("ollama: 'OLLAMA'", self.js)
        self.assertIn("openai_compatible: 'OPENAI COMPATIBLE'", self.js)
        self.assertIn("'LOCAL BACKEND'", self.js)

    def test_reachability_still_uses_selected_provider_status(self) -> None:
        self.assertIn("statusWord(runtimeStatus.ollama)", self.js)
        self.assertIn("ollama: status.ollama", self.js)

    def test_never_interpolates_raw_provider_kind(self) -> None:
        self.assertNotIn("${providerKind}", self.js)
        self.assertNotIn("${status.provider_kind}", self.js)
        self.assertNotIn("${provider_kind}", self.js)


class WebBackendLabelNodeTest(unittest.TestCase):
    def test_node_syntax_check(self) -> None:
        result = subprocess.run(
            ["node", "--check", str(APPJS)],
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            result.returncode, 0, msg=result.stderr or result.stdout
        )

    def test_allowlist_mapping_evaluates_in_node(self) -> None:
        source = APPJS.read_text(encoding="utf-8")
        block = _extract_js_block(
            source, "const BACKEND_DISPLAY_LABELS", "function backendDisplayLabel"
        )
        harness = block + """
const cases = [
  ['ollama', 'OLLAMA'],
  ['openai_compatible', 'OPENAI COMPATIBLE'],
  ['', 'LOCAL BACKEND'],
  [null, 'LOCAL BACKEND'],
  [undefined, 'LOCAL BACKEND'],
  ['OLLAMA', 'LOCAL BACKEND'],
  ['openai', 'LOCAL BACKEND'],
  ['ollama-extra', 'LOCAL BACKEND'],
  ['<img src=x onerror=alert(1)>', 'LOCAL BACKEND'],
];
const failures = [];
for (const [input, expected] of cases) {
  const got = backendDisplayLabel(input);
  if (got !== expected) {
    failures.push(`${JSON.stringify(input)} -> ${JSON.stringify(got)} (want ${JSON.stringify(expected)})`);
  }
}
if (failures.length) {
  console.error('MISMATCH\\n' + failures.join('\\n'));
  process.exit(1);
}
console.log('OK');
"""
        result = subprocess.run(
            ["node", "-e", harness],
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            result.returncode, 0, msg=result.stderr or result.stdout
        )


if __name__ == "__main__":
    unittest.main()