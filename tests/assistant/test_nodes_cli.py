"""src.assistant.nodes CLI validation tests."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "src.assistant.nodes", *args],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )


VALID_INVENTORY = {
    "nodes": [
        {
            "node_id": "node-a",
            "role": "always-on",
            "hostname": "host-a",
            "providers": [
                {
                    "provider_id": "prov-a",
                    "base_url": "http://localhost:11434",
                    "model": "model-a",
                    "local": True,
                    "profiles": ["chat"],
                }
            ],
        },
        {
            "node_id": "node-b",
            "role": "interactive",
            "hostname": "host-b",
            "providers": [
                {
                    "provider_id": "prov-b",
                    "base_url": "http://other:11434",
                    "model": "model-b",
                    "local": True,
                    "profiles": ["voice"],
                }
            ],
        },
    ],
    "default_provider_id": "prov-a",
    "fallback_provider_ids": ["prov-b"],
}


class NodesCLITest(unittest.TestCase):
    def test_valid_inventory_validate_keyword(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(VALID_INVENTORY, f)
            path = f.name

        result = _run_cli("validate", path)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("node node-a", result.stdout)
        self.assertIn("node node-b", result.stdout)
        self.assertIn("OK", result.stdout)

    def test_valid_inventory_bare_path(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(VALID_INVENTORY, f)
            path = f.name

        result = _run_cli(path)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("node node-a", result.stdout)
        self.assertIn("OK", result.stdout)

    def test_invalid_json(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write("{invalid json content")
            path = f.name

        result = _run_cli("validate", path)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Error", result.stderr)

    def test_missing_file(self) -> None:
        result = _run_cli("validate", "/nonexistent/path/inventory.json")
        self.assertNotEqual(result.returncode, 0)

    def test_invalid_inventory_schema(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump({"nodes": []}, f)
            path = f.name

        result = _run_cli("validate", path)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("InvalidNodeInventoryError", result.stderr)

    def test_no_args_returns_error(self) -> None:
        result = _run_cli()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("usage", result.stderr)


if __name__ == "__main__":
    unittest.main()
