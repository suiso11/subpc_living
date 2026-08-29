"""src.assistant.nodes CLI validation tests."""

import copy
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


OPENAI_LOOPBACK_INVENTORY = {
    "nodes": [
        {
            "node_id": "local-node",
            "role": "always-on",
            "hostname": "localhost",
            "providers": [
                {
                    "provider_id": "local-openai",
                    "base_url": "http://localhost:8080/v1",
                    "model": "local-model",
                    "local": True,
                    "provider_kind": "openai_compatible",
                    "profiles": ["chat"],
                }
            ],
        }
    ],
    "default_provider_id": "local-openai",
    "fallback_provider_ids": [],
}


class NodesOpenAICLITest(unittest.TestCase):
    def test_openai_compatible_loopback_inventory_validates(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(OPENAI_LOOPBACK_INVENTORY, f)
            path = f.name

        result = _run_cli("validate", path)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("node local-node", result.stdout)
        self.assertIn("OK", result.stdout)

    def test_openai_compatible_non_loopback_url_rejected(self) -> None:
        data = dict(OPENAI_LOOPBACK_INVENTORY)
        data["nodes"][0]["providers"][0]["base_url"] = "http://main-pc:8080/v1"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(data, f)
            path = f.name

        result = _run_cli("validate", path)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("InvalidNodeInventoryError", result.stderr)

    def test_unknown_provider_kind_rejected(self) -> None:
        data = dict(OPENAI_LOOPBACK_INVENTORY)
        data["nodes"][0]["providers"][0]["provider_kind"] = "anthropic"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(data, f)
            path = f.name

        result = _run_cli("validate", path)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("provider_kind", result.stderr)

    def test_blank_provider_kind_normalizes_to_ollama(self) -> None:
        data = copy.deepcopy(VALID_INVENTORY)
        data["nodes"][0]["providers"][0]["provider_kind"] = "   "
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(data, f)
            path = f.name

        result = _run_cli("validate", path)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("node node-a", result.stdout)
        self.assertIn("OK", result.stdout)

    def test_openai_compatible_local_false_rejected(self) -> None:
        data = copy.deepcopy(OPENAI_LOOPBACK_INVENTORY)
        data["nodes"][0]["providers"][0]["local"] = False
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(data, f)
            path = f.name

        result = _run_cli("validate", path)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("InvalidNodeInventoryError", result.stderr)


if __name__ == "__main__":
    unittest.main()
