from contextlib import closing
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from src.assistant.contracts import AssistantRequest
from src.assistant.nodes import (
    InvalidNodeInventoryError,
    NodeInventory,
    build_node_service,
)
from src.assistant.run_logger import SQLiteRunLogger
from src.llm.providers.fake import FakeProvider


class NodeInventoryTest(unittest.TestCase):
    @staticmethod
    def mapping() -> dict:
        return {
            "nodes": [
                {
                    "node_id": "sub-pc",
                    "role": "always-on",
                    "hostname": "localhost",
                    "providers": [
                        {
                            "provider_id": "local-strong",
                            "base_url": "http://localhost:11434",
                            "model": "strong-model",
                            "local": True,
                            "profiles": ["chat_auto", "deep_reasoning"],
                        }
                    ],
                },
                {
                    "node_id": "main-pc",
                    "role": "interactive",
                    "hostname": "main-pc",
                    "providers": [
                        {
                            "provider_id": "local-fast",
                            "base_url": "http://main-pc:11434",
                            "model": "fast-model",
                            "local": True,
                            "profiles": ["voice_fast", "task_local"],
                        }
                    ],
                },
            ],
            "default_provider_id": "local-strong",
            "fallback_provider_ids": ["local-fast"],
        }

    @staticmethod
    def request(profile: str = "chat_auto") -> AssistantRequest:
        return AssistantRequest(
            text="hello",
            conversation_id="conversation",
            channel="web",
            profile=profile,
        )

    @staticmethod
    def build(inventory: NodeInventory):
        providers: dict[str, FakeProvider] = {}
        specs = []

        def factory(spec):
            specs.append(spec)
            provider = FakeProvider(spec.provider_id, model=spec.model)
            providers[spec.provider_id] = provider
            return provider

        service, registry = build_node_service(
            inventory, provider_factory=factory
        )
        return service, registry, providers, specs

    def test_registers_providers_in_definition_order_with_node_ids(self) -> None:
        inventory = NodeInventory.from_mapping(self.mapping())

        _, registry, _, specs = self.build(inventory)

        self.assertEqual(
            [entry.provider_id for entry in registry.entries()],
            ["local-strong", "local-fast"],
        )
        self.assertEqual(
            [(spec.provider_id, spec.node_id) for spec in specs],
            [("local-strong", "sub-pc"), ("local-fast", "main-pc")],
        )
        strong = registry.get("local-strong")
        fast = registry.get("local-fast")
        self.assertTrue(strong.local)
        self.assertEqual(strong.profiles, ("chat_auto", "deep_reasoning"))
        self.assertTrue(fast.local)
        self.assertEqual(fast.profiles, ("voice_fast", "task_local"))

    def test_flattens_multiple_providers_from_one_node(self) -> None:
        data = self.mapping()
        data["nodes"] = [data["nodes"][0]]
        data["nodes"][0]["providers"].append(
            {
                "provider_id": "local-small",
                "base_url": "http://localhost:11435",
                "model": "small-model",
                "local": True,
                "profiles": ["voice_fast"],
            }
        )
        data["fallback_provider_ids"] = ["local-small"]

        inventory = NodeInventory.from_mapping(data)

        self.assertEqual(
            [spec.provider_id for spec in inventory.providers()],
            ["local-strong", "local-small"],
        )
        self.assertEqual(
            [spec.node_id for spec in inventory.providers()],
            ["sub-pc", "sub-pc"],
        )

    def test_preserves_unknown_metadata(self) -> None:
        data = self.mapping()
        data["nodes"][0]["metadata"] = {"gpu": "P40", "ram_gb": 48}

        inventory = NodeInventory.from_mapping(data)

        self.assertEqual(
            dict(inventory.nodes[0].metadata),
            {"gpu": "P40", "ram_gb": 48},
        )
        with self.assertRaises(TypeError):
            inventory.nodes[0].metadata["gpu"] = "changed"

    def test_profile_route_uses_matching_provider(self) -> None:
        service, _, providers, _ = self.build(
            NodeInventory.from_mapping(self.mapping())
        )

        response = service.generate(
            self.request("voice_fast"),
            [{"role": "user", "content": "hello"}],
        )

        self.assertEqual(response.route.provider_id, "local-fast")
        self.assertEqual(response.text, "local-fast")
        self.assertEqual(providers["local-strong"].calls, [])

    def test_request_without_matching_profile_uses_default_provider(self) -> None:
        service, _, providers, _ = self.build(
            NodeInventory.from_mapping(self.mapping())
        )

        response = service.generate(
            self.request("code_auto"),
            [{"role": "user", "content": "hello"}],
        )

        self.assertEqual(response.route.provider_id, "local-strong")
        self.assertEqual(response.text, "local-strong")
        self.assertEqual(providers["local-fast"].calls, [])

    def test_unavailable_default_uses_inventory_fallback(self) -> None:
        service, _, providers, _ = self.build(
            NodeInventory.from_mapping(self.mapping())
        )
        providers["local-strong"].available = False

        response = service.generate(
            self.request("code_auto"),
            [{"role": "user", "content": "hello"}],
        )

        self.assertEqual(response.route.provider_id, "local-fast")
        self.assertIn("fallback", response.route.reason)
        self.assertIn("local-strong=unavailable", response.route.reason)

    def test_unavailable_profile_primary_falls_back_to_default_provider(self) -> None:
        service, _, providers, _ = self.build(
            NodeInventory.from_mapping(self.mapping())
        )
        providers["local-fast"].available = False

        response = service.generate(
            self.request("voice_fast"),
            [{"role": "user", "content": "hello"}],
        )

        self.assertEqual(response.route.provider_id, "local-strong")
        self.assertEqual(response.text, "local-strong")
        self.assertIn("fallback from local-fast", response.route.reason)
        self.assertIn("local-fast=unavailable", response.route.reason)

    def test_duplicate_profile_keeps_first_defined_provider(self) -> None:
        data = self.mapping()
        data["nodes"][1]["providers"][0]["profiles"].append("chat_auto")
        service, _, _, _ = self.build(NodeInventory.from_mapping(data))

        response = service.generate(
            self.request("chat_auto"),
            [{"role": "user", "content": "hello"}],
        )

        self.assertEqual(response.route.provider_id, "local-strong")

    def test_omitted_run_logger_uses_env_db_path(self) -> None:
        inventory = NodeInventory.from_mapping(self.mapping())
        factory = lambda spec: FakeProvider(spec.provider_id, model=spec.model)
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "nested" / "runs.db"
            with patch.dict(os.environ, {"ASSISTANT_RUN_LOG_DB": str(db_path)}):
                service, _ = build_node_service(
                    inventory, provider_factory=factory
                )
                response = service.generate(
                    self.request(), [{"role": "user", "content": "hello"}]
                )

            self.assertEqual(response.text, "local-strong")
            with closing(sqlite3.connect(db_path)) as connection:
                run = connection.execute(
                    "SELECT success, provider_id FROM model_runs"
                ).fetchone()
                route = connection.execute(
                    "SELECT provider_id FROM route_decisions"
                ).fetchone()
        self.assertEqual(run, (1, "local-strong"))
        self.assertEqual(route, ("local-strong",))

    def test_explicit_none_run_logger_creates_no_db(self) -> None:
        inventory = NodeInventory.from_mapping(self.mapping())
        factory = lambda spec: FakeProvider(spec.provider_id, model=spec.model)
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "unused.db"
            with patch.dict(os.environ, {"ASSISTANT_RUN_LOG_DB": str(db_path)}):
                service, _ = build_node_service(
                    inventory, run_logger=None, provider_factory=factory
                )
                service.generate(
                    self.request(), [{"role": "user", "content": "hello"}]
                )

            self.assertIsNone(service._run_logger)
            self.assertFalse(db_path.exists())

    def test_explicit_run_logger_is_used_as_is(self) -> None:
        inventory = NodeInventory.from_mapping(self.mapping())
        factory = lambda spec: FakeProvider(spec.provider_id, model=spec.model)
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "runs.db"
            logger = SQLiteRunLogger(db_path)
            service, _ = build_node_service(
                inventory, run_logger=logger, provider_factory=factory
            )
            service.generate(
                self.request(), [{"role": "user", "content": "hello"}]
            )

            with closing(sqlite3.connect(db_path)) as connection:
                run = connection.execute(
                    "SELECT success FROM model_runs"
                ).fetchone()
        self.assertEqual(run, (1,))

    def test_default_run_logger_init_failure_falls_back_to_none(self) -> None:
        inventory = NodeInventory.from_mapping(self.mapping())
        factory = lambda spec: FakeProvider(spec.provider_id, model=spec.model)
        for exc in (
            OSError("disk full"),
            sqlite3.OperationalError("disk I/O error"),
        ):
            with self.subTest(exc=exc):
                with tempfile.TemporaryDirectory() as tmp:
                    db_path = Path(tmp) / "nested" / "runs.db"
                    with patch.dict(
                        os.environ, {"ASSISTANT_RUN_LOG_DB": str(db_path)}
                    ):
                        with patch(
                            "src.assistant.factory.SQLiteRunLogger",
                            side_effect=exc,
                        ):
                            with self.assertLogs(
                                "src.assistant.factory", level="WARNING"
                            ) as logs:
                                service, _ = build_node_service(
                                    inventory, provider_factory=factory
                                )
                                response = service.generate(
                                    self.request(),
                                    [{"role": "user", "content": "hello"}],
                                )

                self.assertEqual(response.text, "local-strong")
                self.assertIsNone(service._run_logger)
                self.assertFalse(db_path.exists())
                self.assertTrue(
                    any("run logger" in message for message in logs.output)
                )

    def test_example_json_loads_as_two_nodes_and_two_providers(self) -> None:
        path = Path(__file__).resolve().parents[2] / "config" / "nodes.example.json"

        inventory = NodeInventory.load(path)

        self.assertEqual(len(inventory.nodes), 2)
        self.assertEqual(len(inventory.providers()), 2)
        self.assertEqual(inventory.default_provider_id, "local-strong")
        self.assertEqual(inventory.fallback_provider_ids, ("local-fast",))

    def test_validation_rejects_empty_nodes(self) -> None:
        with self.assertRaisesRegex(InvalidNodeInventoryError, "nodes"):
            NodeInventory.from_mapping(
                {"nodes": [], "default_provider_id": "local-strong"}
            )

    def test_validation_rejects_non_mapping_node_or_provider(self) -> None:
        non_mapping_node = self.mapping()
        non_mapping_node["nodes"][0] = "sub-pc"
        non_mapping_provider = self.mapping()
        non_mapping_provider["nodes"][0]["providers"][0] = "local-strong"

        for data in (non_mapping_node, non_mapping_provider):
            with self.subTest(data=data):
                with self.assertRaisesRegex(
                    InvalidNodeInventoryError, "mapping"
                ):
                    NodeInventory.from_mapping(data)

    def test_validation_rejects_non_string_or_empty_ids(self) -> None:
        cases = []
        for path, value in (
            (("node_id",), 1),
            (("node_id",), " "),
            (("providers", 0, "provider_id"), 1),
            (("providers", 0, "provider_id"), " "),
        ):
            data = self.mapping()
            if path[0] == "node_id":
                data["nodes"][0][path[0]] = value
            else:
                data["nodes"][0][path[0]][path[1]][path[2]] = value
            cases.append(data)

        for data in cases:
            with self.subTest(data=data):
                with self.assertRaises(InvalidNodeInventoryError):
                    NodeInventory.from_mapping(data)

    def test_validation_rejects_duplicate_provider_id(self) -> None:
        data = self.mapping()
        data["nodes"][1]["providers"][0]["provider_id"] = "local-strong"

        with self.assertRaisesRegex(
            InvalidNodeInventoryError, "local-strong"
        ):
            NodeInventory.from_mapping(data)

    def test_validation_rejects_empty_or_non_string_base_url_and_model(self) -> None:
        for field in ("base_url", "model"):
            for value in ("", 1):
                with self.subTest(field=field, value=value):
                    data = self.mapping()
                    data["nodes"][0]["providers"][0][field] = value
                    with self.assertRaisesRegex(
                        InvalidNodeInventoryError, "local-strong"
                    ):
                        NodeInventory.from_mapping(data)

    def test_validation_rejects_non_bool_local(self) -> None:
        for value in ("false", "true", 1, None):
            with self.subTest(value=value):
                data = self.mapping()
                data["nodes"][0]["providers"][0]["local"] = value
                with self.assertRaisesRegex(
                    InvalidNodeInventoryError, "local"
                ):
                    NodeInventory.from_mapping(data)

    def test_validation_rejects_invalid_profiles(self) -> None:
        for value in ("chat_auto", ["chat_auto", 1]):
            with self.subTest(value=value):
                data = self.mapping()
                data["nodes"][0]["providers"][0]["profiles"] = value
                with self.assertRaisesRegex(
                    InvalidNodeInventoryError, "profiles"
                ):
                    NodeInventory.from_mapping(data)

    def test_validation_rejects_duplicate_node_ids_after_normalization(self) -> None:
        data = self.mapping()
        data["nodes"][0]["node_id"] = " a "
        data["nodes"][1]["node_id"] = "a"

        with self.assertRaisesRegex(InvalidNodeInventoryError, "node_id"):
            NodeInventory.from_mapping(data)

    def test_validation_rejects_empty_providers_or_non_mapping_metadata(self) -> None:
        empty_providers = self.mapping()
        empty_providers["nodes"][0]["providers"] = []
        invalid_metadata = self.mapping()
        invalid_metadata["nodes"][0]["metadata"] = ["gpu", "P40"]

        for data, problem in (
            (empty_providers, "providers"),
            (invalid_metadata, "metadata"),
        ):
            with self.subTest(problem=problem):
                with self.assertRaisesRegex(InvalidNodeInventoryError, problem):
                    NodeInventory.from_mapping(data)

    def test_normalizes_node_and_provider_ids(self) -> None:
        data = self.mapping()
        data["nodes"][0]["node_id"] = " sub-pc "
        data["nodes"][1]["providers"][0]["provider_id"] = " local-fast "

        inventory = NodeInventory.from_mapping(data)

        self.assertEqual(inventory.nodes[0].node_id, "sub-pc")
        self.assertEqual(inventory.providers()[1].provider_id, "local-fast")
        self.assertEqual(inventory.providers()[1].node_id, "main-pc")

    def test_validation_rejects_duplicate_provider_ids_after_normalization(self) -> None:
        data = self.mapping()
        data["nodes"][0]["providers"][0]["provider_id"] = " a "
        data["nodes"][1]["providers"][0]["provider_id"] = "a"
        data["default_provider_id"] = "a"

        with self.assertRaisesRegex(InvalidNodeInventoryError, "provider_id"):
            NodeInventory.from_mapping(data)

    def test_validation_rejects_unknown_default_provider(self) -> None:
        data = self.mapping()
        data["default_provider_id"] = "missing-default"

        with self.assertRaisesRegex(
            InvalidNodeInventoryError, "missing-default"
        ):
            NodeInventory.from_mapping(data)

    def test_validation_rejects_unknown_or_duplicate_fallback(self) -> None:
        cases = (
            (["missing-fallback"], "missing-fallback"),
            (["local-fast", " local-fast "], "local-fast"),
        )
        for fallbacks, problem_id in cases:
            with self.subTest(fallbacks=fallbacks):
                data = self.mapping()
                data["fallback_provider_ids"] = fallbacks
                with self.assertRaisesRegex(
                    InvalidNodeInventoryError, problem_id
                ):
                    NodeInventory.from_mapping(data)

    def test_validation_rejects_default_in_fallbacks(self) -> None:
        data = self.mapping()
        data["fallback_provider_ids"] = [" local-strong "]

        with self.assertRaisesRegex(
            InvalidNodeInventoryError, "local-strong"
        ):
            NodeInventory.from_mapping(data)


if __name__ == "__main__":
    unittest.main()
