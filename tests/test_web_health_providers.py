"""Tests for /api/health providers key and AssistantService provider_stats."""

from __future__ import annotations

import asyncio
import json
import types
import unittest
from unittest.mock import patch

from src.assistant.service import AssistantService
from src.llm.errors import ProviderRequestError
from src.llm.providers.fake import FakeProvider
from src.llm.registry import ProviderRegistry
from src.llm.routing.contracts import RouteDecision
from src.web import server


class _FixedRouter:
    def __init__(self, decision: RouteDecision) -> None:
        self.decision = decision

    def route(self, request):
        return self.decision


def _run_health_with_registry(
    registry: ProviderRegistry,
    service: AssistantService | None = None,
    *,
    provider_id: str | None = None,
    provider_kind: str = "ollama",
    provider_base_url: str | None = None,
    provider_api_key_env: str | None = None,
    health_result: dict | None = None,
):
    """Helper: temporarily set server globals, run health(), restore.

    HealthChecker は常にモックする (オフライン)。``health_result`` を渡せば
    ``check_all`` の返却値を固定する。返り値は ``(response, mock_instance)``。
    """
    orig_reg = server.provider_registry
    orig_svc = server.assistant_service
    orig_cfg = server.config
    orig_pid = server.primary_provider_id
    orig_kind = server.primary_provider_kind
    orig_base = server.primary_provider_base_url
    orig_key_env = server.primary_provider_api_key_env
    # Also mock config so ollama_base_url is available and the
    # modules check using provider_registry.get("ollama") does not crash.
    mock_config = types.SimpleNamespace(
        ollama_base_url="http://localhost:11434", model="cfg-model"
    )
    try:
        server.provider_registry = registry
        server.assistant_service = service
        server.config = mock_config
        server.primary_provider_id = provider_id
        server.primary_provider_kind = provider_kind
        server.primary_provider_base_url = provider_base_url
        server.primary_provider_api_key_env = provider_api_key_env
        # If no "ollama" in registry, add one for the modules section
        if "ollama" not in registry:
            _p = FakeProvider("ok", model="ollama-model")
            registry.register("ollama", _p, local=True)
        with patch("src.web.server.HealthChecker") as mock_cls:
            instance = mock_cls.return_value
            if health_result is not None:
                instance.check_all.return_value = health_result
            response = asyncio.run(server.health())
            return response, instance
    finally:
        server.provider_registry = orig_reg
        server.assistant_service = orig_svc
        server.config = orig_cfg
        server.primary_provider_id = orig_pid
        server.primary_provider_kind = orig_kind
        server.primary_provider_base_url = orig_base
        server.primary_provider_api_key_env = orig_key_env


class HealthProvidersTest(unittest.TestCase):
    """Verify /api/health response includes 'providers' with correct shape."""

    def _run_health_with_registry(self, registry, service=None, *, provider_id=None, provider_kind="ollama", provider_base_url=None, health_result=None):
        return _run_health_with_registry(
            registry, service, provider_id=provider_id, provider_kind=provider_kind,
            provider_base_url=provider_base_url, health_result=health_result,
        )

    def test_health_response_includes_providers_key(self) -> None:
        registry = ProviderRegistry()
        provider = FakeProvider("hi", model="test-model")
        registry.register("test-p", provider, local=True)

        svc = AssistantService(
            registry,
            _FixedRouter(
                RouteDecision(
                    provider_id="test-p",
                    model="test-model",
                    local=True,
                    reason="test",
                )
            ),
        )

        result, _ = self._run_health_with_registry(registry, svc)
        body = json.loads(result.body)
        self.assertIn("providers", body)
        providers = body["providers"]
        self.assertIsInstance(providers, list)
        self.assertEqual(len(providers), 2)  # test-p + ollama
        # Find our custom entry
        entry = next(e for e in providers if e["provider_id"] == "test-p")
        expected_keys = {"provider_id", "provider_kind", "local", "available", "error"}
        self.assertEqual(set(entry.keys()), expected_keys)
        self.assertNotIn("model", entry)
        self.assertEqual(entry["provider_id"], "test-p")
        self.assertEqual(entry["provider_kind"], "ollama")
        self.assertTrue(entry["local"])
        self.assertIsInstance(entry["available"], bool)
        self.assertIsNone(entry["error"])

    def test_health_response_providers_available_field(self) -> None:
        registry = ProviderRegistry()
        provider = FakeProvider("hi", model="m", available=False)
        registry.register("unavail", provider, local=True)
        service = AssistantService(
            registry,
            _FixedRouter(
                RouteDecision(
                    provider_id="unavail",
                    model="m",
                    local=True,
                    reason="test",
                )
            ),
        )
        result, _ = self._run_health_with_registry(registry, service)
        body = json.loads(result.body)
        entry = next(e for e in body["providers"] if e["provider_id"] == "unavail")
        self.assertFalse(entry["available"])

    def test_health_providers_no_secrets_in_output(self) -> None:
        """Ensure provider entries contain only the safe health allowlist."""
        registry = ProviderRegistry()
        provider = FakeProvider("secret stuff", model="m")
        registry.register("p", provider, local=True)
        service = AssistantService(
            registry,
            _FixedRouter(
                RouteDecision(provider_id="p", model="m", local=True, reason="t")
            ),
        )
        result, _ = self._run_health_with_registry(registry, service)
        body = json.loads(result.body)
        entry = next(e for e in body["providers"] if e["provider_id"] == "p")
        self.assertEqual(set(entry.keys()), {
            "provider_id", "provider_kind", "local", "available", "error"
        })
        self.assertNotIn("model", entry)


class WebProviderSelectionTest(unittest.TestCase):
    """Backend-neutral status/health selection through the tracked primary provider."""

    @staticmethod
    def _inventory_registry():
        from src.assistant.nodes import NodeInventory, build_node_service

        inventory = NodeInventory.from_mapping({
            "default_provider_id": "local-openai",
            "fallback_provider_ids": ["ollama"],
            "nodes": [
                {
                    "node_id": "main",
                    "providers": [
                        {
                            "provider_id": "ollama",
                            "base_url": "http://localhost:11434",
                            "model": "ollama-model",
                            "provider_kind": "ollama",
                        },
                        {
                            "provider_id": "local-openai",
                            "base_url": "http://localhost:8080/v1",
                            "model": "local-model",
                            "provider_kind": "openai_compatible",
                        },
                    ],
                }
            ],
        })
        return inventory, build_node_service(inventory)

    def _set_globals(self, *, config, registry, service=None, provider_id=None, provider_kind="ollama", provider_base_url=None, provider_api_key_env=None):
        self._orig = (
            server.config,
            server.provider_registry,
            server.assistant_service,
            server.primary_provider_id,
            server.primary_provider_kind,
            server.primary_provider_base_url,
            server.primary_provider_api_key_env,
        )
        server.config = config
        server.provider_registry = registry
        server.assistant_service = service
        server.primary_provider_id = provider_id
        server.primary_provider_kind = provider_kind
        server.primary_provider_base_url = provider_base_url
        server.primary_provider_api_key_env = provider_api_key_env

    def _restore_globals(self) -> None:
        (
            server.config,
            server.provider_registry,
            server.assistant_service,
            server.primary_provider_id,
            server.primary_provider_kind,
            server.primary_provider_base_url,
            server.primary_provider_api_key_env,
        ) = self._orig

    def _run_health_with_registry(self, registry, service=None, *, provider_id=None, provider_kind="ollama", provider_base_url=None, health_result=None):
        return _run_health_with_registry(
            registry, service, provider_id=provider_id, provider_kind=provider_kind,
            provider_base_url=provider_base_url, health_result=health_result,
        )

    def test_health_ollama_backend_uses_selected_provider_mode(self) -> None:
        registry = ProviderRegistry()
        registry.register("ollama", FakeProvider("ok", model="m"), local=True)
        result, instance = self._run_health_with_registry(
            registry, None, provider_id="ollama", provider_kind="ollama",
            provider_base_url="http://localhost:11434",
            health_result={"status": "ok", "checks": {"ollama": {"status": "ok"}}},
        )
        self.assertEqual(instance.check_all.call_args.kwargs.get("provider_kind"), "ollama")
        self.assertEqual(
            instance.check_all.call_args.kwargs.get("provider_url"),
            "http://localhost:11434",
        )
        self.assertFalse(instance.check_all.call_args.kwargs.get("include_web"))
        body = json.loads(result.body)
        self.assertEqual(body["modules"]["provider_id"], "ollama")
        self.assertEqual(body["modules"]["provider_kind"], "ollama")

    def test_health_openai_backend_uses_selected_provider_mode(self) -> None:
        registry = ProviderRegistry()
        registry.register("local-openai", FakeProvider("ok", model="m"), local=True)
        result, instance = self._run_health_with_registry(
            registry, None, provider_id="local-openai", provider_kind="openai_compatible",
            provider_base_url="http://localhost:8080/v1",
            health_result={"status": "ok", "checks": {"local_provider": {"status": "ok"}}},
        )
        self.assertEqual(
            instance.check_all.call_args.kwargs.get("provider_kind"),
            "openai_compatible",
        )
        self.assertEqual(
            instance.check_all.call_args.kwargs.get("provider_url"),
            "http://localhost:8080/v1",
        )
        # Ollama /api/tags プローブに繋がる include_ollama は渡さない
        self.assertNotIn("include_ollama", instance.check_all.call_args.kwargs)
        body = json.loads(result.body)
        self.assertEqual(body["modules"]["provider_id"], "local-openai")
        self.assertEqual(body["modules"]["provider_kind"], "openai_compatible")

    def test_status_uses_inventory_default_provider_id(self) -> None:
        inventory, (service, registry) = self._inventory_registry()
        default_spec = server._inventory_provider_spec(
            inventory, inventory.default_provider_id
        )
        self.assertIsNotNone(default_spec)
        self.assertEqual(default_spec.provider_kind, "openai_compatible")
        cfg = types.SimpleNamespace(model="cfg-model")
        self._set_globals(
            config=cfg,
            registry=registry,
            service=service,
            provider_id=inventory.default_provider_id,
            provider_kind=default_spec.provider_kind,
            provider_base_url=default_spec.base_url,
        )
        try:
            with patch("src.web.server.HealthChecker") as mock_cls:
                mock_cls.return_value.check_all.return_value = {
                    "status": "ok",
                    "checks": {"local_provider": {"status": "ok"}},
                }
                body = asyncio.run(server.status())
        finally:
            self._restore_globals()

        self.assertEqual(body["provider_id"], "local-openai")
        self.assertEqual(body["provider_kind"], "openai_compatible")
        self.assertNotIn("model", body)
        self.assertNotIn("stt_model", body)
        self.assertEqual(body["provider_reachability"], "ok")
        self.assertIs(body["ollama"], True)  # 選択中 (local-openai) の到達性
        self.assertIs(body["local_provider"], True)

    def test_status_legacy_alias_and_neutral_fields_without_secrets(self) -> None:
        registry = ProviderRegistry()
        registry.register("local-openai", FakeProvider("ok", model="m"), local=True)
        cfg = types.SimpleNamespace(model="cfg-model")
        self._set_globals(
            config=cfg,
            registry=registry,
            provider_id="local-openai",
            provider_kind="openai_compatible",
            provider_base_url="http://localhost:8080/v1",
        )
        try:
            with patch("src.web.server.HealthChecker") as mock_cls:
                mock_cls.return_value.check_all.return_value = {
                    "status": "ok",
                    "checks": {"local_provider": {"status": "ok"}},
                }
                body = asyncio.run(server.status())
        finally:
            self._restore_globals()

        self.assertIn("ollama", body)
        self.assertIsInstance(body["ollama"], bool)
        self.assertIn("local_provider", body)
        self.assertIn("provider_id", body)
        self.assertIn("provider_kind", body)
        self.assertIn("provider_reachability", body)
        self.assertIn(body["provider_reachability"], {"ok", "error", "unknown", "unconfigured"})
        self.assertNotIn("base_url", body)
        self.assertNotIn("api_key", body)
        self.assertNotIn("http://localhost:8080", json.dumps(body))

    def test_health_modules_legacy_alias_and_neutral_fields(self) -> None:
        registry = ProviderRegistry()
        registry.register("local-openai", FakeProvider("ok", model="m"), local=True)
        result, _ = self._run_health_with_registry(
            registry, None, provider_id="local-openai", provider_kind="openai_compatible",
            provider_base_url="http://localhost:8080/v1",
            health_result={"status": "ok", "checks": {"local_provider": {"status": "ok"}}},
        )
        body = json.loads(result.body)
        modules = body["modules"]
        self.assertIn("ollama", modules)
        self.assertIsInstance(modules["ollama"], bool)
        self.assertIn("local_provider", modules)
        self.assertIn("provider_id", modules)
        self.assertIn("provider_kind", modules)
        self.assertIn("provider_reachability", modules)
        self.assertIn(modules["provider_reachability"], {"ok", "error", "unknown", "unconfigured"})
        self.assertNotIn("base_url", modules)
        self.assertNotIn("api_key", modules)
        self.assertNotIn("http://localhost:8080", json.dumps(body))


class WebProviderReachabilityTest(unittest.TestCase):
    """Offline tests for selected-provider reachability in /api/health + /api/status."""

    def _registry(self, provider_id: str = "local-openai", available: bool = True):
        registry = ProviderRegistry()
        registry.register(provider_id, FakeProvider("ok", model="m", available=available), local=True)
        return registry

    def test_openai_ok_reachability_without_ollama_probe(self) -> None:
        registry = self._registry()
        result, instance = _run_health_with_registry(
            registry, None, provider_id="local-openai", provider_kind="openai_compatible",
            provider_base_url="http://localhost:8080/v1",
            health_result={"status": "ok", "checks": {"local_provider": {"status": "ok"}}},
        )
        # selected-provider モードでプローブし、Ollama /api/tags に繋がる include_ollama は使わない
        self.assertEqual(instance.check_all.call_args.kwargs.get("provider_kind"), "openai_compatible")
        self.assertEqual(instance.check_all.call_args.kwargs.get("provider_url"), "http://localhost:8080/v1")
        self.assertFalse(instance.check_all.call_args.kwargs.get("include_web"))
        self.assertNotIn("include_ollama", instance.check_all.call_args.kwargs)
        body = json.loads(result.body)
        self.assertEqual(body["modules"]["provider_reachability"], "ok")
        self.assertIs(body["modules"]["ollama"], True)
        self.assertIs(body["modules"]["local_provider"], True)
        self.assertEqual(body["status"], "ok")

    def test_openai_error_reachability(self) -> None:
        registry = self._registry()
        result, _ = _run_health_with_registry(
            registry, None, provider_id="local-openai", provider_kind="openai_compatible",
            provider_base_url="http://localhost:8080/v1",
            health_result={"status": "error", "checks": {"local_provider": {"status": "error"}}},
        )
        body = json.loads(result.body)
        self.assertEqual(body["modules"]["provider_reachability"], "error")
        self.assertIs(body["modules"]["ollama"], False)
        self.assertIs(body["modules"]["local_provider"], False)
        self.assertEqual(body["status"], "error")

    def test_openai_unknown_404_not_reported_ok(self) -> None:
        registry = self._registry()
        result, _ = _run_health_with_registry(
            registry, None, provider_id="local-openai", provider_kind="openai_compatible",
            provider_base_url="http://localhost:8080/v1",
            health_result={"status": "degraded", "checks": {"local_provider": {"status": "unknown"}}},
        )
        body = json.loads(result.body)
        # unknown は ok/true として報告しない。overall は HealthChecker 結果 (degraded) を維持
        self.assertEqual(body["modules"]["provider_reachability"], "unknown")
        self.assertIs(body["modules"]["ollama"], False)
        self.assertIs(body["modules"]["local_provider"], False)
        self.assertEqual(body["status"], "degraded")

    def test_ollama_compatibility_derives_from_checks_ollama(self) -> None:
        registry = self._registry("ollama")
        # checks に local_provider が無く ollama だけの場合も ok になる (= ollama キー由来)
        result, instance = _run_health_with_registry(
            registry, None, provider_id="ollama", provider_kind="ollama",
            provider_base_url="http://localhost:11434",
            health_result={"status": "ok", "checks": {"ollama": {"status": "ok"}}},
        )
        self.assertEqual(instance.check_all.call_args.kwargs.get("provider_kind"), "ollama")
        self.assertEqual(instance.check_all.call_args.kwargs.get("provider_url"), "http://localhost:11434")
        body = json.loads(result.body)
        self.assertEqual(body["modules"]["provider_reachability"], "ok")
        self.assertIs(body["modules"]["ollama"], True)
        self.assertIs(body["modules"]["local_provider"], True)

    def test_inventory_default_uses_actual_backend_kind_and_base(self) -> None:
        inventory, _ = WebProviderSelectionTest._inventory_registry()
        spec = server._inventory_provider_spec(inventory, inventory.default_provider_id)
        self.assertIsNotNone(spec)
        # 既定は openai_compatible (local-openai) であり、実 kind / base を引き継ぐ
        self.assertEqual(spec.provider_kind, "openai_compatible")
        self.assertEqual(spec.base_url, "http://localhost:8080/v1")

    def test_no_url_or_key_exposure_in_health_and_status(self) -> None:
        registry = self._registry()
        health_result = {"status": "ok", "checks": {"local_provider": {"status": "ok"}}}
        result, _ = _run_health_with_registry(
            registry, None, provider_id="local-openai", provider_kind="openai_compatible",
            provider_base_url="http://localhost:8080/v1", health_result=health_result,
        )
        health_text = result.body.decode("utf-8", errors="replace")
        self.assertNotIn("http://localhost:8080", health_text)
        self.assertNotIn("base_url", health_text)
        self.assertNotIn("api_key", health_text)

        orig_cfg, orig_reg, orig_pid, orig_kind, orig_base = (
            server.config, server.provider_registry, server.primary_provider_id,
            server.primary_provider_kind, server.primary_provider_base_url,
        )
        try:
            server.config = types.SimpleNamespace(model="cfg-model")
            server.provider_registry = registry
            server.primary_provider_id = "local-openai"
            server.primary_provider_kind = "openai_compatible"
            server.primary_provider_base_url = "http://localhost:8080/v1"
            with patch("src.web.server.HealthChecker") as mock_cls:
                mock_cls.return_value.check_all.return_value = health_result
                body = asyncio.run(server.status())
        finally:
            server.config, server.provider_registry, server.primary_provider_id, \
                server.primary_provider_kind, server.primary_provider_base_url = \
                orig_cfg, orig_reg, orig_pid, orig_kind, orig_base
        self.assertNotIn("http://localhost:8080", json.dumps(body))
        self.assertNotIn("base_url", body)
        self.assertNotIn("api_key", body)

    def test_uninitialized_state_reports_unconfigured_without_probe(self) -> None:
        orig_cfg, orig_reg, orig_pid, orig_kind, orig_base = (
            server.config, server.provider_registry, server.primary_provider_id,
            server.primary_provider_kind, server.primary_provider_base_url,
        )
        try:
            server.config = None
            server.provider_registry = None
            server.primary_provider_id = None
            server.primary_provider_kind = "ollama"
            server.primary_provider_base_url = None
            with patch("src.web.server.HealthChecker") as mock_cls:
                instance = mock_cls.return_value
                result = asyncio.run(server.health())
                instance.check_all.assert_not_called()
                status_body = asyncio.run(server.status())
        finally:
            server.config, server.provider_registry, server.primary_provider_id, \
                server.primary_provider_kind, server.primary_provider_base_url = \
                orig_cfg, orig_reg, orig_pid, orig_kind, orig_base
        body = json.loads(result.body)
        self.assertEqual(body["modules"]["provider_reachability"], "unconfigured")
        self.assertIs(body["modules"]["ollama"], False)
        self.assertIs(body["modules"]["local_provider"], False)
        self.assertEqual(body["status"], "unconfigured")
        self.assertEqual(status_body["provider_reachability"], "unconfigured")
        self.assertIs(status_body["ollama"], False)
        self.assertIs(status_body["local_provider"], False)


class WebProviderApiKeyTest(unittest.TestCase):
    """API キーは env 名のみで追跡され、各 health/status プローブで実行時解決される。

    キー値・env 名・URL は応答に一切含めない。keyless (Ollama / env未設定・空) では
    ``None`` を渡し、env 変更はリクエストごとに再読込される (キーローテーション)。
    """

    def _registry(self, provider_id: str = "local-openai") -> ProviderRegistry:
        registry = ProviderRegistry()
        registry.register(provider_id, FakeProvider("ok", model="m"), local=True)
        return registry

    def test_health_passes_resolved_key_for_configured_env(self) -> None:
        registry = self._registry()
        with patch.dict("os.environ", {"LOCAL_DUMMY_KEY": "sekrit-value"}, clear=False):
            result, instance = _run_health_with_registry(
                registry, None, provider_id="local-openai",
                provider_kind="openai_compatible",
                provider_base_url="http://localhost:8080/v1",
                provider_api_key_env="LOCAL_DUMMY_KEY",
                health_result={"status": "ok", "checks": {"local_provider": {"status": "ok"}}},
            )
        self.assertEqual(
            instance.check_all.call_args.kwargs.get("provider_api_key"), "sekrit-value"
        )
        body = json.loads(result.body)
        # env 名・キー値・URL は応答に一切含めない (キー値はグローバルにも保持しない)
        self.assertNotIn("sekrit-value", json.dumps(body))
        self.assertNotIn("LOCAL_DUMMY_KEY", json.dumps(body))
        self.assertNotIn("http://localhost:8080", json.dumps(body))

    def test_status_passes_resolved_key_for_configured_env(self) -> None:
        registry = self._registry()
        orig = (
            server.config, server.provider_registry, server.primary_provider_id,
            server.primary_provider_kind, server.primary_provider_base_url,
            server.primary_provider_api_key_env,
        )
        try:
            server.config = types.SimpleNamespace(model="cfg-model")
            server.provider_registry = registry
            server.primary_provider_id = "local-openai"
            server.primary_provider_kind = "openai_compatible"
            server.primary_provider_base_url = "http://localhost:8080/v1"
            server.primary_provider_api_key_env = "LOCAL_DUMMY_KEY"
            with patch.dict("os.environ", {"LOCAL_DUMMY_KEY": "sekrit-value"}, clear=False):
                with patch("src.web.server.HealthChecker") as mock_cls:
                    mock_cls.return_value.check_all.return_value = {
                        "status": "ok",
                        "checks": {"local_provider": {"status": "ok"}},
                    }
                    body = asyncio.run(server.status())
        finally:
            (
                server.config, server.provider_registry, server.primary_provider_id,
                server.primary_provider_kind, server.primary_provider_base_url,
                server.primary_provider_api_key_env,
            ) = orig
        self.assertEqual(
            mock_cls.return_value.check_all.call_args.kwargs.get("provider_api_key"),
            "sekrit-value",
        )
        self.assertNotIn("sekrit-value", json.dumps(body))
        self.assertNotIn("LOCAL_DUMMY_KEY", json.dumps(body))

    def test_unset_env_sends_none(self) -> None:
        registry = self._registry()
        result, instance = _run_health_with_registry(
            registry, None, provider_id="local-openai",
            provider_kind="openai_compatible",
            provider_base_url="http://localhost:8080/v1",
            provider_api_key_env="LOCAL_UNSET_KEY",
            health_result={"status": "ok", "checks": {"local_provider": {"status": "ok"}}},
        )
        self.assertIsNone(instance.check_all.call_args.kwargs.get("provider_api_key"))

    def test_blank_env_name_sends_none(self) -> None:
        registry = self._registry()
        result, instance = _run_health_with_registry(
            registry, None, provider_id="local-openai",
            provider_kind="openai_compatible",
            provider_base_url="http://localhost:8080/v1",
            provider_api_key_env="",
            health_result={"status": "ok", "checks": {"local_provider": {"status": "ok"}}},
        )
        self.assertIsNone(instance.check_all.call_args.kwargs.get("provider_api_key"))

    def test_ollama_kind_ignores_configured_env_name(self) -> None:
        registry = ProviderRegistry()
        registry.register("ollama", FakeProvider("ok", model="m"), local=True)
        with patch.dict("os.environ", {"LOCAL_DUMMY_KEY": "sekrit-value"}, clear=False):
            result, instance = _run_health_with_registry(
                registry, None, provider_id="ollama",
                provider_kind="ollama",
                provider_base_url="http://localhost:11434",
                provider_api_key_env="LOCAL_DUMMY_KEY",
                health_result={"status": "ok", "checks": {"ollama": {"status": "ok"}}},
            )
        self.assertIsNone(instance.check_all.call_args.kwargs.get("provider_api_key"))

    def test_key_rotation_is_read_per_request(self) -> None:
        registry = self._registry()
        result, first_instance = _run_health_with_registry(
            registry, None, provider_id="local-openai",
            provider_kind="openai_compatible",
            provider_base_url="http://localhost:8080/v1",
            provider_api_key_env="LOCAL_DUMMY_KEY",
            health_result={"status": "ok", "checks": {"local_provider": {"status": "ok"}}},
        )
        with patch.dict("os.environ", {"LOCAL_DUMMY_KEY": "old-key"}, clear=False):
            result, second_instance = _run_health_with_registry(
                registry, None, provider_id="local-openai",
                provider_kind="openai_compatible",
                provider_base_url="http://localhost:8080/v1",
                provider_api_key_env="LOCAL_DUMMY_KEY",
                health_result={"status": "ok", "checks": {"local_provider": {"status": "ok"}}},
            )
        with patch.dict("os.environ", {"LOCAL_DUMMY_KEY": "new-key"}, clear=False):
            result, third_instance = _run_health_with_registry(
                registry, None, provider_id="local-openai",
                provider_kind="openai_compatible",
                provider_base_url="http://localhost:8080/v1",
                provider_api_key_env="LOCAL_DUMMY_KEY",
                health_result={"status": "ok", "checks": {"local_provider": {"status": "ok"}}},
            )
        self.assertIsNone(first_instance.check_all.call_args.kwargs.get("provider_api_key"))
        self.assertEqual(
            second_instance.check_all.call_args.kwargs.get("provider_api_key"), "old-key"
        )
        self.assertEqual(
            third_instance.check_all.call_args.kwargs.get("provider_api_key"), "new-key"
        )

    def test_bearer_reaches_transport_for_configured_env(self) -> None:
        """configured dummy env のとき、実 HealthChecker が Bearer ヘッダを送る。

        ネットワークは偽の httpx クライアントで遮断し、送信された URL と
        ヘッダを記録する (offline)。
        """

        class _FakeResponse:
            status_code = 200

            def json(self):
                return {"data": [{"id": "m"}]}

        class _FakeClient:
            def __init__(self, timeout=None):
                self.requests = []

            def get(self, url, headers=None):
                self.requests.append((url, headers or {}))
                return _FakeResponse()

            def close(self):
                pass

        client = _FakeClient()
        fake_httpx = types.SimpleNamespace(
            Client=lambda timeout=None: client,
            TimeoutException=RuntimeError,
            RequestError=RuntimeError,
        )
        orig = (
            server.config, server.provider_registry, server.primary_provider_id,
            server.primary_provider_kind, server.primary_provider_base_url,
            server.primary_provider_api_key_env,
        )
        try:
            registry = self._registry()
            server.config = types.SimpleNamespace(model="cfg-model")
            server.provider_registry = registry
            server.primary_provider_id = "local-openai"
            server.primary_provider_kind = "openai_compatible"
            server.primary_provider_base_url = "http://localhost:8080/v1"
            server.primary_provider_api_key_env = "LOCAL_DUMMY_KEY"
            with patch.dict("os.environ", {"LOCAL_DUMMY_KEY": "sekrit-value"}, clear=False):
                with patch("src.service.healthcheck.httpx", fake_httpx):
                    response = asyncio.run(server.health())
        finally:
            (
                server.config, server.provider_registry, server.primary_provider_id,
                server.primary_provider_kind, server.primary_provider_base_url,
                server.primary_provider_api_key_env,
            ) = orig

        self.assertEqual(len(client.requests), 1)
        url, headers = client.requests[0]
        self.assertEqual(url, "http://localhost:8080/v1/models")
        self.assertEqual(headers.get("Authorization"), "Bearer sekrit-value")
        body_text = response.body.decode("utf-8", errors="replace")
        self.assertNotIn("sekrit-value", body_text)
        self.assertNotIn("LOCAL_DUMMY_KEY", body_text)
        self.assertNotIn("http://localhost:8080", body_text)

    def test_default_provider_api_key_env_selection(self) -> None:
        """lifespan の env 名選択: inventory default spec 優先、config へフォールバック。"""
        cfg = types.SimpleNamespace(local_api_key_env="LOCAL_CFG_KEY")
        spec = types.SimpleNamespace(api_key_env="LOCAL_SPEC_KEY")
        self.assertEqual(server._default_provider_api_key_env(cfg, spec), "LOCAL_SPEC_KEY")
        self.assertEqual(server._default_provider_api_key_env(cfg, None), "LOCAL_CFG_KEY")
        # spec が空なら spec 由来は None (config へはフォールバックしない)
        self.assertIsNone(
            server._default_provider_api_key_env(
                cfg, types.SimpleNamespace(api_key_env="")
            )
        )
        self.assertIsNone(
            server._default_provider_api_key_env(
                types.SimpleNamespace(local_api_key_env=""), None
            )
        )

    def test_inventory_env_name_selection_from_real_spec(self) -> None:
        """NodeInventory の default spec に設定された api_key_env が選ばれる。"""
        from src.assistant.nodes import NodeInventory

        inventory = NodeInventory.from_mapping({
            "default_provider_id": "local-openai",
            "nodes": [
                {
                    "node_id": "main",
                    "providers": [
                        {
                            "provider_id": "local-openai",
                            "base_url": "http://localhost:8080/v1",
                            "model": "m",
                            "provider_kind": "openai_compatible",
                            "api_key_env": "LOCAL_DUMMY_KEY",
                        },
                    ],
                }
            ],
        })
        spec = server._inventory_provider_spec(inventory, inventory.default_provider_id)
        self.assertIsNotNone(spec)
        self.assertEqual(
            server._default_provider_api_key_env(
                types.SimpleNamespace(local_api_key_env=""), spec
            ),
            "LOCAL_DUMMY_KEY",
        )


class ProviderStatsTest(unittest.TestCase):
    """Unit tests for AssistantService.provider_stats()."""

    def _make_service(self, response: str = "ok", *, model: str = "m"):
        registry = ProviderRegistry()
        provider = FakeProvider(response, model=model)
        registry.register("prov", provider, local=True)
        decision = RouteDecision(
            provider_id="prov",
            model=model,
            local=True,
            reason="test",
        )
        service = AssistantService(registry, _FixedRouter(decision))
        request = __import__("src.assistant.contracts", fromlist=["AssistantRequest"]).AssistantRequest(
            text="hi", conversation_id="c", channel="web"
        )
        return service, provider, request

    def test_last_success_at_none_before_generate(self) -> None:
        service, _, _ = self._make_service()
        stats = service.provider_stats()
        self.assertEqual(stats, {})

    def test_last_success_at_set_after_generate(self) -> None:
        service, _, request = self._make_service()
        messages = [{"role": "user", "content": "hi"}]
        service.generate(request, messages)
        stats = service.provider_stats()
        self.assertIn("prov", stats)
        self.assertIsNotNone(stats["prov"]["last_success_at"])
        self.assertIsInstance(stats["prov"]["last_success_at"], float)

    def test_last_success_at_not_set_after_failure(self) -> None:
        from src.llm.providers.fake import FakeProvider as FP

        class FailProvider(FP):
            def generate(self, messages, **options):
                self.calls.append({"kind": "generate", "messages": list(messages), "options": options})
                raise ProviderRequestError("fail", "generate", "test")

        registry = ProviderRegistry()
        fail_provider = FailProvider(model="failing")
        registry.register("fail", fail_provider, local=True)
        service = AssistantService(
            registry,
            _FixedRouter(
                RouteDecision(
                    provider_id="fail",
                    model="failing",
                    local=True,
                    reason="test",
                )
            ),
        )
        request = __import__("src.assistant.contracts", fromlist=["AssistantRequest"]).AssistantRequest(
            text="hi", conversation_id="c", channel="web"
        )
        messages = [{"role": "user", "content": "hi"}]
        with self.assertRaises(Exception):
            service.generate(request, messages)
        stats = service.provider_stats()
        self.assertNotIn("fail", stats)

    def test_last_success_at_updated_on_stream(self) -> None:
        from src.assistant.service import AssistantService
        from src.llm.providers.fake import FakeProvider as FP

        registry = ProviderRegistry()
        provider = FakeProvider("ok", model="stream-m", stream_chunks=("a", "b"))
        registry.register("sp", provider, local=True)
        service = AssistantService(
            registry,
            _FixedRouter(
                RouteDecision(
                    provider_id="sp",
                    model="stream-m",
                    local=True,
                    reason="test",
                )
            ),
        )
        request = __import__("src.assistant.contracts", fromlist=["AssistantRequest"]).AssistantRequest(
            text="hi", conversation_id="c", channel="web"
        )
        messages = [{"role": "user", "content": "hi"}]
        result = service.generate_stream(request, messages)
        list(result)
        stats = service.provider_stats()
        self.assertIn("sp", stats)
        self.assertIsNotNone(stats["sp"]["last_success_at"])


class WebStartupLogTest(unittest.TestCase):
    """起動時のLLM availabilityログが lifecycle を到達性として誤認させない。

    openai_compatible の ``is_available()`` は lifecycle-only で到達性を検証
    しないため、起動ログで ``"LLM OK"`` とは断言せず 'configured; reachability
    checked by health/generation' だけを出す。Ollama の probe-based 動作は維持。
    """

    def _capture(self, llm, *, provider_id, provider_kind, model="m"):
        with patch.object(server.logger, "info") as mock_info, patch.object(
            server.logger, "warning"
        ) as mock_warning:
            server._log_llm_startup_status(
                llm, provider_id=provider_id, provider_kind=provider_kind, model=model
            )
            return mock_info, mock_warning

    @staticmethod
    def _all_text(mock_info, mock_warning) -> str:
        parts = []
        for call in mock_info.call_args_list + mock_warning.call_args_list:
            args = call.args
            parts.append(args[0] % args[1:] if args else "")
        return " ".join(parts)

    def test_openai_startup_log_is_configured_not_llm_ok(self) -> None:
        info, warning = self._capture(
            FakeProvider("ok", model="m"),
            provider_id="local-openai",
            provider_kind="openai_compatible",
        )
        text = self._all_text(info, warning)
        self.assertIn("reachability checked by health/generation", text)
        self.assertNotIn("LLM OK", text)
        self.assertFalse(warning.called)

    def test_openai_startup_log_unavailable_lifecycle_warns_without_reachability_claim(self) -> None:
        info, warning = self._capture(
            FakeProvider("ok", model="m", available=False),
            provider_id="local-openai",
            provider_kind="openai_compatible",
        )
        text = self._all_text(info, warning)
        self.assertIn("利用可能な状態にありません", text)
        self.assertNotIn("LLM OK", text)
        # 到達性は未検証なので "接続できません" とは断言しない
        self.assertNotIn("接続できません", text)
        self.assertFalse(info.called)

    def test_openai_startup_log_no_url_or_key_exposure(self) -> None:
        info, warning = self._capture(
            FakeProvider("ok", model="m"),
            provider_id="local-openai",
            provider_kind="openai_compatible",
        )
        text = self._all_text(info, warning)
        self.assertNotIn("http://", text)
        self.assertNotIn("localhost:8080", text)
        self.assertNotIn("api_key", text)
        self.assertNotIn("Bearer", text)

    def test_ollama_startup_log_preserves_probe_based_result(self) -> None:
        info, warning = self._capture(
            FakeProvider("ok", model="m"),
            provider_id="ollama",
            provider_kind="ollama",
        )
        text = self._all_text(info, warning)
        self.assertIn("LLM OK", text)

    def test_ollama_startup_log_failure_preserved(self) -> None:
        info, warning = self._capture(
            FakeProvider("ok", model="m", available=False),
            provider_id="ollama",
            provider_kind="ollama",
        )
        text = self._all_text(info, warning)
        self.assertIn("接続できません", text)
        self.assertNotIn("LLM OK", text)

    def test_no_provider_startup_warning(self) -> None:
        info, warning = self._capture(None, provider_id=None, provider_kind="ollama")
        text = self._all_text(info, warning)
        self.assertIn("Providerが1つも登録されていません", text)
        self.assertNotIn("LLM OK", text)


class WebProvidersAvailabilityTest(unittest.TestCase):
    """providers[] の選択中Providerが is_available() でなく到達性で available を決める。"""

    @staticmethod
    def _selected_entry(*, health_result, provider_base_url="http://localhost:8080/v1"):
        registry = ProviderRegistry()
        registry.register(
            "local-openai", FakeProvider("ok", model="m", available=True), local=True
        )
        result, _ = _run_health_with_registry(
            registry,
            None,
            provider_id="local-openai",
            provider_kind="openai_compatible",
            provider_base_url=provider_base_url,
            health_result=health_result,
        )
        body = json.loads(result.body)
        return next(e for e in body["providers"] if e["provider_id"] == "local-openai")

    def test_selected_provider_available_from_ok_reachability(self) -> None:
        entry = self._selected_entry(
            health_result={"status": "ok", "checks": {"local_provider": {"status": "ok"}}}
        )
        self.assertIs(entry["available"], True)
        self.assertNotIn("model", entry)
        self.assertIsNone(entry["error"])

    def test_selected_provider_error_is_not_available(self) -> None:
        entry = self._selected_entry(
            health_result={"status": "error", "checks": {"local_provider": {"status": "error"}}}
        )
        self.assertIs(entry["available"], False)
        self.assertNotIn("model", entry)
        self.assertEqual(entry["error"], None)

    def test_selected_provider_unknown_is_not_available(self) -> None:
        entry = self._selected_entry(
            health_result={"status": "degraded", "checks": {"local_provider": {"status": "unknown"}}}
        )
        self.assertIs(entry["available"], False)
        self.assertNotIn("model", entry)
        self.assertIsNone(entry["error"])

    def test_selected_provider_unconfigured_is_not_available(self) -> None:
        # base_url 未解決 (None) → プローブせず unconfigured。
        registry = ProviderRegistry()
        registry.register(
            "local-openai", FakeProvider("ok", model="m", available=True), local=True
        )
        result, _ = _run_health_with_registry(
            registry,
            None,
            provider_id="local-openai",
            provider_kind="openai_compatible",
            provider_base_url=None,
            health_result=None,
        )
        body = json.loads(result.body)
        entry = next(e for e in body["providers"] if e["provider_id"] == "local-openai")
        self.assertIs(entry["available"], False)
        self.assertNotIn("model", entry)
        self.assertIsNone(entry["error"])

    def test_selected_provider_with_lifecycle_false_but_reachable_is_available(self) -> None:
        # is_available() が False でも到達性 ok なら available=True (到達性優先)
        registry = ProviderRegistry()
        registry.register(
            "local-openai", FakeProvider("ok", model="m", available=False), local=True
        )
        result, _ = _run_health_with_registry(
            registry,
            None,
            provider_id="local-openai",
            provider_kind="openai_compatible",
            provider_base_url="http://localhost:8080/v1",
            health_result={"status": "ok", "checks": {"local_provider": {"status": "ok"}}},
        )
        body = json.loads(result.body)
        entry = next(e for e in body["providers"] if e["provider_id"] == "local-openai")
        self.assertIs(entry["available"], True)

    def test_non_selected_provider_keeps_lifecycle_available_and_safe_schema(self) -> None:
        registry = ProviderRegistry()
        registry.register(
            "local-openai", FakeProvider("ok", model="m", available=True), local=True
        )
        registry.register(
            "secondary", FakeProvider("ok", model="s", available=False), local=True
        )
        result, _ = _run_health_with_registry(
            registry,
            None,
            provider_id="local-openai",
            provider_kind="openai_compatible",
            provider_base_url="http://localhost:8080/v1",
            health_result={"status": "ok", "checks": {"local_provider": {"status": "ok"}}},
        )
        body = json.loads(result.body)
        secondary = next(e for e in body["providers"] if e["provider_id"] == "secondary")
        # is_available() (lifecycle) 由来のまま
        self.assertIs(secondary["available"], False)
        self.assertEqual(
            set(secondary.keys()),
            {"provider_id", "provider_kind", "local", "available", "error"},
        )
        self.assertNotIn("model", secondary)
        self.assertIsNone(secondary["error"])

    def test_providers_list_no_url_or_key_exposure(self) -> None:
        entry = self._selected_entry(
            health_result={"status": "ok", "checks": {"local_provider": {"status": "ok"}}}
        )
        text = json.dumps(entry)
        self.assertNotIn("http://", text)
        self.assertNotIn("localhost:8080", text)
        self.assertNotIn("api_key", text)
        self.assertNotIn("Bearer", text)


if __name__ == "__main__":
    unittest.main()
