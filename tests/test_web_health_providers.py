"""Tests for /api/health providers key and AssistantService provider_stats."""

from __future__ import annotations

import asyncio
import json
import types
import unittest

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


class HealthProvidersTest(unittest.TestCase):
    """Verify /api/health response includes 'providers' with correct shape."""

    def _run_health_with_registry(
        self, registry: ProviderRegistry, service: AssistantService | None = None
    ):
        """Helper: temporarily set server globals, run health(), restore."""
        orig_reg = server.provider_registry
        orig_svc = server.assistant_service
        orig_cfg = server.config
        # Also mock config so ollama_base_url is available and the
        # modules check using provider_registry.get("ollama") does not crash.
        mock_config = types.SimpleNamespace(ollama_base_url="http://localhost:11434")
        try:
            server.provider_registry = registry
            server.assistant_service = service
            server.config = mock_config
            # If no "ollama" in registry, add one for the modules section
            if "ollama" not in registry:
                _p = FakeProvider("ok", model="ollama-model")
                registry.register("ollama", _p, local=True)
            return asyncio.run(server.health())
        finally:
            server.provider_registry = orig_reg
            server.assistant_service = orig_svc
            server.config = orig_cfg

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

        result = self._run_health_with_registry(registry, svc)
        body = json.loads(result.body)
        self.assertIn("providers", body)
        providers = body["providers"]
        self.assertIsInstance(providers, list)
        self.assertEqual(len(providers), 2)  # test-p + ollama
        # Find our custom entry
        entry = next(e for e in providers if e["provider_id"] == "test-p")
        expected_keys = {
            "provider_id",
            "model",
            "local",
            "available",
            "last_success_at",
        }
        self.assertEqual(set(entry.keys()), expected_keys)
        self.assertEqual(entry["provider_id"], "test-p")
        self.assertEqual(entry["model"], "test-model")
        self.assertTrue(entry["local"])
        self.assertIsInstance(entry["available"], bool)
        self.assertIsNone(entry["last_success_at"])

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
        result = self._run_health_with_registry(registry, service)
        body = json.loads(result.body)
        entry = next(e for e in body["providers"] if e["provider_id"] == "unavail")
        self.assertFalse(entry["available"])

    def test_health_providers_no_secrets_in_output(self) -> None:
        """Ensure provider entries contain only the 5 declared keys (no messages/prompts)."""
        registry = ProviderRegistry()
        provider = FakeProvider("secret stuff", model="m")
        registry.register("p", provider, local=True)
        service = AssistantService(
            registry,
            _FixedRouter(
                RouteDecision(provider_id="p", model="m", local=True, reason="t")
            ),
        )
        result = self._run_health_with_registry(registry, service)
        body = json.loads(result.body)
        entry = next(e for e in body["providers"] if e["provider_id"] == "p")
        self.assertEqual(set(entry.keys()), {
            "provider_id", "model", "local", "available", "last_success_at"
        })


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


if __name__ == "__main__":
    unittest.main()
