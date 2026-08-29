from contextlib import closing
from dataclasses import dataclass
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import httpx

from src.assistant.contracts import AssistantRequest
from src.assistant.factory import (
    build_assistant_service,
    build_local_provider,
    build_local_service,
)
from src.assistant.run_logger import SQLiteRunLogger
from src.chat.config import ChatConfig
from src.llm.cloud_config import CloudConfig
from src.llm.providers.fake import FakeProvider
from src.llm.providers.local_openai import LocalOpenAICompatibleProvider
from src.llm.providers.ollama import OllamaProvider
from src.llm.registry import UnknownProviderError


class _ChatOnlyTransport(httpx.BaseTransport):
    """Chat-completions only: serves /chat/completions and records requests.

    A ``GET /models`` is intentionally *not* served; any request to it would
    hang unless explicitly rejected. Used to prove generation performs no
    /models preflight.
    """

    def __init__(self, json_data: dict) -> None:
        self.json_data = json_data
        self.requests: list[httpx.Request] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path.endswith("/models"):
            raise AssertionError("/models must not be probed during chat")
        return httpx.Response(200, json=self.json_data, request=request)


@dataclass
class FactoryConfig:
    ollama_base_url: str = "http://unused.invalid"
    model: str = "unused-model"
    temperature: float = 0.23
    top_p: float = 0.81
    top_k: int = 17
    repeat_penalty: float = 1.37
    num_ctx: int = 3456
    num_predict: int | None = 123


class BuildLocalServiceTest(unittest.TestCase):
    @staticmethod
    def request() -> AssistantRequest:
        return AssistantRequest(
            text="質問",
            conversation_id="factory-test",
            channel="internal",
        )

    def test_injected_provider_receives_configured_options(self) -> None:
        config = FactoryConfig()
        provider = FakeProvider(response="応答")

        service, registry = build_local_service(config, provider=provider)
        response = service.generate(
            self.request(), [{"role": "user", "content": "質問"}]
        )

        self.assertEqual(response.text, "応答")
        self.assertEqual(
            provider.calls[0]["options"],
            {
                "temperature": 0.23,
                "top_p": 0.81,
                "top_k": 17,
                "repeat_penalty": 1.37,
                "num_ctx": 3456,
                "num_predict": 123,
            },
        )
        entry = registry.get("ollama")
        self.assertIs(entry.provider, provider)
        self.assertTrue(entry.local)

    def test_custom_provider_id_is_registered_and_used_as_default(self) -> None:
        provider = FakeProvider(response="custom response")

        service, registry = build_local_service(
            FactoryConfig(), provider=provider, provider_id="local-custom"
        )
        response = service.generate(
            self.request(), [{"role": "user", "content": "質問"}]
        )

        self.assertIs(registry.get("local-custom").provider, provider)
        self.assertEqual(response.text, "custom response")
        self.assertEqual(response.route.provider_id, "local-custom")
        self.assertEqual(response.route.reason, "default route")

    def test_omitted_run_logger_uses_env_db_path(self) -> None:
        provider = FakeProvider(response="応答")
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "nested" / "runs.db"
            with patch.dict(os.environ, {"ASSISTANT_RUN_LOG_DB": str(db_path)}):
                service, _ = build_local_service(
                    FactoryConfig(), provider=provider
                )
                response = service.generate(
                    self.request(), [{"role": "user", "content": "質問"}]
                )

            self.assertEqual(response.text, "応答")
            with closing(sqlite3.connect(db_path)) as connection:
                run = connection.execute(
                    "SELECT success, provider_id FROM model_runs"
                ).fetchone()
                route = connection.execute(
                    "SELECT provider_id FROM route_decisions"
                ).fetchone()
        self.assertEqual(run, (1, "ollama"))
        self.assertEqual(route, ("ollama",))

    def test_explicit_none_run_logger_creates_no_db(self) -> None:
        provider = FakeProvider(response="応答")
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "unused.db"
            with patch.dict(os.environ, {"ASSISTANT_RUN_LOG_DB": str(db_path)}):
                service, _ = build_local_service(
                    FactoryConfig(), provider=provider, run_logger=None
                )
                response = service.generate(
                    self.request(), [{"role": "user", "content": "質問"}]
                )

            self.assertEqual(response.text, "応答")
            self.assertIsNone(service._run_logger)
            self.assertFalse(db_path.exists())

    def test_explicit_run_logger_is_used_as_is(self) -> None:
        provider = FakeProvider(response="応答")
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "runs.db"
            logger = SQLiteRunLogger(db_path)
            service, _ = build_local_service(
                FactoryConfig(), provider=provider, run_logger=logger
            )
            service.generate(self.request(), [{"role": "user", "content": "質問"}])

            with closing(sqlite3.connect(db_path)) as connection:
                run = connection.execute(
                    "SELECT success FROM model_runs"
                ).fetchone()
        self.assertEqual(run, (1,))

    def test_default_run_logger_init_failure_falls_back_to_none(self) -> None:
        provider = FakeProvider(response="応答")
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
                                service, _ = build_local_service(
                                    FactoryConfig(), provider=provider
                                )
                                response = service.generate(
                                    self.request(),
                                    [{"role": "user", "content": "質問"}],
                                )

                self.assertEqual(response.text, "応答")
                self.assertIsNone(service._run_logger)
                self.assertFalse(db_path.exists())
                self.assertTrue(
                    any("run logger" in message for message in logs.output)
                )


class BuildLocalProviderTest(unittest.TestCase):
    """公開API build_local_provider: Service/Registry/loggerを経ずbackendを解決する"""

    def test_default_config_resolves_ollama_provider(self) -> None:
        config = ChatConfig(model="qwen2.5:7b-instruct-q4_K_M")
        provider_id, provider = build_local_provider(config)
        self.assertEqual(provider_id, "ollama")
        self.assertIsInstance(provider, OllamaProvider)
        self.assertEqual(provider.provider_id, "ollama")
        self.assertEqual(provider.model, "qwen2.5:7b-instruct-q4_K_M")

    def test_openai_compatible_resolves_local_openai_provider(self) -> None:
        config = ChatConfig(
            model="local-model",
            local_provider_kind="openai_compatible",
        )
        provider_id, provider = build_local_provider(config)
        self.assertEqual(provider_id, "local-openai")
        self.assertIsInstance(provider, LocalOpenAICompatibleProvider)
        self.assertEqual(provider.provider_id, "local-openai")
        self.assertEqual(provider.model, "local-model")

    def test_injected_provider_returned_as_is_with_default_id(self) -> None:
        provider = FakeProvider(response="injected")
        provider_id, resolved = build_local_provider(FactoryConfig(), provider=provider)
        self.assertEqual(provider_id, "ollama")
        self.assertIs(resolved, provider)

    def test_injected_provider_with_custom_id(self) -> None:
        provider = FakeProvider(response="custom")
        provider_id, resolved = build_local_provider(
            FactoryConfig(), provider=provider, provider_id="manual-job"
        )
        self.assertEqual(provider_id, "manual-job")
        self.assertIs(resolved, provider)

    def test_custom_ollama_id_from_config(self) -> None:
        config = ChatConfig(local_provider_id="custom-ollama")
        provider_id, provider = build_local_provider(config)
        self.assertEqual(provider_id, "custom-ollama")
        self.assertIsInstance(provider, OllamaProvider)
        self.assertEqual(provider.provider_id, "custom-ollama")

    def test_custom_openai_compatible_id_from_config(self) -> None:
        config = ChatConfig(
            local_provider_kind="openai_compatible",
            local_provider_id="llama-server",
        )
        provider_id, provider = build_local_provider(config)
        self.assertEqual(provider_id, "llama-server")
        self.assertIsInstance(provider, LocalOpenAICompatibleProvider)
        self.assertEqual(provider.provider_id, "llama-server")

    def test_openai_compatible_url_validation_rejects_non_loopback(self) -> None:
        config = ChatConfig(
            local_provider_kind="openai_compatible",
            local_base_url="http://192.168.1.5:8080/v1",
        )
        with self.assertRaises(ValueError):
            build_local_provider(config)

    def test_unknown_kind_raises(self) -> None:
        config = ChatConfig(local_provider_kind="banana")
        with self.assertRaises(ValueError):
            build_local_provider(config)

    def test_blank_kind_normalizes_to_ollama(self) -> None:
        for blank in ("", "   ", "\t\n "):
            with self.subTest(kind=blank):
                config = ChatConfig(local_provider_kind=blank)
                provider_id, provider = build_local_provider(config)
                self.assertEqual(provider_id, "ollama")
                self.assertIsInstance(provider, OllamaProvider)
                self.assertEqual(provider.provider_id, "ollama")

    def test_no_logger_or_database_side_effects(self) -> None:
        provider = FakeProvider(response="x")
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "runs.db"
            with patch.dict(os.environ, {"ASSISTANT_RUN_LOG_DB": str(db_path)}):
                provider_id, resolved = build_local_provider(
                    FactoryConfig(), provider=provider
                )
                build_local_provider(ChatConfig())
            self.assertEqual(provider_id, "ollama")
            self.assertIs(resolved, provider)
            self.assertFalse(db_path.exists())


class BuildLocalServiceFromConfigTest(unittest.TestCase):
    """P0-2: config 駆動のローカルbackend選択 (provider注入なし)"""

    def test_default_config_builds_ollama_provider(self):
        config = ChatConfig(model="qwen2.5:7b-instruct-q4_K_M")
        service, registry = build_local_service(config, run_logger=None)
        entry = registry.get("ollama")
        self.assertIsInstance(entry.provider, OllamaProvider)
        self.assertTrue(entry.local)
        self.assertNotIn("local-openai", registry)

    def test_openai_compatible_builds_local_openai_provider(self):
        config = ChatConfig(
            model="local-model",
            local_provider_kind="openai_compatible",
        )
        service, registry = build_local_service(config, run_logger=None)
        entry = registry.get("local-openai")
        self.assertIsInstance(entry.provider, LocalOpenAICompatibleProvider)
        self.assertTrue(entry.local)
        self.assertEqual(entry.provider.model, "local-model")
        # Ollama は登録されない
        with self.assertRaises(UnknownProviderError):
            registry.get("ollama")

    def test_openai_compatible_honors_local_base_url(self):
        config = ChatConfig(
            model="m",
            local_provider_kind="openai_compatible",
            local_base_url="http://localhost:4321/v1",
        )
        service, registry = build_local_service(config, run_logger=None)
        provider = registry.get("local-openai").provider
        self.assertEqual(str(provider._client.base_url), "http://localhost:4321/v1/")

    def test_custom_local_provider_id_is_registered(self):
        config = ChatConfig(
            local_provider_kind="openai_compatible",
            local_provider_id="llama-server",
        )
        service, registry = build_local_service(config, run_logger=None)
        self.assertIn("llama-server", registry)
        self.assertIsInstance(
            registry.get("llama-server").provider, LocalOpenAICompatibleProvider
        )

    def test_custom_ollama_provider_id_agrees_with_registry(self):
        config = ChatConfig(local_provider_id="custom-ollama")
        service, registry = build_local_service(config, run_logger=None)
        self.assertIn("custom-ollama", registry)
        entry = registry.get("custom-ollama")
        self.assertIsInstance(entry.provider, OllamaProvider)
        # エラー・ログ用 provider_id が Registry キーと一致する
        self.assertEqual(entry.provider.provider_id, "custom-ollama")
        self.assertTrue(entry.local)

    def test_chat_only_local_openai_generates_without_models_preflight(self):
        transport = _ChatOnlyTransport(
            json_data={
                "choices": [{"message": {"content": "local reply"}}],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 2,
                    "total_tokens": 3,
                },
            }
        )
        client = httpx.Client(
            transport=transport, base_url="http://localhost:8080/v1"
        )
        config = ChatConfig(
            model="local-model",
            local_provider_kind="openai_compatible",
            local_base_url="http://localhost:8080/v1",
        )
        with patch("src.llm.providers.local_openai.httpx.Client", return_value=client):
            service, registry = build_local_service(config, run_logger=None)
            response = service.generate(
                AssistantRequest(
                    text="質問",
                    conversation_id="factory-test",
                    channel="internal",
                ),
                [{"role": "user", "content": "質問"}],
            )

        self.assertEqual(response.text, "local reply")
        self.assertEqual(response.route.provider_id, "local-openai")
        self.assertEqual(
            registry.get("local-openai").provider.provider_id, "local-openai"
        )
        paths = [request.url.path for request in transport.requests]
        self.assertEqual(paths, ["/v1/chat/completions"])

    def test_non_loopback_local_base_url_rejected(self):
        config = ChatConfig(
            local_provider_kind="openai_compatible",
            local_base_url="http://192.168.1.5:8080/v1",
        )
        with self.assertRaises(ValueError):
            build_local_service(config, run_logger=None)
        with self.assertRaises(ValueError):
            build_assistant_service(config, run_logger=None)

    def test_unknown_kind_raises(self):
        config = ChatConfig(local_provider_kind="banana")
        with self.assertRaises(ValueError):
            build_local_service(config, run_logger=None)

    def test_blank_kind_builds_default_ollama_service(self):
        for blank in ("", "  "):
            with self.subTest(kind=blank):
                config = ChatConfig(local_provider_kind=blank)
                service, registry = build_local_service(config, run_logger=None)
                entry = registry.get("ollama")
                self.assertIsInstance(entry.provider, OllamaProvider)
                self.assertTrue(entry.local)
                self.assertNotIn("local-openai", registry)

    def test_resolved_api_key_passed_to_provider_only(self):
        config = ChatConfig(
            local_provider_kind="openai_compatible",
            local_api_key_env="LOCAL_FACTORY_TEST_TOKEN",
        )
        with patch.dict(os.environ, {"LOCAL_FACTORY_TEST_TOKEN": "dummy-token"}):
            service, registry = build_local_service(config, run_logger=None)
        self.assertEqual(
            registry.get("local-openai").provider._api_key, "dummy-token"
        )

    def test_keyless_when_env_unset(self):
        config = ChatConfig(
            local_provider_kind="openai_compatible",
            local_api_key_env="LOCAL_FACTORY_TEST_TOKEN_MISSING",
        )
        with patch.dict(os.environ):
            os.environ.pop("LOCAL_FACTORY_TEST_TOKEN_MISSING", None)
            service, registry = build_local_service(config, run_logger=None)
        self.assertEqual(registry.get("local-openai").provider._api_key, "")

    def test_injected_provider_ignores_config_kind(self):
        provider = FakeProvider(response="injected")
        config = ChatConfig(local_provider_kind="openai_compatible")
        service, registry = build_local_service(
            config, provider=provider, run_logger=None
        )
        entry = registry.get("ollama")
        self.assertIs(entry.provider, provider)
        self.assertTrue(entry.local)

    def test_injected_provider_with_explicit_provider_id(self):
        provider = FakeProvider(response="x")
        service, registry = build_local_service(
            ChatConfig(),
            provider=provider,
            provider_id="local-custom",
            run_logger=None,
        )
        self.assertIs(registry.get("local-custom").provider, provider)

    def test_cloud_coexistence_with_local_openai(self):
        config = ChatConfig(local_provider_kind="openai_compatible")
        cloud = CloudConfig(enabled=True, model="cloud-m", provider_id="cloud")
        service, registry, bridge = build_assistant_service(
            config, cloud_config=cloud, run_logger=None
        )
        self.assertIsNotNone(bridge)
        local_entry = registry.get("local-openai")
        self.assertIsInstance(local_entry.provider, LocalOpenAICompatibleProvider)
        self.assertTrue(local_entry.local)
        cloud_entry = registry.get("cloud")
        self.assertFalse(cloud_entry.local)
        # ローカルproviderはcloud経路へ混ざらず、defaultはローカルbackend
        self.assertEqual(service._router.default_provider_id, "local-openai")

    def test_build_assistant_service_default_local_ollama(self):
        service, registry, bridge = build_assistant_service(
            ChatConfig(), run_logger=None
        )
        self.assertIsNone(bridge)
        entry = registry.get("ollama")
        self.assertIsInstance(entry.provider, OllamaProvider)
        self.assertTrue(entry.local)


if __name__ == "__main__":
    unittest.main()
