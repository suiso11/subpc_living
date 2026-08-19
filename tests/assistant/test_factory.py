from contextlib import closing
from dataclasses import dataclass
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from src.assistant.contracts import AssistantRequest
from src.assistant.factory import build_local_service
from src.assistant.run_logger import SQLiteRunLogger
from src.llm.providers.fake import FakeProvider


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


if __name__ == "__main__":
    unittest.main()
