from contextlib import closing
import sqlite3
import tempfile
from pathlib import Path
import unittest

from src.assistant.run_logger import SQLiteRunLogger, _sanitize_error
from src.llm.routing.contracts import RouteDecision


class SQLiteRunLoggerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tempdir.name) / "nested" / "runs.db"
        self.logger = SQLiteRunLogger(self.db_path, clock=lambda: 123.5)
        self.route = RouteDecision(
            provider_id="local-strong",
            model="model-a",
            local=True,
            reason="default route",
            fallback_provider_ids=("local-fast",),
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def rows(self, table: str):
        with closing(sqlite3.connect(self.db_path)) as connection:
            return connection.execute(f"SELECT * FROM {table}").fetchall()

    def test_records_route_and_run_without_content_columns(self) -> None:
        self.logger.record_route(
            "request-1", channel="web", profile="chat_auto", decision=self.route
        )
        self.logger.record_run(
            "request-1",
            channel="web",
            profile="chat_auto",
            route=self.route,
            latency_ms=42,
            success=True,
            error=None,
        )

        with closing(sqlite3.connect(self.db_path)) as connection:
            route = connection.execute(
                "SELECT provider_id, model, local, reason, fallback_provider_ids "
                "FROM route_decisions"
            ).fetchone()
            run = connection.execute(
                "SELECT provider_id, latency_ms, success, error FROM model_runs"
            ).fetchone()
            columns = {
                row[1]
                for table in ("route_decisions", "model_runs")
                for row in connection.execute(f"PRAGMA table_info({table})")
            }

        self.assertEqual(
            route,
            ("local-strong", "model-a", 1, "default route", '["local-fast"]'),
        )
        self.assertEqual(run, ("local-strong", 42, 1, None))
        self.assertTrue(
            {"content", "text", "messages", "system_prompt", "conversation_id"}.isdisjoint(columns)
        )

    def test_duplicate_request_id_is_first_write_wins(self) -> None:
        changed = RouteDecision("other", "other-model", True, "changed")
        for route in (self.route, changed):
            self.logger.record_route(
                "same", channel="cli", profile="chat_auto", decision=route
            )
            self.logger.record_run(
                "same",
                channel="cli",
                profile="chat_auto",
                route=route,
                latency_ms=10 if route is self.route else 99,
                success=route is self.route,
                error=None if route is self.route else "changed",
            )

        with closing(sqlite3.connect(self.db_path)) as connection:
            route = connection.execute(
                "SELECT provider_id FROM route_decisions WHERE request_id='same'"
            ).fetchone()
            run = connection.execute(
                "SELECT provider_id, latency_ms, success FROM model_runs "
                "WHERE request_id='same'"
            ).fetchone()
        self.assertEqual(route, ("local-strong",))
        self.assertEqual(run, ("local-strong", 10, 1))

    def test_failed_run_can_be_recorded_without_route(self) -> None:
        self.logger.record_run(
            "no-route",
            channel="voice",
            profile="voice_fast",
            route=None,
            latency_ms=-1,
            success=False,
            error="NoRouteError",
        )

        with closing(sqlite3.connect(self.db_path)) as connection:
            row = connection.execute(
                "SELECT provider_id, local, latency_ms, success, error FROM model_runs"
            ).fetchone()
        self.assertEqual(row, (None, None, 0, 0, "NoRouteError"))

    def test_error_outside_allowed_format_is_redacted(self) -> None:
        self.logger.record_run(
            "bad-error",
            channel="voice",
            profile="voice_fast",
            route=None,
            latency_ms=1,
            success=False,
            error="ProviderRequestError: planned failure",
        )

        with closing(sqlite3.connect(self.db_path)) as connection:
            row = connection.execute(
                "SELECT error FROM model_runs WHERE request_id='bad-error'"
            ).fetchone()
        self.assertEqual(row, ("redacted_error",))

    def test_error_is_truncated_to_max_length(self) -> None:
        error = "X" * 200
        self.assertEqual(len(_sanitize_error(error)), 64)

    def test_allowed_error_code_is_preserved(self) -> None:
        self.assertEqual(_sanitize_error("AssistantGenerationError"), "AssistantGenerationError")
        self.assertIsNone(_sanitize_error(None))

    def test_journal_mode_is_wal(self) -> None:
        with closing(sqlite3.connect(self.db_path)) as connection:
            mode = connection.execute("PRAGMA journal_mode").fetchone()
        self.assertEqual(mode[0], "wal")

    def test_wal_not_supported_raises_operational_error(self) -> None:
        with self.assertRaises(sqlite3.OperationalError):
            SQLiteRunLogger(":memory:")


if __name__ == "__main__":
    unittest.main()
