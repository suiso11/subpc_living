from __future__ import annotations

import unittest
from unittest import mock

from src.perception import ActivityRuntime
from src.perception.activity import ActivitySample
from src.web import server

_COMPANION_ENV = {
    "COMPANION_ACTIVITY_ENABLED": "true",
    "COMPANION_ACTIVITY_POLL_INTERVAL_SECONDS": "5",
    "COMPANION_ACTIVITY_IDLE_THRESHOLD_SECONDS": "300",
    "COMPANION_ACTIVITY_AWAY_THRESHOLD_SECONDS": "1800",
}


class _FakeSource:
    def __init__(self) -> None:
        self.calls = 0

    def sample(self) -> ActivitySample:
        self.calls += 1
        return ActivitySample(
            timestamp=float(self.calls),
            idle_seconds=0.0,
            app_category="work",
        )


def _patch_env(overrides: dict[str, str] | None = None) -> dict:
    values = dict(_COMPANION_ENV)
    if overrides:
        values.update(overrides)
    return mock.patch.dict(
        server.os.environ,
        values,
        clear=True,
    )


class CompanionStartupHelperTest(unittest.TestCase):
    def setUp(self) -> None:
        self.original_runtime = server.activity_runtime

    def tearDown(self) -> None:
        if server.activity_runtime is not None:
            server.activity_runtime.stop()
        server.activity_runtime = self.original_runtime

    def test_disabled_when_env_is_not_true(self) -> None:
        with _patch_env({"COMPANION_ACTIVITY_ENABLED": "false"}), mock.patch.object(
            server, "create_activity_source", side_effect=AssertionError("must not create source")
        ):
            self.assertIsNone(server._start_companion_activity_runtime())
        self.assertIsNone(server.activity_runtime)

    def test_disabled_when_env_is_unset(self) -> None:
        with _patch_env({"COMPANION_ACTIVITY_ENABLED": ""}), mock.patch.object(
            server, "create_activity_source", side_effect=AssertionError("must not create source")
        ):
            self.assertIsNone(server._start_companion_activity_runtime())
        self.assertIsNone(server.activity_runtime)

    def test_disabled_by_default_with_no_activity_env(self) -> None:
        with mock.patch.dict(
            server.os.environ,
            {},
            clear=True,
        ), mock.patch(
            "src.perception.bootstrap.create_activity_source",
            side_effect=AssertionError("must not create source"),
        ):
            self.assertIsNone(server._start_companion_activity_runtime())
        self.assertIsNone(server.activity_runtime)

    def test_canonical_sensor_activity_true_enables(self) -> None:
        fake = _FakeSource()
        with mock.patch.dict(
            server.os.environ,
            {
                "SENSOR_ACTIVITY_ENABLED": "true",
                "COMPANION_ACTIVITY_POLL_INTERVAL_SECONDS": "5",
                "COMPANION_ACTIVITY_IDLE_THRESHOLD_SECONDS": "300",
                "COMPANION_ACTIVITY_AWAY_THRESHOLD_SECONDS": "1800",
            },
            clear=True,
        ), mock.patch("src.perception.bootstrap.create_activity_source", return_value=fake):
            runtime = server._start_companion_activity_runtime()
        self.assertIsNotNone(runtime)
        self.assertIs(server.activity_runtime, runtime)
        self.assertTrue(runtime.is_running)

    def test_canonical_false_overrides_legacy_true(self) -> None:
        with _patch_env(
            {
                "COMPANION_ACTIVITY_ENABLED": "true",
                "SENSOR_ACTIVITY_ENABLED": "false",
            }
        ), mock.patch(
            "src.perception.bootstrap.create_activity_source",
            side_effect=AssertionError("must not create source"),
        ):
            self.assertIsNone(server._start_companion_activity_runtime())
        self.assertIsNone(server.activity_runtime)

    def test_legacy_activity_true_alone_still_enables(self) -> None:
        fake = _FakeSource()
        with mock.patch.dict(
            server.os.environ,
            {"COMPANION_ACTIVITY_ENABLED": "true"},
            clear=True,
        ), mock.patch("src.perception.bootstrap.create_activity_source", return_value=fake):
            runtime = server._start_companion_activity_runtime()
        self.assertIsNotNone(runtime)
        self.assertIs(server.activity_runtime, runtime)
        self.assertTrue(runtime.is_running)

    def test_invalid_canonical_activity_values_fail_closed(self) -> None:
        for value in ("", "0", "1", "yes", "on", "no", "false"):
            with self.subTest(value=value), mock.patch.dict(
                server.os.environ,
                {
                    "SENSOR_ACTIVITY_ENABLED": value,
                    "COMPANION_ACTIVITY_ENABLED": "true",
                },
                clear=True,
            ), mock.patch(
                "src.perception.bootstrap.create_activity_source",
                side_effect=AssertionError("must not create source"),
            ):
                self.assertIsNone(server._start_companion_activity_runtime())
            self.assertIsNone(server.activity_runtime)

    def test_enabled_creates_source_and_starts_runtime(self) -> None:
        fake = _FakeSource()
        with _patch_env(), mock.patch("src.perception.bootstrap.create_activity_source", return_value=fake):
            runtime = server._start_companion_activity_runtime()
        self.assertIsNotNone(runtime)
        self.assertIs(server.activity_runtime, runtime)
        self.assertTrue(runtime.is_running)
        self.assertEqual(runtime._source, fake)

    def test_valid_numeric_settings_are_used(self) -> None:
        fake = _FakeSource()
        with _patch_env(
            {
                "COMPANION_ACTIVITY_POLL_INTERVAL_SECONDS": "12.5",
                "COMPANION_ACTIVITY_IDLE_THRESHOLD_SECONDS": "60",
                "COMPANION_ACTIVITY_AWAY_THRESHOLD_SECONDS": "120",
            }
        ), mock.patch("src.perception.bootstrap.create_activity_source", return_value=fake):
            runtime = server._start_companion_activity_runtime()
        self.assertIsNotNone(runtime)
        self.assertEqual(runtime._poll_interval, 12.5)
        self.assertEqual(runtime._collector.idle_threshold, 60)
        self.assertEqual(runtime._collector.away_threshold, 120)

    def test_defaults_used_when_settings_unset(self) -> None:
        fake = _FakeSource()
        with _patch_env(
            {
                "COMPANION_ACTIVITY_POLL_INTERVAL_SECONDS": "",
                "COMPANION_ACTIVITY_IDLE_THRESHOLD_SECONDS": "",
                "COMPANION_ACTIVITY_AWAY_THRESHOLD_SECONDS": "",
            }
        ), mock.patch("src.perception.bootstrap.create_activity_source", return_value=fake):
            runtime = server._start_companion_activity_runtime()
        self.assertIsNotNone(runtime)
        self.assertEqual(runtime._poll_interval, 5.0)
        self.assertEqual(runtime._collector.idle_threshold, 300)
        self.assertEqual(runtime._collector.away_threshold, 1800)

    def test_invalid_setting_disables_only_companion(self) -> None:
        for bad in ("abc", "0", "-5", "nan", "inf"):
            with self.subTest(bad=bad):
                with _patch_env({"COMPANION_ACTIVITY_POLL_INTERVAL_SECONDS": bad}), (
                    mock.patch(
                        "src.perception.bootstrap.create_activity_source",
                        side_effect=AssertionError("must not create source"),
                    )
                ):
                    self.assertIsNone(server._start_companion_activity_runtime())
                self.assertIsNone(server.activity_runtime)

    def test_invalid_idle_or_away_setting_disables_companion(self) -> None:
        for name in (
            "COMPANION_ACTIVITY_IDLE_THRESHOLD_SECONDS",
            "COMPANION_ACTIVITY_AWAY_THRESHOLD_SECONDS",
        ):
            with self.subTest(name=name):
                with _patch_env({name: "oops"}), mock.patch(
                    "src.perception.bootstrap.create_activity_source",
                    side_effect=AssertionError("must not create source"),
                ):
                    self.assertIsNone(server._start_companion_activity_runtime())
                self.assertIsNone(server.activity_runtime)

    def test_startup_failure_logs_only_exception_type_and_continues(self) -> None:
        with _patch_env(), mock.patch(
            "src.perception.bootstrap.create_activity_source", side_effect=RuntimeError("secret process data")
        ), mock.patch.object(server.logger, "warning") as warn:
            runtime = server._start_companion_activity_runtime()
        self.assertIsNone(runtime)
        self.assertIsNone(server.activity_runtime)
        self.assertTrue(warn.called)
        logged = str([c.args for c in warn.call_args_list])
        self.assertIn("RuntimeError", logged)
        self.assertNotIn("secret process data", logged)

    def test_activity_runtime_construction_failure_also_disables(self) -> None:
        fake = _FakeSource()
        with _patch_env(), mock.patch("src.perception.bootstrap.create_activity_source", return_value=fake), (
            mock.patch("src.perception.bootstrap.ActivityRuntime", side_effect=ValueError("boom"))
        ), mock.patch.object(server.logger, "warning") as warn:
            runtime = server._start_companion_activity_runtime()
        self.assertIsNone(runtime)
        logged = str([c.args for c in warn.call_args_list])
        self.assertIn("ValueError", logged)
        self.assertNotIn("boom", logged)


class CompanionStateEndpointTest(unittest.TestCase):
    def setUp(self) -> None:
        from starlette.testclient import TestClient

        self.original_runtime = server.activity_runtime
        server.activity_runtime = None
        self.client = TestClient(server.app)

    def tearDown(self) -> None:
        if server.activity_runtime is not None:
            server.activity_runtime.stop()
        server.activity_runtime = self.original_runtime

    def test_returns_enabled_false_when_disabled(self) -> None:
        r = self.client.get("/api/companion/state")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"enabled": False})

    def test_returns_counters_and_serialized_state_when_enabled(self) -> None:
        runtime = ActivityRuntime(_FakeSource(), poll_interval=5.0)
        runtime.collect_once()
        server.activity_runtime = runtime
        r = self.client.get("/api/companion/state")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertTrue(body["enabled"])
        self.assertFalse(body["running"])
        self.assertEqual(body["failure_count"], 0)
        self.assertEqual(body["consecutive_failures"], 0)
        self.assertIsNone(body["last_error_type"])
        self.assertIsNotNone(body["last_update_at"])
        self.assertEqual(
            body["state"],
            {
                "activity_mode": "focused",
                "present": True,
                "focused_since": 1.0,
                "interruptible": False,
                "display_state": "focused",
                "updated_at": 1.0,
            },
        )

    def test_state_never_exposes_raw_or_sensitive_fields(self) -> None:
        runtime = ActivityRuntime(_FakeSource(), poll_interval=5.0)
        runtime.collect_once()
        server.activity_runtime = runtime
        body = self.client.get("/api/companion/state").json()
        self.assertEqual(
            set(body),
            {
                "enabled",
                "running",
                "last_update_at",
                "failure_count",
                "consecutive_failures",
                "last_error_type",
                "state",
            },
        )
        self.assertEqual(
            set(body["state"]),
            {
                "activity_mode",
                "present",
                "focused_since",
                "interruptible",
                "display_state",
                "updated_at",
            },
        )
        serialized = str(body)
        for forbidden in (
            "process",
            "pid",
            "app_category",
            "window_title",
            "title",
            "path",
            "sample",
            "event",
            "error text",
            "raw",
        ):
            self.assertNotIn(forbidden, serialized)

    def test_endpoint_is_unauthenticated_and_read_only(self) -> None:
        self.assertEqual(self.client.get("/api/companion/state").status_code, 200)
        self.assertEqual(self.client.post("/api/companion/state").status_code, 405)


if __name__ == "__main__":
    unittest.main()