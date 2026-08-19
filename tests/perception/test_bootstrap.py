from __future__ import annotations

import unittest
from unittest import mock

from src.perception import ActivityRuntime
from src.perception import bootstrap
from src.perception.activity import ActivitySample


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


_ENV_TRUE = {
    "COMPANION_ACTIVITY_ENABLED": "true",
    "COMPANION_ACTIVITY_POLL_INTERVAL_SECONDS": "5",
    "COMPANION_ACTIVITY_IDLE_THRESHOLD_SECONDS": "300",
    "COMPANION_ACTIVITY_AWAY_THRESHOLD_SECONDS": "1800",
}


def _env(overrides: dict[str, str] | None = None) -> dict[str, str]:
    values = dict(_ENV_TRUE)
    if overrides:
        values.update(overrides)
    return values


class CreateActivityRuntimeFromEnvTest(unittest.TestCase):
    def test_disabled_when_env_unset(self) -> None:
        with mock.patch.object(
            bootstrap,
            "create_activity_source",
            side_effect=AssertionError("must not create source"),
        ):
            self.assertIsNone(bootstrap.create_activity_runtime_from_env({}))
        self.assertIsNone(bootstrap.create_activity_runtime_from_env({}))

    def test_disabled_when_env_not_true(self) -> None:
        for value in ("false", "FALSE", "0", "", "no"):
            with self.subTest(value=value), mock.patch.object(
                bootstrap,
                "create_activity_source",
                side_effect=AssertionError("must not create source"),
            ):
                self.assertIsNone(
                    bootstrap.create_activity_runtime_from_env(
                        _env({"COMPANION_ACTIVITY_ENABLED": value})
                    )
                )

    def test_invalid_numeric_settings_return_none(self) -> None:
        names = (
            "COMPANION_ACTIVITY_POLL_INTERVAL_SECONDS",
            "COMPANION_ACTIVITY_IDLE_THRESHOLD_SECONDS",
            "COMPANION_ACTIVITY_AWAY_THRESHOLD_SECONDS",
        )
        for name in names:
            for bad in ("-1", "0", "abc", "nan", "inf", "-inf"):
                with self.subTest(name=name, bad=bad), mock.patch.object(
                    bootstrap,
                    "create_activity_source",
                    side_effect=AssertionError("must not create source"),
                ):
                    self.assertIsNone(
                        bootstrap.create_activity_runtime_from_env(_env({name: bad}))
                    )

    def test_true_creates_source_and_returns_started_runtime(self) -> None:
        fake = _FakeSource()
        with mock.patch.object(bootstrap, "create_activity_source", return_value=fake):
            runtime = bootstrap.create_activity_runtime_from_env(_env())
        self.assertIsNotNone(runtime)
        self.assertIs(runtime._source, fake)
        self.assertTrue(runtime.is_running)
        runtime.stop(timeout=1.0)
        self.assertFalse(runtime.is_running)

    def test_numeric_settings_are_forwarded(self) -> None:
        fake = _FakeSource()
        with mock.patch.object(bootstrap, "create_activity_source", return_value=fake):
            runtime = bootstrap.create_activity_runtime_from_env(
                _env(
                    {
                        "COMPANION_ACTIVITY_POLL_INTERVAL_SECONDS": "12.5",
                        "COMPANION_ACTIVITY_IDLE_THRESHOLD_SECONDS": "60",
                        "COMPANION_ACTIVITY_AWAY_THRESHOLD_SECONDS": "120",
                    }
                )
            )
        self.assertIsNotNone(runtime)
        self.assertEqual(runtime._poll_interval, 12.5)
        self.assertEqual(runtime._collector.idle_threshold, 60)
        self.assertEqual(runtime._collector.away_threshold, 120)
        runtime.stop(timeout=1.0)

    def test_defaults_used_when_settings_unset(self) -> None:
        fake = _FakeSource()
        with mock.patch.object(bootstrap, "create_activity_source", return_value=fake):
            runtime = bootstrap.create_activity_runtime_from_env(
                _env(
                    {
                        "COMPANION_ACTIVITY_POLL_INTERVAL_SECONDS": "",
                        "COMPANION_ACTIVITY_IDLE_THRESHOLD_SECONDS": "",
                        "COMPANION_ACTIVITY_AWAY_THRESHOLD_SECONDS": "",
                    }
                )
            )
        self.assertIsNotNone(runtime)
        self.assertEqual(runtime._poll_interval, 5.0)
        self.assertEqual(runtime._collector.idle_threshold, 300)
        self.assertEqual(runtime._collector.away_threshold, 1800)
        runtime.stop(timeout=1.0)

    def test_source_failure_returns_none_and_logs_only_type(self) -> None:
        logger = mock.Mock()
        with mock.patch.object(
            bootstrap,
            "create_activity_source",
            side_effect=RuntimeError("secret process data"),
        ):
            runtime = bootstrap.create_activity_runtime_from_env(_env(), logger=logger)
        self.assertIsNone(runtime)
        self.assertTrue(logger.warning.called)
        logged = str([c.args for c in logger.warning.call_args_list])
        self.assertIn("RuntimeError", logged)
        self.assertNotIn("secret process data", logged)

    def test_runtime_construction_failure_returns_none_and_logs_only_type(self) -> None:
        logger = mock.Mock()
        fake = _FakeSource()
        with mock.patch.object(
            bootstrap, "create_activity_source", return_value=fake
        ), mock.patch.object(
            bootstrap, "ActivityRuntime", side_effect=ValueError("boom")
        ):
            runtime = bootstrap.create_activity_runtime_from_env(_env(), logger=logger)
        self.assertIsNone(runtime)
        logged = str([c.args for c in logger.warning.call_args_list])
        self.assertIn("ValueError", logged)
        self.assertNotIn("boom", logged)

    def test_uses_os_environ_when_env_is_none(self) -> None:
        fake = _FakeSource()
        with mock.patch.dict(
            bootstrap.os.environ, _env(), clear=False
        ), mock.patch.object(bootstrap, "create_activity_source", return_value=fake):
            runtime = bootstrap.create_activity_runtime_from_env()
        self.assertIsNotNone(runtime)
        runtime.stop(timeout=1.0)


class CompanionStatePayloadTest(unittest.TestCase):
    def test_none_runtime_returns_disabled(self) -> None:
        self.assertEqual(bootstrap.companion_state_payload(None), {"enabled": False})

    def test_last_state_none_serializes_state_as_none(self) -> None:
        runtime = ActivityRuntime(_FakeSource(), poll_interval=5.0)
        payload = bootstrap.companion_state_payload(runtime)
        self.assertEqual(
            set(payload),
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
        self.assertTrue(payload["enabled"])
        self.assertFalse(payload["running"])
        self.assertIsNone(payload["state"])

    def test_last_state_payload_contains_only_state_fields(self) -> None:
        runtime = ActivityRuntime(_FakeSource(), poll_interval=5.0)
        runtime.collect_once()
        payload = bootstrap.companion_state_payload(runtime)
        self.assertEqual(
            set(payload),
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
            set(payload["state"]),
            {
                "activity_mode",
                "present",
                "focused_since",
                "interruptible",
                "display_state",
                "updated_at",
            },
        )
        serialized = str(payload)
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

    def test_failure_counters_are_serialized(self) -> None:
        source = _FakeSource()
        runtime = ActivityRuntime(source, poll_interval=5.0)
        runtime.collect_once()
        payload = bootstrap.companion_state_payload(runtime)
        self.assertEqual(payload["failure_count"], 0)
        self.assertEqual(payload["consecutive_failures"], 0)
        self.assertIsNone(payload["last_error_type"])
        self.assertIsNotNone(payload["last_update_at"])


if __name__ == "__main__":
    unittest.main()
