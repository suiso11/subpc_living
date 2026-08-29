from __future__ import annotations

import os
import unittest
from dataclasses import FrozenInstanceError
from unittest import mock

from src.perception import policy

ALL_SENSORS = (
    "camera",
    "screen_capture",
    "screen_ingest",
    "activity",
    "monitor",
    "microphone",
    "process_details",
)

CANONICAL = policy.CANONICAL_ENV_NAMES
LEGACY = policy.LEGACY_ENV_ALIASES


class ParseOptInTest(unittest.TestCase):
    def test_only_explicit_true_enables(self) -> None:
        for value in ("true", "TRUE", " True ", "tRuE"):
            with self.subTest(value=value):
                self.assertTrue(policy.parse_opt_in(value))

    def test_everything_else_is_false(self) -> None:
        for value in (
            None,
            1,
            True,
            "",
            "false",
            "FALSE",
            "0",
            "1",
            "yes",
            "on",
            "no",
            " enabled ",
            "tru",
        ):
            with self.subTest(value=value):
                self.assertFalse(policy.parse_opt_in(value))


class ResolveDefaultsTest(unittest.TestCase):
    def test_empty_env_defaults_all_off(self) -> None:
        resolved = policy.resolve_sensor_policy({})
        for sensor in ALL_SENSORS:
            self.assertFalse(resolved.is_enabled(sensor))
        self.assertEqual(resolved.enabled_sensor_ids(), ())

    def test_canonical_true_enables_each_sensor(self) -> None:
        for sensor in ALL_SENSORS:
            with self.subTest(sensor=sensor):
                resolved = policy.resolve_sensor_policy({CANONICAL[sensor]: "true"})
                self.assertTrue(resolved.is_enabled(sensor))
                self.assertEqual(resolved.enabled_sensor_ids(), (sensor,))

    def test_canonical_false_keeps_sensor_off(self) -> None:
        for sensor in ALL_SENSORS:
            with self.subTest(sensor=sensor):
                resolved = policy.resolve_sensor_policy({CANONICAL[sensor]: "false"})
                self.assertFalse(resolved.is_enabled(sensor))
                self.assertEqual(resolved.enabled_sensor_ids(), ())

    def test_other_sensors_stay_off_when_one_enabled(self) -> None:
        resolved = policy.resolve_sensor_policy({CANONICAL["camera"]: "true"})
        self.assertTrue(resolved.is_enabled("camera"))
        for sensor in ALL_SENSORS:
            if sensor != "camera":
                self.assertFalse(resolved.is_enabled(sensor))

    def test_invalid_canonical_values_fail_closed(self) -> None:
        for value in ("", "false", "0", "1", "yes", "on", "no"):
            with self.subTest(value=value):
                resolved = policy.resolve_sensor_policy(
                    {CANONICAL["camera"]: value}
                )
                self.assertFalse(resolved.is_enabled("camera"))

    def test_uses_os_environ_when_env_is_none(self) -> None:
        with mock.patch.dict(os.environ, {CANONICAL["microphone"]: "true"}):
            resolved = policy.resolve_sensor_policy()
        self.assertTrue(resolved.is_enabled("microphone"))


class LegacyAliasTest(unittest.TestCase):
    def test_web_screen_context_enables_screen_capture_when_canonical_absent(
        self,
    ) -> None:
        resolved = policy.resolve_sensor_policy({"WEB_SCREEN_CONTEXT_ENABLED": "true"})
        self.assertTrue(resolved.is_enabled("screen_capture"))
        for sensor in ALL_SENSORS:
            if sensor != "screen_capture":
                self.assertFalse(resolved.is_enabled(sensor))

    def test_companion_activity_enables_activity_when_canonical_absent(self) -> None:
        resolved = policy.resolve_sensor_policy({"COMPANION_ACTIVITY_ENABLED": "true"})
        self.assertTrue(resolved.is_enabled("activity"))
        for sensor in ALL_SENSORS:
            if sensor != "activity":
                self.assertFalse(resolved.is_enabled(sensor))

    def test_legacy_true_still_needs_explicit_true(self) -> None:
        for value in ("false", "", "0", "yes", "on"):
            with self.subTest(value=value):
                resolved = policy.resolve_sensor_policy(
                    {"WEB_SCREEN_CONTEXT_ENABLED": value}
                )
                self.assertFalse(resolved.is_enabled("screen_capture"))

    def test_canonical_false_overrides_legacy_true(self) -> None:
        resolved = policy.resolve_sensor_policy(
            {
                CANONICAL["screen_capture"]: "false",
                "WEB_SCREEN_CONTEXT_ENABLED": "true",
            }
        )
        self.assertFalse(resolved.is_enabled("screen_capture"))

    def test_canonical_activity_false_overrides_legacy_true(self) -> None:
        resolved = policy.resolve_sensor_policy(
            {
                CANONICAL["activity"]: "false",
                "COMPANION_ACTIVITY_ENABLED": "true",
            }
        )
        self.assertFalse(resolved.is_enabled("activity"))

    def test_canonical_blank_presence_overrides_legacy_true(self) -> None:
        resolved = policy.resolve_sensor_policy(
            {
                CANONICAL["screen_capture"]: "",
                "WEB_SCREEN_CONTEXT_ENABLED": "true",
            }
        )
        self.assertFalse(resolved.is_enabled("screen_capture"))

    def test_invalid_canonical_activity_overrides_legacy_true(self) -> None:
        for value in ("", "0", "1", "yes", "on", "no"):
            with self.subTest(value=value):
                resolved = policy.resolve_sensor_policy(
                    {
                        CANONICAL["activity"]: value,
                        "COMPANION_ACTIVITY_ENABLED": "true",
                    }
                )
                self.assertFalse(resolved.is_enabled("activity"))

    def test_legacy_true_does_not_leak_to_other_sensors(self) -> None:
        resolved = policy.resolve_sensor_policy(
            {
                "WEB_SCREEN_CONTEXT_ENABLED": "true",
                "COMPANION_ACTIVITY_ENABLED": "true",
            }
        )
        self.assertTrue(resolved.is_enabled("screen_capture"))
        self.assertTrue(resolved.is_enabled("activity"))
        for sensor in ("camera", "screen_ingest", "monitor", "microphone",
                       "process_details"):
            self.assertFalse(resolved.is_enabled(sensor))


class ScreenIngestTokenTest(unittest.TestCase):
    def test_token_presence_never_enables_screen_ingest(self) -> None:
        for env in (
            {"SCREEN_INGEST_TOKEN": "secret"},
            {"SCREEN_INGEST_TOKEN": "secret", "WEB_SCREEN_CONTEXT_ENABLED": "true"},
            {"SCREEN_INGEST_TOKEN": "secret", "COMPANION_ACTIVITY_ENABLED": "true"},
            {"SCREEN_INGEST_TOKEN": "secret", CANONICAL["screen_capture"]: "true"},
        ):
            with self.subTest(env=env):
                resolved = policy.resolve_sensor_policy(env)
                self.assertFalse(resolved.is_enabled("screen_ingest"))

    def test_screen_ingest_requires_canonical_true(self) -> None:
        resolved = policy.resolve_sensor_policy(
            {CANONICAL["screen_ingest"]: "true", "SCREEN_INGEST_TOKEN": "secret"}
        )
        self.assertTrue(resolved.is_enabled("screen_ingest"))


class SensorPolicyApiTest(unittest.TestCase):
    def test_typed_lookup_returns_booleans(self) -> None:
        resolved = policy.resolve_sensor_policy({CANONICAL["activity"]: "true"})
        self.assertIsInstance(resolved.is_enabled("activity"), bool)
        self.assertTrue(resolved.is_enabled("activity"))
        self.assertFalse(resolved.is_enabled("camera"))

    def test_is_enabled_unknown_sensor_raises(self) -> None:
        resolved = policy.resolve_sensor_policy({})
        with self.assertRaises(ValueError):
            resolved.is_enabled("webcam")

    def test_status_payload_contains_only_source_names_and_booleans(self) -> None:
        resolved = policy.resolve_sensor_policy(
            {
                CANONICAL["camera"]: "true",
                CANONICAL["monitor"]: "true",
            }
        )
        payload = resolved.as_status_payload()
        self.assertEqual(set(payload), set(ALL_SENSORS))
        for sensor, value in payload.items():
            self.assertIsInstance(value, bool)
            self.assertIn(sensor, ALL_SENSORS)
        self.assertTrue(payload["camera"])
        self.assertTrue(payload["monitor"])
        self.assertFalse(payload["screen_capture"])
        self.assertFalse(payload["screen_ingest"])

    def test_status_payload_never_leaks_env_names_or_values(self) -> None:
        resolved = policy.resolve_sensor_policy(
            {
                CANONICAL["camera"]: "true",
                "SCREEN_INGEST_TOKEN": "supersecret",
                "WEB_SCREEN_CONTEXT_ENABLED": "true",
                "COMPANION_ACTIVITY_ENABLED": "true",
            }
        )
        serialized = str(resolved.as_status_payload())
        for forbidden in (
            "SENSOR_",
            "SCREEN_INGEST_TOKEN",
            "supersecret",
            "WEB_SCREEN_CONTEXT_ENABLED",
            "COMPANION_ACTIVITY_ENABLED",
            "ENABLED",
        ):
            self.assertNotIn(forbidden, serialized)

    def test_repr_contains_no_env_names_or_values(self) -> None:
        resolved = policy.resolve_sensor_policy(
            {CANONICAL["camera"]: "true", "SCREEN_INGEST_TOKEN": "supersecret"}
        )
        serialized = repr(resolved)
        self.assertNotIn("supersecret", serialized)
        self.assertNotIn("SCREEN_INGEST_TOKEN", serialized)
        self.assertNotIn("SENSOR_", serialized)
        self.assertNotIn("ENABLED", serialized)


class ImmutabilityTest(unittest.TestCase):
    def test_policy_is_frozen(self) -> None:
        resolved = policy.resolve_sensor_policy({})
        with self.assertRaises(FrozenInstanceError):
            resolved.camera = True  # type: ignore[misc]

    def test_policy_does_not_change_when_env_mutates(self) -> None:
        env: dict[str, str] = {}
        resolved = policy.resolve_sensor_policy(env)
        env[CANONICAL["camera"]] = "true"
        self.assertFalse(resolved.is_enabled("camera"))

    def test_constructor_rejects_non_bool_flags(self) -> None:
        with self.assertRaises(TypeError):
            policy.SensorPolicy(camera=1)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            policy.SensorPolicy(microphone="true")  # type: ignore[arg-type]


class PolicyMetadataTest(unittest.TestCase):
    def test_sensor_ids_match_canonical_env_names(self) -> None:
        self.assertEqual(set(policy.SENSOR_IDS), set(ALL_SENSORS))
        self.assertEqual(set(policy.VALID_SENSOR_IDS), set(policy.SENSOR_IDS))
        self.assertEqual(set(CANONICAL), set(policy.SENSOR_IDS))

    def test_legacy_aliases_are_subset_of_sensors(self) -> None:
        self.assertLessEqual(set(LEGACY), set(policy.VALID_SENSOR_IDS))
        self.assertEqual(set(LEGACY), {"screen_capture", "activity"})

    def test_sensor_ids_are_ordered_and_unique(self) -> None:
        self.assertEqual(len(policy.SENSOR_IDS), len(set(policy.SENSOR_IDS)))
        self.assertEqual(len(policy.SENSOR_IDS), 7)


if __name__ == "__main__":
    unittest.main()