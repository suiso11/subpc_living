"""Pure Python unit tests for the overlay shell state logic (Phase 6a)."""
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from src.desktop.shell import (
    ShellVisualState,
    decide_shell_state,
    overlay_visibility,
    sensor_provenance,
)


class TestDecideShellState(unittest.TestCase):
    """Tests for decide_shell_state priority mapping."""

    def test_error_beats_conversation(self) -> None:
        result = decide_shell_state(
            {"activity_mode": "idle", "present": True},
            conversation_active=True,
            has_error=True,
        )
        self.assertEqual(result, ShellVisualState.ERROR)

    def test_conversation_beats_schedule_near(self) -> None:
        result = decide_shell_state(
            {"activity_mode": "idle", "present": True},
            conversation_active=True,
            schedule_near=True,
        )
        self.assertEqual(result, ShellVisualState.CONVERSING)

    def test_schedule_near_beats_away(self) -> None:
        result = decide_shell_state(
            {"activity_mode": "away", "present": False},
            schedule_near=True,
        )
        self.assertEqual(result, ShellVisualState.SCHEDULE_NEAR)

    def test_none_state_returns_idle(self) -> None:
        self.assertEqual(decide_shell_state(None), ShellVisualState.IDLE)

    def test_empty_state_returns_idle(self) -> None:
        self.assertEqual(decide_shell_state({}), ShellVisualState.IDLE)

    def test_present_false_returns_away(self) -> None:
        result = decide_shell_state({"activity_mode": "idle", "present": False})
        self.assertEqual(result, ShellVisualState.AWAY)

    def test_activity_mode_away_returns_away(self) -> None:
        result = decide_shell_state({"activity_mode": "away", "present": True})
        self.assertEqual(result, ShellVisualState.AWAY)

    def test_focused_returns_working(self) -> None:
        result = decide_shell_state({"activity_mode": "focused", "present": True})
        self.assertEqual(result, ShellVisualState.WORKING)

    def test_idle_mode_returns_idle(self) -> None:
        result = decide_shell_state({"activity_mode": "idle", "present": True})
        self.assertEqual(result, ShellVisualState.IDLE)

    def test_unknown_mode_returns_idle(self) -> None:
        result = decide_shell_state({"activity_mode": "something", "present": True})
        self.assertEqual(result, ShellVisualState.IDLE)

    def test_focused_not_interruptible(self) -> None:
        result = decide_shell_state(
            {"activity_mode": "focused", "present": True, "interruptible": False}
        )
        self.assertEqual(result, ShellVisualState.WORKING)

    def test_conversation_active_with_none_state(self) -> None:
        result = decide_shell_state(None, conversation_active=True)
        self.assertEqual(result, ShellVisualState.CONVERSING)

    def test_error_only_needs_flag(self) -> None:
        result = decide_shell_state(None, has_error=True)
        self.assertEqual(result, ShellVisualState.ERROR)


class TestOverlayVisibility(unittest.TestCase):
    """Tests for overlay_visibility."""

    def test_error_no_shrink(self) -> None:
        result = overlay_visibility("error")
        self.assertEqual(result, {"visible": True, "shrink": False})

    def test_working_not_interruptible_shrink(self) -> None:
        result = overlay_visibility("working", interruptible=False)
        self.assertEqual(result, {"visible": True, "shrink": True})

    def test_working_interruptible_no_shrink(self) -> None:
        result = overlay_visibility("working", interruptible=True)
        self.assertEqual(result, {"visible": True, "shrink": False})

    def test_away_shrink(self) -> None:
        result = overlay_visibility("away")
        self.assertEqual(result, {"visible": True, "shrink": True})

    def test_idle_no_shrink(self) -> None:
        result = overlay_visibility("idle")
        self.assertEqual(result, {"visible": True, "shrink": False})

    def test_conversing_no_shrink(self) -> None:
        result = overlay_visibility("conversing")
        self.assertEqual(result, {"visible": True, "shrink": False})

    def test_schedule_near_no_shrink(self) -> None:
        result = overlay_visibility("schedule_near")
        self.assertEqual(result, {"visible": True, "shrink": False})

    def test_away_with_interruptible_shrink(self) -> None:
        result = overlay_visibility("away", interruptible=True)
        self.assertEqual(result, {"visible": True, "shrink": True})


class TestSensorProvenance(unittest.TestCase):
    """Tests for sensor_provenance."""

    def test_activity_source(self) -> None:
        result = sensor_provenance("activity", 100.0)
        self.assertEqual(result["source"], "activity")
        self.assertEqual(result["source_label"], "PC活動")
        self.assertEqual(result["fetched_at"], 100.0)
        self.assertFalse(result["saved"])

    def test_calendar_source(self) -> None:
        result = sensor_provenance("calendar", 200.0)
        self.assertEqual(result["source_label"], "予定")

    def test_tasks_source(self) -> None:
        result = sensor_provenance("tasks", 300.0)
        self.assertEqual(result["source_label"], "タスク")

    def test_monitor_source(self) -> None:
        result = sensor_provenance("monitor", 400.0)
        self.assertEqual(result["source_label"], "PC状態")

    def test_unknown_source_fallback(self) -> None:
        result = sensor_provenance("unknown_source", 500.0)
        self.assertEqual(result["source_label"], "unknown_source")

    def test_saved_defaults_false(self) -> None:
        result = sensor_provenance("activity", 100.0)
        self.assertFalse(result["saved"])


class TestDeterminism(unittest.TestCase):
    """Same inputs must produce the same outputs."""

    def test_decide_shell_state_deterministic(self) -> None:
        state = {"activity_mode": "focused", "present": True}
        a = decide_shell_state(state)
        b = decide_shell_state(state)
        self.assertEqual(a, b)

    def test_overlay_visibility_deterministic(self) -> None:
        a = overlay_visibility("working", interruptible=False)
        b = overlay_visibility("working", interruptible=False)
        self.assertEqual(a, b)

    def test_sensor_provenance_deterministic(self) -> None:
        a = sensor_provenance("activity", 100.0)
        b = sensor_provenance("activity", 100.0)
        self.assertEqual(a, b)


class TestShellVisualStateConstants(unittest.TestCase):
    """Verify all expected states are defined."""

    def test_all_states_present(self) -> None:
        expected = {"idle", "working", "conversing", "away", "schedule_near", "error"}
        actual = {
            ShellVisualState.IDLE,
            ShellVisualState.WORKING,
            ShellVisualState.CONVERSING,
            ShellVisualState.AWAY,
            ShellVisualState.SCHEDULE_NEAR,
            ShellVisualState.ERROR,
        }
        self.assertEqual(actual, expected)

    def test_all_states_are_strings(self) -> None:
        for attr in (
            ShellVisualState.IDLE,
            ShellVisualState.WORKING,
            ShellVisualState.CONVERSING,
            ShellVisualState.AWAY,
            ShellVisualState.SCHEDULE_NEAR,
            ShellVisualState.ERROR,
        ):
            self.assertIsInstance(attr, str)


class TestApplyClickThrough(unittest.TestCase):
    """Tests for apply_click_through without real Win32 calls."""

    def test_returns_false_when_hwnd_is_zero(self) -> None:
        from src.desktop.windows import apply_click_through
        self.assertFalse(apply_click_through(0, True))

    def test_returns_false_on_non_windows(self) -> None:
        from src.desktop.windows import apply_click_through
        with patch("src.desktop.windows.os") as mock_os:
            mock_os.name = "posix"
            self.assertFalse(apply_click_through(12345, True))


if __name__ == "__main__":
    unittest.main()
