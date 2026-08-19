from __future__ import annotations

import unittest

from src.companion.contracts import (
    ACTIVITY_MODES,
    VALID_ACTIVITY_MODES,
    VALID_DISPLAY_STATES,
    VALID_STATES,
    CompanionState,
    PerceptionEvent,
)
from src.companion.state import (
    PrivacyViolationError,
    StateAggregator,
    StateAggregatorError,
)


def event(
    state: str,
    timestamp: float,
    confidence: float = 1.0,
    source: str = "test",
    raw_data_retained: bool = False,
) -> PerceptionEvent:
    return PerceptionEvent(
        state=state,
        timestamp=timestamp,
        confidence=confidence,
        source=source,
        raw_data_retained=raw_data_retained,
    )


class PerceptionEventContractTest(unittest.TestCase):
    def test_unknown_state_rejected(self) -> None:
        with self.assertRaises(ValueError):
            PerceptionEvent(state="sleeping", timestamp=1.0)

    def test_accepts_float_timestamp(self) -> None:
        PerceptionEvent(state="idle", timestamp=1.5)

    def test_rejects_non_numeric_timestamp(self) -> None:
        with self.assertRaises(TypeError):
            PerceptionEvent(state="idle", timestamp="1.0")

    def test_rejects_bool_timestamp(self) -> None:
        with self.assertRaises(TypeError):
            PerceptionEvent(state="idle", timestamp=True)

    def test_rejects_confidence_below_zero(self) -> None:
        with self.assertRaises(ValueError):
            PerceptionEvent(state="idle", timestamp=1.0, confidence=-0.1)

    def test_rejects_confidence_above_one(self) -> None:
        with self.assertRaises(ValueError):
            PerceptionEvent(state="idle", timestamp=1.0, confidence=1.1)

    def test_accepts_confidence_bounds(self) -> None:
        for confidence in (0.0, 0.5, 1.0):
            PerceptionEvent(state="idle", timestamp=1.0, confidence=confidence)

    def test_rejects_non_bool_confidence(self) -> None:
        with self.assertRaises(ValueError):
            PerceptionEvent(state="idle", timestamp=1.0, confidence="high")

    def test_rejects_empty_source(self) -> None:
        with self.assertRaises(ValueError):
            PerceptionEvent(state="idle", timestamp=1.0, source="   ")

    def test_rejects_non_bool_raw_data_retained(self) -> None:
        with self.assertRaises(TypeError):
            PerceptionEvent(state="idle", timestamp=1.0, raw_data_retained=1)

    def test_is_frozen(self) -> None:
        e = event("idle", 1.0)
        with self.assertRaises(AttributeError):
            e.state = "away"

    def test_no_raw_payload_field(self) -> None:
        self.assertNotIn("payload", PerceptionEvent.__dataclass_fields__)
        self.assertNotIn("raw", PerceptionEvent.__dataclass_fields__)
        self.assertNotIn("metadata", PerceptionEvent.__dataclass_fields__)

    def test_exact_fields(self) -> None:
        self.assertEqual(
            set(PerceptionEvent.__dataclass_fields__),
            {"state", "timestamp", "confidence", "source", "raw_data_retained"},
        )


class CompanionStateContractTest(unittest.TestCase):
    def test_idle_default_shape(self) -> None:
        s = CompanionState(
            activity_mode="idle",
            present=True,
            focused_since=None,
            interruptible=True,
            display_state="idle",
            updated_at=1.0,
        )
        self.assertEqual(s.activity_mode, "idle")
        self.assertTrue(s.present)
        self.assertIsNone(s.focused_since)
        self.assertTrue(s.interruptible)
        self.assertEqual(s.display_state, "idle")

    def test_rejects_unknown_activity_mode(self) -> None:
        with self.assertRaises(ValueError):
            CompanionState(
                activity_mode="busy",
                present=True,
                focused_since=None,
                interruptible=True,
                display_state="idle",
                updated_at=1.0,
            )

    def test_rejects_unknown_display_state(self) -> None:
        with self.assertRaises(ValueError):
            CompanionState(
                activity_mode="idle",
                present=True,
                focused_since=None,
                interruptible=True,
                display_state="busy",
                updated_at=1.0,
            )

    def test_rejects_non_bool_present(self) -> None:
        with self.assertRaises(TypeError):
            CompanionState(
                activity_mode="idle",
                present=1,
                focused_since=None,
                interruptible=True,
                display_state="idle",
                updated_at=1.0,
            )

    def test_is_frozen(self) -> None:
        s = CompanionState(
            activity_mode="idle",
            present=True,
            focused_since=None,
            interruptible=True,
            display_state="idle",
            updated_at=1.0,
        )
        with self.assertRaises(AttributeError):
            s.present = False


class StateAggregatorInitialTest(unittest.TestCase):
    def test_initial_state_is_idle(self) -> None:
        agg = StateAggregator()
        s = agg.state
        self.assertEqual(s.activity_mode, "idle")
        self.assertTrue(s.present)
        self.assertIsNone(s.focused_since)
        self.assertTrue(s.interruptible)
        self.assertEqual(s.display_state, "idle")

    def test_default_min_confidence(self) -> None:
        self.assertEqual(StateAggregator().min_confidence, 0.5)

    def test_rejects_invalid_min_confidence(self) -> None:
        for value in (-0.1, 1.1, "high"):
            with self.assertRaises(ValueError):
                StateAggregator(min_confidence=value)


class StateAggregatorTransitionsTest(unittest.TestCase):
    def test_focused_sets_full_state(self) -> None:
        s = StateAggregator().apply(event("focused", 10.0))
        self.assertEqual(s.activity_mode, "focused")
        self.assertTrue(s.present)
        self.assertEqual(s.focused_since, 10.0)
        self.assertFalse(s.interruptible)
        self.assertEqual(s.display_state, "focused")

    def test_idle_sets_full_state(self) -> None:
        s = StateAggregator().apply(event("idle", 10.0))
        self.assertEqual(s.activity_mode, "idle")
        self.assertTrue(s.present)
        self.assertIsNone(s.focused_since)
        self.assertTrue(s.interruptible)
        self.assertEqual(s.display_state, "idle")

    def test_away_sets_full_state(self) -> None:
        s = StateAggregator().apply(event("away", 10.0))
        self.assertEqual(s.activity_mode, "away")
        self.assertFalse(s.present)
        self.assertIsNone(s.focused_since)
        self.assertFalse(s.interruptible)
        self.assertEqual(s.display_state, "away")

    def test_focus_continues_preserves_focused_since(self) -> None:
        agg = StateAggregator()
        agg.apply(event("focused", 10.0))
        s = agg.apply(event("focused", 20.0))
        self.assertEqual(s.focused_since, 10.0)
        self.assertEqual(s.activity_mode, "focused")

    def test_focus_restarts_focused_since_after_idle(self) -> None:
        agg = StateAggregator()
        agg.apply(event("focused", 10.0))
        agg.apply(event("idle", 20.0))
        s = agg.apply(event("focused", 30.0))
        self.assertEqual(s.focused_since, 30.0)

    def test_away_clears_focused_since(self) -> None:
        agg = StateAggregator()
        agg.apply(event("focused", 10.0))
        s = agg.apply(event("away", 20.0))
        self.assertIsNone(s.focused_since)


class StateAggregatorConfidenceTest(unittest.TestCase):
    def test_low_confidence_does_not_change_state(self) -> None:
        agg = StateAggregator(min_confidence=0.6)
        before = agg.state
        s = agg.apply(event("away", 10.0, confidence=0.4))
        self.assertIs(s, before)
        self.assertEqual(s.activity_mode, "idle")

    def test_high_confidence_changes_state(self) -> None:
        agg = StateAggregator(min_confidence=0.6)
        s = agg.apply(event("away", 10.0, confidence=0.9))
        self.assertEqual(s.activity_mode, "away")

    def test_equal_confidence_is_accepted(self) -> None:
        agg = StateAggregator(min_confidence=0.5)
        s = agg.apply(event("away", 10.0, confidence=0.5))
        self.assertEqual(s.activity_mode, "away")

    def test_confidence_boundary_just_below(self) -> None:
        agg = StateAggregator(min_confidence=0.5)
        s = agg.apply(event("away", 10.0, confidence=0.4999))
        self.assertEqual(s.activity_mode, "idle")


class StateAggregatorOrderingTest(unittest.TestCase):
    def test_out_of_order_timestamp_ignored(self) -> None:
        agg = StateAggregator()
        agg.apply(event("focused", 10.0))
        before = agg.state
        s = agg.apply(event("idle", 5.0))
        self.assertIs(s, before)
        self.assertEqual(s.activity_mode, "focused")

    def test_equal_timestamp_ignored(self) -> None:
        agg = StateAggregator()
        agg.apply(event("focused", 10.0))
        before = agg.state
        s = agg.apply(event("idle", 10.0))
        self.assertIs(s, before)
        self.assertEqual(s.activity_mode, "focused")

    def test_newer_timestamp_applied(self) -> None:
        agg = StateAggregator()
        agg.apply(event("focused", 10.0))
        s = agg.apply(event("idle", 20.0))
        self.assertEqual(s.activity_mode, "idle")
        self.assertEqual(s.updated_at, 20.0)


class StateAggregatorPrivacyTest(unittest.TestCase):
    def test_raw_data_retained_rejected(self) -> None:
        agg = StateAggregator()
        with self.assertRaises(PrivacyViolationError):
            agg.apply(event("idle", 10.0, raw_data_retained=True))

    def test_privacy_violation_is_aggregator_error(self) -> None:
        self.assertTrue(issubclass(PrivacyViolationError, StateAggregatorError))
        self.assertTrue(issubclass(StateAggregatorError, ValueError))

    def test_rejection_does_not_change_state(self) -> None:
        agg = StateAggregator()
        agg.apply(event("focused", 10.0))
        before = agg.state
        with self.assertRaises(PrivacyViolationError):
            agg.apply(event("idle", 20.0, raw_data_retained=True))
        self.assertIs(agg.state, before)


class StateAggregatorImmutabilityTest(unittest.TestCase):
    def test_input_event_unchanged(self) -> None:
        e = event("focused", 10.0)
        before = (e.state, e.timestamp, e.confidence, e.source, e.raw_data_retained)
        StateAggregator().apply(e)
        self.assertEqual(
            (e.state, e.timestamp, e.confidence, e.source, e.raw_data_retained),
            before,
        )

    def test_output_is_frozen(self) -> None:
        s = StateAggregator().apply(event("focused", 10.0))
        with self.assertRaises(AttributeError):
            s.activity_mode = "idle"

    def test_time_comes_from_event_only(self) -> None:
        agg = StateAggregator()
        s = agg.apply(event("focused", 123.0))
        self.assertEqual(s.updated_at, 123.0)
        self.assertEqual(s.focused_since, 123.0)


class StateAggregatorExportsTest(unittest.TestCase):
    def test_exports_from_companion_package(self) -> None:
        import src.companion as companion

        self.assertIs(companion.PerceptionEvent, PerceptionEvent)
        self.assertIs(companion.CompanionState, CompanionState)
        self.assertIs(companion.StateAggregator, StateAggregator)
        self.assertIs(companion.PrivacyViolationError, PrivacyViolationError)

    def test_valid_constants(self) -> None:
        self.assertEqual(VALID_STATES, frozenset({"focused", "idle", "away"}))
        self.assertEqual(VALID_ACTIVITY_MODES, frozenset({"focused", "idle", "away"}))
        self.assertEqual(VALID_DISPLAY_STATES, frozenset({"focused", "idle", "away"}))
        self.assertEqual(ACTIVITY_MODES, VALID_ACTIVITY_MODES)


if __name__ == "__main__":
    unittest.main()
