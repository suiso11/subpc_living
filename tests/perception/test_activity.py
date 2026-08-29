from __future__ import annotations

import unittest

from src.companion.contracts import PerceptionEvent
from src.companion.state import StateAggregator
from src.perception.activity import (
    VALID_APP_CATEGORIES,
    ActivityEventCollector,
    ActivitySample,
)
from src.perception import ActivityEventCollector as PkgActivityEventCollector
from src.perception import ActivitySample as PkgActivitySample


def sample(
    timestamp: float,
    idle_seconds: float,
    app_category: str = "work",
) -> ActivitySample:
    return ActivitySample(
        timestamp=timestamp,
        idle_seconds=idle_seconds,
        app_category=app_category,
    )


class ActivitySampleContractTest(unittest.TestCase):
    def test_is_frozen(self) -> None:
        s = sample(1.0, 10.0)
        with self.assertRaises(AttributeError):
            s.idle_seconds = 20.0

    def test_exact_fields(self) -> None:
        self.assertEqual(
            set(ActivitySample.__dataclass_fields__),
            {"timestamp", "idle_seconds", "app_category"},
        )

    def test_no_raw_fields(self) -> None:
        fields = ActivitySample.__dataclass_fields__
        for forbidden in ("app_name", "window_title", "title", "text", "path", "pid", "raw"):
            self.assertNotIn(forbidden, fields)

    def test_rejects_unknown_app_category(self) -> None:
        with self.assertRaises(ValueError):
            ActivitySample(timestamp=1.0, idle_seconds=0, app_category="gaming")

    def test_rejects_non_numeric_timestamp(self) -> None:
        with self.assertRaises(TypeError):
            ActivitySample(timestamp="1.0", idle_seconds=0, app_category="work")

    def test_rejects_negative_idle(self) -> None:
        with self.assertRaises(ValueError):
            ActivitySample(timestamp=1.0, idle_seconds=-1, app_category="work")

    def test_accepts_all_categories(self) -> None:
        self.assertEqual(
            VALID_APP_CATEGORIES,
            frozenset({"work", "communication", "media", "system", "unknown"}),
        )
        for cat in VALID_APP_CATEGORIES:
            ActivitySample(timestamp=1.0, idle_seconds=0, app_category=cat)


class ActivityEventCollectorValidationTest(unittest.TestCase):
    def test_defaults(self) -> None:
        c = ActivityEventCollector()
        self.assertEqual(c.idle_threshold, 300)
        self.assertEqual(c.away_threshold, 1800)
        self.assertEqual(c.source, "pc_activity")

    def test_rejects_negative_threshold(self) -> None:
        with self.assertRaises(ValueError):
            ActivityEventCollector(idle_threshold=-1)
        with self.assertRaises(ValueError):
            ActivityEventCollector(away_threshold=-1)

    def test_rejects_away_below_idle(self) -> None:
        with self.assertRaises(ValueError):
            ActivityEventCollector(idle_threshold=500, away_threshold=100)

    def test_rejects_invalid_source(self) -> None:
        with self.assertRaises(ValueError):
            ActivityEventCollector(source="   ")


class ActivityEventCollectorStateTest(unittest.TestCase):
    def test_focused_below_idle(self) -> None:
        e = ActivityEventCollector().update(sample(1.0, 0))
        self.assertIsNotNone(e)
        assert e is not None
        self.assertEqual(e.state, "focused")

    def test_idle_at_threshold(self) -> None:
        e = ActivityEventCollector(idle_threshold=300).update(sample(1.0, 300))
        self.assertIsNotNone(e)
        assert e is not None
        self.assertEqual(e.state, "idle")

    def test_focused_just_below_threshold(self) -> None:
        e = ActivityEventCollector(idle_threshold=300).update(sample(1.0, 299.999))
        self.assertIsNotNone(e)
        assert e is not None
        self.assertEqual(e.state, "focused")

    def test_away_at_threshold(self) -> None:
        e = ActivityEventCollector(away_threshold=1800).update(sample(1.0, 1800))
        self.assertIsNotNone(e)
        assert e is not None
        self.assertEqual(e.state, "away")

    def test_idle_between_thresholds(self) -> None:
        e = ActivityEventCollector().update(sample(1.0, 900))
        self.assertIsNotNone(e)
        assert e is not None
        self.assertEqual(e.state, "idle")

    def test_away_above_threshold(self) -> None:
        e = ActivityEventCollector().update(sample(1.0, 2000))
        self.assertIsNotNone(e)
        assert e is not None
        self.assertEqual(e.state, "away")

    def test_transitions_focused_to_idle_to_away(self) -> None:
        c = ActivityEventCollector()
        self.assertEqual(c.update(sample(1.0, 0)).state, "focused")
        self.assertEqual(c.update(sample(2.0, 400)).state, "idle")
        self.assertEqual(c.update(sample(3.0, 2000)).state, "away")

    def test_same_state_returns_none(self) -> None:
        c = ActivityEventCollector()
        self.assertIsNotNone(c.update(sample(1.0, 0)))
        self.assertIsNone(c.update(sample(2.0, 100)))
        self.assertIsNone(c.update(sample(3.0, 200)))

    def test_first_sample_always_emits(self) -> None:
        c = ActivityEventCollector()
        self.assertIsNotNone(c.update(sample(1.0, 0)))


class ActivityEventCollectorConfidenceTest(unittest.TestCase):
    def _collect(self, idle: float, cat: str) -> PerceptionEvent:
        c = ActivityEventCollector()
        e = c.update(sample(1.0, idle, cat))
        assert e is not None
        return e

    def test_work_focused(self) -> None:
        self.assertEqual(self._collect(0, "work").confidence, 0.95)

    def test_known_non_work_focused(self) -> None:
        for cat in ("communication", "media", "system"):
            self.assertEqual(self._collect(0, cat).confidence, 0.8)

    def test_unknown_focused(self) -> None:
        self.assertEqual(self._collect(0, "unknown").confidence, 0.6)

    def test_idle_confidence(self) -> None:
        self.assertEqual(self._collect(400, "work").confidence, 0.9)

    def test_away_confidence(self) -> None:
        self.assertEqual(self._collect(2000, "unknown").confidence, 0.9)


class ActivityEventCollectorEventShapeTest(unittest.TestCase):
    def test_event_shape(self) -> None:
        c = ActivityEventCollector()
        e = c.update(sample(123.0, 0, "work"))
        assert e is not None
        self.assertIsInstance(e, PerceptionEvent)
        self.assertEqual(e.timestamp, 123.0)
        self.assertEqual(e.source, "pc_activity")
        self.assertFalse(e.raw_data_retained)
        self.assertEqual(e.state, "focused")

    def test_input_sample_unchanged(self) -> None:
        c = ActivityEventCollector()
        s = sample(10.0, 0)
        c.update(s)
        self.assertEqual((s.timestamp, s.idle_seconds, s.app_category), (10.0, 0, "work"))


class ActivityEventCollectorResetTest(unittest.TestCase):
    def test_reset_clears_last_state(self) -> None:
        c = ActivityEventCollector()
        c.update(sample(1.0, 0))
        self.assertEqual(c.last_state, "focused")
        c.reset()
        self.assertIsNone(c.last_state)
        e = c.update(sample(2.0, 0))
        self.assertIsNotNone(e)
        assert e is not None
        self.assertEqual(e.state, "focused")

    def test_reset_does_not_keep_raw_data(self) -> None:
        c = ActivityEventCollector()
        c.update(sample(1.0, 0))
        c.reset()
        self.assertEqual(vars(c)["_last_state"], None)


class ActivityEventCollectorStateAggregatorIntegrationTest(unittest.TestCase):
    def test_full_pipeline(self) -> None:
        c = ActivityEventCollector()
        agg = StateAggregator()
        for ts, idle in enumerate((0, 100, 400, 2000, 50)):
            e = c.update(sample(float(ts), float(idle)))
            if e is not None:
                agg.apply(e)
        self.assertEqual(agg.state.activity_mode, "focused")

    def test_idle_flows_into_aggregator(self) -> None:
        c = ActivityEventCollector()
        agg = StateAggregator()
        agg.apply(c.update(sample(0.0, 0.0)))
        agg.apply(c.update(sample(1.0, 400.0)))
        self.assertEqual(agg.state.activity_mode, "idle")

    def test_away_flows_into_aggregator(self) -> None:
        c = ActivityEventCollector()
        agg = StateAggregator()
        agg.apply(c.update(sample(0.0, 0.0)))
        agg.apply(c.update(sample(1.0, 2000.0)))
        self.assertEqual(agg.state.activity_mode, "away")


class ActivityEventCollectorExportsTest(unittest.TestCase):
    def test_exports_from_package(self) -> None:
        self.assertIs(PkgActivityEventCollector, ActivityEventCollector)
        self.assertIs(PkgActivitySample, ActivitySample)


if __name__ == "__main__":
    unittest.main()
