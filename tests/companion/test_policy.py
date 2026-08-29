from __future__ import annotations

import unittest

from src.companion.contracts import CompanionState
from src.companion.policy import (
    DeterministicProactivePolicy,
    PolicyContext,
    PolicyDecision,
)


def make_state(
    *,
    activity_mode: str = "idle",
    present: bool = True,
    focused_since: float | None = None,
    interruptible: bool = True,
    display_state: str | None = None,
    updated_at: float = 0.0,
) -> CompanionState:
    return CompanionState(
        activity_mode=activity_mode,
        present=present,
        focused_since=focused_since,
        interruptible=interruptible,
        display_state=display_state if display_state is not None else activity_mode,
        updated_at=updated_at,
    )


def make_context(
    state: CompanionState,
    *,
    now: float = 1000.0,
    next_event_at: float | None = None,
    next_event_title: str | None = None,
    last_fired: dict[str, float] | None = None,
) -> PolicyContext:
    return PolicyContext(
        state=state,
        now=now,
        next_event_at=next_event_at,
        next_event_title=next_event_title,
        last_fired=last_fired if last_fired is not None else {},
    )


def make_policy() -> DeterministicProactivePolicy:
    return DeterministicProactivePolicy(
        long_work_seconds=600.0,
        schedule_lead_seconds=600.0,
        away_return_seconds=60.0,
    )


class PolicyDecisionContractTest(unittest.TestCase):
    def test_decision_is_frozen(self) -> None:
        d = PolicyDecision(
            should_act=False,
            action_kind="silent",
            message_hint="",
            requires_approval=False,
            cooldown_key="silent",
            reason="silent",
        )
        with self.assertRaises(AttributeError):
            d.should_act = True

    def test_context_is_frozen(self) -> None:
        ctx = make_context(make_state(updated_at=980.0))
        with self.assertRaises(AttributeError):
            ctx.now = 1.0


class DeterministicProactivePolicyTest(unittest.TestCase):
    def test_focused_is_silent(self) -> None:
        state = make_state(
            activity_mode="focused",
            interruptible=False,
            focused_since=100.0,
            updated_at=900.0,
        )
        decision = make_policy().decide(make_context(state))
        self.assertFalse(decision.should_act)
        self.assertEqual(decision.action_kind, "silent")
        self.assertEqual(decision.reason, "focused")

    def test_focused_with_approaching_schedule_and_interruptible_reminds(self) -> None:
        state = make_state(
            activity_mode="focused",
            interruptible=True,
            focused_since=100.0,
            updated_at=900.0,
        )
        decision = make_policy().decide(
            make_context(state, next_event_at=1500.0)
        )
        self.assertTrue(decision.should_act)
        self.assertEqual(decision.action_kind, "schedule_remind")

    def test_long_work_suggests_break(self) -> None:
        state = make_state(focused_since=100.0, updated_at=200.0)
        decision = make_policy().decide(make_context(state))
        self.assertTrue(decision.should_act)
        self.assertEqual(decision.action_kind, "break_suggest")
        self.assertFalse(decision.requires_approval)

    def test_schedule_approaching_reminds(self) -> None:
        state = make_state(updated_at=200.0)
        decision = make_policy().decide(
            make_context(state, next_event_at=1500.0)
        )
        self.assertTrue(decision.should_act)
        self.assertEqual(decision.action_kind, "schedule_remind")

    def test_schedule_past_does_not_remind(self) -> None:
        state = make_state(updated_at=200.0)
        decision = make_policy().decide(
            make_context(state, next_event_at=500.0)
        )
        self.assertFalse(decision.should_act)

    def test_schedule_far_future_does_not_remind(self) -> None:
        state = make_state(updated_at=200.0)
        decision = make_policy().decide(
            make_context(state, next_event_at=5000.0)
        )
        self.assertFalse(decision.should_act)

    def test_away_return(self) -> None:
        state = make_state(activity_mode="idle", present=True, updated_at=980.0)
        decision = make_policy().decide(make_context(state))
        self.assertTrue(decision.should_act)
        self.assertEqual(decision.action_kind, "away_return")

    def test_cooldown_silences(self) -> None:
        state = make_state(updated_at=200.0)
        ctx = make_context(
            state,
            next_event_at=1500.0,
            last_fired={"schedule_remind": 900.0},
        )
        decision = make_policy().decide(ctx)
        self.assertFalse(decision.should_act)
        self.assertEqual(decision.action_kind, "silent")
        self.assertEqual(decision.reason, "cooldown")
        self.assertEqual(decision.cooldown_key, "schedule_remind")

    def test_not_interruptible_is_silent(self) -> None:
        state = make_state(
            activity_mode="idle",
            present=True,
            focused_since=100.0,
            interruptible=False,
            updated_at=200.0,
        )
        decision = make_policy().decide(
            make_context(state, next_event_at=1500.0)
        )
        self.assertFalse(decision.should_act)
        self.assertEqual(decision.action_kind, "silent")

    def test_requires_approval_is_always_false(self) -> None:
        policy = make_policy()
        cases = [
            make_context(make_state(focused_since=100.0, updated_at=200.0)),
            make_context(
                make_state(updated_at=200.0), next_event_at=1500.0
            ),
            make_context(make_state(updated_at=980.0)),
            make_context(
                make_state(
                    activity_mode="focused",
                    interruptible=False,
                    updated_at=900.0,
                )
            ),
        ]
        for ctx in cases:
            self.assertFalse(policy.decide(ctx).requires_approval)

    def test_decide_is_pure(self) -> None:
        policy = make_policy()
        ctx = make_context(make_state(focused_since=100.0, updated_at=200.0))
        self.assertEqual(policy.decide(ctx), policy.decide(ctx))

    def test_message_hint_does_not_leak_event_title(self) -> None:
        state = make_state(updated_at=200.0)
        title = "病院の予約 15:00"
        decision = make_policy().decide(
            make_context(
                state,
                next_event_at=1500.0,
                next_event_title=title,
            )
        )
        self.assertTrue(decision.should_act)
        self.assertNotIn(title, decision.message_hint)
        self.assertNotIn("病院", decision.message_hint)


if __name__ == "__main__":
    unittest.main()
