# Phase 4: PerceptionEvent と CompanionState 契約、決定的StateAggregator
from .contracts import (
    ACTIVITY_MODES,
    VALID_ACTIVITY_MODES,
    VALID_DISPLAY_STATES,
    VALID_STATES,
    CompanionState,
    PerceptionEvent,
)
from .state import PrivacyViolationError, StateAggregator, StateAggregatorError
from .calendar import CalendarSource, NextEvent

__all__ = [
    "ACTIVITY_MODES",
    "VALID_ACTIVITY_MODES",
    "VALID_DISPLAY_STATES",
    "VALID_STATES",
    "CompanionState",
    "PerceptionEvent",
    "PrivacyViolationError",
    "StateAggregator",
    "StateAggregatorError",
    "CalendarSource",
    "NextEvent",
]
