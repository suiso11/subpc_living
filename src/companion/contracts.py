"""Companion契約。

PerceptionEvent はセンサー・生データへ依存しない、StateAggregator の入力となる
知覚イベント契約。本文・metadata・raw payload は保持せず、状態遷移に必要な
最小のフィールドだけを持つ。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

State = Literal["focused", "idle", "away"]
ActivityMode = Literal["focused", "idle", "away"]
DisplayState = Literal["focused", "idle", "away"]

VALID_STATES: frozenset[str] = frozenset(get_args(State))
VALID_ACTIVITY_MODES: frozenset[str] = frozenset(get_args(ActivityMode))
VALID_DISPLAY_STATES: frozenset[str] = frozenset(get_args(DisplayState))

ACTIVITY_MODES: frozenset[str] = VALID_ACTIVITY_MODES


@dataclass(frozen=True)
class PerceptionEvent:
    """StateAggregator への入力となる知覚イベント。

    raw payload・本文・metadata は持たない。センサーや生データへ依存せず、
    状態遷移に必要な最小限のフィールドだけを保持する。
    """

    state: State
    timestamp: float
    confidence: float = 1.0
    source: str = "unknown"
    raw_data_retained: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.state, str) or self.state not in VALID_STATES:
            raise ValueError(f"unknown state: {self.state!r}")
        if not isinstance(self.timestamp, (int, float)) or isinstance(self.timestamp, bool):
            raise TypeError(
                f"timestamp must be int or float, got {type(self.timestamp).__name__}"
            )
        if (
            not isinstance(self.confidence, (int, float))
            or isinstance(self.confidence, bool)
            or not 0.0 <= self.confidence <= 1.0
        ):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence!r}")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError(f"source must be a non-empty str, got {self.source!r}")
        if not isinstance(self.raw_data_retained, bool):
            raise TypeError(
                f"raw_data_retained must be bool, got {type(self.raw_data_retained).__name__}"
            )


@dataclass(frozen=True)
class CompanionState:
    """StateAggregator の出力となる不変な状態スナップショット。"""

    activity_mode: ActivityMode
    present: bool
    focused_since: float | None
    interruptible: bool
    display_state: DisplayState
    updated_at: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.activity_mode, str)
            or self.activity_mode not in VALID_ACTIVITY_MODES
        ):
            raise ValueError(f"unknown activity_mode: {self.activity_mode!r}")
        if not isinstance(self.present, bool):
            raise TypeError(f"present must be bool, got {type(self.present).__name__}")
        if self.focused_since is not None and (
            not isinstance(self.focused_since, (int, float))
            or isinstance(self.focused_since, bool)
        ):
            raise TypeError(
                f"focused_since must be int, float or None, "
                f"got {type(self.focused_since).__name__}"
            )
        if not isinstance(self.interruptible, bool):
            raise TypeError(
                f"interruptible must be bool, got {type(self.interruptible).__name__}"
            )
        if (
            not isinstance(self.display_state, str)
            or self.display_state not in VALID_DISPLAY_STATES
        ):
            raise ValueError(f"unknown display_state: {self.display_state!r}")
        if not isinstance(self.updated_at, (int, float)) or isinstance(self.updated_at, bool):
            raise TypeError(
                f"updated_at must be int or float, got {type(self.updated_at).__name__}"
            )
