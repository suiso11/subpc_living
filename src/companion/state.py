"""決定的StateAggregator。

PerceptionEvent を適用して CompanionState を決める。センサーや生データへは依存せず、
イベントの state / timestamp / confidence という注入値だけで状態遷移を決定する。
時刻は注入値 (イベントの timestamp) だけで決まり、現在時刻は参照しない。
"""

from __future__ import annotations

from src.companion.contracts import CompanionState, PerceptionEvent


class StateAggregatorError(ValueError):
    """許可されないイベントが渡された。"""


class PrivacyViolationError(StateAggregatorError):
    """raw_data_retained=True のイベントはprivacy違反として拒否する。"""


class StateAggregator:
    """focused / idle / away 遷移を決定的に計算する状態集約器。

    - 初期状態は idle。
    - low confidence (min_confidence 未満) のイベントは状態を変えない。
    - 古い timestamp (現在の updated_at 以下) のイベントは状態を変えない
      (out-of-order 拒否)。
    - raw_data_retained=True のイベントは PrivacyViolationError で拒否。
    - 入力 event を変更しない。出力は不変な CompanionState。
    """

    def __init__(self, min_confidence: float = 0.5) -> None:
        if (
            not isinstance(min_confidence, (int, float))
            or isinstance(min_confidence, bool)
            or not 0.0 <= min_confidence <= 1.0
        ):
            raise ValueError(f"min_confidence must be in [0, 1], got {min_confidence!r}")
        self.min_confidence = min_confidence
        self._state = CompanionState(
            activity_mode="idle",
            present=True,
            focused_since=None,
            interruptible=True,
            display_state="idle",
            updated_at=0.0,
        )

    @property
    def state(self) -> CompanionState:
        return self._state

    def apply(self, event: PerceptionEvent) -> CompanionState:
        if event.raw_data_retained:
            raise PrivacyViolationError(
                f"raw_data_retained=True is a privacy violation (source={event.source!r})"
            )
        if event.confidence < self.min_confidence:
            return self._state
        if event.timestamp <= self._state.updated_at:
            return self._state
        self._state = self._transition(event)
        return self._state

    def _transition(self, event: PerceptionEvent) -> CompanionState:
        if event.state == "focused":
            focused_since = (
                self._state.focused_since
                if self._state.activity_mode == "focused"
                else event.timestamp
            )
            return CompanionState(
                activity_mode="focused",
                present=True,
                focused_since=focused_since,
                interruptible=False,
                display_state="focused",
                updated_at=event.timestamp,
            )
        if event.state == "idle":
            return CompanionState(
                activity_mode="idle",
                present=True,
                focused_since=None,
                interruptible=True,
                display_state="idle",
                updated_at=event.timestamp,
            )
        return CompanionState(
            activity_mode="away",
            present=False,
            focused_since=None,
            interruptible=False,
            display_state="away",
            updated_at=event.timestamp,
        )
