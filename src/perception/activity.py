"""分類済みアプリカテゴリと idle 秒数から focused/idle/away 遷移イベントへ変換する。

生のアプリ名・window title・text・path・pid・raw input は一切扱わない。
受け取るのは「アプリカテゴリ」と「idle 秒数」という、既に分類・集計された値だけ。
生データを保存せず、状態遷移に必要な最小情報のみで PerceptionEvent を生成する。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

from src.companion.contracts import PerceptionEvent

AppCategory = Literal["work", "communication", "media", "system", "unknown"]

VALID_APP_CATEGORIES: frozenset[str] = frozenset(get_args(AppCategory))

# work だけ既知の作業カテゴリ。それ以外の既知カテゴリ (communication/media/system) は
# 作業ではない。unknown は分類不能。
_WORK_CATEGORIES: frozenset[str] = frozenset({"work"})
_NON_WORK_KNOWN_CATEGORIES: frozenset[str] = frozenset(
    {"communication", "media", "system"}
)

_IDLE_CONFIDENCE = 0.9
_AWAY_CONFIDENCE = 0.9
_WORK_CONFIDENCE = 0.95
_NON_WORK_KNOWN_CONFIDENCE = 0.8
_UNKNOWN_CONFIDENCE = 0.6


@dataclass(frozen=True)
class ActivitySample:
    """PC activity の最小サンプル。

    raw なアプリ名・window title・text・path・pid・raw input は持たない。
    """

    timestamp: float
    idle_seconds: float
    app_category: AppCategory

    def __post_init__(self) -> None:
        if (
            not isinstance(self.timestamp, (int, float))
            or isinstance(self.timestamp, bool)
        ):
            raise TypeError(
                f"timestamp must be int or float, got {type(self.timestamp).__name__}"
            )
        if (
            not isinstance(self.idle_seconds, (int, float))
            or isinstance(self.idle_seconds, bool)
            or self.idle_seconds < 0
        ):
            raise ValueError(
                f"idle_seconds must be a non-negative number, got {self.idle_seconds!r}"
            )
        if (
            not isinstance(self.app_category, str)
            or self.app_category not in VALID_APP_CATEGORIES
        ):
            raise ValueError(f"unknown app_category: {self.app_category!r}")


class ActivityEventCollector:
    """focused/idle/away 遷移イベントを決定論的に生成する。

    - idle_seconds >= away_threshold -> away
    - idle_seconds >= idle_threshold  -> idle
    - それ未満                         -> focused
    - 最初のサンプルと state が変化したときだけ PerceptionEvent を返す。
      同一 state が続く場合は None を返す。
    - イベントの timestamp はサンプルの timestamp 由来。
    - 生データは保存しない。reset は last state だけを消す。
    """

    def __init__(
        self,
        idle_threshold: float = 300,
        away_threshold: float = 1800,
        source: str = "pc_activity",
    ) -> None:
        for name, value in (("idle_threshold", idle_threshold), ("away_threshold", away_threshold)):
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or value < 0
            ):
                raise ValueError(
                    f"{name} must be a non-negative number, got {value!r}"
                )
        if away_threshold < idle_threshold:
            raise ValueError(
                f"away_threshold ({away_threshold!r}) must be >= "
                f"idle_threshold ({idle_threshold!r})"
            )
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"source must be a non-empty str, got {source!r}")
        self.idle_threshold = idle_threshold
        self.away_threshold = away_threshold
        self.source = source
        self._last_state: str | None = None

    @property
    def last_state(self) -> str | None:
        return self._last_state

    def update(self, sample: ActivitySample) -> PerceptionEvent | None:
        if sample.idle_seconds >= self.away_threshold:
            state = "away"
        elif sample.idle_seconds >= self.idle_threshold:
            state = "idle"
        else:
            state = "focused"

        if state == self._last_state:
            return None

        self._last_state = state
        return PerceptionEvent(
            state=state,
            timestamp=sample.timestamp,
            confidence=self._confidence(state, sample.app_category),
            source=self.source,
            raw_data_retained=False,
        )

    def reset(self) -> None:
        self._last_state = None

    @staticmethod
    def _confidence(state: str, app_category: str) -> float:
        if state in ("idle", "away"):
            return _IDLE_CONFIDENCE if state == "idle" else _AWAY_CONFIDENCE
        if app_category in _WORK_CATEGORIES:
            return _WORK_CONFIDENCE
        if app_category in _NON_WORK_KNOWN_CATEGORIES:
            return _NON_WORK_KNOWN_CONFIDENCE
        return _UNKNOWN_CONFIDENCE
