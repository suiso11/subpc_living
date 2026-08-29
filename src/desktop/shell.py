"""Companion overlay shell state logic (Phase 6a).

Deterministic mapping from the privacy-safe companion state payload to a
finite visual state for the overlay window (roadmap section 7: 待機 / 作業中 /
会話中 / 離席 / 予定接近 / エラー). No raw data, no LLM decisions.
"""
from __future__ import annotations

from typing import Any


class ShellVisualState:
    """String constants for the overlay's finite visual states."""

    IDLE = "idle"
    WORKING = "working"
    CONVERSING = "conversing"
    AWAY = "away"
    SCHEDULE_NEAR = "schedule_near"
    ERROR = "error"


def decide_shell_state(
    state: dict[str, Any] | None,
    *,
    conversation_active: bool = False,
    schedule_near: bool = False,
    has_error: bool = False,
) -> str:
    """Deterministic priority mapping from companion payload to overlay state.

    Priority (highest to lowest):
    1. has_error -> "error"
    2. conversation_active -> "conversing"
    3. schedule_near -> "schedule_near"
    4. state is None or empty dict -> "idle"
    5. state["present"] is False or activity_mode == "away" -> "away"
    6. activity_mode == "focused" -> "working"
    7. otherwise -> "idle"
    """
    if has_error:
        return ShellVisualState.ERROR
    if conversation_active:
        return ShellVisualState.CONVERSING
    if schedule_near:
        return ShellVisualState.SCHEDULE_NEAR
    if state is None or not state:
        return ShellVisualState.IDLE
    if state.get("present") is False or state.get("activity_mode") == "away":
        return ShellVisualState.AWAY
    if state.get("activity_mode") == "focused":
        return ShellVisualState.WORKING
    return ShellVisualState.IDLE


def overlay_visibility(shell_state: str, *, interruptible: bool = True) -> dict[str, bool]:
    """Return visibility and shrink flags for the overlay window.

    - "error" -> visible, no shrink
    - "working" and not interruptible (集中モード) -> visible, shrink
    - "away" -> visible, shrink
    - everything else -> visible, no shrink
    """
    if shell_state == ShellVisualState.ERROR:
        return {"visible": True, "shrink": False}
    if shell_state == ShellVisualState.WORKING and not interruptible:
        return {"visible": True, "shrink": True}
    if shell_state == ShellVisualState.AWAY:
        return {"visible": True, "shrink": True}
    return {"visible": True, "shrink": False}


_SOURCE_LABELS: dict[str, str] = {
    "activity": "PC活動",
    "calendar": "予定",
    "tasks": "タスク",
    "monitor": "PC状態",
}


def sensor_provenance(source: str, fetched_at: float, *, saved: bool = False) -> dict[str, Any]:
    """Build a sensor-provenance display dict.

    生データは保存しないため saved は常に False (現状)。
    """
    return {
        "source": source,
        "source_label": _SOURCE_LABELS.get(source, source),
        "fetched_at": fetched_at,
        "saved": saved,
    }
