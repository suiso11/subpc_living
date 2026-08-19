"""Companion activity の env 駆動起動と privacy-safe 状態 payload の共通ヘルパ。

Web / Discord / Voice / Desktop から共通で使う。グローバル状態は持たず、
呼び出し側が ActivityRuntime のライフサイクルを管理する。
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Mapping

from src.perception import ActivityRuntime, create_activity_source

_POLL_DEFAULT = 5.0
_IDLE_DEFAULT = 300.0
_AWAY_DEFAULT = 1800.0


def _positive_float(
    name: str,
    default: float,
    env: Mapping[str, str],
    logger: logging.Logger,
) -> float | None:
    """環境変数から正の有限 float を読む。未設定は default、不正は None (機能無効化)。"""
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        logger.warning(
            "Companion activity setting %s invalid (%s): companion disabled",
            name,
            type(exc).__name__,
        )
        return None
    if not math.isfinite(value) or value <= 0:
        logger.warning(
            "Companion activity setting %s invalid (must be a positive number): companion disabled",
            name,
        )
        return None
    return value


def create_activity_runtime_from_env(
    env: Mapping[str, str] | None = None,
    *,
    logger: logging.Logger | None = None,
) -> ActivityRuntime | None:
    """COMPANION_ACTIVITY_ENABLED=true のときだけ ActivityRuntime を生成・start して返す。

    env は None なら os.environ を使う。数値設定不正・source 生成失敗は None を返し、
    例外の型名だけ logger.warning に残し、メッセージ・環境・プロセス情報は記録しない。
    無効時は None。start() 呼び出し済みの runtime を返す。
    """
    resolved = os.environ if env is None else env
    resolved_logger = logger if logger is not None else logging.getLogger(__name__)
    if resolved.get("COMPANION_ACTIVITY_ENABLED", "").strip().lower() != "true":
        return None

    poll_interval = _positive_float(
        "COMPANION_ACTIVITY_POLL_INTERVAL_SECONDS",
        _POLL_DEFAULT,
        resolved,
        resolved_logger,
    )
    idle_threshold = _positive_float(
        "COMPANION_ACTIVITY_IDLE_THRESHOLD_SECONDS",
        _IDLE_DEFAULT,
        resolved,
        resolved_logger,
    )
    away_threshold = _positive_float(
        "COMPANION_ACTIVITY_AWAY_THRESHOLD_SECONDS",
        _AWAY_DEFAULT,
        resolved,
        resolved_logger,
    )
    if None in (poll_interval, idle_threshold, away_threshold):
        return None

    try:
        runtime = ActivityRuntime(
            create_activity_source(),
            poll_interval=poll_interval,
            idle_threshold=idle_threshold,
            away_threshold=away_threshold,
        )
        runtime.start()
    except Exception as exc:
        resolved_logger.warning(
            "Companion activity startup failed (%s): feature disabled",
            type(exc).__name__,
        )
        return None
    return runtime


def companion_state_payload(runtime: ActivityRuntime | None) -> dict:
    """GET /api/companion/state と同じ privacy-safe payload を返す。

    runtime が None なら {"enabled": False}。それ以外は enabled/running/last_update_at/
    failure_count/consecutive_failures/last_error_type と state (CompanionState の
    フィールドのみ、または None) を返す。プロセス名・PID・アプリ分類・window title・
    エラー本文・生サンプル/イベントは一切含めない。
    """
    if runtime is None:
        return {"enabled": False}
    status = runtime.status
    last_state = status.last_state
    payload: dict = {
        "enabled": True,
        "running": status.running,
        "last_update_at": status.last_update_at,
        "failure_count": status.failure_count,
        "consecutive_failures": status.consecutive_failures,
        "last_error_type": status.last_error_type,
    }
    if last_state is not None:
        payload["state"] = {
            "activity_mode": last_state.activity_mode,
            "present": last_state.present,
            "focused_since": last_state.focused_since,
            "interruptible": last_state.interruptible,
            "display_state": last_state.display_state,
            "updated_at": last_state.updated_at,
        }
    else:
        payload["state"] = None
    return payload
