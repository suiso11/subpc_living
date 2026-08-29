"""Companion activity の env 駆動起動と privacy-safe 状態 payload の共通ヘルパ。

Web / Discord / Voice / Desktop から共通で使う。グローバル状態は持たず、
呼び出し側が ActivityRuntime のライフサイクルを管理する。
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Mapping

from src.perception import (
    ActivityRuntime,
    create_activity_source,
    resolve_sensor_policy,
)

_POLL_DEFAULT = 5.0
_IDLE_DEFAULT = 300.0
_AWAY_DEFAULT = 1800.0

# 外部向け (HTTP / WebSocket / companion payload) のエラーコード allowlist。
# 例外クラス名は内部ログでのみ使い、外部 JSON へはこの固定コードだけを載せる。
# 未知・カスタム例外は常に "internal_error" へ写像する (fail closed)。
SENSOR_ERROR_CODES = ("timeout", "invalid_input", "unavailable", "internal_error")

_SENSOR_ERROR_ALLOWLIST = {
    # timeout
    "TimeoutError": "timeout",
    "TimeoutExpired": "timeout",
    # invalid_input
    "ValueError": "invalid_input",
    "TypeError": "invalid_input",
    "KeyError": "invalid_input",
    "IndexError": "invalid_input",
    "AttributeError": "invalid_input",
    "JSONDecodeError": "invalid_input",
    "UnicodeDecodeError": "invalid_input",
    # unavailable
    "ConnectionError": "unavailable",
    "ConnectionRefusedError": "unavailable",
    "ConnectionResetError": "unavailable",
    "BrokenPipeError": "unavailable",
    "OSError": "unavailable",
    "FileNotFoundError": "unavailable",
    "PermissionError": "unavailable",
}


def sensor_error_code_from_name(name: str | None) -> str:
    """例外クラス名を外部向け固定 error code へ写像する。

    allowlist に一致すれば対応コードを、None・未知・カスタムクラスは
    ``internal_error`` を返す。例外クラス名そのものは外部へ載せない。
    """
    if not name:
        return "internal_error"
    return _SENSOR_ERROR_ALLOWLIST.get(name, "internal_error")


def sensor_error_code(exc: BaseException) -> str:
    """例外を外部向け固定 error code へ写像する。

    型名は内部ログで使えるが、外部 JSON にはこの関数の戻り値だけを載せる。
    """
    return sensor_error_code_from_name(type(exc).__name__)


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


def _best_effort_stop(runtime: ActivityRuntime) -> None:
    """start に失敗した runtime を停止試行する。失敗は握りつぶす (best effort)。"""
    try:
        runtime.stop()
    except Exception:
        pass


def create_activity_runtime_from_env(
    env: Mapping[str, str] | None = None,
    *,
    logger: logging.Logger | None = None,
) -> ActivityRuntime | None:
    """SensorPolicy.activity が有効のときだけ ActivityRuntime を生成・start して返す。

    有効判定は共有 resolve_sensor_policy の SensorPolicy.activity を使う。canonical の
    SENSOR_ACTIVITY_ENABLED が最優先で、未設定のときだけ legacy の
    COMPANION_ACTIVITY_ENABLED を参照する。無効時は source/collector/runtime を一切
    生成せず None。env は None なら os.environ を使う。数値設定不正・source 生成失敗は
    None を返し、例外の型名だけ logger.warning に残し、メッセージ・環境・プロセス情報は
    記録しない。start() が真を返した場合のみ runtime を返し、start が False のときや
    start が raise したときは生成済み runtime を best-effort で stop して None を返す。
    """
    resolved = os.environ if env is None else env
    resolved_logger = logger if logger is not None else logging.getLogger(__name__)
    if not resolve_sensor_policy(resolved).activity:
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

    runtime: ActivityRuntime | None = None
    try:
        runtime = ActivityRuntime(
            create_activity_source(),
            poll_interval=poll_interval,
            idle_threshold=idle_threshold,
            away_threshold=away_threshold,
        )
        started = runtime.start()
    except Exception as exc:
        resolved_logger.warning(
            "Companion activity startup failed (%s): feature disabled",
            type(exc).__name__,
        )
        if runtime is not None:
            _best_effort_stop(runtime)
        return None
    if not started:
        resolved_logger.warning(
            "Companion activity startup incomplete: feature disabled",
        )
        _best_effort_stop(runtime)
        return None
    return runtime


def companion_state_payload(runtime: ActivityRuntime | None) -> dict:
    """GET /api/companion/state と同じ privacy-safe payload を返す。

    runtime が None なら {"enabled": False}。それ以外は enabled/running/last_update_at/
    failure_count/consecutive_failures/last_error_type と state (CompanionState の
    フィールドのみ、または None) を返す。プロセス名・PID・アプリ分類・window title・
    エラー本文・生サンプル/イベントは一切含めない。last_error_type はランタイム内部の
    型名ではなく、allowlist の固定コード (unavailable / internal_error 等) へ写像する。
    """
    if runtime is None:
        return {"enabled": False}
    status = runtime.status
    last_state = status.last_state
    # last_error_type はランタイム内部の型名ではなく allowlist の固定コードへ写像する。
    # エラーが無い (None) ときは None のまま維持し、未知/カスタム型名は internal_error へ落とす。
    last_error_type = status.last_error_type
    if last_error_type is not None:
        last_error_type = sensor_error_code_from_name(last_error_type)
    payload: dict = {
        "enabled": True,
        "running": status.running,
        "last_update_at": status.last_update_at,
        "failure_count": status.failure_count,
        "consecutive_failures": status.consecutive_failures,
        "last_error_type": last_error_type,
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
