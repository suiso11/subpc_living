"""センサー opt-in 方針の共有・不変 (immutable) 解決器。

カメラ・画面・活動など機微なセンサーは既定で無効とし、明示的なオプトイン
(canonical 環境変数の `true`) でのみ有効化する。全入口 (Web / Discord / Voice /
CLI / Desktop) がこの単一解決器を共有する。

- 既定は全て False (safe default / fail closed)。
- 有効化は canonical 名の明示 `true` のみ。それ以外の値・未設定・不正値は False。
- 後方互換: canonical 名が存在しないときだけ legacy 名
  (`WEB_SCREEN_CONTEXT_ENABLED` → screen_capture、`COMPANION_ACTIVITY_ENABLED` →
  activity) を参照する。canonical 名が存在する場合はその値が確定値で、false は
  legacy の true を上書きする。
- token (`SCREEN_INGEST_TOKEN`) の存在だけでは screen_ingest を有効化しない。
- 公開 payload / lookup は boolean とセンサー source 名のみ。env 名・env 値・
  token・secret は一切含めない。

グローバル状態を持たず、`resolve_sensor_policy` が env から一度だけ解決して
frozen な `SensorPolicy` を返す。解決後は env の変更に影響されない。
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, get_args

SensorId = Literal[
    "camera",
    "screen_capture",
    "screen_ingest",
    "activity",
    "monitor",
    "microphone",
    "process_details",
]

SENSOR_IDS: tuple[str, ...] = get_args(SensorId)
VALID_SENSOR_IDS: frozenset[str] = frozenset(SENSOR_IDS)

CANONICAL_ENV_NAMES: dict[str, str] = {
    "camera": "SENSOR_CAMERA_ENABLED",
    "screen_capture": "SENSOR_SCREEN_CAPTURE_ENABLED",
    "screen_ingest": "SENSOR_SCREEN_INGEST_ENABLED",
    "activity": "SENSOR_ACTIVITY_ENABLED",
    "monitor": "SENSOR_MONITOR_ENABLED",
    "microphone": "SENSOR_MICROPHONE_ENABLED",
    "process_details": "SENSOR_PROCESS_DETAILS_ENABLED",
}

LEGACY_ENV_ALIASES: dict[str, str] = {
    "screen_capture": "WEB_SCREEN_CONTEXT_ENABLED",
    "activity": "COMPANION_ACTIVITY_ENABLED",
}


def parse_opt_in(value: object) -> bool:
    """明示的な `true` だけを True とみなす fail-closed parser。

    前後空白を除き、大小文字を無視して `true` と完全一致するときだけ True。
    None・非文字列・その他の値 (`1` / `yes` / `on` / `false` / 空) は False。
    不正値で例外を上げず、必ず False に倒れる (fail closed)。
    """
    if not isinstance(value, str):
        return False
    return value.strip().lower() == "true"


@dataclass(frozen=True)
class SensorPolicy:
    """解決済みのセンサー opt-in 方針。不変 (frozen)。

    Boolean の status payload (`as_status_payload`) と typed lookup (`is_enabled`)
    だけを提供し、env 名・env 値・token・secret は公開しない。
    """

    camera: bool = False
    screen_capture: bool = False
    screen_ingest: bool = False
    activity: bool = False
    monitor: bool = False
    microphone: bool = False
    process_details: bool = False

    def __post_init__(self) -> None:
        for sensor in SENSOR_IDS:
            value = getattr(self, sensor)
            if not isinstance(value, bool):
                raise TypeError(
                    f"{sensor} must be a bool, got {type(value).__name__}"
                )

    def is_enabled(self, sensor: str) -> bool:
        """sensor の opt-in 状態を返す typed lookup。未知 sensor は ValueError。"""
        if sensor not in VALID_SENSOR_IDS:
            raise ValueError(
                f"unknown sensor: {sensor!r} (valid: {', '.join(SENSOR_IDS)})"
            )
        return bool(getattr(self, sensor))

    def enabled_sensor_ids(self) -> tuple[str, ...]:
        """有効化されたセンサー source 名の tuple (宣言順)。"""
        return tuple(sensor for sensor in SENSOR_IDS if getattr(self, sensor))

    def as_status_payload(self) -> dict[str, bool]:
        """センサー source 名 → bool の status payload。

        env 名・env 値・token・secret は含めない。boolean と sensor source 名のみ。
        """
        return {sensor: bool(getattr(self, sensor)) for sensor in SENSOR_IDS}


def resolve_sensor_policy(env: Mapping[str, str] | None = None) -> SensorPolicy:
    """env から SensorPolicy を不変に解決する。

    env は None なら os.environ を使う。canonical 名が存在すればその値が確定値で、
    存在しなければ legacy 名 (screen_capture: WEB_SCREEN_CONTEXT_ENABLED /
    activity: COMPANION_ACTIVITY_ENABLED) を参照する。token の存在はどのセンサーも
    有効化しない。
    """
    resolved = os.environ if env is None else env
    flags: dict[str, bool] = {}
    for sensor in SENSOR_IDS:
        canonical = CANONICAL_ENV_NAMES[sensor]
        if canonical in resolved:
            flags[sensor] = parse_opt_in(resolved[canonical])
        else:
            legacy = LEGACY_ENV_ALIASES.get(sensor)
            if legacy is not None and legacy in resolved:
                flags[sensor] = parse_opt_in(resolved[legacy])
            else:
                flags[sensor] = False
    return SensorPolicy(**flags)