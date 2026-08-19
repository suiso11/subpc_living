"""本文を保存しないAssistant経路・実行ログ。"""

from __future__ import annotations

from contextlib import closing
import json
from pathlib import Path
import re
import sqlite3
import threading
import time
from typing import Protocol

from src.llm.routing.contracts import RouteDecision

_ALLOWED_ERROR_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
MAX_ERROR_LENGTH = 64
REDACTED_ERROR = "redacted_error"


def _sanitize_error(error: str | None) -> str | None:
    """安定したcodeのみ保存し、許可形式外はredacted_errorへ置換する。"""
    if error is None:
        return None
    if not _ALLOWED_ERROR_RE.match(error):
        return REDACTED_ERROR
    return error[:MAX_ERROR_LENGTH]


class RunLogger(Protocol):
    """AssistantServiceから受け取る最小限の実行ログ契約。"""

    def record_route(
        self,
        request_id: str,
        *,
        channel: str,
        profile: str,
        decision: RouteDecision,
    ) -> None:
        """最初に決定した経路を記録する。"""
        ...

    def record_run(
        self,
        request_id: str,
        *,
        channel: str,
        profile: str,
        route: RouteDecision | None,
        latency_ms: int,
        success: bool,
        error: str | None,
    ) -> None:
        """生成の最終結果を記録する。"""
        ...


class SQLiteRunLogger:
    """経路と結果だけをSQLiteへ保存するfirst-write-wins logger。"""

    def __init__(
        self,
        db_path: str | Path = "data/assistant/model_runs.db",
        *,
        busy_timeout_ms: int = 5_000,
        clock=time.time,
    ) -> None:
        self.db_path = Path(db_path)
        self.busy_timeout_ms = max(0, int(busy_timeout_ms))
        self._clock = clock
        self._lock = threading.Lock()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            str(self.db_path), timeout=self.busy_timeout_ms / 1000.0
        )
        connection.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
        return connection

    def _initialize(self) -> None:
        with self._lock, closing(self._connect()) as connection:
            mode = connection.execute("PRAGMA journal_mode=WAL").fetchone()
            if mode is None or str(mode[0]).lower() != "wal":
                raise sqlite3.OperationalError("journal_mode=WAL is not enabled")
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS route_decisions (
                    request_id TEXT PRIMARY KEY,
                    channel TEXT NOT NULL,
                    profile TEXT NOT NULL,
                    provider_id TEXT NOT NULL,
                    model TEXT NOT NULL,
                    local INTEGER NOT NULL,
                    reason TEXT NOT NULL,
                    fallback_provider_ids TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS model_runs (
                    request_id TEXT PRIMARY KEY,
                    channel TEXT NOT NULL,
                    profile TEXT NOT NULL,
                    provider_id TEXT,
                    model TEXT,
                    local INTEGER,
                    latency_ms INTEGER NOT NULL,
                    success INTEGER NOT NULL,
                    error TEXT,
                    created_at REAL NOT NULL
                );
                """
            )
            connection.commit()

    def record_route(
        self,
        request_id: str,
        *,
        channel: str,
        profile: str,
        decision: RouteDecision,
    ) -> None:
        with self._lock, closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO route_decisions (
                    request_id, channel, profile, provider_id, model, local,
                    reason, fallback_provider_ids, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    channel,
                    profile,
                    decision.provider_id,
                    decision.model,
                    int(decision.local),
                    decision.reason,
                    json.dumps(decision.fallback_provider_ids),
                    float(self._clock()),
                ),
            )
            connection.commit()

    def record_run(
        self,
        request_id: str,
        *,
        channel: str,
        profile: str,
        route: RouteDecision | None,
        latency_ms: int,
        success: bool,
        error: str | None,
    ) -> None:
        with self._lock, closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO model_runs (
                    request_id, channel, profile, provider_id, model, local,
                    latency_ms, success, error, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    channel,
                    profile,
                    route.provider_id if route else None,
                    route.model if route else None,
                    int(route.local) if route else None,
                    max(0, int(latency_ms)),
                    int(success),
                    _sanitize_error(error),
                    float(self._clock()),
                ),
            )
            connection.commit()
