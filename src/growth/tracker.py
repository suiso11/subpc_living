"""全会話経路で共有する、説明可能な成長イベント台帳。

モデルの重みが毎ターン更新されるとは表現せず、実際に増えた会話例、検索可能な
記憶、評価、修正候補、個人化事実を数える。会話本文は保存しない。
"""
from __future__ import annotations

import json
import math
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo

UTC = timezone.utc
SIGNAL_POINTS = {
    "training_turn": 5,
    "feedback": 15,
    "correction": 35,
    "profile_fact": 25,
    "quest_reward": 0,
}
ASSET_POINT_WEIGHTS = {
    "retrievable_memories": 2,
    "knowledge_items": 5,
    "training_turns": 1,
    "feedback_items": 8,
    "correction_candidates": 25,
    "profile_facts": 10,
    "conversation_summaries": 5,
}


class GrowthTracker:
    """SQLite WALで複数プロセスから成長イベントを追記・集計する。"""

    def __init__(
        self,
        db_path: str | Path = "data/growth/growth.db",
        *,
        timezone_name: str = "Asia/Tokyo",
        busy_timeout_ms: int = 5000,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.tz = ZoneInfo(timezone_name)
        self.busy_timeout_ms = busy_timeout_ms
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=self.busy_timeout_ms / 1000.0,
            isolation_level=None,
        )
        conn.row_factory = sqlite3.Row
        conn.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS growth_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_key TEXT NOT NULL UNIQUE,
                    occurred_at TEXT NOT NULL,
                    local_date TEXT NOT NULL,
                    source TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    points INTEGER NOT NULL,
                    memory_saved INTEGER NOT NULL DEFAULT 0,
                    user_chars INTEGER NOT NULL DEFAULT 0,
                    assistant_chars INTEGER NOT NULL DEFAULT 0,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_growth_date ON growth_events(local_date)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_growth_kind ON growth_events(kind)"
            )

    def _insert(
        self,
        *,
        event_key: str,
        source: str,
        kind: str,
        points: int,
        memory_saved: bool,
        user_chars: int,
        assistant_chars: int,
        metadata: Optional[dict[str, Any]],
        now: Optional[datetime],
    ) -> bool:
        now = now or datetime.now(UTC)
        if now.tzinfo is None:
            now = now.replace(tzinfo=UTC)
        now = now.astimezone(UTC)
        local_date = now.astimezone(self.tz).date().isoformat()
        safe_metadata = metadata if isinstance(metadata, dict) else {}
        try:
            metadata_json = json.dumps(
                safe_metadata, ensure_ascii=False, separators=(",", ":"), sort_keys=True
            )
        except (TypeError, ValueError):
            metadata_json = "{}"
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                cur = conn.execute(
                    """
                    INSERT OR IGNORE INTO growth_events
                        (event_key, occurred_at, local_date, source, kind, points,
                         memory_saved, user_chars, assistant_chars, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event_key,
                        now.isoformat(),
                        local_date,
                        (source or "unknown")[:80],
                        kind,
                        max(0, int(points)),
                        1 if memory_saved else 0,
                        max(0, int(user_chars)),
                        max(0, int(assistant_chars)),
                        metadata_json,
                    ),
                )
                conn.execute("COMMIT")
                return cur.rowcount == 1
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def record_conversation(
        self,
        *,
        source: str,
        session_id: str,
        user_chars: int,
        assistant_chars: int,
        memory_saved: bool,
        event_key: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> bool:
        """成功した1往復を記録。検索可能なRAG保存成功時は追加点。"""
        return self._insert(
            event_key=event_key or f"conversation:{uuid.uuid4().hex}",
            source=source,
            kind="conversation",
            points=10 + (10 if memory_saved else 0),
            memory_saved=memory_saved,
            user_chars=user_chars,
            assistant_chars=assistant_chars,
            metadata={"session_id": str(session_id)[:128]},
            now=now,
        )

    def record_signal(
        self,
        *,
        kind: str,
        source: str,
        event_key: str,
        points: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None,
        now: Optional[datetime] = None,
    ) -> bool:
        """評価・修正・訓練例・個人化事実の増加を重複なく記録。"""
        if kind not in SIGNAL_POINTS:
            raise ValueError(f"unsupported growth signal: {kind}")
        return self._insert(
            event_key=event_key,
            source=source,
            kind=kind,
            points=SIGNAL_POINTS[kind] if points is None else points,
            memory_saved=False,
            user_chars=0,
            assistant_chars=0,
            metadata=metadata,
            now=now,
        )

    def existing_event_keys(self, event_keys: list[str]) -> set[str]:
        """指定したイベントキーのうち、すでに記録済みのものを返す。"""
        safe_keys = [str(key) for key in event_keys if key]
        if not safe_keys:
            return set()
        placeholders = ",".join("?" for _ in safe_keys)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT event_key FROM growth_events WHERE event_key IN ({placeholders})",
                safe_keys,
            ).fetchall()
        return {str(row["event_key"]) for row in rows}

    @staticmethod
    def _level(points: int) -> tuple[int, int, int]:
        # level N の開始点は 100*(N-1)^2。
        level = int(math.sqrt(max(0, points) / 100)) + 1
        start = 100 * (level - 1) ** 2
        end = 100 * level**2
        progress = 0 if end == start else int((points - start) * 100 / (end - start))
        return level, max(0, min(100, progress)), end

    def summary(
        self,
        *,
        now: Optional[datetime] = None,
        days: int = 14,
        asset_counts: Optional[dict[str, int]] = None,
    ) -> dict[str, Any]:
        now = now or datetime.now(UTC)
        if now.tzinfo is None:
            now = now.replace(tzinfo=UTC)
        local_today = now.astimezone(self.tz).date()
        days = max(1, min(int(days), 90))
        first_day = local_today - timedelta(days=days - 1)

        with self._connect() as conn:
            totals = conn.execute(
                """
                SELECT COALESCE(SUM(points), 0) AS points,
                       SUM(CASE WHEN kind='conversation' THEN 1 ELSE 0 END) AS turns,
                       SUM(CASE WHEN kind='conversation' AND memory_saved=1 THEN 1 ELSE 0 END) AS memory_turns,
                       MIN(occurred_at) AS started_at
                FROM growth_events
                """
            ).fetchone()
            signal_rows = conn.execute(
                "SELECT kind, COUNT(*) AS n FROM growth_events "
                "WHERE kind != 'conversation' GROUP BY kind"
            ).fetchall()
            source_rows = conn.execute(
                "SELECT source, COUNT(*) AS n FROM growth_events GROUP BY source ORDER BY n DESC"
            ).fetchall()
            daily_rows = conn.execute(
                """
                SELECT local_date, SUM(points) AS points,
                       SUM(CASE WHEN kind='conversation' THEN 1 ELSE 0 END) AS turns,
                       SUM(CASE WHEN kind='conversation' AND memory_saved=1 THEN 1 ELSE 0 END) AS memory_turns,
                       SUM(CASE WHEN kind='conversation' THEN user_chars + assistant_chars ELSE 0 END) AS chars
                FROM growth_events WHERE local_date >= ?
                GROUP BY local_date ORDER BY local_date
                """,
                (first_day.isoformat(),),
            ).fetchall()

        daily_map = {
            row["local_date"]: {
                "points": int(row["points"]),
                "turns": int(row["turns"]),
                "memory_turns": int(row["memory_turns"]),
                "chars": int(row["chars"]),
            }
            for row in daily_rows
        }
        daily = []
        for offset in range(days):
            current = first_day + timedelta(days=offset)
            values = daily_map.get(
                current.isoformat(),
                {"points": 0, "turns": 0, "memory_turns": 0, "chars": 0},
            )
            daily.append({"date": current.isoformat(), **values})

        streak = 0
        cursor = local_today
        if daily_map.get(cursor.isoformat(), {}).get("turns", 0) == 0:
            cursor -= timedelta(days=1)
        while daily_map.get(cursor.isoformat(), {}).get("turns", 0) > 0:
            streak += 1
            cursor -= timedelta(days=1)

        tracked_points = int(totals["points"] or 0)
        safe_asset_counts = {
            name: max(0, int((asset_counts or {}).get(name, 0)))
            for name in ASSET_POINT_WEIGHTS
        }
        asset_points = sum(
            safe_asset_counts[name] * weight
            for name, weight in ASSET_POINT_WEIGHTS.items()
        )
        points = tracked_points + asset_points
        level, level_progress, next_level = self._level(points)
        today = daily_map.get(
            local_today.isoformat(),
            {"points": 0, "turns": 0, "memory_turns": 0, "chars": 0},
        )
        signals = {name: 0 for name in SIGNAL_POINTS}
        signals.update({row["kind"]: int(row["n"]) for row in signal_rows})
        return {
            "growth_points": points,
            "tracked_points": tracked_points,
            "asset_points": asset_points,
            "total_turns": int(totals["turns"] or 0),
            "memory_turns": int(totals["memory_turns"] or 0),
            "signals": signals,
            "today_points": int(today["points"]),
            "today_turns": int(today["turns"]),
            "today_memory_turns": int(today["memory_turns"]),
            "today_chars": int(today["chars"]),
            "streak_days": streak,
            "level": level,
            "level_progress": level_progress,
            "next_level_points": next_level,
            "daily": daily,
            "sources": {row["source"]: int(row["n"]) for row in source_rows},
            "tracking_started_at": totals["started_at"],
            "asset_counts": safe_asset_counts,
        }
