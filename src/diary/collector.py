"""Collect local signals used for daily diary generation."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import date, datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from src.integrations.google_calendar import GoogleCalendarMCPClient


@dataclass(frozen=True)
class DiarySources:
    target_date: str
    timezone: str
    generated_at: str
    calendar: dict[str, Any]
    manual_schedule: list[dict[str, Any]]
    discord_turns: list[dict[str, Any]]
    recent_summaries: list[dict[str, Any]]
    metrics_summary: dict[str, Any]
    profile: dict[str, Any]


class DiaryCollector:
    """Load diary source data from local stores and optional Google Calendar."""

    def __init__(
        self,
        project_root: str | Path,
        *,
        calendar_client: GoogleCalendarMCPClient | None = None,
        timezone: str = "Asia/Tokyo",
    ):
        self.project_root = Path(project_root)
        self.calendar_client = calendar_client
        self.timezone = timezone

    def collect(
        self,
        target_date: date,
        *,
        calendar_id: str | list[str] = "primary",
        calendar_account: str | list[str] | None = None,
        include_calendar: bool = True,
    ) -> DiarySources:
        profile = self._load_profile()
        calendar = self._collect_calendar(
            target_date,
            calendar_id=calendar_id,
            calendar_account=calendar_account,
            include_calendar=include_calendar,
        )
        return DiarySources(
            target_date=target_date.isoformat(),
            timezone=self.timezone,
            generated_at=datetime.now(ZoneInfo(self.timezone)).isoformat(timespec="seconds"),
            calendar=calendar,
            manual_schedule=self._manual_schedule_for_day(profile, target_date),
            discord_turns=self._load_discord_turns(target_date),
            recent_summaries=self._load_recent_summaries(limit=8),
            metrics_summary=self._load_metrics_summary(target_date),
            profile=self._profile_digest(profile),
        )

    def _collect_calendar(
        self,
        target_date: date,
        *,
        calendar_id: str | list[str],
        calendar_account: str | list[str] | None,
        include_calendar: bool,
    ) -> dict[str, Any]:
        if not include_calendar:
            return {"enabled": False, "events": [], "error": ""}
        if self.calendar_client is None:
            return {"enabled": False, "events": [], "error": "calendar client is not configured"}

        result = self.calendar_client.list_events_for_day(
            target_date,
            timezone=self.timezone,
            calendar_id=calendar_id,
            account=calendar_account,
        )
        return {
            "enabled": True,
            "ok": result.ok,
            "source": result.source,
            "error": result.error,
            "events": [asdict(event) for event in result.events],
        }

    def _load_profile(self) -> dict[str, Any]:
        path = self.project_root / "data" / "profile" / "user_profile.json"
        if not path.exists():
            return {}
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _manual_schedule_for_day(profile: dict[str, Any], target_date: date) -> list[dict[str, Any]]:
        schedule = profile.get("schedule", [])
        if not isinstance(schedule, list):
            return []
        day = target_date.isoformat()
        return [
            item
            for item in schedule
            if isinstance(item, dict) and str(item.get("date", "")) == day
        ]

    def _load_discord_turns(self, target_date: date, *, limit: int = 80) -> list[dict[str, Any]]:
        path = self.project_root / "data" / "discord_training" / "conversations.jsonl"
        if not path.exists():
            return []

        tz = ZoneInfo(self.timezone)
        turns: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(item, dict):
                    continue
                created_at = self._parse_datetime(item.get("created_at"), tz)
                if created_at is None or created_at.date() != target_date:
                    continue
                turns.append(
                    {
                        "created_at": created_at.isoformat(timespec="seconds"),
                        "channel_id": item.get("channel_id"),
                        "user": self._truncate(str(item.get("user", "")), 500),
                        "assistant": self._truncate(str(item.get("assistant", "")), 500),
                    }
                )

        turns.sort(key=lambda item: str(item.get("created_at", "")))
        return turns[-limit:]

    def _load_recent_summaries(self, *, limit: int = 8) -> list[dict[str, Any]]:
        summaries_dir = self.project_root / "data" / "profile" / "summaries"
        if not summaries_dir.exists():
            return []
        summaries: list[dict[str, Any]] = []
        for path in sorted(summaries_dir.glob("summary_*.json"), reverse=True)[:limit]:
            try:
                with path.open("r", encoding="utf-8") as f:
                    item = json.load(f)
            except Exception:
                continue
            if not isinstance(item, dict):
                continue
            summaries.append(
                {
                    "session_id": item.get("session_id"),
                    "summarized_at": item.get("summarized_at"),
                    "turn_count": item.get("turn_count"),
                    "summary": self._truncate(str(item.get("summary", "")), 600),
                }
            )
        return summaries

    def _load_metrics_summary(self, target_date: date) -> dict[str, Any]:
        db_path = self.project_root / "data" / "metrics" / "system_metrics.db"
        if not db_path.exists():
            return {"available": False, "error": "metrics database not found"}

        tz = ZoneInfo(self.timezone)
        start_ts = datetime.combine(target_date, time.min, tzinfo=tz).timestamp()
        end_ts = datetime.combine(target_date, time.max, tzinfo=tz).timestamp()

        try:
            conn = sqlite3.connect(str(db_path))
            cur = conn.cursor()
            cur.execute(
                """
                SELECT
                    COUNT(*),
                    AVG(cpu_percent),
                    MAX(cpu_percent),
                    AVG(mem_percent),
                    MAX(mem_percent),
                    AVG(gpu_util_percent),
                    MAX(gpu_util_percent),
                    AVG(gpu_power_w),
                    MAX(gpu_temp_c),
                    MAX(cpu_temp_c)
                FROM metrics
                WHERE timestamp >= ? AND timestamp <= ?
                """,
                (start_ts, end_ts),
            )
            row = cur.fetchone()
            conn.close()
        except Exception as exc:
            return {"available": False, "error": str(exc)}

        if not row or row[0] == 0:
            return {"available": True, "sample_count": 0}

        return {
            "available": True,
            "sample_count": row[0],
            "cpu_avg": self._round(row[1]),
            "cpu_max": self._round(row[2]),
            "mem_avg": self._round(row[3]),
            "mem_max": self._round(row[4]),
            "gpu_avg": self._round(row[5]),
            "gpu_max": self._round(row[6]),
            "gpu_power_avg_w": self._round(row[7]),
            "gpu_temp_max_c": self._round(row[8]),
            "cpu_temp_max_c": self._round(row[9]),
        }

    @staticmethod
    def _profile_digest(profile: dict[str, Any]) -> dict[str, Any]:
        return {
            "name": profile.get("name", ""),
            "nickname": profile.get("nickname", ""),
            "habits": profile.get("habits", {}) if isinstance(profile.get("habits"), dict) else {},
            "notes": profile.get("notes", [])[-8:] if isinstance(profile.get("notes"), list) else [],
            "extracted_facts": (
                profile.get("extracted_facts", [])[-12:]
                if isinstance(profile.get("extracted_facts"), list)
                else []
            ),
        }

    @staticmethod
    def _parse_datetime(value: Any, tz: ZoneInfo) -> datetime | None:
        if not isinstance(value, str) or not value:
            return None
        try:
            dt = datetime.fromisoformat(value)
        except ValueError:
            return None
        if dt.tzinfo is None:
            return dt.replace(tzinfo=tz)
        return dt.astimezone(tz)

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        text = text.strip()
        if len(text) <= limit:
            return text
        return text[: limit - 1].rstrip() + "…"

    @staticmethod
    def _round(value: Any) -> float | None:
        if value is None:
            return None
        return round(float(value), 1)
