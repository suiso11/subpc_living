"""Google Calendar access through the existing MCP server."""
from __future__ import annotations

import json
import os
import shlex
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Sequence
from zoneinfo import ZoneInfo

from src.integrations.mcp_stdio import MCPStdioClient, MCPStdioError


DEFAULT_GOOGLE_OAUTH_CREDENTIALS = Path.home() / ".config" / "google-calendar-mcp" / "gcp-oauth.keys.json"
DEFAULT_GOOGLE_CALENDAR_MCP_COMMAND = ("npx", "-y", "@cocal/google-calendar-mcp")


@dataclass(frozen=True)
class CalendarEvent:
    title: str
    start: str
    end: str
    calendar_id: str = ""
    account_id: str = ""
    location: str = ""
    description: str = ""
    status: str = ""
    html_link: str = ""

    @property
    def sort_key(self) -> str:
        return self.start or self.end or self.title

    @classmethod
    def from_mcp_event(cls, raw: dict[str, Any]) -> "CalendarEvent":
        start_obj = raw.get("start") if isinstance(raw.get("start"), dict) else {}
        end_obj = raw.get("end") if isinstance(raw.get("end"), dict) else {}
        return cls(
            title=str(raw.get("summary") or "(無題)"),
            start=str(start_obj.get("dateTime") or start_obj.get("date") or ""),
            end=str(end_obj.get("dateTime") or end_obj.get("date") or ""),
            calendar_id=str(raw.get("calendarId") or ""),
            account_id=str(raw.get("accountId") or ""),
            location=str(raw.get("location") or ""),
            description=str(raw.get("description") or ""),
            status=str(raw.get("status") or ""),
            html_link=str(raw.get("htmlLink") or ""),
        )


@dataclass(frozen=True)
class CalendarFetchResult:
    events: list[CalendarEvent]
    error: str = ""
    source: str = "google-calendar-mcp"

    @property
    def ok(self) -> bool:
        return not self.error


class GoogleCalendarMCPClient:
    """Read calendar events via @cocal/google-calendar-mcp."""

    def __init__(
        self,
        *,
        command: Sequence[str] = DEFAULT_GOOGLE_CALENDAR_MCP_COMMAND,
        credentials_path: str | Path | None = None,
        token_path: str | Path | None = None,
        timeout_sec: float = 60.0,
    ):
        self.command = list(command)
        self.credentials_path = Path(
            credentials_path
            or os.environ.get("GOOGLE_OAUTH_CREDENTIALS", str(DEFAULT_GOOGLE_OAUTH_CREDENTIALS))
        ).expanduser()
        token_value = token_path or os.environ.get("GOOGLE_CALENDAR_MCP_TOKEN_PATH", "")
        self.token_path = Path(token_value).expanduser() if token_value else None
        self.timeout_sec = timeout_sec

    @classmethod
    def from_env(cls) -> "GoogleCalendarMCPClient":
        command_text = os.environ.get("GOOGLE_CALENDAR_MCP_COMMAND", "").strip()
        command = shlex.split(command_text) if command_text else DEFAULT_GOOGLE_CALENDAR_MCP_COMMAND
        timeout = float(os.environ.get("GOOGLE_CALENDAR_MCP_TIMEOUT_SEC", "60"))
        return cls(command=command, timeout_sec=timeout)

    def list_events_for_day(
        self,
        target_date: date,
        *,
        timezone: str = "Asia/Tokyo",
        calendar_id: str | list[str] = "primary",
        account: str | list[str] | None = None,
    ) -> CalendarFetchResult:
        if not self.credentials_path.exists():
            return CalendarFetchResult(
                events=[],
                error=(
                    "Google OAuth credentials not found: "
                    f"{self.credentials_path}. Run Google Calendar MCP auth first."
                ),
            )

        tz = ZoneInfo(timezone)
        start_dt = datetime.combine(target_date, time.min, tzinfo=tz)
        end_dt = start_dt + timedelta(days=1)
        arguments: dict[str, Any] = {
            "calendarId": calendar_id,
            "timeMin": start_dt.replace(tzinfo=None).isoformat(timespec="seconds"),
            "timeMax": end_dt.replace(tzinfo=None).isoformat(timespec="seconds"),
            "timeZone": timezone,
        }
        if account:
            arguments["account"] = account

        env = {
            "GOOGLE_OAUTH_CREDENTIALS": str(self.credentials_path),
            "ENABLED_TOOLS": "list-events,list-calendars,get-current-time",
        }
        if self.token_path is not None:
            env["GOOGLE_CALENDAR_MCP_TOKEN_PATH"] = str(self.token_path)

        try:
            result = MCPStdioClient(
                self.command,
                env=env,
                timeout_sec=self.timeout_sec,
            ).call_tool("list-events", arguments)
        except MCPStdioError as exc:
            return CalendarFetchResult(events=[], error=str(exc))

        try:
            data = self._extract_json_payload(result)
            raw_events = data.get("events", [])
            if not isinstance(raw_events, list):
                raw_events = []
            events = [
                CalendarEvent.from_mcp_event(event)
                for event in raw_events
                if isinstance(event, dict)
            ]
            events.sort(key=lambda event: event.sort_key)
            return CalendarFetchResult(events=events)
        except Exception as exc:
            return CalendarFetchResult(events=[], error=f"Failed to parse Google Calendar MCP response: {exc}")

    @staticmethod
    def _extract_json_payload(result: dict[str, Any]) -> dict[str, Any]:
        structured = result.get("structuredContent")
        if isinstance(structured, dict):
            return structured

        content = result.get("content", [])
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    parsed = json.loads(item["text"])
                    if isinstance(parsed, dict):
                        return parsed
        return {}
