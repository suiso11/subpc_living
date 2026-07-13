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
    event_id: str = ""

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
            event_id=str(raw.get("id") or raw.get("eventId") or ""),
        )


@dataclass(frozen=True)
class CalendarMutationResult:
    """create/update/delete イベントの結果。"""

    ok: bool
    event_id: str = ""
    error: str = ""
    source: str = "google-calendar-mcp"


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

    # --- 書き込み系 (create/update/delete) と範囲取得 ---

    _READ_TOOLS = "list-events,list-calendars,get-current-time"
    _WRITE_TOOLS = "create-event,update-event,delete-event,list-events,list-calendars,get-current-time"

    def _env(self, enabled_tools: str) -> dict[str, str]:
        env = {
            "GOOGLE_OAUTH_CREDENTIALS": str(self.credentials_path),
            "ENABLED_TOOLS": enabled_tools,
        }
        if self.token_path is not None:
            env["GOOGLE_CALENDAR_MCP_TOKEN_PATH"] = str(self.token_path)
        return env

    def _credentials_error(self) -> str:
        return (
            "Google OAuth credentials not found: "
            f"{self.credentials_path}. Run Google Calendar MCP auth first."
        )

    def _call_tool(self, tool: str, arguments: dict[str, Any], enabled_tools: str) -> dict[str, Any]:
        return MCPStdioClient(
            self.command,
            env=self._env(enabled_tools),
            timeout_sec=self.timeout_sec,
        ).call_tool(tool, arguments)

    def create_event(
        self,
        *,
        summary: str,
        start: str,
        end: str,
        calendar_id: str = "primary",
        description: str = "",
        timezone: str = "Asia/Tokyo",
        location: str = "",
    ) -> CalendarMutationResult:
        """イベントを作成し、作成された event_id を返す。

        start/end は ISO8601 文字列。終日イベントは "YYYY-MM-DD"、
        時刻付きイベントはローカルの naive ISO ("YYYY-MM-DDTHH:MM:SS") を渡し、
        timezone (IANA) を併せて指定する。
        """
        if not self.credentials_path.exists():
            return CalendarMutationResult(ok=False, error=self._credentials_error())
        arguments: dict[str, Any] = {
            "calendarId": calendar_id,
            "summary": summary,
            "start": start,
            "end": end,
            "timeZone": timezone,
        }
        if description:
            arguments["description"] = description
        if location:
            arguments["location"] = location
        try:
            result = self._call_tool("create-event", arguments, self._WRITE_TOOLS)
        except MCPStdioError as exc:
            return CalendarMutationResult(ok=False, error=str(exc))
        return CalendarMutationResult(ok=True, event_id=self._extract_event_id(result))

    def update_event(
        self,
        event_id: str,
        *,
        calendar_id: str = "primary",
        summary: str | None = None,
        start: str | None = None,
        end: str | None = None,
        description: str | None = None,
        timezone: str = "Asia/Tokyo",
        location: str | None = None,
    ) -> CalendarMutationResult:
        """既存イベントを更新する。渡したフィールドのみ変更する。"""
        if not self.credentials_path.exists():
            return CalendarMutationResult(ok=False, error=self._credentials_error())
        arguments: dict[str, Any] = {
            "calendarId": calendar_id,
            "eventId": event_id,
            "timeZone": timezone,
        }
        if summary is not None:
            arguments["summary"] = summary
        if start is not None:
            arguments["start"] = start
        if end is not None:
            arguments["end"] = end
        if description is not None:
            arguments["description"] = description
        if location is not None:
            arguments["location"] = location
        try:
            result = self._call_tool("update-event", arguments, self._WRITE_TOOLS)
        except MCPStdioError as exc:
            return CalendarMutationResult(ok=False, error=str(exc))
        found = self._extract_event_id(result)
        return CalendarMutationResult(ok=True, event_id=found or event_id)

    def delete_event(
        self,
        event_id: str,
        *,
        calendar_id: str = "primary",
        send_updates: str = "none",
    ) -> CalendarMutationResult:
        """イベントを削除する。"""
        if not self.credentials_path.exists():
            return CalendarMutationResult(ok=False, error=self._credentials_error())
        arguments: dict[str, Any] = {
            "calendarId": calendar_id,
            "eventId": event_id,
            "sendUpdates": send_updates,
        }
        try:
            self._call_tool("delete-event", arguments, self._WRITE_TOOLS)
        except MCPStdioError as exc:
            return CalendarMutationResult(ok=False, error=str(exc))
        return CalendarMutationResult(ok=True, event_id=event_id)

    def list_events_range(
        self,
        start_date: date,
        end_date: date,
        *,
        timezone: str = "Asia/Tokyo",
        calendar_id: str | list[str] = "primary",
        account: str | list[str] | None = None,
    ) -> CalendarFetchResult:
        """start_date 〜 end_date (両端含む) のイベントを取得する。"""
        if not self.credentials_path.exists():
            return CalendarFetchResult(events=[], error=self._credentials_error())

        tz = ZoneInfo(timezone)
        start_dt = datetime.combine(start_date, time.min, tzinfo=tz)
        # end_date を含めるため timeMax は翌日 0:00。
        end_dt = datetime.combine(end_date, time.min, tzinfo=tz) + timedelta(days=1)
        arguments: dict[str, Any] = {
            "calendarId": calendar_id,
            "timeMin": start_dt.replace(tzinfo=None).isoformat(timespec="seconds"),
            "timeMax": end_dt.replace(tzinfo=None).isoformat(timespec="seconds"),
            "timeZone": timezone,
        }
        if account:
            arguments["account"] = account

        try:
            result = self._call_tool("list-events", arguments, self._READ_TOOLS)
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

    @classmethod
    def _extract_event_id(cls, result: dict[str, Any]) -> str:
        """create/update レスポンスから event id を頑健に抽出する。

        @cocal/google-calendar-mcp の create-event は
        structuredContent = {"event": {...}, "conflicts": ...} を返し、
        event.id に Google のイベントIDが入る想定。念のため複数箇所を探す。
        """
        candidates: list[dict[str, Any]] = []
        structured = result.get("structuredContent")
        if isinstance(structured, dict):
            candidates.append(structured)
            ev = structured.get("event")
            if isinstance(ev, dict):
                candidates.append(ev)
        payload = cls._extract_json_payload(result)
        if payload:
            candidates.append(payload)
            ev = payload.get("event")
            if isinstance(ev, dict):
                candidates.append(ev)
        for cand in candidates:
            for key in ("id", "eventId"):
                value = cand.get(key)
                if isinstance(value, str) and value:
                    return value
        return ""

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
