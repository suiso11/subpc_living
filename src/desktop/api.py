"""Small, testable HTTP client used by the native QML frontend."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import httpx


class DesktopApiError(RuntimeError):
    """A backend request failed with a user-displayable message."""


@dataclass(frozen=True)
class BackendAddress:
    http: str
    websocket: str


def normalize_server_url(value: str) -> BackendAddress:
    url = str(value or "").strip().rstrip("/")
    if not url:
        url = "http://127.0.0.1:8000"
    if not url.startswith(("http://", "https://")):
        url = f"http://{url}"
    websocket = (
        f"wss://{url.removeprefix('https://')}"
        if url.startswith("https://")
        else f"ws://{url.removeprefix('http://')}"
    )
    return BackendAddress(http=url, websocket=websocket)


class DesktopApi:
    """Synchronous API facade; the Qt bridge runs calls on its thread pool."""

    def __init__(
        self,
        server_url: str = "http://127.0.0.1:8000",
        *,
        timeout: float = 20.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.timeout = timeout
        self.transport = transport
        self._client: httpx.Client | None = None
        self.set_server_url(server_url)

    @property
    def server_url(self) -> str:
        return self.address.http

    @property
    def websocket_url(self) -> str:
        return f"{self.address.websocket}/ws/chat"

    def set_server_url(self, value: str) -> None:
        address = normalize_server_url(value)
        old = self._client
        self.address = address
        self._client = httpx.Client(
            base_url=address.http,
            timeout=self.timeout,
            transport=self.transport,
            headers={"User-Agent": "SUBPC-BUDDY-Desktop/1.0"},
        )
        if old is not None:
            old.close()

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        if self._client is None:
            raise DesktopApiError("バックエンド接続は終了しています")
        try:
            response = self._client.request(method, path, **kwargs)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as exc:
            try:
                detail = exc.response.json().get("error")
            except Exception:
                detail = None
            raise DesktopApiError(detail or f"HTTP {exc.response.status_code}") from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise DesktopApiError(f"バックエンドに接続できません: {exc}") from exc
        if not isinstance(data, dict):
            raise DesktopApiError("バックエンドから不正な応答を受け取りました")
        return data

    def status(self) -> dict[str, Any]:
        return self._request("GET", "/api/status")

    def resume_chat(self, session_id: str | None = None) -> dict[str, Any]:
        params = {"session_id": session_id} if session_id else None
        return self._request("GET", "/api/chat/resume", params=params)

    def tasks(self, status: str = "open") -> dict[str, Any]:
        return self._request("GET", "/api/tasks", params={"status": status, "limit": 200})

    def add_task(self, text: str, priority: str = "normal", note: str = "") -> dict[str, Any]:
        payload: dict[str, Any] = {"text": text, "priority": priority}
        if note.strip():
            payload["note"] = note.strip()
        return self._request("POST", "/api/tasks", json=payload)

    def update_task(self, task_id: int, fields: dict[str, Any]) -> dict[str, Any]:
        return self._request("PATCH", f"/api/tasks/{int(task_id)}", json=fields)

    def complete_task(self, task_id: int) -> dict[str, Any]:
        return self._request("POST", f"/api/tasks/{int(task_id)}/done")

    def drop_task(self, task_id: int) -> dict[str, Any]:
        return self._request("POST", f"/api/tasks/{int(task_id)}/drop")

    def regenerate_task(self, task_id: int) -> dict[str, Any]:
        return self._request("POST", f"/api/tasks/{int(task_id)}/breakdown")

    def game(self) -> dict[str, Any]:
        return self._request("GET", "/api/game")

    def claim_mission(self, mission_id: str) -> dict[str, Any]:
        return self._request("POST", "/api/game/claim", json={"mission_id": mission_id})

    def journal(self, unit: str = "subpc-web", lines: int = 200) -> dict[str, Any]:
        return self._request(
            "GET",
            "/api/logs/journal",
            params={"unit": unit, "lines": max(10, min(int(lines), 1000))},
        )

    def histories(self) -> dict[str, Any]:
        return self._request("GET", "/api/history/sessions")

    def history(self, filename: str) -> dict[str, Any]:
        return self._request("GET", f"/api/history/sessions/{quote(filename, safe='')}")

    def delete_history(self, filename: str) -> dict[str, Any]:
        return self._request("DELETE", f"/api/history/sessions/{quote(filename, safe='')}")
