"""Persistent desktop-only settings with no Qt dependency."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path


def default_settings_path() -> Path:
    if os.name == "nt":
        root = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        root = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return root / "SUBPC BUDDY" / "desktop.json"


@dataclass
class DesktopSettings:
    server_url: str = "http://127.0.0.1:8000"
    session_id: str = ""
    start_hidden: bool = False
    close_to_tray: bool = True
    tts_enabled: bool = False

    @classmethod
    def load(cls, path: str | Path | None = None) -> "DesktopSettings":
        target = Path(path) if path else default_settings_path()
        try:
            raw = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raw = {}
        if not isinstance(raw, dict):
            raw = {}
        values = {key: raw[key] for key in cls.__dataclass_fields__ if key in raw}
        settings = cls(**values)
        env_url = os.environ.get("SUBPC_DESKTOP_SERVER_URL", "").strip()
        if env_url:
            settings.server_url = env_url
        return settings

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path else default_settings_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(".tmp")
        temp.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(target)
        return target
