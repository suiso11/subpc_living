"""会話履歴 (data/chat_history/*.json) の一覧・閲覧・削除・整理。"""
from __future__ import annotations

import json
import re
from pathlib import Path

_SESSION_FILE_RE = re.compile(r"^session_[\w.-]+\.json$")
_SAFE_SESSION_ID_RE = re.compile(r"^[\w.-]{1,128}$")


def _is_session_file(name: str) -> bool:
    return bool(_SESSION_FILE_RE.match(name)) and "/" not in name and "\\" not in name


def is_safe_session_id(session_id: str) -> bool:
    """セッションIDとして安全か検証する (path traversal / 空文字 / 記号を拒否)。"""
    if not isinstance(session_id, str) or not session_id:
        return False
    return bool(_SAFE_SESSION_ID_RE.match(session_id))


def session_file_for(history_dir: str | Path, session_id: str) -> Path | None:
    """安全なIDの保存先を返す。不正IDでは None。"""
    if not is_safe_session_id(session_id):
        return None
    return Path(history_dir) / f"session_{session_id}.json"


def read_session_by_id(history_dir: str | Path, session_id: str) -> dict | None:
    """安全な session_id から session_<id>.json を読み取る。不正・不在・破損は None。"""
    if not is_safe_session_id(session_id):
        return None
    path = session_file_for(history_dir, session_id)
    if path is None or not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict) or data.get("session_id") != session_id:
        return None
    return data


def read_latest_valid_session(history_dir: str | Path) -> dict | None:
    """履歴ディレクトリから最新の有効（読取可能な）セッションを返す。無ければ None。"""
    directory = Path(history_dir)
    if not directory.is_dir():
        return None
    candidates: list[tuple[str, dict]] = []
    for path in directory.glob("session_*.json"):
        if not _is_session_file(path.name):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        session_id = data.get("session_id") if isinstance(data, dict) else None
        expected_path = session_file_for(directory, session_id)
        if expected_path is None or expected_path != path:
            continue
        sort_key = data.get("saved_at") or data.get("created_at") or ""
        candidates.append((sort_key, data))
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][1]


def list_sessions(history_dir: str | Path) -> list[dict]:
    """履歴ファイルの一覧をメタデータ付きで返す (新しい順)。"""
    directory = Path(history_dir)
    if not directory.is_dir():
        return []
    entries = []
    for path in directory.glob("session_*.json"):
        entry = {
            "file": path.name,
            "size_bytes": path.stat().st_size,
            "session_id": None,
            "created_at": None,
            "saved_at": None,
            "turn_count": None,
            "preview": "",
        }
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            entry["session_id"] = data.get("session_id")
            entry["created_at"] = data.get("created_at")
            entry["saved_at"] = data.get("saved_at")
            entry["turn_count"] = data.get("turn_count")
            for message in data.get("messages", []):
                if message.get("role") == "user" and message.get("content"):
                    entry["preview"] = str(message["content"])[:80]
                    break
        except (json.JSONDecodeError, OSError):
            entry["preview"] = "(読み込み不可)"
        entries.append(entry)
    entries.sort(key=lambda e: e["saved_at"] or e["created_at"] or "", reverse=True)
    return entries


def read_session(history_dir: str | Path, filename: str) -> dict | None:
    """履歴ファイルの中身を返す。不正なファイル名は None。"""
    if not _is_session_file(filename):
        return None
    path = Path(history_dir) / filename
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def delete_session(history_dir: str | Path, filename: str) -> bool:
    """履歴ファイルを削除する。成功したら True。"""
    if not _is_session_file(filename):
        return False
    path = Path(history_dir) / filename
    if not path.is_file():
        return False
    path.unlink()
    return True


def prune_sessions(history_dir: str | Path, max_files: int) -> int:
    """古い履歴ファイルを削除して max_files 件までに抑える。削除件数を返す。"""
    if max_files <= 0:
        return 0
    directory = Path(history_dir)
    if not directory.is_dir():
        return 0
    files = sorted(
        directory.glob("session_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    removed = 0
    for path in files[max_files:]:
        try:
            path.unlink()
            removed += 1
        except OSError:
            pass
    return removed
