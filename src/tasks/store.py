"""
タスクストア (SQLite)

Discord bot / 音声パイプラインなど複数プロセスからの同時アクセスを前提に、
src/monitor/storage.py の WAL パターンを踏襲する。

- due_at は UTC ISO で保存し、表示は Asia/Tokyo。
- granularity='date' のタスクは「ローカル23:59締切」に正規化された due_at を持つ
  (正規化はパーサ側 src/discord_bot/task_ui.py が行い、ここでは保存された値を使う)。
- 通知の重複送信を防ぐため、claim_due_notifications() は BEGIN IMMEDIATE で
  lease を取ってから対象行を返す。
"""
from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional
from zoneinfo import ZoneInfo

UTC = timezone.utc
DEFAULT_TZ = "Asia/Tokyo"

VALID_PRIORITY = ("high", "normal", "low")
VALID_STATUS = ("open", "done", "dropped")
VALID_GRANULARITY = ("date", "datetime")
VALID_SOURCE = ("command", "chat", "voice", "context_menu", "board", "web")

# priority の並び順 (数値が小さいほど優先)
_PRIORITY_RANK = {"high": 0, "normal": 1, "low": 2}


def utc_now() -> datetime:
    return datetime.now(UTC)


def to_iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).isoformat()


def from_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


class TaskStore:
    """SQLite ベースのタスクストア。"""

    def __init__(
        self,
        db_path: str = "data/tasks/tasks.db",
        *,
        timezone_name: str = DEFAULT_TZ,
        busy_timeout_ms: int = 5000,
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.timezone_name = timezone_name
        self.busy_timeout_ms = busy_timeout_ms
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()  # プロセス内スレッド間の直列化
        # タスク変更 (add/update/done/drop) をコミット後に通知するフック。
        # カレンダー同期など外部連携がここに subscribe する。全変更経路が
        # TaskStore を通るため、呼び出し側を個別に計装せずに済む。
        # コールバックは (task_id, event) を受け取り、例外は握り潰される。
        self.on_change: Optional[Callable[[int, str], None]] = None

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone_name)

    # --- ライフサイクル ---

    def initialize(self) -> "TaskStore":
        # isolation_level=None で autocommit にし、トランザクションは手動制御する。
        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            isolation_level=None,
            timeout=self.busy_timeout_ms / 1000.0,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()
        return self

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def _require(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("TaskStore 未初期化。initialize() を先に呼んでください。")
        return self._conn

    @contextmanager
    def _tx(self, *, immediate: bool = False):
        """手動トランザクション。immediate=True で BEGIN IMMEDIATE (書込ロック先取り)。"""
        conn = self._require()
        with self._lock:
            conn.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            try:
                yield conn
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def _create_tables(self) -> None:
        with self._tx() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    note TEXT,
                    action_hint TEXT,
                    due_at TEXT,
                    due_granularity TEXT,
                    priority TEXT NOT NULL DEFAULT 'normal',
                    status TEXT NOT NULL DEFAULT 'open',
                    source TEXT NOT NULL DEFAULT 'command',
                    created_at TEXT NOT NULL,
                    completed_at TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tasks_status_due ON tasks (status, due_at)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS task_notifications (
                    task_id INTEGER PRIMARY KEY,
                    last_stage TEXT,
                    last_notified_at TEXT,
                    next_notify_at TEXT,
                    snoozed_until TEXT,
                    repeat_count INTEGER NOT NULL DEFAULT 0,
                    lease_owner TEXT,
                    lease_until TEXT,
                    FOREIGN KEY (task_id) REFERENCES tasks (id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_notif_next ON task_notifications (next_notify_at)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS task_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id INTEGER,
                    event TEXT NOT NULL,
                    detail TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            self._migrate(conn)

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        """既存DBを壊さずにカラムを追加する冪等マイグレーション。"""
        cols = {row["name"] for row in conn.execute("PRAGMA table_info(tasks)").fetchall()}
        if "calendar_event_id" not in cols:
            conn.execute("ALTER TABLE tasks ADD COLUMN calendar_event_id TEXT")
        if "calendar_synced_at" not in cols:
            conn.execute("ALTER TABLE tasks ADD COLUMN calendar_synced_at TEXT")

    # --- 内部ユーティリティ ---

    def _fire_change(self, task_id: int, event: str) -> None:
        """変更フックを呼ぶ。トランザクション外・コミット後に呼ぶこと。"""
        cb = self.on_change
        if cb is None:
            return
        try:
            cb(task_id, event)
        except Exception:
            # 外部連携の失敗でタスク操作自体を失敗させない。
            pass

    @staticmethod
    def _log_event(conn: sqlite3.Connection, task_id: Optional[int], event: str, detail: str, now: datetime) -> None:
        conn.execute(
            "INSERT INTO task_events (task_id, event, detail, created_at) VALUES (?, ?, ?, ?)",
            (task_id, event, detail, to_iso(now)),
        )

    @staticmethod
    def _row_to_task(row: sqlite3.Row) -> dict:
        return {
            "id": row["id"],
            "title": row["title"],
            "note": row["note"],
            "action_hint": row["action_hint"],
            "due_at": from_iso(row["due_at"]),
            "due_granularity": row["due_granularity"],
            "priority": row["priority"],
            "status": row["status"],
            "source": row["source"],
            "created_at": from_iso(row["created_at"]),
            "completed_at": from_iso(row["completed_at"]),
            "calendar_event_id": row["calendar_event_id"],
            "calendar_synced_at": from_iso(row["calendar_synced_at"]),
        }

    # --- CRUD ---

    def add(
        self,
        title: str,
        *,
        note: Optional[str] = None,
        action_hint: Optional[str] = None,
        due_at: Optional[datetime] = None,
        due_granularity: Optional[str] = None,
        priority: str = "normal",
        source: str = "command",
        now: Optional[datetime] = None,
    ) -> int:
        title = (title or "").strip()
        if not title:
            raise ValueError("title は必須です")
        if priority not in VALID_PRIORITY:
            priority = "normal"
        if source not in VALID_SOURCE:
            source = "command"
        if due_at is not None and due_granularity not in VALID_GRANULARITY:
            due_granularity = "datetime"
        now = now or utc_now()

        with self._tx(immediate=True) as conn:
            cur = conn.execute(
                """
                INSERT INTO tasks
                    (title, note, action_hint, due_at, due_granularity,
                     priority, status, source, created_at, completed_at)
                VALUES (?, ?, ?, ?, ?, ?, 'open', ?, ?, NULL)
                """,
                (
                    title, note, action_hint, to_iso(due_at), due_granularity,
                    priority, source, to_iso(now),
                ),
            )
            task_id = int(cur.lastrowid)
            # due_at があれば通知対象。next_notify_at=NULL は「即評価」を意味する。
            conn.execute(
                """
                INSERT INTO task_notifications
                    (task_id, last_stage, last_notified_at, next_notify_at,
                     snoozed_until, repeat_count, lease_owner, lease_until)
                VALUES (?, NULL, NULL, NULL, NULL, 0, NULL, NULL)
                """,
                (task_id,),
            )
            self._log_event(conn, task_id, "add", f"title={title!r} due={to_iso(due_at)}", now)
        self._fire_change(task_id, "add")
        return task_id

    def get(self, task_id: int) -> Optional[dict]:
        conn = self._require()
        with self._lock:
            row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        return self._row_to_task(row) if row else None

    def list(self, status: str = "open", limit: int = 100) -> list[dict]:
        """status のタスクを期限順 (期限ありが先、近い順)、次に優先度・作成順で返す。"""
        conn = self._require()
        with self._lock:
            rows = conn.execute(
                "SELECT * FROM tasks WHERE status = ?",
                (status,),
            ).fetchall()
        tasks = [self._row_to_task(r) for r in rows]

        def sort_key(t: dict):
            has_due = 0 if t["due_at"] is not None else 1
            due_val = t["due_at"] or datetime.max.replace(tzinfo=UTC)
            prio = _PRIORITY_RANK.get(t["priority"], 1)
            return (has_due, due_val, prio, t["id"])

        tasks.sort(key=sort_key)
        return tasks[:limit]

    def update(
        self,
        task_id: int,
        *,
        title: Optional[str] = None,
        note: Optional[str] = None,
        action_hint: Optional[str] = None,
        due_at: Optional[datetime] = None,
        due_granularity: Optional[str] = None,
        priority: Optional[str] = None,
        clear_due: bool = False,
        now: Optional[datetime] = None,
    ) -> bool:
        now = now or utc_now()
        fields: list[str] = []
        params: list[Any] = []
        if title is not None:
            fields.append("title = ?")
            params.append(title.strip())
        if note is not None:
            fields.append("note = ?")
            params.append(note)
        if action_hint is not None:
            fields.append("action_hint = ?")
            params.append(action_hint)
        due_changed = False
        if clear_due:
            fields.append("due_at = NULL")
            fields.append("due_granularity = NULL")
            due_changed = True
        elif due_at is not None:
            fields.append("due_at = ?")
            params.append(to_iso(due_at))
            fields.append("due_granularity = ?")
            params.append(due_granularity if due_granularity in VALID_GRANULARITY else "datetime")
            due_changed = True
        if priority is not None and priority in VALID_PRIORITY:
            fields.append("priority = ?")
            params.append(priority)
        if not fields:
            return False
        params.append(task_id)
        with self._tx(immediate=True) as conn:
            cur = conn.execute(
                f"UPDATE tasks SET {', '.join(fields)} WHERE id = ? AND status = 'open'",
                params,
            )
            changed = cur.rowcount > 0
            if changed and due_changed:
                # 期限を変えたら通知状態をリセットして再評価させる。
                conn.execute(
                    """
                    UPDATE task_notifications
                    SET last_stage = NULL, next_notify_at = NULL, repeat_count = 0,
                        lease_owner = NULL, lease_until = NULL
                    WHERE task_id = ?
                    """,
                    (task_id,),
                )
            if changed:
                self._log_event(conn, task_id, "update", ", ".join(fields), now)
        if changed:
            self._fire_change(task_id, "update")
        return changed

    def done(self, task_id: int, *, now: Optional[datetime] = None) -> bool:
        now = now or utc_now()
        with self._tx(immediate=True) as conn:
            cur = conn.execute(
                "UPDATE tasks SET status = 'done', completed_at = ? WHERE id = ? AND status = 'open'",
                (to_iso(now), task_id),
            )
            changed = cur.rowcount > 0
            if changed:
                self._clear_notifications(conn, task_id)
                self._log_event(conn, task_id, "done", "", now)
        if changed:
            self._fire_change(task_id, "done")
        return changed

    def drop(self, task_id: int, *, now: Optional[datetime] = None) -> bool:
        now = now or utc_now()
        with self._tx(immediate=True) as conn:
            cur = conn.execute(
                "UPDATE tasks SET status = 'dropped', completed_at = ? WHERE id = ? AND status = 'open'",
                (to_iso(now), task_id),
            )
            changed = cur.rowcount > 0
            if changed:
                self._clear_notifications(conn, task_id)
                self._log_event(conn, task_id, "drop", "", now)
        if changed:
            self._fire_change(task_id, "drop")
        return changed

    def snooze(self, task_id: int, until: datetime, *, now: Optional[datetime] = None) -> bool:
        now = now or utc_now()
        with self._tx(immediate=True) as conn:
            row = conn.execute(
                "SELECT status FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
            if row is None or row["status"] != "open":
                return False
            conn.execute(
                """
                UPDATE task_notifications
                SET snoozed_until = ?, next_notify_at = ?, lease_owner = NULL, lease_until = NULL
                WHERE task_id = ?
                """,
                (to_iso(until), to_iso(until), task_id),
            )
            self._log_event(conn, task_id, "snooze", f"until={to_iso(until)}", now)
        return True

    @staticmethod
    def _clear_notifications(conn: sqlite3.Connection, task_id: int) -> None:
        conn.execute(
            """
            UPDATE task_notifications
            SET next_notify_at = NULL, snoozed_until = NULL, lease_owner = NULL, lease_until = NULL
            WHERE task_id = ?
            """,
            (task_id,),
        )

    # --- カレンダー同期マッピング ---

    def set_calendar_event(
        self, task_id: int, event_id: str, *, now: Optional[datetime] = None
    ) -> None:
        """タスクに対応する Google Calendar イベントIDを記録する。"""
        now = now or utc_now()
        with self._tx(immediate=True) as conn:
            conn.execute(
                "UPDATE tasks SET calendar_event_id = ?, calendar_synced_at = ? WHERE id = ?",
                (event_id, to_iso(now), task_id),
            )

    def clear_calendar_event(self, task_id: int, *, now: Optional[datetime] = None) -> None:
        """カレンダーイベントの対応付けを解除する (イベント削除後)。"""
        now = now or utc_now()
        with self._tx(immediate=True) as conn:
            conn.execute(
                "UPDATE tasks SET calendar_event_id = NULL, calendar_synced_at = ? WHERE id = ?",
                (to_iso(now), task_id),
            )

    # --- コンテキスト注入用 ---

    def get_context_tasks(self, limit: int = 8, *, now: Optional[datetime] = None) -> list[dict]:
        """未完了タスクを 期限超過→今日→近日→高優先度 の順で最大 limit 件返す。"""
        now = now or utc_now()
        tasks = self.list(status="open", limit=1000)
        tz = self.tz
        today_local = now.astimezone(tz).date()

        overdue: list[dict] = []
        today: list[dict] = []
        soon: list[dict] = []
        high: list[dict] = []
        for t in tasks:
            due = t["due_at"]
            if due is not None and due < now:
                overdue.append(t)
            elif due is not None and due.astimezone(tz).date() == today_local:
                today.append(t)
            elif due is not None and due <= now + timedelta(days=3):
                soon.append(t)
            elif t["priority"] == "high":
                high.append(t)

        ordered: list[dict] = []
        seen: set[int] = set()
        for bucket in (overdue, today, soon, high):
            for t in bucket:
                if t["id"] in seen:
                    continue
                seen.add(t["id"])
                ordered.append(t)
                if len(ordered) >= limit:
                    return ordered
        return ordered

    # --- 通知 (リマインドエンジン用) ---

    def claim_due_notifications(
        self,
        owner: str,
        now: Optional[datetime] = None,
        *,
        lease_seconds: int = 120,
    ) -> list[dict]:
        """通知評価が必要な open タスクを lease してから返す (BEGIN IMMEDIATE)。

        - due_at がある open タスク
        - next_notify_at が NULL (=即評価) または now 以前
        - snoozed_until が NULL または now 以前
        - lease が空 / 期限切れ / 自分のもの
        """
        now = now or utc_now()
        now_iso = to_iso(now)
        lease_until_iso = to_iso(now + timedelta(seconds=lease_seconds))

        with self._tx(immediate=True) as conn:
            rows = conn.execute(
                """
                SELECT t.*, n.last_stage AS n_last_stage, n.last_notified_at AS n_last_notified_at,
                       n.next_notify_at AS n_next_notify_at, n.snoozed_until AS n_snoozed_until,
                       n.repeat_count AS n_repeat_count
                FROM tasks t
                JOIN task_notifications n ON n.task_id = t.id
                WHERE t.status = 'open'
                  AND t.due_at IS NOT NULL
                  AND (n.next_notify_at IS NULL OR n.next_notify_at <= ?)
                  AND (n.snoozed_until IS NULL OR n.snoozed_until <= ?)
                  AND (n.lease_until IS NULL OR n.lease_until <= ? OR n.lease_owner = ?)
                """,
                (now_iso, now_iso, now_iso, owner),
            ).fetchall()

            claimed: list[dict] = []
            for row in rows:
                conn.execute(
                    "UPDATE task_notifications SET lease_owner = ?, lease_until = ? WHERE task_id = ?",
                    (owner, lease_until_iso, row["id"]),
                )
                task = self._row_to_task(row)
                task["notification"] = {
                    "last_stage": row["n_last_stage"],
                    "last_notified_at": from_iso(row["n_last_notified_at"]),
                    "next_notify_at": from_iso(row["n_next_notify_at"]),
                    "snoozed_until": from_iso(row["n_snoozed_until"]),
                    "repeat_count": row["n_repeat_count"] or 0,
                }
                claimed.append(task)
        return claimed

    def record_notification(
        self,
        task_id: int,
        owner: str,
        *,
        stage: Optional[str],
        next_notify_at: Optional[datetime],
        repeat_count: int,
        fired: bool,
        now: Optional[datetime] = None,
    ) -> None:
        """claim 済みタスクの通知状態を更新し、lease を解放する。

        fired=True のとき last_notified_at と last_stage を更新する。
        fired=False (未発火・繰り越し等) のときは next_notify_at のみ更新する。
        """
        now = now or utc_now()
        with self._tx(immediate=True) as conn:
            if fired:
                conn.execute(
                    """
                    UPDATE task_notifications
                    SET last_stage = ?, last_notified_at = ?, next_notify_at = ?,
                        repeat_count = ?, lease_owner = NULL, lease_until = NULL
                    WHERE task_id = ?
                    """,
                    (stage, to_iso(now), to_iso(next_notify_at), repeat_count, task_id),
                )
                self._log_event(conn, task_id, "notify", f"stage={stage}", now)
            else:
                conn.execute(
                    """
                    UPDATE task_notifications
                    SET next_notify_at = ?, repeat_count = ?, lease_owner = NULL, lease_until = NULL
                    WHERE task_id = ?
                    """,
                    (to_iso(next_notify_at), repeat_count, task_id),
                )

    def release_lease(self, task_id: int, owner: str) -> None:
        with self._tx(immediate=True) as conn:
            conn.execute(
                "UPDATE task_notifications SET lease_owner = NULL, lease_until = NULL "
                "WHERE task_id = ? AND lease_owner = ?",
                (task_id, owner),
            )

    def events(self, task_id: Optional[int] = None, limit: int = 100) -> list[dict]:
        conn = self._require()
        with self._lock:
            if task_id is None:
                rows = conn.execute(
                    "SELECT * FROM task_events ORDER BY id DESC LIMIT ?", (limit,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM task_events WHERE task_id = ? ORDER BY id DESC LIMIT ?",
                    (task_id, limit),
                ).fetchall()
        return [
            {
                "id": r["id"],
                "task_id": r["task_id"],
                "event": r["event"],
                "detail": r["detail"],
                "created_at": from_iso(r["created_at"]),
            }
            for r in rows
        ]


# --- LLM コンテキスト用フォーマッタ ---

def format_local_due(due_at: Optional[datetime], granularity: Optional[str], tz: ZoneInfo, now: datetime) -> str:
    """due_at をローカル表示の短い文字列にする。"""
    if due_at is None:
        return "期限なし"
    local = due_at.astimezone(tz)
    if granularity == "date":
        return local.strftime("%-m/%-d")
    return local.strftime("%-m/%-d %H:%M")


def build_task_context(store: "TaskStore", limit: int = 8, *, now: Optional[datetime] = None) -> str:
    """未完了タスクを「--- 未完了タスク ---」ブロックとして返す。0件なら空文字。"""
    now = now or utc_now()
    try:
        tasks = store.get_context_tasks(limit=limit, now=now)
    except Exception:
        return ""
    if not tasks:
        return ""
    tz = store.tz
    lines = ["\n--- 未完了タスク ---"]
    for t in tasks:
        due = t["due_at"]
        if due is None:
            due_str = "期限なし"
        elif due < now:
            due_str = f"期限超過 ({format_local_due(due, t['due_granularity'], tz, now)})"
        else:
            due_str = format_local_due(due, t["due_granularity"], tz, now)
        prio = {"high": "[高]", "low": "[低]", "normal": ""}.get(t["priority"], "")
        line = f"- {prio}{t['title']} (期限: {due_str})"
        if t["action_hint"]:
            line += f" 次の一手: {t['action_hint']}"
        lines.append(line)
    return "\n".join(lines)
