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

import hashlib
import json
import sqlite3
import threading
import unicodedata
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional
from zoneinfo import ZoneInfo

from src.tasks.decomposer import decompose_task
from src.tasks.formatting import format_short_due
from src.tasks.safety import is_sensitive_text

UTC = timezone.utc
DEFAULT_TZ = "Asia/Tokyo"

VALID_PRIORITY = ("high", "normal", "low")
VALID_STATUS = ("open", "done", "dropped")
VALID_GRANULARITY = ("date", "datetime")
VALID_SOURCE = ("command", "chat", "voice", "context_menu", "board", "web")
VALID_CANDIDATE_STATUS = ("pending", "accepted", "dismissed")
# 受け入れ/却下済み候補と等価な候補の再送信を抑制する日数。
CANDIDATE_SUPPRESS_DAYS = 30

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
        self._backfill_breakdowns()
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
                    completed_at TEXT,
                    breakdown_json TEXT NOT NULL DEFAULT '[]',
                    rev INTEGER NOT NULL DEFAULT 0
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
            # --- タスク候補 Inbox ---
            # 抽出器 (Discord/Web/音声) が提案する未確定タスクを置く場所。
            # 生の会話テキストは一切保存しない。title+due+priority を NFKC 正規化して
            # SHA-256 指紋を取り、同指紋の pending 重複と 30 日以内の accepted/dismissed
            # を抑制する。
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS task_candidates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    fingerprint TEXT NOT NULL,
                    title TEXT NOT NULL,
                    due_at TEXT,
                    due_granularity TEXT,
                    priority TEXT NOT NULL DEFAULT 'normal',
                    status TEXT NOT NULL DEFAULT 'pending',
                    task_id INTEGER,
                    created_at TEXT NOT NULL,
                    decided_at TEXT,
                    FOREIGN KEY (task_id) REFERENCES tasks (id) ON DELETE SET NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_candidates_fp_status "
                "ON task_candidates (fingerprint, status)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_candidates_status_created "
                "ON task_candidates (status, created_at)"
            )
            self._migrate(conn)

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """既存DBを壊さずにカラムを追加する冪等マイグレーション。"""
        cols = {row["name"] for row in conn.execute("PRAGMA table_info(tasks)").fetchall()}
        if "calendar_event_id" not in cols:
            conn.execute("ALTER TABLE tasks ADD COLUMN calendar_event_id TEXT")
        if "calendar_synced_at" not in cols:
            conn.execute("ALTER TABLE tasks ADD COLUMN calendar_synced_at TEXT")
        if "breakdown_json" not in cols:
            conn.execute("ALTER TABLE tasks ADD COLUMN breakdown_json TEXT NOT NULL DEFAULT '[]'")
        if "rev" not in cols:
            # 楽観的並行制御用のリビジョン。リマインドを無効化しうる変更が
            # 同一トランザクションで rev = rev + 1 する。冪等 (存在すれば何もしない)。
            conn.execute("ALTER TABLE tasks ADD COLUMN rev INTEGER NOT NULL DEFAULT 0")
        # task_notifications の lease カラム (重複送信防止) は旧DBに無いことがある。
        # CREATE TABLE IF NOT EXISTS は既存テーブルを変えないため、不足分だけ ALTER する。
        # 新規DBでは CREATE 側に両カラムが含まれており、ここは何もしない (冪等)。
        notif_cols = {
            row["name"] for row in conn.execute("PRAGMA table_info(task_notifications)").fetchall()
        }
        if notif_cols:
            if "lease_owner" not in notif_cols:
                conn.execute("ALTER TABLE task_notifications ADD COLUMN lease_owner TEXT")
            if "lease_until" not in notif_cols:
                conn.execute("ALTER TABLE task_notifications ADD COLUMN lease_until TEXT")
        # task_candidates は CREATE TABLE IF NOT EXISTS で新規DBには due_granularity 付きで
        # 作られる。旧DBに同テーブルが既に存在してカラム無しの場合だけ ALTER する。
        # テーブル自体が無い場合は上の CREATE で新スキーマが作られているので何もしない。
        cand_cols = {
            row["name"] for row in conn.execute("PRAGMA table_info(task_candidates)").fetchall()
        }
        if cand_cols and "due_granularity" not in cand_cols:
            conn.execute("ALTER TABLE task_candidates ADD COLUMN due_granularity TEXT")
            # 旧fingerprintは granularity を含まない。全旧行を同じ新規則へ移し、
            # pending重複・accepted/dismissedの30日抑制をアップグレード後も維持する。
            rows = conn.execute(
                "SELECT id, title, due_at, priority FROM task_candidates"
            ).fetchall()
            for row in rows:
                due_at = from_iso(row["due_at"])
                granularity = "datetime" if due_at is not None else None
                priority = row["priority"] if row["priority"] in VALID_PRIORITY else "normal"
                fingerprint = self._candidate_fingerprint(
                    str(row["title"]), due_at, priority, granularity
                )
                conn.execute(
                    "UPDATE task_candidates SET due_granularity = ?, fingerprint = ? WHERE id = ?",
                    (granularity, fingerprint, int(row["id"])),
                )

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
    def _decode_breakdown(value: Any) -> list[dict[str, Any]]:
        """旧形式の文字列配列も、完了状態つきの手順として読み込む。"""
        try:
            raw = json.loads(value or "[]")
        except (TypeError, ValueError, json.JSONDecodeError):
            return []
        if not isinstance(raw, list):
            return []

        result: list[dict[str, Any]] = []
        for step in raw:
            if isinstance(step, dict):
                text = str(step.get("text") or "").strip()[:240]
                done = step.get("done") is True
            else:
                text = str(step).strip()[:240]
                done = False
            if text:
                result.append({"text": text, "done": done})
            if len(result) >= 3:
                break
        return result

    @staticmethod
    def _decode_steps(value: Any) -> list[str]:
        return [step["text"] for step in TaskStore._decode_breakdown(value)]

    @staticmethod
    def _encode_breakdown(steps: Iterable[str], done: Optional[Iterable[bool]] = None) -> str:
        done_values = list(done or [])
        payload = []
        for index, step in enumerate(steps):
            text = str(step).strip()[:240]
            if not text:
                continue
            payload.append({
                "text": text,
                "done": bool(done_values[index]) if index < len(done_values) else False,
            })
            if len(payload) >= 3:
                break
        return json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def _generated_breakdown(
        title: str,
        note: Optional[str] = None,
        action_hint: Optional[str] = None,
    ) -> tuple[str, list[str]]:
        generated = decompose_task(title, note=note, action_hint=action_hint)
        first = str(action_hint or generated.first_step).strip()[:240]
        tail = [step for step in generated.steps if step != first]
        steps = ([first] if first else []) + tail
        return first, steps[:3]

    def _backfill_breakdowns(self) -> int:
        """既存の未完了タスクだけを、冪等に細分化する。"""
        changed = 0
        now = utc_now()
        with self._tx(immediate=True) as conn:
            rows = conn.execute(
                "SELECT id, title, note, action_hint, breakdown_json FROM tasks WHERE status = 'open'"
            ).fetchall()
            for row in rows:
                existing_steps = self._decode_steps(row["breakdown_json"])
                existing_hint = str(row["action_hint"] or "").strip()
                if existing_hint and existing_steps:
                    continue
                hint, steps = self._generated_breakdown(
                    str(row["title"]), row["note"], existing_hint or None
                )
                conn.execute(
                    "UPDATE tasks SET action_hint = ?, breakdown_json = ? WHERE id = ?",
                    (existing_hint or hint, self._encode_breakdown(steps), row["id"]),
                )
                self._log_event(conn, int(row["id"]), "decompose", "automatic backfill", now)
                changed += 1
        return changed

    @staticmethod
    def _row_to_task(row: sqlite3.Row) -> dict:
        breakdown = TaskStore._decode_breakdown(row["breakdown_json"])
        return {
            "id": row["id"],
            "title": row["title"],
            "note": row["note"],
            "action_hint": row["action_hint"],
            "steps": [step["text"] for step in breakdown],
            "step_done": [step["done"] for step in breakdown],
            "due_at": from_iso(row["due_at"]),
            "due_granularity": row["due_granularity"],
            "priority": row["priority"],
            "status": row["status"],
            "source": row["source"],
            "created_at": from_iso(row["created_at"]),
            "completed_at": from_iso(row["completed_at"]),
            "calendar_event_id": row["calendar_event_id"],
            "calendar_synced_at": from_iso(row["calendar_synced_at"]),
            "rev": row["rev"] or 0,
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
        generated_hint, generated_steps = self._generated_breakdown(title, note, action_hint)

        with self._tx(immediate=True) as conn:
            cur = conn.execute(
                """
                INSERT INTO tasks
                    (title, note, action_hint, due_at, due_granularity,
                     priority, status, source, created_at, completed_at, breakdown_json)
                VALUES (?, ?, ?, ?, ?, ?, 'open', ?, ?, NULL, ?)
                """,
                (
                    title, note, generated_hint, to_iso(due_at), due_granularity,
                    priority, source, to_iso(now), self._encode_breakdown(generated_steps),
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
            clean_hint = action_hint.strip()[:240]
            params.append(clean_hint)
            fields.append("breakdown_json = ?")
            if clean_hint:
                current = self.get(task_id)
                current_steps = list(current.get("steps") or []) if current else []
                current_done = list(current.get("step_done") or []) if current else []
                tail = current_steps[1:]
                first_done = bool(current_done[0]) if current_steps and current_steps[0] == clean_hint else False
                done_values = [first_done] + current_done[1:]
                params.append(self._encode_breakdown(([clean_hint] + tail)[:3], done_values))
            else:
                params.append("[]")
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
        fields.append("rev = rev + 1")  # リマインドを無効化しうる変更を記録
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

    def regenerate_breakdown(self, task_id: int, *, now: Optional[datetime] = None) -> bool:
        """現在のタイトル・メモから細分化を作り直す。子タスクは作らない。"""
        now = now or utc_now()
        with self._tx(immediate=True) as conn:
            row = conn.execute(
                "SELECT title, note FROM tasks WHERE id = ? AND status = 'open'",
                (task_id,),
            ).fetchone()
            if row is None:
                return False
            hint, steps = self._generated_breakdown(str(row["title"]), row["note"])
            conn.execute(
                "UPDATE tasks SET action_hint = ?, breakdown_json = ?, rev = rev + 1 WHERE id = ?",
                (hint, self._encode_breakdown(steps), task_id),
            )
            self._log_event(conn, task_id, "decompose", "manual regenerate", now)
        return True

    def set_step_done(
        self,
        task_id: int,
        step_index: int,
        done: bool,
        *,
        now: Optional[datetime] = None,
    ) -> bool:
        """分割された手順の完了状態を更新する。タスク本体は自動完了しない。"""
        now = now or utc_now()
        changed = False
        with self._tx(immediate=True) as conn:
            row = conn.execute(
                "SELECT breakdown_json FROM tasks WHERE id = ? AND status = 'open'",
                (task_id,),
            ).fetchone()
            if row is None:
                return False
            breakdown = self._decode_breakdown(row["breakdown_json"])
            if step_index < 0 or step_index >= len(breakdown):
                return False
            if breakdown[step_index]["done"] != done:
                breakdown[step_index]["done"] = done
                conn.execute(
                    "UPDATE tasks SET breakdown_json = ? WHERE id = ?",
                    (
                        self._encode_breakdown(
                            (step["text"] for step in breakdown),
                            (step["done"] for step in breakdown),
                        ),
                        task_id,
                    ),
                )
                self._log_event(
                    conn,
                    task_id,
                    "step_done",
                    f"index={step_index} done={done}",
                    now,
                )
                changed = True
        if changed:
            self._fire_change(task_id, "step_done")
        return True

    def done(self, task_id: int, *, now: Optional[datetime] = None) -> bool:
        now = now or utc_now()
        with self._tx(immediate=True) as conn:
            cur = conn.execute(
                "UPDATE tasks SET status = 'done', completed_at = ?, rev = rev + 1 "
                "WHERE id = ? AND status = 'open'",
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
                "UPDATE tasks SET status = 'dropped', completed_at = ?, rev = rev + 1 "
                "WHERE id = ? AND status = 'open'",
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
            conn.execute(
                "UPDATE tasks SET rev = rev + 1 WHERE id = ? AND status = 'open'",
                (task_id,),
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

    def revalidate_notification_lease(
        self,
        task_id: int,
        owner: str,
        expected_rev: int,
        *,
        lease_seconds: int = 120,
        now: Optional[datetime] = None,
    ) -> bool:
        """外部コールバック直前の lease と rev の再検証 (BEGIN IMMEDIATE)。

        - task が open かつ rev が expected_rev と一致し、かつ lease_owner が
          owner のときだけ lease を延長して True を返す。
        - それ以外 (存在しない / done/dropped / rev 不一致) は owner の
          stale lease だけを安全に解除して False を返す。他 owner の lease は
          触らない。
        """
        now = now or utc_now()
        now_iso = to_iso(now)
        lease_until_iso = to_iso(now + timedelta(seconds=lease_seconds))
        with self._tx(immediate=True) as conn:
            row = conn.execute(
                "SELECT status, rev FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
            if row is None or row["status"] != "open" or (row["rev"] or 0) != expected_rev:
                conn.execute(
                    "UPDATE task_notifications SET lease_owner = NULL, lease_until = NULL "
                    "WHERE task_id = ? AND lease_owner = ?",
                    (task_id, owner),
                )
                return False
            cur = conn.execute(
                "UPDATE task_notifications SET lease_owner = ?, lease_until = ? "
                "WHERE task_id = ? AND lease_owner = ?",
                (owner, lease_until_iso, task_id, owner),
            )
            return cur.rowcount > 0

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
        expected_rev: Optional[int] = None,
    ) -> bool:
        """claim 済みタスクの通知状態を更新し、lease を解放する。

        fired=True のとき last_notified_at と last_stage を更新する。
        fired=False (未発火・繰り越し等) のときは next_notify_at のみ更新する。

        expected_rev を指定した場合、task が open かつ rev が expected_rev と
        一致し、かつ lease_owner が owner のときだけ更新して True を返す。
        それ以外は何も更新せず、owner の stale lease だけを安全に解除して
        False を返す (並行の done/drop/update/snooze を上書きしない)。
        省略時は旧挙動のまま無条件更新して True を返す。
        """
        now = now or utc_now()
        with self._tx(immediate=True) as conn:
            if expected_rev is not None:
                row = conn.execute(
                    "SELECT status, rev FROM tasks WHERE id = ?", (task_id,)
                ).fetchone()
                if row is None or row["status"] != "open" or (row["rev"] or 0) != expected_rev:
                    conn.execute(
                        "UPDATE task_notifications SET lease_owner = NULL, lease_until = NULL "
                        "WHERE task_id = ? AND lease_owner = ?",
                        (task_id, owner),
                    )
                    return False
                cur = conn.execute(
                    "SELECT lease_owner FROM task_notifications WHERE task_id = ?",
                    (task_id,),
                ).fetchone()
                if cur is None or cur["lease_owner"] != owner:
                    return False
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
        return True

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

    # --- タスク候補 Inbox ---

    @staticmethod
    def _candidate_fingerprint(
        title: str,
        due_at: Optional[datetime],
        priority: str,
        due_granularity: Optional[str],
    ) -> str:
        """NFKC 正規化した title+due+priority+due_granularity の SHA-256 指紋を返す。
        生の会話テキストは含めず、候補の等価性判定だけに使う。
        """
        norm_title = unicodedata.normalize("NFKC", title)
        norm_due = unicodedata.normalize("NFKC", to_iso(due_at) or "")
        norm_prio = unicodedata.normalize("NFKC", priority)
        norm_gran = unicodedata.normalize("NFKC", due_granularity or "")
        payload = f"{norm_title}|{norm_due}|{norm_prio}|{norm_gran}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _row_to_candidate(row: sqlite3.Row) -> dict:
        return {
            "id": int(row["id"]),
            "source": row["source"],
            "fingerprint": row["fingerprint"],
            "title": row["title"],
            "due_at": from_iso(row["due_at"]),
            "due_granularity": row["due_granularity"],
            "priority": row["priority"],
            "status": row["status"],
            "task_id": row["task_id"],
            "created_at": from_iso(row["created_at"]),
            "decided_at": from_iso(row["decided_at"]),
        }

    def create_candidate(
        self,
        *,
        title: str,
        due_at: Optional[datetime] = None,
        due_granularity: Optional[str] = None,
        priority: str = "normal",
        source: str = "chat",
        now: Optional[datetime] = None,
    ) -> Optional[int]:
        """タスク候補を Inbox に追加する。

        - title は NFKC 正規化して保存する (生の会話テキストは保存しない)。
        - due_granularity は due_at があるときだけ意味を持つ。未指定の場合は
          'datetime' とみなして指紋・保存する (旧呼び出し側との後方互換)。
          due_at が無ければ granularity は保存しない (NULL)。
        - 同指紋の pending 候補が既に存在すれば、新規作成せずにその id を返す (重複抑制)。
        - 直近30日以内に同じ指紋の accepted/dismissed 候補が存在すれば、新規作成を
          抑制して None を返す。
        戻り値: 新規作成候補 id、dedup で既存を使った場合その id、抑制された場合は None。
        """
        norm_title = unicodedata.normalize("NFKC", (title or "").strip())[:200]
        if not norm_title:
            raise ValueError("title は必須です")
        if is_sensitive_text(norm_title):
            raise ValueError("秘密情報を含む候補は保存できません")
        if priority not in VALID_PRIORITY:
            priority = "normal"
        if source not in VALID_SOURCE:
            source = "chat"
        if due_at is not None and due_granularity not in VALID_GRANULARITY:
            due_granularity = "datetime"
        elif due_at is None:
            due_granularity = None
        fingerprint = self._candidate_fingerprint(norm_title, due_at, priority, due_granularity)
        now = now or utc_now()
        now_iso = to_iso(now)
        with self._tx(immediate=True) as conn:
            # 同指紋の pending 候補があれば dedup (同一/横断どちらも既存を使う)
            existing = conn.execute(
                "SELECT id FROM task_candidates WHERE fingerprint = ? AND status = 'pending' "
                "ORDER BY id DESC LIMIT 1",
                (fingerprint,),
            ).fetchone()
            if existing is not None:
                return int(existing["id"])
            # 30日以内に accepted/dismissed の同指紋候補があれば抑制
            cutoff_iso = to_iso(now - timedelta(days=CANDIDATE_SUPPRESS_DAYS))
            decided = conn.execute(
                "SELECT id FROM task_candidates "
                "WHERE fingerprint = ? AND status IN ('accepted','dismissed') "
                "AND decided_at IS NOT NULL AND decided_at >= ? "
                "ORDER BY id DESC LIMIT 1",
                (fingerprint, cutoff_iso),
            ).fetchone()
            if decided is not None:
                return None
            cur = conn.execute(
                """
                INSERT INTO task_candidates
                    (source, fingerprint, title, due_at, due_granularity,
                     priority, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)
                """,
                (source, fingerprint, norm_title, to_iso(due_at), due_granularity,
                 priority, now_iso),
            )
            return int(cur.lastrowid)

    def list_candidates(self, status: str = "pending", limit: int = 100) -> list[dict]:
        """指定 status の候補を新しい順に最大 limit 件返す。"""
        conn = self._require()
        with self._lock:
            rows = conn.execute(
                "SELECT * FROM task_candidates WHERE status = ? ORDER BY id DESC LIMIT ?",
                (status, limit),
            ).fetchall()
        return [self._row_to_candidate(r) for r in rows]

    def get_candidate(self, candidate_id: int) -> Optional[dict]:
        conn = self._require()
        with self._lock:
            row = conn.execute(
                "SELECT * FROM task_candidates WHERE id = ?",
                (candidate_id,),
            ).fetchone()
        return self._row_to_candidate(row) if row else None

    def accept_candidate(
        self,
        candidate_id: int,
        *,
        now: Optional[datetime] = None,
    ) -> tuple[Optional[int], bool]:
        """候補を受け入れ、通常タスクを1件作成する。

        - pending 候補: 新規に task/notification/breakdown を1つずつ作り、
          task_events に 'accept' を1件記録し、候補を status='accepted' に更新する。
          トランザクションコミット後に on_change を1回だけ発火する。
        - 既に accepted: 既存の task_id と created=False を返す (冪等・再作成しない)。
        - dismissed: ValueError を送出する (却下済との衝突)。
        戻り値: (task_id, created)。created=False は on_change を発火しない。
        """
        now = now or utc_now()
        now_iso = to_iso(now)
        task_id: int
        created: bool
        with self._tx(immediate=True) as conn:
            row = conn.execute(
                "SELECT * FROM task_candidates WHERE id = ?",
                (candidate_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"候補 {candidate_id} が見つかりません")
            status = row["status"]
            if status == "accepted":
                return (int(row["task_id"]) if row["task_id"] is not None else None), False
            if status == "dismissed":
                raise ValueError("dismissed 済みの候補は受け入れられません")
            # pending -> 正規タスクを1件作成
            title = str(row["title"])
            due_at = from_iso(row["due_at"])
            priority = row["priority"] if row["priority"] in VALID_PRIORITY else "normal"
            source = row["source"] if row["source"] in VALID_SOURCE else "chat"
            due_granularity = row["due_granularity"]
            if due_at is not None and due_granularity not in VALID_GRANULARITY:
                due_granularity = "datetime"
            elif due_at is None:
                due_granularity = None
            generated_hint, generated_steps = self._generated_breakdown(title, None, None)
            breakdown_json = self._encode_breakdown(generated_steps)
            cur = conn.execute(
                """
                INSERT INTO tasks
                    (title, note, action_hint, due_at, due_granularity,
                     priority, status, source, created_at, completed_at, breakdown_json)
                VALUES (?, NULL, ?, ?, ?, ?, 'open', ?, ?, NULL, ?)
                """,
                (title, generated_hint, to_iso(due_at), due_granularity,
                 priority, source, now_iso, breakdown_json),
            )
            task_id = int(cur.lastrowid)
            conn.execute(
                """
                INSERT INTO task_notifications
                    (task_id, last_stage, last_notified_at, next_notify_at,
                     snoozed_until, repeat_count, lease_owner, lease_until)
                VALUES (?, NULL, NULL, NULL, NULL, 0, NULL, NULL)
                """,
                (task_id,),
            )
            self._log_event(conn, task_id, "accept", f"candidate={candidate_id}", now)
            conn.execute(
                "UPDATE task_candidates SET status = 'accepted', task_id = ?, decided_at = ? "
                "WHERE id = ?",
                (task_id, now_iso, candidate_id),
            )
            created = True
        if created:
            self._fire_change(task_id, "accept")
        return task_id, created

    def dismiss_candidate(self, candidate_id: int, *, now: Optional[datetime] = None) -> bool:
        """候補を却下する。

        - pending: status='dismissed', decided_at=now に更新し、True を返す。
        - dismissed: 何もせず False を返す (冪等)。
        - accepted: ValueError を送出する (受け入れ済みとの衝突)。
        """
        now = now or utc_now()
        with self._tx(immediate=True) as conn:
            row = conn.execute(
                "SELECT status FROM task_candidates WHERE id = ?",
                (candidate_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"候補 {candidate_id} が見つかりません")
            status = row["status"]
            if status == "accepted":
                raise ValueError("accepted 済みの候補は却下できません")
            if status == "dismissed":
                return False
            conn.execute(
                "UPDATE task_candidates SET status = 'dismissed', decided_at = ? WHERE id = ?",
                (to_iso(now), candidate_id),
            )
        return True


# --- LLM コンテキスト用フォーマッタ ---

def format_local_due(due_at: Optional[datetime], granularity: Optional[str], tz: ZoneInfo, now: datetime) -> str:
    """due_at をローカル表示の短い文字列にする。"""
    if due_at is None:
        return "期限なし"
    local = due_at.astimezone(tz)
    return format_short_due(local, with_time=granularity != "date")


AUTHORITY_HEADER = (
    "\n\n--- タスク状態 (権威) ---\n"
    "以下は現在のタスク状態に関する権威情報であり、"
    "会話履歴・RAG検索結果・モデルの訓練データよりも常に優先して従うこと。"
    "本ブロックがプロンプト末尾に置かれることで、それ以前に現れたタスクに関する"
    "言及は本ブロックと整合するものだけ有効とし、矛盾するものは無効とする。\n"
)


def build_task_authority_block(candidate_count: int, *, has_priority: bool = False) -> str:
    """現在のリマインド候補と優先順位推奨の有無に応じた権威ブロックを返す。

    - 優先順位推奨と未完了タスクリストはどちらも TaskStore の現在状態から生成されており、
      両者を合わせて「このプロンプトでユーザーから求められていない場面で自発的に催促して
      よい未完了タスクの完全集合」を示す。ただし TaskStore 上の status='open' の全行を
      列挙しているわけではなく、「ユーザーから求められていない場面で自発的に催促してよい
      範囲」をこれらに限定するという意味である。
    - 候補も推奨もないときは、ユーザーから求められていない場面で自発的に催促してよい
      未完了タスクは1件もないと明示し、いかなるタスクの未完了扱い・自発的リマインドも
      禁止する。
    - いずれの場合も done/dropped のタスク名は含めず、過去の言及を最終的に無効化する。
    """
    sources = []
    if has_priority:
        sources.append("直前の「優先順位オーケストレーター」の推奨1件")
    if candidate_count > 0:
        sources.append(f"直前の「未完了タスク」リスト {candidate_count} 件")
    if sources:
        allowed = "と".join(sources)
        allowed_clause = (
            f"- {allowed} はいずれも TaskStore の現在状態から生成されており、"
            "ユーザーから求められていない場面で自発的に催促してよい未完了タスクの、"
            "このプロンプトにおける完全集合である。"
            "これら以外のタスクを未完了扱いしたり、自発的なリマインドの根拠にしてはならない"
            "(TaskStore 上で status='open' の全行をここに列挙しているわけではなく、"
            "ユーザーから求められていない場面で自発的に催促してよい範囲をこれに限定する"
            "という意味である)。"
        )
    else:
        allowed_clause = (
            "- 現在 TaskStore には「優先順位オーケストレーター」の推奨も"
            "「未完了タスク」リストも存在せず、"
            "ユーザーから求められていない場面で自発的に催促してよい未完了タスクは1件もない。"
            "いかなるタスクについても未完了扱いや自発的なリマインドを提案してはならない"
            "(会話履歴・RAG・訓練データに基づく提案も禁止する)。"
        )
    done_clause = (
        "- status='done'(完了) または 'dropped'(破棄) のタスクは TaskStore 上で"
        "完全に処理済みであり、未完了扱いやリマインド候補として絶対に提示してはならない。"
        "会話履歴・RAG・訓練データにそれらのタスク名が残っていても無視すること。"
        "本ブロックはそれら過去の言及を最終的に無効化する。"
    )
    return AUTHORITY_HEADER + allowed_clause + "\n" + done_clause


def build_task_context(store: "TaskStore", limit: int = 8, *, now: Optional[datetime] = None) -> str:
    """優先順位の推奨と未完了タスクをチャット用コンテキストにし、
    末尾に現在のタスク完了・リマインド状態の権威ブロックを付ける。

    0 件のときも権威ブロックは必ず返し、完了/破棄タスクの扱いを上書き指定する。
    ただし現在のタスク状態を読めなかったとき (get_context_tasks が例外を送ったとき) は
    空文字列を返す。このとき権威ブロックも置かれないため、呼び出し側・設定は
    「タスク状態が取得できない」という前提で状態不明時の非催促ルールを適用すること
    (権威ブロックによる完了/破棄の無効化も行われないので、履歴に基づく催促は抑制する)。
    """
    now = now or utc_now()
    try:
        tasks = store.get_context_tasks(limit=limit, now=now)
    except Exception:
        # タスク状態の読み取りに失敗したときは空文字列を返す。
        # 権威ブロックを置かないことで呼び出し側に状態不明を伝え、
        # 状態不明時の非催促ルールを適用させる。
        return ""
    try:
        from src.tasks.prioritizer import build_priority_context

        priority_text = build_priority_context(store, now=now)
    except Exception:
        priority_text = ""
    tz = store.tz
    head = priority_text
    if tasks:
        lines = ["\n--- 未完了タスク ---"]
        for t in tasks:
            due = t["due_at"]
            if due is None:
                due_str = "期限なし"
            elif due < now:
                due_str = f"期限超過 ({format_local_due(due, t['due_granularity'], tz, now)})"
            else:
                due_str = format_local_due(due, t['due_granularity'], tz, now)
            prio = {"high": "[高]", "low": "[低]", "normal": ""}.get(t["priority"], "")
            line = f"- {prio}{t['title']} (期限: {due_str})"
            if t["action_hint"]:
                line += f" 次の一手: {t['action_hint']}"
            lines.append(line)
        head = head + "\n".join(lines)
    return head + build_task_authority_block(len(tasks), has_priority=bool(priority_text))
