"""Web会話から、ルールベースでタスクを安全に編集する。

LLMの文章生成とDB書き込みを切り離し、明示的なタスク操作だけを
TaskStoreに反映する。名前が曖昧な場合は候補IDを返して停止し、削除は
セッション単位の確認を必須とする。
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from src.discord_bot.task_ui import parse_due, split_quick_input
from src.tasks.store import TaskStore

UTC = timezone.utc
CONFIRM_TTL = timedelta(minutes=5)

_LIST_RE = re.compile(
    r"(?:タスク|やること).*(?:一覧|リスト|見せて|見たい|教えて|確認)"
)
_HELP_RE = re.compile(r"(?:タスク).*(?:編集したい|操作|使い方|どうやって)")
_ADD_PREFIX_RE = re.compile(r"^\s*タスク\s*[:：]\s*(?P<body>.+)$", re.DOTALL)
_ADD_AFTER_RE = re.compile(
    r"^\s*タスク(?:に|へ)\s*(?P<body>.+?)\s*(?:を)?"
    r"(?:追加|登録)(?:して|してください|しといて|お願い)?[!！?？。\s]*$",
    re.DOTALL,
)
_ADD_BEFORE_RE = re.compile(
    r"^\s*(?P<body>.+?)\s*(?:を)?タスク(?:に|へ)\s*"
    r"(?:追加|登録)(?:して|してください|しといて|お願い)?[!！?？。\s]*$",
    re.DOTALL,
)
_TASK_ID_RE = re.compile(r"タスク(?:ID)?\s*[#＃]?\s*(?P<id>\d+)", re.IGNORECASE)
_QUOTED_RE = re.compile(r"[「『\"](?P<title>[^」』\"]+)[」』\"](?:の)?タスク")
_TITLE_TASK_RE = re.compile(r"^\s*(?P<title>.+?)(?:の)?タスク(?:を|は|の|に|、|,|\s)")
_RENAME_RE = re.compile(
    r"(?:名前|タイトル)を\s*[「『\"]?(?P<title>.+?)[」』\"]?\s*"
    r"に(?:変更|変えて|して)(?:して|ください|ね|よ)?[!！?？。\s]*$"
)
_NOTE_RE = re.compile(
    r"メモを\s*[「『\"]?(?P<note>.+?)[」』\"]?\s*"
    r"に(?:変更|変えて|して)(?:して|ください|ね|よ)?[!！?？。\s]*$"
)
_DONE_RE = re.compile(r"(?:完了|完了に|終わった|終わり|済み|できた|やった)(?:にして|にする|にしといて|に変更|にしてください|といて)?")
_DROP_RE = re.compile(r"(?:削除|消して|取り消し|取消し|取り下げ)")
_CLEAR_DUE_RE = re.compile(r"(?:期限|締切)(?:を|は)?\s*(?:なし|無し|外して|消して)")
_CLEAR_NOTE_RE = re.compile(r"メモ(?:を|は)?\s*(?:なし|無し|空|消して|削除)")
_BREAKDOWN_RE = re.compile(r"(?:細分化|分解|手順(?:を)?(?:作|考え|やり直|作り直))")
_SHOW_STEPS_RE = re.compile(r"(?:最初の一歩|次の一手|手順).*(?:見せ|教え|確認|なに|何)")
_DUE_HINT_RE = re.compile(
    r"(?:期限|締切|今日|明日|あした|明後日|あさって|今夜|来週|今週|週末|"
    r"[月火水木金土日](?:曜日|曜)?|\d{1,4}[/-]\d{1,2}|\d{1,2}月\d{1,2}日|"
    r"\d{1,2}日(?:まで)?|\d{1,2}時|\d+分後|\d+時間後|\d+日後)"
)
_TASK_ACTION_RE = re.compile(
    r"(?:完了|終わった|終わり|済み|できた|やった|"
    r"削除|消して|取り消し|取消し|取り下げ|"
    r"変更|変えて|延期|移動|ずらし|期限|締切|優先|だいじ|大事|重要|後回し|名前|タイトル|メモ|"
    r"細分化|分解|最初の一歩|次の一手|手順)"
)


@dataclass(frozen=True)
class PendingDrop:
    task_id: int
    title: str
    expires_at: datetime


def _normalize(value: str) -> str:
    return re.sub(r"[\s　、,。・!！?？「」『』\"']+", "", value).casefold()


def _task_line(task: dict, store: TaskStore) -> str:
    due_at = task.get("due_at")
    if due_at is None:
        due = "期限なし"
    else:
        local = due_at.astimezone(store.tz)
        due = local.strftime("%-m/%-d")
        if task.get("due_granularity") != "date":
            due += local.strftime(" %H:%M")
    priority = {"high": "・だいじ", "low": "・あとで"}.get(task.get("priority"), "")
    line = f"#{task['id']} {task['title']}（{due}{priority}）"
    if task.get("action_hint"):
        line += f"／まず: {task['action_hint']}"
    return line


def _breakdown_text(task: dict) -> str:
    steps = [str(step) for step in task.get("steps") or [] if str(step).strip()][:3]
    lines = [f"「{task['title']}」（#{task['id']}）は、ここから始められます。"]
    if task.get("action_hint"):
        lines.append(f"まず5分: {task['action_hint']}")
    lines.extend(f"{index}. {step}" for index, step in enumerate(steps, 1))
    return "\n".join(lines)


def _help_text() -> str:
    return (
        "話すだけでタスクを編集できます。\n"
        "・「タスクを見せて」\n"
        "・「タスク: 明日 牛乳を買う」\n"
        "・「タスク13を明日に変更」\n"
        "・「タスク13をだいじにして」\n"
        "・「タスク13を細分化して」\n"
        "・「タスク13の最初の一歩を教えて」\n"
        "・「タスク13を完了」\n"
        "・「タスク13を削除」"
    )


class TaskChatEditor:
    """セッション単位の削除確認を持つ、スレッドセーフな会話編集器。"""

    def __init__(self) -> None:
        self._pending: dict[str, PendingDrop] = {}
        self._lock = threading.Lock()

    def handle(
        self,
        text: str,
        *,
        store: Optional[TaskStore],
        session_id: str,
        now: Optional[datetime] = None,
    ) -> Optional[str]:
        text = str(text or "").strip()
        if not text:
            return None
        now = now or datetime.now(UTC)
        if now.tzinfo is None:
            now = now.replace(tzinfo=UTC)

        pending_reply = self._handle_pending(text, store, session_id, now)
        if pending_reply is not None:
            return pending_reply

        if _HELP_RE.search(text):
            return _help_text()
        if _LIST_RE.search(text) and not (_SHOW_STEPS_RE.search(text) or _BREAKDOWN_RE.search(text)):
            if store is None:
                return "タスク管理が無効のため、一覧を読み込めません。"
            return self._list_tasks(store)

        add_body = self._extract_add_body(text)
        if add_body is not None:
            if store is None:
                return "タスク管理が無効のため、追加できません。"
            return self._add_task(add_body, store, now)

        # 普通の会話を誤検出しないよう、編集は「タスク」と操作語の両方を必須にする。
        if "タスク" not in text or not _TASK_ACTION_RE.search(text):
            return None
        if store is None:
            return "タスク管理が無効のため、編集できません。"

        task, error = self._resolve_task(text, store)
        if error is not None:
            return error
        assert task is not None

        if _BREAKDOWN_RE.search(text):
            if not store.regenerate_breakdown(int(task["id"]), now=now):
                return "そのタスクはすでに変更されています。"
            refreshed = store.get(int(task["id"]))
            assert refreshed is not None
            return _breakdown_text(refreshed)
        if _SHOW_STEPS_RE.search(text):
            return _breakdown_text(task)

        clear_due = _CLEAR_DUE_RE.search(text)
        clear_note = _CLEAR_NOTE_RE.search(text)
        if _DROP_RE.search(text) and not clear_due and not clear_note:
            with self._lock:
                self._pending[session_id] = PendingDrop(
                    task_id=int(task["id"]),
                    title=str(task["title"]),
                    expires_at=now + CONFIRM_TTL,
                )
            return (
                f"「{task['title']}」（#{task['id']}）を削除しますか？\n"
                "よければ「削除する」、やめるなら「キャンセル」と言ってください。"
            )

        if _DONE_RE.search(text):
            if store.done(int(task["id"]), now=now):
                return f"「{task['title']}」（#{task['id']}）を完了にしました。"
            return "そのタスクはすでに変更されています。"

        changes: dict = {}
        descriptions: list[str] = []
        rename = _RENAME_RE.search(text)
        if rename:
            title = rename.group("title").strip()
            if title:
                changes["title"] = title[:200]
                descriptions.append(f"名前を「{title[:200]}」")
        note_match = _NOTE_RE.search(text)
        if clear_note:
            changes["note"] = ""
            descriptions.append("メモを「なし」")
        elif note_match:
            note = note_match.group("note").strip()
            changes["note"] = note
            descriptions.append("メモ")

        priority = self._parse_priority(text)
        if priority is not None:
            changes["priority"] = priority
            label = {"high": "だいじ", "normal": "ふつう", "low": "あとで"}[priority]
            descriptions.append(f"優先度を「{label}」")

        if clear_due:
            changes["clear_due"] = True
            descriptions.append("期限を「なし」")
        elif _DUE_HINT_RE.search(text) and rename is None and note_match is None:
            due_at, granularity = parse_due(text, now, store.tz)
            if due_at is not None:
                changes["due_at"] = due_at
                changes["due_granularity"] = granularity
                local = due_at.astimezone(store.tz)
                display = local.strftime("%-m/%-d")
                if granularity != "date":
                    display += local.strftime(" %H:%M")
                descriptions.append(f"期限を「{display}」")

        if not changes:
            return (
                "どこを変えるかわかりませんでした。\n"
                "例: 「タスク%dを明日に変更」「タスク%dをだいじにして」"
                % (task["id"], task["id"])
            )
        if store.update(int(task["id"]), now=now, **changes):
            return f"「{task['title']}」（#{task['id']}）の" + "、".join(descriptions) + "に変更しました。"
        return "そのタスクはすでに変更されています。"

    def _handle_pending(
        self,
        text: str,
        store: Optional[TaskStore],
        session_id: str,
        now: datetime,
    ) -> Optional[str]:
        normalized = _normalize(text)
        with self._lock:
            pending = self._pending.get(session_id)
            if pending is None:
                return None
            if pending.expires_at < now:
                self._pending.pop(session_id, None)
                return "削除の確認が時間切れになりました。もう一度タスクを指定してください。"
            if normalized in {"キャンセル", "いいえ", "やめる", "やめとく", "取り消し"}:
                self._pending.pop(session_id, None)
                return "削除をキャンセルしました。"
            if normalized not in {"削除する", "消す", "はい削除する", "お願い", "ok", "yes"}:
                # 別の話題へ移ったら、後の「はい」で誤削除しないよう確認を破棄する。
                self._pending.pop(session_id, None)
                return None
            self._pending.pop(session_id, None)

        if store is None:
            return "タスク管理が無効のため、削除できません。"
        current = store.get(pending.task_id)
        if current is None or current.get("status") != "open" or current.get("title") != pending.title:
            return "確認中にタスクが変更されたため、削除しませんでした。"
        if store.drop(pending.task_id, now=now):
            return f"「{pending.title}」（#{pending.task_id}）を削除しました。"
        return "タスクを削除できませんでした。"

    @staticmethod
    def _extract_add_body(text: str) -> Optional[str]:
        for pattern in (_ADD_PREFIX_RE, _ADD_AFTER_RE, _ADD_BEFORE_RE):
            match = pattern.match(text)
            if match:
                return match.group("body").strip()
        return None

    @staticmethod
    def _add_task(body: str, store: TaskStore, now: datetime) -> str:
        parsed = split_quick_input(body, now, store.tz)
        title = str(parsed.get("title") or "").strip()
        if not title:
            return "タスクの内容がわかりませんでした。例: 「タスク: 明日 牛乳を買う」"
        task_id = store.add(
            title[:200],
            due_at=parsed.get("due_at"),
            due_granularity=parsed.get("due_granularity"),
            priority=str(parsed.get("priority") or "normal"),
            source="web",
            now=now,
        )
        task = store.get(task_id)
        assert task is not None
        return "タスクを追加しました: " + _task_line(task, store) + "\n" + _breakdown_text(task)

    @staticmethod
    def _list_tasks(store: TaskStore) -> str:
        tasks = store.list("open", 20)
        if not tasks:
            return "未完了のタスクはありません。\n追加は「タスク: 明日 牛乳を買う」のように話してください。"
        lines = [f"未完了のタスクは{len(tasks)}件です。"]
        lines.extend(f"・{_task_line(task, store)}" for task in tasks)
        lines.append("変更は「タスク13を明日に変更」のように番号で話せます。")
        return "\n".join(lines)

    @staticmethod
    def _resolve_task(text: str, store: TaskStore) -> tuple[Optional[dict], Optional[str]]:
        id_match = _TASK_ID_RE.search(text)
        if id_match:
            task_id = int(id_match.group("id"))
            task = store.get(task_id)
            if task is None or task.get("status") != "open":
                return None, f"未完了のタスク#{task_id}は見つかりません。"
            return task, None

        title_match = _QUOTED_RE.search(text) or _TITLE_TASK_RE.search(text)
        if title_match is None:
            return None, "どのタスクかわかりませんでした。「タスクを見せて」で番号を確認できます。"
        target = _normalize(title_match.group("title"))
        tasks = store.list("open", 200)
        exact = [task for task in tasks if _normalize(str(task["title"])) == target]
        candidates = exact or [
            task for task in tasks
            if target in _normalize(str(task["title"])) or _normalize(str(task["title"])) in target
        ]
        if not candidates:
            return None, f"「{title_match.group('title').strip()}」に一致する未完了タスクは見つかりません。"
        if len(candidates) > 1:
            lines = ["複数のタスクが一致したため、まだ変更していません。"]
            lines.extend(f"・{_task_line(task, store)}" for task in candidates[:8])
            lines.append("「タスク13を完了」のように番号で指定してください。")
            return None, "\n".join(lines)
        return candidates[0], None

    @staticmethod
    def _parse_priority(text: str) -> Optional[str]:
        if re.search(r"(?:あとで|後回し|優先度を?低|優先を?下げ)", text):
            return "low"
        if re.search(r"(?:ふつう|普通|通常|優先度を?戻)", text):
            return "normal"
        if re.search(r"(?:だいじ|大事|重要|最優先|優先度を?高|優先を?上げ)", text):
            return "high"
        return None
