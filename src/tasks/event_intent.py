"""自然言語からの Google Calendar 予定登録。

「予定: 明日15時 歯医者」(明示プレフィックス) と、
「明日15時に歯医者の予定入れて」(自然文トリガー) の両方を扱う。

日時解釈はローカル LLM を使わず、タスクと同じルールベースパーサ
(src/discord_bot/task_ui.py の split_quick_input / parse_due) を再利用する
(ローカルモデルの日付・TZ計算は信頼できないため)。

呼び出し側 (Discord bot / Web chat / 音声パイプライン) は LLM 応答の前に
try_register_event() を呼び、None 以外が返ったらその定型文をそのまま返信する
(「タスク:」即登録と同じ作法。口調の言い換えは配送側の自由)。
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo

from src.discord_bot.task_ui import split_quick_input

UTC = timezone.utc

DEFAULT_EVENT_DURATION_MIN = 60

# 明示プレフィックス: 「予定: ...」「よてい: ...」
_PREFIX_RE = re.compile(r"^\s*(?:予定|よてい)\s*[::]\s*(?P<body>.+)$", re.DOTALL)

# 自然文トリガー: 「〜(の)予定(を)入れて/追加して/登録して/作成して」
# 「カレンダーに〜(を)入れて/追加して/登録して」
# 依頼のしっぽ: 「〜入れておいてくれる?」「〜登録してください」などをまとめて消す
_TAIL = r"(?:\s*(?:おいて|置いて|もらって))?(?:ください|下さい|くれる?|もらえる?|ほしい|欲しい|ね|よ)?[??!!。〜~]*"
_NL_EVENT_RE = re.compile(
    r"(?:の)?予定(?:を|も)?\s*(?:入れて|いれて|追加して|登録して|作成して|作って|セットして)" + _TAIL
)
_NL_CALENDAR_RE = re.compile(
    r"カレンダー(?:に|へ|にも)\s*(?P<body>.*?)(?:を|も)?\s*(?:入れて|いれて|追加して|登録して|作成して|作って)" + _TAIL
)

# タイトル前後に残りがちな助詞・付属語
_TITLE_LEAD_RE = re.compile(r"^(?:に|へ|で|から|、|,|\s)+")
_TITLE_TAIL_RE = re.compile(r"(?:を|の|に|へ|、|,|\s)+$")


def detect_event_intent(text: str) -> Optional[str]:
    """予定登録の意図を検出し、日時+タイトルを含む本文を返す。意図が無ければ None。"""
    if not text or not text.strip():
        return None
    m = _PREFIX_RE.match(text)
    if m:
        return m.group("body").strip()
    m = _NL_CALENDAR_RE.search(text)
    if m:
        body = m.group("body").strip()
        # 「明日15時にカレンダーに歯医者入れて」のように日時がトリガーの外に
        # あることが多いので、トリガー部分を除いた全文を渡す。
        rest = (text[: m.start()] + " " + body + " " + text[m.end():]).strip()
        return rest or None
    m = _NL_EVENT_RE.search(text)
    if m:
        rest = (text[: m.start()] + " " + text[m.end():]).strip()
        return rest or None
    return None


def parse_event_request(body: str, now: datetime, tz: ZoneInfo) -> Optional[dict]:
    """本文から {title, start, end, all_day, display} を作る。日時が取れなければ None。

    start/end は GoogleCalendarMCPClient.create_event に渡せる形式:
    終日は "YYYY-MM-DD"、時刻付きはローカル naive ISO。
    """
    result = split_quick_input(body, now, tz)
    due_at = result.get("due_at")
    if due_at is None:
        return None
    title = _clean_title(str(result.get("title") or ""))
    if not title:
        return None

    local = due_at.astimezone(tz)
    if result.get("due_granularity") == "date":
        d = local.date()
        return {
            "title": title,
            "start": d.isoformat(),
            "end": (d + timedelta(days=1)).isoformat(),
            "all_day": True,
            "display": f"{d.month}/{d.day} 終日",
        }
    end_local = local + timedelta(minutes=DEFAULT_EVENT_DURATION_MIN)
    return {
        "title": title,
        "start": local.replace(tzinfo=None).isoformat(timespec="seconds"),
        "end": end_local.replace(tzinfo=None).isoformat(timespec="seconds"),
        "all_day": False,
        "display": f"{local.month}/{local.day} {local.strftime('%H:%M')}",
    }


def _clean_title(title: str) -> str:
    title = _TITLE_LEAD_RE.sub("", title.strip())
    title = _TITLE_TAIL_RE.sub("", title)
    # 「歯医者の予定」→「歯医者」
    for tail in ("の予定", "予定"):
        if title.endswith(tail) and len(title) > len(tail):
            title = title[: -len(tail)]
            break
    return title.strip()


def try_register_event(
    text: str,
    *,
    client: Any,
    calendar_id: str = "primary",
    timezone_name: str = "Asia/Tokyo",
    upcoming_path: str | Path | None = None,
    now: Optional[datetime] = None,
) -> Optional[str]:
    """text が予定登録要求なら Google Calendar に登録し、結果の定型文を返す。

    予定登録の意図が無いテキストには None を返す (呼び出し側は通常の
    LLM 応答へフォールスルーする)。意図はあるが日時が読めない場合や
    登録失敗時も、None ではなく案内文を返す (LLM が「登録したつもり」の
    応答を捏造するのを防ぐため)。
    """
    body = detect_event_intent(text)
    if body is None:
        return None

    now = now or datetime.now(UTC)
    tz = ZoneInfo(timezone_name)
    req = parse_event_request(body, now, tz)
    if req is None:
        return "予定の日時がわかりませんでした。「予定: 明日15時 歯医者」のように日時つきで言ってください。"

    if client is None:
        return "カレンダー連携が無効のため予定を登録できません (TASKS_CALENDAR_SYNC_ENABLED を確認してください)。"

    try:
        res = client.create_event(
            calendar_id=calendar_id,
            summary=req["title"],
            start=req["start"],
            end=req["end"],
            timezone=timezone_name,
        )
    except Exception as e:
        return f"予定の登録に失敗しました: {e}"
    if not getattr(res, "ok", False):
        return f"予定の登録に失敗しました: {getattr(res, 'error', '不明なエラー')}"

    if upcoming_path is not None:
        _append_upcoming(
            upcoming_path,
            {
                "title": req["title"],
                "start": req["start"],
                "end": req["end"],
                "location": "",
                "description": "",
                "event_id": getattr(res, "event_id", "") or "",
            },
        )
    return f"予定を登録しました: {req['display']} {req['title']}"


def _append_upcoming(upcoming_path: str | Path, event: dict) -> None:
    """upcoming.json へ楽観 append (best-effort、失敗しても無視)。

    Google には登録済みなので、失敗しても次回 pull で表示に追いつく。
    """
    path = Path(upcoming_path)
    try:
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        events = payload.get("events")
        if not isinstance(events, list):
            events = []
        events.append(event)
        events.sort(key=lambda e: str(e.get("start") or "") if isinstance(e, dict) else "")
        payload["events"] = events
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception as e:
        print(f"[EventIntent] upcoming.json 追記失敗 (無視): {e}")
