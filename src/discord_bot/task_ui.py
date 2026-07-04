"""
タスク管理の Discord UI とパーサ。

- parse_due:   「明日」「7/10」「7/10 15:00」程度の簡易日時パーサ (dateutil/LLM 不使用)
- parse_snooze: 「30m」「2h」「明日」等のスヌーズ時間パーサ
- validate_extraction: 自然会話から LLM が抽出した strict JSON の検証
- TaskConfirmView:  会話から抽出したタスクの「登録する/無視」確認ボタン
- TaskReminderView: リマインド配送メッセージの [完了][+30分][+2時間] ボタン

確認ボタンUIは bot.py の CorrectionPickView (👎修正候補ピック) の実装パターンを踏襲する。
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo

import discord

from src.tasks.store import VALID_PRIORITY

UTC = timezone.utc

TASK_PREFIX_RE = re.compile(r"^タスク\s*[:：]\s*(?P<body>.+)$", re.DOTALL)

_MD_RE = re.compile(r"(?P<month>\d{1,2})\s*[/／月]\s*(?P<day>\d{1,2})日?")
_HM_RE = re.compile(r"(?P<hour>\d{1,2})\s*[:：時]\s*(?P<minute>\d{1,2})?分?")
_REL_MIN_RE = re.compile(r"(?P<n>\d+)\s*分後")
_REL_HOUR_RE = re.compile(r"(?P<n>\d+)\s*時間後")
_REL_DAY_RE = re.compile(r"(?P<n>\d+)\s*日後")


def _local_to_utc(local_naive: datetime, tz: ZoneInfo) -> datetime:
    return local_naive.replace(tzinfo=tz).astimezone(UTC)


def parse_due(text: str, now: datetime, tz: ZoneInfo) -> tuple[Optional[datetime], Optional[str]]:
    """自由文の期限を (due_at_utc, granularity) に変換する。解釈不能なら (None, None)。

    granularity='date' の場合は due_at をローカル 23:59 に正規化する。
    """
    if not text or not text.strip():
        return (None, None)
    s = text.strip()
    local_now = now.astimezone(tz)

    # --- 相対表現 ---
    m = _REL_MIN_RE.search(s)
    if m:
        return (now + timedelta(minutes=int(m.group("n"))), "datetime")
    m = _REL_HOUR_RE.search(s)
    if m:
        return (now + timedelta(hours=int(m.group("n"))), "datetime")
    m = _REL_DAY_RE.search(s)
    if m:
        base = (local_now + timedelta(days=int(m.group("n")))).replace(
            hour=23, minute=59, second=0, microsecond=0
        )
        return (_local_to_utc(base.replace(tzinfo=None), tz), "date")

    # --- 日付ワード ---
    base_date = None
    if "今日" in s or "本日" in s:
        base_date = local_now.date()
    elif "明後日" in s:
        base_date = (local_now + timedelta(days=2)).date()
    elif "明日" in s:
        base_date = (local_now + timedelta(days=1)).date()

    # --- M/D ---
    md = _MD_RE.search(s)
    if md and base_date is None:
        month = int(md.group("month"))
        day = int(md.group("day"))
        year = local_now.year
        try:
            candidate = local_now.date().replace(month=month, day=day, year=year)
        except ValueError:
            return (None, None)
        # 過去日付なら翌年扱い
        if candidate < local_now.date():
            try:
                candidate = candidate.replace(year=year + 1)
            except ValueError:
                pass
        base_date = candidate

    # --- 時刻 ---
    hm = _HM_RE.search(s)
    time_hour = time_minute = None
    if hm:
        h = int(hm.group("hour"))
        mi = int(hm.group("minute")) if hm.group("minute") else 0
        if 0 <= h <= 23 and 0 <= mi <= 59:
            time_hour, time_minute = h, mi

    if base_date is not None:
        if time_hour is not None:
            local_naive = datetime(
                base_date.year, base_date.month, base_date.day, time_hour, time_minute
            )
            return (_local_to_utc(local_naive, tz), "datetime")
        # 時刻なし → その日の 23:59 締切
        local_naive = datetime(base_date.year, base_date.month, base_date.day, 23, 59)
        return (_local_to_utc(local_naive, tz), "date")

    # --- 時刻のみ (今日/翌日の HH:MM) ---
    if time_hour is not None:
        local_naive = datetime(
            local_now.year, local_now.month, local_now.day, time_hour, time_minute
        )
        due = _local_to_utc(local_naive, tz)
        if due <= now:
            due = due + timedelta(days=1)
        return (due, "datetime")

    return (None, None)


def parse_snooze(text: str, now: datetime, tz: ZoneInfo) -> Optional[datetime]:
    """スヌーズ時間を until(UTC) に変換する。解釈不能なら None。"""
    if not text or not text.strip():
        return None
    s = text.strip().lower()
    local_now = now.astimezone(tz)

    m = re.fullmatch(r"(\d+)\s*(m|min|分)", s)
    if m:
        return now + timedelta(minutes=int(m.group(1)))
    m = re.fullmatch(r"(\d+)\s*(h|hr|時間)", s)
    if m:
        return now + timedelta(hours=int(m.group(1)))
    m = re.fullmatch(r"(\d+)\s*(d|day|日)", s)
    if m:
        return now + timedelta(days=int(m.group(1)))
    if s in ("明日", "tomorrow"):
        tomorrow = (local_now + timedelta(days=1)).replace(
            hour=9, minute=0, second=0, microsecond=0
        )
        return _local_to_utc(tomorrow.replace(tzinfo=None), tz)
    if "明後日" in s:
        d = (local_now + timedelta(days=2)).replace(hour=9, minute=0, second=0, microsecond=0)
        return _local_to_utc(d.replace(tzinfo=None), tz)
    # 「30分後」等も許容
    due, _ = parse_due(s, now, tz)
    return due


def validate_extraction(raw: Any, assume_tz: Optional[ZoneInfo] = None) -> Optional[dict]:
    """LLM 抽出結果 (strict JSON) を検証・正規化する。

    期待形: {"is_task": bool, "title": str, "due": ISO or null, "priority": str}
    不正・is_task=false・title 空 の場合は None。
    タイムゾーンなしの due は assume_tz (既定 Asia/Tokyo) の時刻として解釈する —
    抽出プロンプトが現在日時を Asia/Tokyo で提示しているため。
    """
    if isinstance(raw, str):
        raw = raw.strip()
        # ```json ... ``` フェンスを剥がす
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw).strip()
        try:
            obj = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return None
    elif isinstance(raw, dict):
        obj = raw
    else:
        return None

    if not isinstance(obj, dict):
        return None
    if obj.get("is_task") is not True:
        return None
    title = obj.get("title")
    if not isinstance(title, str) or not title.strip():
        return None

    due_at: Optional[datetime] = None
    due_val = obj.get("due")
    if isinstance(due_val, str) and due_val.strip():
        try:
            dt = datetime.fromisoformat(due_val.strip().replace("Z", "+00:00"))
            local_tz = assume_tz or ZoneInfo("Asia/Tokyo")
            due_at = dt if dt.tzinfo else dt.replace(tzinfo=local_tz)
            due_at = due_at.astimezone(UTC)
        except (ValueError, TypeError):
            due_at = None

    priority = obj.get("priority")
    if priority not in VALID_PRIORITY:
        priority = "normal"

    return {
        "title": title.strip()[:200],
        "due_at": due_at,
        "priority": priority,
    }


def build_extraction_prompt(now_local: datetime) -> str:
    """自然会話からタスクを抽出させるためのシステムプロンプト。"""
    weekday_ja = "月火水木金土日"[now_local.weekday()]
    now_str = now_local.strftime("%Y-%m-%d %H:%M")
    # LLMの日付計算は信用できないので、相対表現の換算表を計算済みで渡す
    tomorrow = now_local + timedelta(days=1)
    next_monday = now_local + timedelta(days=7 - now_local.weekday())
    next_friday = next_monday + timedelta(days=4)
    return (
        "あなたはユーザーの発言から『やるべきタスク』を抽出する抽出器です。"
        f"現在日時は {now_str}、今日は{weekday_ja}曜日です (Asia/Tokyo)。\n"
        "出力は必ず次の厳密なJSONのみ。前置き・説明・コードフェンスは禁止。\n"
        '{"is_task": true/false, "title": "簡潔なタスク名", '
        '"due": "ISO8601の絶対日時 または null", "priority": "high/normal/low"}\n'
        "- 依頼・締切・予定・やること が含まれるときだけ is_task=true。\n"
        "- 相対表現(明日・来週など)は現在日時を基準に絶対日時へ変換する。\n"
        f"- 日付換算表 (これに従うこと): 明日={tomorrow:%Y-%m-%d}、"
        f"来週の月曜={next_monday:%Y-%m-%d}、来週の金曜={next_friday:%Y-%m-%d}。"
        "他の曜日も「来週の月曜」からの日数で数える。\n"
        "- 時刻が不明で日付だけなら、その日の 23:59 を due にする。\n"
        "- 期限が全く不明なら due は null。\n"
        "- 雑談・感想・質問など、やることでないものは is_task=false。"
    )


def _refresh_board(state: Any) -> None:
    """常設タスクボードがあれば更新を要求する (無ければ何もしない)。"""
    refresh = getattr(state, "refresh_task_board", None)
    if callable(refresh):
        try:
            refresh()
        except Exception:
            pass


def make_due_summary(due_at: Optional[datetime], tz: ZoneInfo) -> str:
    if due_at is None:
        return "期限なし"
    return due_at.astimezone(tz).strftime("%-m/%-d %H:%M")


class TaskConfirmView(discord.ui.View):
    """会話から抽出したタスクの「登録する/無視」確認UI。黙って自動登録はしない。"""

    def __init__(
        self,
        state: Any,
        *,
        title: str,
        due_at: Optional[datetime],
        due_granularity: Optional[str],
        priority: str,
        source: str,
        timeout: float = 300.0,
    ):
        super().__init__(timeout=timeout)
        self.state = state
        self.title = title
        self.due_at = due_at
        self.due_granularity = due_granularity
        self.priority = priority
        self.source = source
        self.message: discord.Message | None = None

        register_btn = discord.ui.Button(label="登録する", style=discord.ButtonStyle.success)
        register_btn.callback = self._register
        self.add_item(register_btn)

        ignore_btn = discord.ui.Button(label="無視", style=discord.ButtonStyle.secondary)
        ignore_btn.callback = self._ignore
        self.add_item(ignore_btn)

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        channel_id = interaction.channel_id or 0
        if self.state.is_allowed_feedback(interaction.user.id, channel_id):
            return True
        await interaction.response.send_message("この操作は許可されていません。", ephemeral=True)
        return False

    async def _register(self, interaction: discord.Interaction) -> None:
        store = getattr(self.state, "task_store", None)
        if store is None:
            await interaction.response.edit_message(content="タスクストアが無効です。", view=None)
            self.stop()
            return
        try:
            import asyncio

            task_id = await asyncio.to_thread(
                store.add,
                self.title,
                due_at=self.due_at,
                due_granularity=self.due_granularity,
                priority=self.priority,
                source=self.source,
            )
        except Exception as e:
            await interaction.response.edit_message(content=f"登録に失敗しました: `{e}`", view=None)
            self.stop()
            return
        due_str = make_due_summary(self.due_at, ZoneInfo(getattr(store, "timezone_name", "Asia/Tokyo")))
        self.stop()
        await interaction.response.edit_message(
            content=f"タスクを登録しました (#{task_id}): {self.title} (期限: {due_str})",
            view=None,
        )
        _refresh_board(self.state)

    async def _ignore(self, interaction: discord.Interaction) -> None:
        self.stop()
        await interaction.response.edit_message(content="タスク登録を見送りました。", view=None)

    async def on_timeout(self) -> None:
        if self.message is not None:
            try:
                await self.message.edit(content="タスク確認は期限切れになりました。", view=None)
            except discord.HTTPException:
                pass


class TaskReminderView(discord.ui.View):
    """リマインド配送メッセージに付ける [完了][+30分][+2時間] ボタン。"""

    def __init__(self, state: Any, task_id: int, *, timeout: float = 86400.0):
        super().__init__(timeout=timeout)
        self.state = state
        self.task_id = task_id

        done_btn = discord.ui.Button(label="完了", style=discord.ButtonStyle.success)
        done_btn.callback = self._done
        self.add_item(done_btn)

        snooze30_btn = discord.ui.Button(label="+30分", style=discord.ButtonStyle.secondary)
        snooze30_btn.callback = self._make_snooze(timedelta(minutes=30), "+30分")
        self.add_item(snooze30_btn)

        snooze2h_btn = discord.ui.Button(label="+2時間", style=discord.ButtonStyle.secondary)
        snooze2h_btn.callback = self._make_snooze(timedelta(hours=2), "+2時間")
        self.add_item(snooze2h_btn)

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        channel_id = interaction.channel_id or 0
        if self.state.is_allowed_feedback(interaction.user.id, channel_id):
            return True
        await interaction.response.send_message("この操作は許可されていません。", ephemeral=True)
        return False

    def _store(self):
        return getattr(self.state, "task_store", None)

    async def _done(self, interaction: discord.Interaction) -> None:
        import asyncio

        store = self._store()
        if store is None:
            await interaction.response.send_message("タスクストアが無効です。", ephemeral=True)
            return
        ok = await asyncio.to_thread(store.done, self.task_id)
        self.stop()
        text = "完了にしました。おつかれさま。" if ok else "すでに完了/削除済みでした。"
        await interaction.response.edit_message(content=text, view=None)
        if ok:
            _refresh_board(self.state)

    def _make_snooze(self, delta: timedelta, label: str):
        async def _snooze(interaction: discord.Interaction) -> None:
            import asyncio

            store = self._store()
            if store is None:
                await interaction.response.send_message("タスクストアが無効です。", ephemeral=True)
                return
            until = datetime.now(UTC) + delta
            ok = await asyncio.to_thread(store.snooze, self.task_id, until)
            self.stop()
            text = f"{label}スヌーズしました。また声かけますね。" if ok else "すでに完了/削除済みでした。"
            await interaction.response.edit_message(content=text, view=None)
            if ok:
                _refresh_board(self.state)

        return _snooze
