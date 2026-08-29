"""
タスク管理の Discord UI とパーサ。

- parse_due:   「明日」「金曜」「来週水曜」「午後3時」「7/10 15:00」等の簡易日時パーサ (dateutil/LLM 不使用)
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

from src.tasks.extractor import build_extraction_prompt, validate_extraction
from src.tasks.formatting import format_short_due

UTC = timezone.utc

TASK_PREFIX_RE = re.compile(r"^タスク\s*[:：]\s*(?P<body>.+)$", re.DOTALL)

_ZEN2HAN = str.maketrans("０１２３４５６７８９：／−", "0123456789:/-")

# STT等でひらがなになりがちな表現を漢字表記へ寄せる (長い語から先に置換する)
_KANA_ALIASES = (
    ("あさって", "明後日"),
    ("あした", "明日"),
    ("あす", "明日"),
    ("きょう", "今日"),
    ("こんや", "今夜"),
    ("らいしゅう", "来週"),
    ("しゅうまつ", "週末"),
)

_YMD_RE = re.compile(
    r"(?P<year>20\d{2})\s*[-/／年]\s*(?P<month>\d{1,2})\s*[-/／月]\s*(?P<day>\d{1,2})日?"
)
_MD_RE = re.compile(r"(?P<month>\d{1,2})\s*[/／月]\s*(?P<day>\d{1,2})日?")
_DAY_ONLY_RE = re.compile(r"(?<![\d/／月])(?P<day>\d{1,2})日(?![後分\d])")
_HM_RE = re.compile(
    r"(?P<ampm>午前|午後)?\s*(?P<hour>\d{1,2})\s*"
    r"(?:[:：]\s*(?P<minute>\d{1,2})?|時\s*(?:(?P<half>半)|(?P<minute2>\d{1,2})\s*分?)?)"
)
_REL_MIN_RE = re.compile(r"(?P<n>\d+)\s*分後")
_REL_HOUR_RE = re.compile(r"(?P<n>\d+)\s*時間後")
_REL_DAY_RE = re.compile(r"(?P<n>\d+)\s*日後")
_REL_WEEK_RE = re.compile(r"(?P<n>\d+)\s*週間後")

_WEEKDAY_IDX = {"月": 0, "火": 1, "水": 2, "木": 3, "金": 4, "土": 5, "日": 6}
_WEEKDAY_RE = re.compile(r"(?P<week>再来週|来週|今週)?\s*の?\s*(?P<wd>[月火水木金土日])曜日?")

# 「朝」「夜」等のあいまい時刻の既定値 (hour, minute)
_TIME_WORDS = (
    ("正午", (12, 0)),
    ("朝", (9, 0)),
    ("昼", (12, 0)),
    ("夕方", (17, 0)),
    ("今夜", (20, 0)),
    ("夜", (20, 0)),
    ("晩", (20, 0)),
)


def _local_to_utc(local_naive: datetime, tz: ZoneInfo) -> datetime:
    return local_naive.replace(tzinfo=tz).astimezone(UTC)


def parse_due(text: str, now: datetime, tz: ZoneInfo) -> tuple[Optional[datetime], Optional[str]]:
    """自由文の期限を (due_at_utc, granularity) に変換する。解釈不能なら (None, None)。

    granularity='date' の場合は due_at をローカル 23:59 に正規化する。
    """
    if not text or not text.strip():
        return (None, None)
    s = text.strip().translate(_ZEN2HAN)
    for kana, kanji in _KANA_ALIASES:
        s = s.replace(kana, kanji)
    local_now = now.astimezone(tz)

    # --- 相対表現 ---
    m = _REL_MIN_RE.search(s)
    if m:
        return (now + timedelta(minutes=int(m.group("n"))), "datetime")
    m = _REL_HOUR_RE.search(s)
    if m:
        return (now + timedelta(hours=int(m.group("n"))), "datetime")
    m = _REL_DAY_RE.search(s) or _REL_WEEK_RE.search(s)
    if m:
        days = int(m.group("n")) * (7 if "週間後" in m.group(0) else 1)
        base = (local_now + timedelta(days=days)).replace(
            hour=23, minute=59, second=0, microsecond=0
        )
        return (_local_to_utc(base.replace(tzinfo=None), tz), "date")

    # --- 日付ワード ---
    base_date = None
    if "今日" in s or "本日" in s or "今夜" in s:
        base_date = local_now.date()
    elif "明後日" in s:
        base_date = (local_now + timedelta(days=2)).date()
    elif "明日" in s:
        base_date = (local_now + timedelta(days=1)).date()

    # --- 曜日 (「金曜」「来週水曜」「今週土曜日」) ---
    if base_date is None:
        wd = _WEEKDAY_RE.search(s)
        if wd:
            target = _WEEKDAY_IDX[wd.group("wd")]
            monday = local_now.date() - timedelta(days=local_now.weekday())
            week = wd.group("week")
            if week == "今週":
                base_date = monday + timedelta(days=target)
            elif week == "来週":
                base_date = monday + timedelta(days=7 + target)
            elif week == "再来週":
                base_date = monday + timedelta(days=14 + target)
            else:
                # 修飾なしは「次にくるその曜日」(今日を含む)
                base_date = local_now.date() + timedelta(
                    days=(target - local_now.weekday()) % 7
                )

    # --- 週の単独表現 (「週末」「来週」「再来週」) ---
    if base_date is None:
        monday = local_now.date() - timedelta(days=local_now.weekday())
        if "来週末" in s:
            base_date = monday + timedelta(days=7 + 5)
        elif "週末" in s:
            # 次にくる土曜 (今日を含む)
            base_date = local_now.date() + timedelta(days=(5 - local_now.weekday()) % 7)
        elif "再来週" in s:
            # 「再来週中に」= 再来週の日曜まで
            base_date = monday + timedelta(days=14 + 6)
        elif "来週" in s:
            # 「来週中に」= 来週の日曜まで
            base_date = monday + timedelta(days=7 + 6)

    # --- YYYY/MM/DD・YYYY-MM-DD・YYYY年M月D日 ---
    if base_date is None:
        ymd = _YMD_RE.search(s)
        if ymd:
            try:
                base_date = local_now.date().replace(
                    year=int(ymd.group("year")),
                    month=int(ymd.group("month")),
                    day=int(ymd.group("day")),
                )
            except ValueError:
                return (None, None)

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

    # --- 日のみ (「10日」= 今月の10日、過ぎていれば来月) ---
    if base_date is None:
        dm = _DAY_ONLY_RE.search(s)
        if dm:
            day = int(dm.group("day"))
            for months_ahead in (0, 1):
                year = local_now.year + (local_now.month - 1 + months_ahead) // 12
                month = (local_now.month - 1 + months_ahead) % 12 + 1
                try:
                    candidate = local_now.date().replace(year=year, month=month, day=day)
                except ValueError:
                    continue
                if candidate >= local_now.date():
                    base_date = candidate
                    break

    # --- 時刻 (「15:00」「15時半」「午後3時」) ---
    hm = _HM_RE.search(s)
    time_hour = time_minute = None
    if hm:
        h = int(hm.group("hour"))
        if hm.group("half"):
            mi = 30
        else:
            mi_raw = hm.group("minute") or hm.group("minute2")
            mi = int(mi_raw) if mi_raw else 0
        if hm.group("ampm") == "午後" and h < 12:
            h += 12
        elif hm.group("ampm") == "午前" and h == 12:
            h = 0
        if 0 <= h <= 23 and 0 <= mi <= 59:
            time_hour, time_minute = h, mi

    # --- あいまい時刻ワード (「朝」「夕方」「今夜」) ---
    if time_hour is None:
        for word, (h, mi) in _TIME_WORDS:
            if word in s:
                time_hour, time_minute = h, mi
                break

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


def split_quick_input(text: str, now: datetime, tz: ZoneInfo) -> dict:
    """自由文1行からタイトル・期限・優先度を分離する。

    戻り値: {"title": str, "due_at": datetime|None, "due_granularity": str|None, "priority": str}

    処理:
    1. 正規化 (_ZEN2HAN, _KANA_ALIASES)
    2. 優先度抽出 (「!」「至急」「急ぎ」「重要」→ high; 「あとで」「いつか」→ low)
    3. 期限表現スパン検出と除去
    4. parse_due で due_at 計算
    5. title は除去後の残りを strip して連続空白を畳む
    """
    if not text or not text.strip():
        return {
            "title": "",
            "due_at": None,
            "due_granularity": None,
            "priority": "normal",
        }

    # Step 1: 正規化
    s = text.strip().translate(_ZEN2HAN)
    for kana, kanji in _KANA_ALIASES:
        s = s.replace(kana, kanji)

    # Step 2: 優先度抽出
    priority = "normal"

    # 高優先度マーカー: 先頭の ! / ！
    high_prefix = re.match(r"^[!！]+\s*", s)
    if high_prefix:
        priority = "high"
        s = s[high_prefix.end():]

    # 高優先度マーカー: 末尾の ! / ！
    high_suffix = re.search(r"\s*[!！]+$", s)
    if high_suffix:
        priority = "high"
        s = s[:high_suffix.start()]

    # 高優先度キーワード: 「至急」「急ぎ」「重要」(複数可、全て除去)
    for word in ("至急", "急ぎ", "重要"):
        if word in s:
            priority = "high"
            s = s.replace(word, "")

    # 低優先度キーワード: 「あとで」「いつか」
    for word in ("あとで", "いつか"):
        if word in s:
            priority = "low"
            s = s.replace(word, "")

    # Step 3: 期限表現スパン検出と除去
    # スパン除去前のテキストを due_at 計算用に保存
    s_for_due = s.strip()

    # 期限表現マッチパターンを収集 (parse_due が使うのと同じ)
    # 複数マッチの場合は最後に統合
    spans_to_remove = []

    # 相対表現: N分後、N時間後、N日後、N週間後
    for pattern in (_REL_MIN_RE, _REL_HOUR_RE, _REL_DAY_RE, _REL_WEEK_RE):
        for m in pattern.finditer(s):
            spans_to_remove.append((m.start(), m.end()))

    # 日付ワード: 今日、本日、明日、明後日、今夜
    for word in ("今日", "本日", "明日", "明後日", "今夜"):
        for m in re.finditer(re.escape(word), s):
            spans_to_remove.append((m.start(), m.end()))

    # 曜日表現: _WEEKDAY_RE
    for m in _WEEKDAY_RE.finditer(s):
        spans_to_remove.append((m.start(), m.end()))

    # 週表現: 週末、来週末、来週、再来週 (長い順に)
    for word in ("来週末", "週末", "再来週", "来週"):
        for m in re.finditer(re.escape(word), s):
            spans_to_remove.append((m.start(), m.end()))

    # 日付: YYYY/MM/DD、YYYY-MM-DD、YYYY年M月D日
    for m in _YMD_RE.finditer(s):
        spans_to_remove.append((m.start(), m.end()))

    # 日付: M/D、M月D日
    for m in _MD_RE.finditer(s):
        spans_to_remove.append((m.start(), m.end()))

    # 日のみ: N日 (「日分」には対応していない _DAY_ONLY_RE ではなく _DAY_ONLY_NO_SUFFIX_RE を使用)
    for m in _DAY_ONLY_RE.finditer(s):
        spans_to_remove.append((m.start(), m.end()))

    # 時刻: HH:MM、H時M分、午後3時等
    for m in _HM_RE.finditer(s):
        spans_to_remove.append((m.start(), m.end()))

    # あいまい時刻ワード: 朝、昼、夕方、夜、晩、正午
    for word, _ in _TIME_WORDS:
        for m in re.finditer(re.escape(word), s):
            spans_to_remove.append((m.start(), m.end()))

    # スパンの重複排除と統合
    if spans_to_remove:
        spans_to_remove.sort()
        merged = []
        for start, end in spans_to_remove:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))

        # 除去するスパンの直後の助詞も一緒に除去
        particles = ("までに", "まで", "に", "の")
        final_spans = []
        for start, end in merged:
            extended_end = end
            rest = s[end:]
            for particle in particles:
                if rest.startswith(particle):
                    extended_end = end + len(particle)
                    break
            final_spans.append((start, extended_end))

        # テキストから除去 (逆順で処理)
        for start, end in reversed(final_spans):
            s = s[:start] + s[end:]

    # Step 4: due_at を計算 (スパン除去前のテキスト全体を parse_due に渡す)
    due_at, due_granularity = parse_due(s_for_due, now, tz)

    # Step 5: title 生成
    title = s.strip()
    # 連続空白を1つに畳む
    title = re.sub(r"\s+", " ", title)

    # title が空の場合は正規化済み入力全体を title にする (due は計算済みをそのまま返す)
    if not title:
        title = s_for_due.strip()
        title = re.sub(r"\s+", " ", title)

    return {
        "title": title,
        "due_at": due_at,
        "due_granularity": due_granularity,
        "priority": priority,
    }



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
    return format_short_due(due_at.astimezone(tz), with_time=True)


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
