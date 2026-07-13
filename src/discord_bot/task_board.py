"""
常設タスクボード + 低摩擦タスク登録UI。

- TaskModal:        右クリック登録 / ボード[＋追加] の共通入力フォーム (機能A)
- 純関数の整形器:    format_remaining / format_board_line / build_board_description /
                    select_option_label / build_select_options  (テスト対象)
- TaskBoardView:    ボードメッセージに載る永続View ([＋追加] ボタン + タスク選択 Select)
- TaskActionView:   Select で選んだ1件への [完了][編集][+30分][+2時間][削除] (ephemeral・揮発)
- TaskBoardManager: ボードメッセージのライフサイクル (探索/作成/ピン) と
                    デバウンス refresh、15分毎の定期 refresh を管理

永続Viewの custom_id 体系:
    taskboard:add:v1     ＋追加ボタン
    taskboard:select:v1  タスク選択 Select
アクションボタンは ephemeral メッセージ上にのみ出るため揮発Viewでよい
(再起動で ephemeral メッセージ自体が消えるので永続化しても意味がない)。
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo

import discord

from src.discord_bot.task_ui import make_due_summary, parse_due
from src.tasks.prioritizer import format_focus_decision
from src.tasks.store import VALID_PRIORITY

UTC = timezone.utc

BOARD_TITLE = "📋 タスクボード"
BOARD_MAX_ITEMS = 15
SELECT_MAX_OPTIONS = 25

ADD_BUTTON_CUSTOM_ID = "taskboard:add:v1"
SELECT_CUSTOM_ID = "taskboard:select:v1"
FOCUS_BUTTON_CUSTOM_ID = "taskboard:focus:v1"

_PRIORITY_EMOJI = {"high": "🔴", "normal": "⚪", "low": "🔵"}


# --------------------------------------------------------------------------
# 純関数の整形器 (テスト対象。discord オブジェクトに依存しない)
# --------------------------------------------------------------------------

def format_remaining(due_at: Optional[datetime], now: datetime) -> str:
    """期限までの残り時間 / 超過時間を短い日本語にする。期限なしは ""。"""
    if due_at is None:
        return ""
    delta = due_at - now
    overdue = delta.total_seconds() < 0
    secs = abs(delta.total_seconds())
    days = int(secs // 86400)
    hours = int((secs % 86400) // 3600)
    minutes = int((secs % 3600) // 60)
    if days >= 1:
        span = f"{days}日"
    elif hours >= 1:
        span = f"{hours}時間"
    else:
        span = f"{max(1, minutes)}分"
    return f"超過{span}" if overdue else f"あと{span}"


def _due_label(due_at: Optional[datetime], granularity: Optional[str], tz: ZoneInfo) -> str:
    if due_at is None:
        return "期限なし"
    local = due_at.astimezone(tz)
    if granularity == "date":
        return local.strftime("%-m/%-d")
    return local.strftime("%-m/%-d %H:%M")


def format_due_input(due_at: Optional[datetime], granularity: Optional[str], tz: ZoneInfo) -> str:
    """編集モーダルの期限欄へ入れる短い表記。空欄は期限なしを表す。"""
    if due_at is None:
        return ""
    local = due_at.astimezone(tz)
    if granularity == "date":
        return local.strftime("%-m/%-d")
    return local.strftime("%-m/%-d %H:%M")


def _truncate(text: str, limit: int) -> str:
    text = (text or "").strip().replace("\n", " ")
    if len(text) <= limit:
        return text
    return text[: max(1, limit - 1)] + "…"


def format_board_line(task: dict, tz: ZoneInfo, now: datetime, *, title_limit: int = 42) -> str:
    """1タスクを1行に整形する。例:
    `#12` 🔴 レポート提出 — 7/10 15:00 (あと2日)
    ⚠️ `#7` ⚪ 家賃振込 — 7/1 (超過3日)
    """
    due = task["due_at"]
    prio = _PRIORITY_EMOJI.get(task.get("priority"), "⚪")
    title = _truncate(task.get("title", ""), title_limit)
    due_label = _due_label(due, task.get("due_granularity"), tz)
    overdue = due is not None and due < now
    remaining = format_remaining(due, now)
    head = "⚠️ " if overdue else ""
    body = f"`#{task['id']}` {prio} {title} — {due_label}"
    if remaining:
        body += f" ({remaining})"
    return head + body


def build_board_description(
    tasks: list[dict], tz: ZoneInfo, now: datetime, *, limit: int = BOARD_MAX_ITEMS
) -> str:
    """ボード embed の本文テキスト。0件なら「タスクなし」表示。"""
    if not tasks:
        return "タスクはありません ✨"
    shown = tasks[:limit]
    lines = [format_board_line(t, tz, now) for t in shown]
    hidden = len(tasks) - len(shown)
    if hidden > 0:
        lines.append(f"…ほか {hidden} 件")
    return "\n".join(lines)


def select_option_label(task: dict, tz: ZoneInfo, now: datetime) -> tuple[str, str]:
    """Select 1件分の (label, description)。label<=100, description<=100 を保証。"""
    prio = _PRIORITY_EMOJI.get(task.get("priority"), "⚪")
    label = _truncate(f"#{task['id']} {task.get('title', '')}", 100)
    due = task["due_at"]
    due_label = _due_label(due, task.get("due_granularity"), tz)
    remaining = format_remaining(due, now)
    overdue = due is not None and due < now
    desc_parts = [f"{prio}", due_label]
    if remaining:
        desc_parts.append(f"({'⚠️' if overdue else ''}{remaining})")
    return label, _truncate(" ".join(desc_parts), 100)


def build_select_options(
    tasks: list[dict], tz: ZoneInfo, now: datetime, *, limit: int = SELECT_MAX_OPTIONS
) -> list[discord.SelectOption]:
    """未完了タスクから Select の options を作る (25件上限)。0件なら無し。"""
    options: list[discord.SelectOption] = []
    for task in tasks[:limit]:
        label, description = select_option_label(task, tz, now)
        options.append(
            discord.SelectOption(label=label, value=str(task["id"]), description=description)
        )
    return options


def resolve_task_input(
    *,
    title: str,
    due_raw: str,
    priority_raw: str,
    note_raw: str,
    now: datetime,
    tz: ZoneInfo,
) -> tuple[Optional[dict], Optional[str]]:
    """モーダル入力を store.add 用の kwargs に正規化する。

    成功時 (kwargs, None)、失敗時 (None, ephemeral で出すエラー文) を返す。
    期限が解釈不能なら登録しない (エラーを返す)。優先度は不正/空欄なら normal。
    """
    title = (title or "").strip()
    if not title:
        return None, "タイトルを入力してください。"

    due_at = None
    granularity = None
    due_raw = (due_raw or "").strip()
    if due_raw:
        due_at, granularity = parse_due(due_raw, now, tz)
        if due_at is None:
            return None, (
                f"期限「{due_raw}」を解釈できませんでした。登録していません。\n"
                "例: 明日 18時 / 金曜 / 来週水曜 / 7/10 15:00"
            )

    priority = (priority_raw or "").strip().lower()
    if priority not in VALID_PRIORITY:
        priority = "normal"
    note = (note_raw or "").strip() or None

    return (
        {
            "title": title[:200],
            "due_at": due_at,
            "due_granularity": granularity,
            "priority": priority,
            "note": note,
        },
        None,
    )


def resolve_task_edit_input(
    *,
    current_task: dict,
    title: str,
    due_raw: str,
    default_due_raw: str,
    priority_raw: str,
    note_raw: str,
    now: datetime,
    tz: ZoneInfo,
) -> tuple[Optional[dict], Optional[str]]:
    """編集モーダル入力を store.update 用の kwargs に正規化する。

    期限欄は空欄で期限削除。既定値から変更されていなければ、期限超過タスクでも
    parse_due の「過去日付は翌年」補正を踏まず、保存済み due_at を維持する。
    """
    title = (title or "").strip()
    if not title:
        return None, "タイトルを入力してください。"

    due_raw = (due_raw or "").strip()
    default_due_raw = (default_due_raw or "").strip()
    due_at = None
    granularity = None
    clear_due = False
    if not due_raw:
        clear_due = current_task.get("due_at") is not None
    elif due_raw == default_due_raw and current_task.get("due_at") is not None:
        due_at = current_task.get("due_at")
        granularity = current_task.get("due_granularity")
    else:
        due_at, granularity = parse_due(due_raw, now, tz)
        if due_at is None:
            return None, (
                f"期限「{due_raw}」を解釈できませんでした。更新していません。\n"
                "例: 明日 18時 / 金曜 / 来週水曜 / 7/10 15:00"
            )

    priority = (priority_raw or "").strip().lower()
    if priority not in VALID_PRIORITY:
        priority = "normal"

    return (
        {
            "title": title[:200],
            "due_at": due_at,
            "due_granularity": granularity,
            "clear_due": clear_due,
            "priority": priority,
            "note": (note_raw or "").strip(),
        },
        None,
    )


def build_board_embed(
    tasks: list[dict], tz: ZoneInfo, now: datetime, *, limit: int = BOARD_MAX_ITEMS
) -> discord.Embed:
    embed = discord.Embed(
        title=BOARD_TITLE,
        description=build_board_description(tasks, tz, now, limit=limit),
        color=0x5865F2,
    )
    updated = now.astimezone(tz).strftime("%-m/%-d %H:%M")
    embed.set_footer(text=f"更新 {updated} · 未完了 {len(tasks)} 件")
    return embed


# --------------------------------------------------------------------------
# 共通入力フォーム (機能A)
# --------------------------------------------------------------------------

class TaskModal(discord.ui.Modal, title="タスクに登録"):
    """右クリック登録 / ボード[＋追加] で開く共通モーダル。

    重い処理 (LLM等) はここで行わない (モーダルは3秒以内に表示する必要があるため、
    呼び出し側は send_modal だけを速やかに実行する)。
    """

    def __init__(self, state: Any, *, prefill_title: str = "", source: str = "context_menu"):
        super().__init__()
        self.state = state
        self.source = source

        self.title_input: discord.ui.TextInput = discord.ui.TextInput(
            label="タイトル",
            default=(prefill_title or "")[:120],
            max_length=200,
            required=True,
        )
        self.due_input: discord.ui.TextInput = discord.ui.TextInput(
            label="期限 (空欄可)",
            placeholder="明日 / 7/10 / 7/10 15:00",
            required=False,
            max_length=50,
        )
        self.priority_input: discord.ui.TextInput = discord.ui.TextInput(
            label="優先度 (空欄=normal)",
            placeholder="high / normal / low",
            required=False,
            max_length=10,
        )
        self.note_input: discord.ui.TextInput = discord.ui.TextInput(
            label="メモ (任意)",
            style=discord.TextStyle.paragraph,
            required=False,
            max_length=1000,
        )
        for item in (self.title_input, self.due_input, self.priority_input, self.note_input):
            self.add_item(item)

    def _tz(self) -> ZoneInfo:
        return ZoneInfo(getattr(self.state, "tasks_timezone", "Asia/Tokyo"))

    async def on_submit(self, interaction: discord.Interaction) -> None:
        store = getattr(self.state, "task_store", None)
        if store is None:
            await interaction.response.send_message("タスクストアが無効です。", ephemeral=True)
            return

        tz = self._tz()
        kwargs, error = resolve_task_input(
            title=str(self.title_input.value or ""),
            due_raw=str(self.due_input.value or ""),
            priority_raw=str(self.priority_input.value or ""),
            note_raw=str(self.note_input.value or ""),
            now=datetime.now(UTC),
            tz=tz,
        )
        if error is not None:
            await interaction.response.send_message(error, ephemeral=True)
            return

        try:
            task_id = await asyncio.to_thread(
                store.add,
                kwargs["title"],
                note=kwargs["note"],
                due_at=kwargs["due_at"],
                due_granularity=kwargs["due_granularity"],
                priority=kwargs["priority"],
                source=self.source,
            )
        except Exception as e:
            await interaction.response.send_message(f"登録に失敗しました: `{e}`", ephemeral=True)
            return

        due_str = make_due_summary(kwargs["due_at"], tz)
        await interaction.response.send_message(
            f"タスクを登録しました (#{task_id}): {kwargs['title']}"
            f" (期限: {due_str}, 優先度: {kwargs['priority']})",
            ephemeral=True,
        )
        self.state.refresh_task_board()


class TaskEditModal(discord.ui.Modal, title="タスクを編集"):
    """選択済みタスクを既存値入りで編集するモーダル。"""

    def __init__(self, manager: "TaskBoardManager", task: dict):
        super().__init__()
        self.manager = manager
        self.task = task
        tz = self._tz()
        self.default_due_raw = format_due_input(
            task.get("due_at"), task.get("due_granularity"), tz
        )

        self.title_input: discord.ui.TextInput = discord.ui.TextInput(
            label="タイトル",
            default=(task.get("title") or "")[:120],
            max_length=200,
            required=True,
        )
        self.due_input: discord.ui.TextInput = discord.ui.TextInput(
            label="期限 (空欄で期限なし)",
            default=self.default_due_raw,
            placeholder="明日 / 7/10 / 7/10 15:00",
            required=False,
            max_length=50,
        )
        self.priority_input: discord.ui.TextInput = discord.ui.TextInput(
            label="優先度 (high / normal / low)",
            default=(task.get("priority") or "normal")[:10],
            required=False,
            max_length=10,
        )
        self.note_input: discord.ui.TextInput = discord.ui.TextInput(
            label="メモ (空欄で削除)",
            default=(task.get("note") or "")[:1000],
            style=discord.TextStyle.paragraph,
            required=False,
            max_length=1000,
        )
        for item in (self.title_input, self.due_input, self.priority_input, self.note_input):
            self.add_item(item)

    def _tz(self) -> ZoneInfo:
        return ZoneInfo(getattr(self.manager.state, "tasks_timezone", "Asia/Tokyo"))

    async def on_submit(self, interaction: discord.Interaction) -> None:
        store = getattr(self.manager.state, "task_store", None)
        if store is None:
            await interaction.response.send_message("タスクストアが無効です。", ephemeral=True)
            return

        task = await asyncio.to_thread(store.get, int(self.task["id"]))
        if task is None or task["status"] != "open":
            await interaction.response.send_message(
                f"タスク #{self.task['id']} は見つからないか、すでに完了/削除済みです。",
                ephemeral=True,
            )
            self.manager.request_refresh()
            return

        tz = self._tz()
        kwargs, error = resolve_task_edit_input(
            current_task=self.task,
            title=str(self.title_input.value or ""),
            due_raw=str(self.due_input.value or ""),
            default_due_raw=self.default_due_raw,
            priority_raw=str(self.priority_input.value or ""),
            note_raw=str(self.note_input.value or ""),
            now=datetime.now(UTC),
            tz=tz,
        )
        if error is not None:
            await interaction.response.send_message(error, ephemeral=True)
            return

        try:
            ok = await asyncio.to_thread(
                store.update,
                int(self.task["id"]),
                title=kwargs["title"],
                note=kwargs["note"],
                due_at=kwargs["due_at"],
                due_granularity=kwargs["due_granularity"],
                priority=kwargs["priority"],
                clear_due=kwargs["clear_due"],
            )
        except Exception as e:
            await interaction.response.send_message(f"更新に失敗しました: `{e}`", ephemeral=True)
            return

        if not ok:
            await interaction.response.send_message(
                f"タスク #{self.task['id']} が見つからないか、すでに完了/削除済みです。",
                ephemeral=True,
            )
            self.manager.request_refresh()
            return

        due_str = make_due_summary(kwargs["due_at"], tz)
        if kwargs["clear_due"]:
            due_str = "未設定"
        await interaction.response.send_message(
            f"タスク #{self.task['id']} を更新しました: {kwargs['title']}"
            f" (期限: {due_str}, 優先度: {kwargs['priority']})",
            ephemeral=True,
        )
        self.manager.request_refresh()


# --------------------------------------------------------------------------
# ボード上の操作View
# --------------------------------------------------------------------------

class _BoardSelect(discord.ui.Select):
    def __init__(self, options: list[discord.SelectOption]):
        disabled = not options
        if not options:
            options = [discord.SelectOption(label="(タスクなし)", value="__none__")]
        super().__init__(
            placeholder="タスクを選んで操作…",
            min_values=1,
            max_values=1,
            options=options,
            disabled=disabled,
            custom_id=SELECT_CUSTOM_ID,
        )

    async def callback(self, interaction: discord.Interaction) -> None:
        view: "TaskBoardView" = self.view  # type: ignore[assignment]
        await view.handle_select(interaction, self.values[0] if self.values else None)


class TaskBoardView(discord.ui.View):
    """ボードメッセージに載る永続View。timeout=None + 固定 custom_id。

    Select の options は refresh 時に作り直したインスタンスをメッセージへ貼り直す。
    起動時 add_view で登録するテンプレートは options 空でよい
    (dispatch は custom_id で行われ、選択値は interaction payload から読まれる)。
    """

    def __init__(self, manager: "TaskBoardManager", options: Optional[list[discord.SelectOption]] = None):
        super().__init__(timeout=None)
        self.manager = manager
        self.add_item(_BoardSelect(options or []))

    @discord.ui.button(
        label="＋追加", style=discord.ButtonStyle.success, custom_id=ADD_BUTTON_CUSTOM_ID, row=1
    )
    async def add_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        if not self._allowed(interaction):
            await interaction.response.send_message("この操作は許可されていません。", ephemeral=True)
            return
        await interaction.response.send_modal(TaskModal(self.manager.state, source="board"))

    @discord.ui.button(
        label="🎯 今やる",
        style=discord.ButtonStyle.primary,
        custom_id=FOCUS_BUTTON_CUSTOM_ID,
        row=1,
    )
    async def focus_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        if not self._allowed(interaction):
            await interaction.response.send_message("この操作は許可されていません。", ephemeral=True)
            return
        controller = getattr(self.manager.state, "priority_controller", None)
        if controller is None:
            await interaction.response.send_message(
                "優先順位オーケストレーターが無効です。", ephemeral=True
            )
            return
        decision = await asyncio.to_thread(controller.recommend)
        await interaction.response.send_message(
            format_focus_decision(decision, self.manager._tz()), ephemeral=True
        )

    def _allowed(self, interaction: discord.Interaction) -> bool:
        return self.manager.state.is_allowed_feedback(
            interaction.user.id, interaction.channel_id or 0
        )

    async def handle_select(self, interaction: discord.Interaction, value: Optional[str]) -> None:
        if not self._allowed(interaction):
            await interaction.response.send_message("この操作は許可されていません。", ephemeral=True)
            return
        store = getattr(self.manager.state, "task_store", None)
        if store is None or not value or value == "__none__":
            await interaction.response.send_message("タスクストアが無効です。", ephemeral=True)
            return
        try:
            task_id = int(value)
        except ValueError:
            await interaction.response.send_message("不正な選択です。", ephemeral=True)
            return
        task = await asyncio.to_thread(store.get, task_id)
        if task is None or task["status"] != "open":
            await interaction.response.send_message(
                f"タスク #{task_id} は見つからないか、すでに完了/削除済みです。", ephemeral=True
            )
            self.manager.request_refresh()
            return
        tz = ZoneInfo(getattr(self.manager.state, "tasks_timezone", "Asia/Tokyo"))
        summary = format_board_line(task, tz, datetime.now(UTC))
        await interaction.response.send_message(
            f"選択: {summary}\n操作を選んでください。",
            view=TaskActionView(self.manager, task_id),
            ephemeral=True,
        )


class TaskActionView(discord.ui.View):
    """Select で選んだ1件への操作 (ephemeral・揮発)。"""

    def __init__(self, manager: "TaskBoardManager", task_id: int, *, timeout: float = 300.0):
        super().__init__(timeout=timeout)
        self.manager = manager
        self.task_id = task_id

        done_btn = discord.ui.Button(label="完了", style=discord.ButtonStyle.success)
        done_btn.callback = self._done
        self.add_item(done_btn)

        edit_btn = discord.ui.Button(label="編集", style=discord.ButtonStyle.primary)
        edit_btn.callback = self._edit
        self.add_item(edit_btn)

        s30 = discord.ui.Button(label="+30分", style=discord.ButtonStyle.secondary)
        s30.callback = self._make_snooze(timedelta(minutes=30), "+30分")
        self.add_item(s30)

        s2h = discord.ui.Button(label="+2時間", style=discord.ButtonStyle.secondary)
        s2h.callback = self._make_snooze(timedelta(hours=2), "+2時間")
        self.add_item(s2h)

        del_btn = discord.ui.Button(label="削除", style=discord.ButtonStyle.danger)
        del_btn.callback = self._drop
        self.add_item(del_btn)

    def _store(self):
        return getattr(self.manager.state, "task_store", None)

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if self.manager.state.is_allowed_feedback(
            interaction.user.id, interaction.channel_id or 0
        ):
            return True
        await interaction.response.send_message("この操作は許可されていません。", ephemeral=True)
        return False

    async def _finish(self, interaction: discord.Interaction, text: str) -> None:
        self.stop()
        await interaction.response.edit_message(content=text, view=None)
        self.manager.request_refresh()

    async def _done(self, interaction: discord.Interaction) -> None:
        store = self._store()
        if store is None:
            await interaction.response.edit_message(content="タスクストアが無効です。", view=None)
            return
        ok = await asyncio.to_thread(store.done, self.task_id)
        await self._finish(
            interaction,
            f"タスク #{self.task_id} を完了にしました。" if ok else "すでに完了/削除済みでした。",
        )

    async def _drop(self, interaction: discord.Interaction) -> None:
        store = self._store()
        if store is None:
            await interaction.response.edit_message(content="タスクストアが無効です。", view=None)
            return
        ok = await asyncio.to_thread(store.drop, self.task_id)
        await self._finish(
            interaction,
            f"タスク #{self.task_id} を削除しました。" if ok else "すでに完了/削除済みでした。",
        )

    async def _edit(self, interaction: discord.Interaction) -> None:
        store = self._store()
        if store is None:
            await interaction.response.edit_message(content="タスクストアが無効です。", view=None)
            return
        task = await asyncio.to_thread(store.get, self.task_id)
        if task is None or task["status"] != "open":
            await interaction.response.edit_message(
                content=f"タスク #{self.task_id} は見つからないか、すでに完了/削除済みです。",
                view=None,
            )
            self.manager.request_refresh()
            return
        await interaction.response.send_modal(TaskEditModal(self.manager, task))

    def _make_snooze(self, delta: timedelta, label: str):
        async def _snooze(interaction: discord.Interaction) -> None:
            store = self._store()
            if store is None:
                await interaction.response.edit_message(content="タスクストアが無効です。", view=None)
                return
            until = datetime.now(UTC) + delta
            ok = await asyncio.to_thread(store.snooze, self.task_id, until)
            await self._finish(
                interaction,
                f"タスク #{self.task_id} を{label}先送りしました。" if ok else "すでに完了/削除済みでした。",
            )

        return _snooze


# --------------------------------------------------------------------------
# ボードのライフサイクル管理
# --------------------------------------------------------------------------

class TaskBoardManager:
    """ボードメッセージの探索・作成・更新を担う。

    - setup():           起動後に1度、既存ボードを探すか新規作成してピン留めする
    - request_refresh():  デバウンス付きで embed / Select を更新する (bot ループ上で呼ぶ)
    - periodic_loop():    15分毎の定期 refresh (残り時間表示の鮮度維持)
    """

    def __init__(
        self,
        bot: Any,
        state: Any,
        *,
        channel_id: Optional[int],
        enabled: bool = True,
        state_path: Optional[Path] = None,
        debounce_sec: float = 1.5,
        periodic_sec: float = 900.0,
    ):
        self.bot = bot
        self.state = state
        self.channel_id = channel_id
        self.enabled = enabled
        self.state_path = state_path
        self.debounce_sec = debounce_sec
        self.periodic_sec = periodic_sec

        self.message_id: Optional[int] = None
        self._message: Optional[discord.Message] = None
        self._dirty = False
        self._debounce_task: Optional[asyncio.Task] = None
        self._periodic_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

    # --- 永続化 (message_id) ---

    def _load_state(self) -> None:
        if not self.state_path or not self.state_path.exists():
            return
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if data.get("channel_id") == self.channel_id:
            mid = data.get("message_id")
            self.message_id = int(mid) if mid else None

    def _save_state(self) -> None:
        if not self.state_path:
            return
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(
                json.dumps({"channel_id": self.channel_id, "message_id": self.message_id}),
                encoding="utf-8",
            )
        except OSError as e:
            print(f"[TaskBoard] state 保存失敗: {e}")

    def _tz(self) -> ZoneInfo:
        return ZoneInfo(getattr(self.state, "tasks_timezone", "Asia/Tokyo"))

    async def _get_channel(self):
        if self.channel_id is None:
            return None
        channel = self.bot.get_channel(self.channel_id)
        if channel is None:
            try:
                channel = await self.bot.fetch_channel(self.channel_id)
            except discord.HTTPException:
                return None
        return channel if hasattr(channel, "send") else None

    def _is_own_board(self, message: discord.Message) -> bool:
        me = self.bot.user
        if me is None or message.author.id != me.id:
            return False
        return bool(message.embeds) and (message.embeds[0].title == BOARD_TITLE)

    async def _load_tasks(self) -> list[dict]:
        store = getattr(self.state, "task_store", None)
        if store is None:
            return []
        return await asyncio.to_thread(store.list, "open", 1000)

    async def _build_embed(self, tasks: list[dict], tz: ZoneInfo, now: datetime) -> discord.Embed:
        embed = build_board_embed(tasks, tz, now)
        controller = getattr(self.state, "priority_controller", None)
        if controller is None:
            return embed
        try:
            decision = await asyncio.to_thread(
                controller.recommend, now=now, select=False
            )
        except Exception:
            return embed
        if decision is not None:
            task = decision.task
            mode = "固定中" if decision.current else "推奨"
            reason = " / ".join(decision.ranked.reasons[:2])
            embed.insert_field_at(
                0,
                name=f"🎯 今やる ({mode})",
                value=f"`#{task['id']}` {task['title']}\n{reason}",
                inline=False,
            )
        return embed

    async def setup(self) -> None:
        """既存ボードを探して無ければ作る。起動後 (on_ready) に1度呼ぶ。"""
        if not self.enabled:
            print("[TaskBoard] disabled")
            return
        channel = await self._get_channel()
        if channel is None:
            print(f"[TaskBoard] 配送先チャンネルが見つかりません (channel_id={self.channel_id})")
            return

        self._load_state()
        # 1) state に message_id があれば復元を試みる
        if self.message_id is not None:
            try:
                msg = await channel.fetch_message(self.message_id)
                if self._is_own_board(msg):
                    self._message = msg
            except discord.HTTPException:
                self._message = None

        # 2) ピン留めから自分のボードを探す
        if self._message is None:
            try:
                pins = await channel.pins()
            except discord.HTTPException:
                pins = []
            for msg in pins:
                if self._is_own_board(msg):
                    self._message = msg
                    self.message_id = msg.id
                    break

        # 3) 無ければ新規作成してピン留め
        if self._message is None:
            tasks = await self._load_tasks()
            tz = self._tz()
            now = datetime.now(UTC)
            embed = await self._build_embed(tasks, tz, now)
            view = TaskBoardView(self, build_select_options(tasks, tz, now))
            try:
                self._message = await channel.send(embed=embed, view=view)
                self.message_id = self._message.id
                try:
                    await self._message.pin()
                except discord.HTTPException as e:
                    print(f"[TaskBoard] ピン留め失敗 (続行): {e}")
            except discord.HTTPException as e:
                print(f"[TaskBoard] ボード作成失敗: {e}")
                return

        self._save_state()
        print(f"[TaskBoard] ready: channel={self.channel_id} message={self.message_id}")
        await self.refresh()
        if self._periodic_task is None:
            self._periodic_task = asyncio.create_task(self.periodic_loop())

    def request_refresh(self) -> None:
        """デバウンス付き refresh 要求。bot ループ上のコルーチンから呼ぶこと。"""
        if not self.enabled or self._message is None:
            return
        self._dirty = True
        if self._debounce_task is None or self._debounce_task.done():
            self._debounce_task = asyncio.create_task(self._debounced())

    def request_refresh_threadsafe(self) -> None:
        loop = getattr(self.bot, "loop", None)
        if loop is not None:
            loop.call_soon_threadsafe(self.request_refresh)

    async def _debounced(self) -> None:
        # 連打・多重変更を coalesce する。処理中に来た要求は _dirty で拾って再ループ。
        while self._dirty:
            self._dirty = False
            await asyncio.sleep(self.debounce_sec)
            await self.refresh()

    async def refresh(self) -> None:
        if not self.enabled or self._message is None:
            return
        async with self._lock:
            tasks = await self._load_tasks()
            tz = self._tz()
            now = datetime.now(UTC)
            embed = await self._build_embed(tasks, tz, now)
            view = TaskBoardView(self, build_select_options(tasks, tz, now))
            try:
                await self._message.edit(embed=embed, view=view)
            except discord.NotFound:
                # ボードが消された。再作成する。
                self._message = None
                self.message_id = None
                self._save_state()
                await self.setup()
            except discord.HTTPException as e:
                print(f"[TaskBoard] refresh 失敗: {e}")

    async def periodic_loop(self) -> None:
        try:
            while not self.bot.is_closed():
                await asyncio.sleep(self.periodic_sec)
                await self.refresh()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"[TaskBoard] periodic loop error: {e}")

    def stop(self) -> None:
        for task in (self._debounce_task, self._periodic_task):
            if task is not None:
                task.cancel()
