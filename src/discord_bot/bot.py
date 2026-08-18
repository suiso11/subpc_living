"""
Discord 操作コンソール。

Slash commands:
- /ask: Ollama に質問
- /tts: TTS backend で WAV を生成して添付
- /status: ヘルスチェック
- /service: service_ctl.sh 経由の限定サービス操作

通常メッセージ:
- DISCORD_AUTO_REPLY_CHANNEL_IDS のチャンネルでは、投稿されたテキストへ自動返信する。
- bot返答への 👍/👎 と `修正: ...` 返信をファインチューニング候補として記録する。
"""
from __future__ import annotations

import asyncio
import io
import os
import re
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import discord
from discord import app_commands
from discord.ext import commands

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.audio.tts_factory import all_tts_voices, backend_name, create_tts_backend
from src.assistant.contracts import AssistantRequest
from src.assistant.service import AssistantService
from src.chat.config import ChatConfig
from src.chat.emotion import parse_emotion_tag
from src.chat.session import ChatSession
from src.chat.web_search import WebSearchContext, create_web_search_context
from src.discord_bot.training import (
    BAD_REACTION,
    GOOD_REACTION,
    DiscordTrainingLog,
    parse_correction,
)
from src.discord_bot.proactive_bridge import ProactiveBridge, create_proactive_bridge, rewrite_message
from src.discord_bot.task_ui import (
    TASK_PREFIX_RE,
    TaskConfirmView,
    TaskReminderView,
    build_extraction_prompt,
    make_due_summary,
    parse_due,
    parse_snooze,
    validate_extraction,
)
from src.discord_bot.task_board import TaskBoardManager, TaskBoardView, TaskModal
from src.discord_bot.voice_reply_debouncer import VoiceReplyDebouncer
from src.discord_bot.terminal import DiscordTerminal
from src.discord_bot.voice_stt import DiscordVoiceSTT, VoiceSTTConfig, VoiceSTTError
from src.discord_bot.voice_tts import VoiceTTSConfig, VoiceTTSError, VoiceTTSPlayer
from src.screen.context import ScreenContext
from src.screen import create_screen_context
from src.diary.collector import DiaryCollector
from src.diary.service import DailyDiaryResult, DailyDiaryService
from src.integrations.google_calendar import GoogleCalendarMCPClient
from src.persona.daily_personalizer import DailyPersonalizer, PersonalizationResult
from src.service.healthcheck import HealthChecker
from src.tasks.reminder import TaskReminderEngine, parse_quiet_hours
from src.tasks.prioritizer import PriorityController, format_focus_decision
from src.tasks.store import TaskStore
from src.growth.tracker import GrowthTracker
from src.llm.contracts import GenerationOptions
from src.llm.provider import LLMProvider
from src.llm.providers.ollama import OllamaProvider
from src.llm.registry import ProviderRegistry
from src.llm.routing.static import StaticRouter

from src.service.log_setup import setup_logging
logger = setup_logging("subpc-discord")


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
VOICE_TRANSCRIPT_RE = re.compile(
    r"^\[\d{2}:\d{2}:\d{2}\]\s+(?P<speaker>.+?):\s*(?P<text>.+)$", re.DOTALL
)
MAX_DISCORD_MESSAGE = 1900


def load_env_file(path: Path) -> None:
    """Tiny .env loader. Existing environment variables win."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def parse_id_set(value: str | None) -> set[int]:
    if not value:
        return set()
    result: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            result.add(int(item))
        except ValueError:
            logger.warning("[Discord] ID を無視します: %s", item)
    return result


def parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def parse_optional_id(value: str | None) -> int | None:
    if not value:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        logger.warning("[Discord] ID を無視します: %s", value)
        return None


def parse_hhmm(value: str | None, default: str = "23:50") -> time:
    raw = (value or default).strip()
    try:
        hour_text, minute_text = raw.split(":", 1)
        hour = int(hour_text)
        minute = int(minute_text)
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            raise ValueError
        return time(hour=hour, minute=minute)
    except ValueError:
        logger.warning("[Discord] DIARY_POST_TIME を無視します: %s", raw)
        hour_text, minute_text = default.split(":", 1)
        return time(hour=int(hour_text), minute=int(minute_text))


def parse_float(value: str | None, default: float) -> float:
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning("[Discord] float 設定値を無視します: %s", value)
        return default


def clean_output(text: str, limit: int = 3500) -> str:
    text = ANSI_RE.sub("", text).strip()
    if len(text) <= limit:
        return text
    return text[-limit:]


def split_message(text: str) -> list[str]:
    if not text:
        return ["(empty)"]
    chunks = []
    while text:
        chunks.append(text[:MAX_DISCORD_MESSAGE])
        text = text[MAX_DISCORD_MESSAGE:]
    return chunks


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_optional_int(value: Any, default: int | None = None) -> int | None:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class DiscordLLMProfile:
    name: str
    ollama_base_url: str
    model: str
    system_prompt: str
    max_history_turns: int
    temperature: float
    top_p: float
    top_k: int
    repeat_penalty: float
    num_ctx: int
    num_predict: int | None = None

    @classmethod
    def from_config(
        cls,
        config: ChatConfig,
        *,
        name: str = "default",
        overrides: dict[str, Any] | None = None,
    ) -> "DiscordLLMProfile":
        raw = overrides or {}
        # 基本プロンプト: profile が明示 system_prompt を持てばそれを、無ければ
        # profile の model に対する model_prompt_overrides (無ければ base) を使う。
        if "system_prompt" in raw:
            system_prompt = str(raw.get("system_prompt", ""))
        else:
            profile_model = str(raw.get("model", config.model))
            system_prompt = config.effective_system_prompt(profile_model)
        suffix = str(raw.get("system_prompt_suffix", "") or "")
        if suffix:
            system_prompt = f"{system_prompt.rstrip()}\n{suffix}"

        return cls(
            name=name,
            ollama_base_url=str(raw.get("ollama_base_url", config.ollama_base_url)),
            model=str(raw.get("model", config.model)),
            system_prompt=system_prompt,
            max_history_turns=_coerce_int(
                raw.get("max_history_turns", config.max_history_turns),
                config.max_history_turns,
            ),
            temperature=_coerce_float(raw.get("temperature", config.temperature), config.temperature),
            top_p=_coerce_float(raw.get("top_p", config.top_p), config.top_p),
            top_k=_coerce_int(raw.get("top_k", config.top_k), config.top_k),
            repeat_penalty=_coerce_float(
                raw.get("repeat_penalty", config.repeat_penalty),
                config.repeat_penalty,
            ),
            num_ctx=_coerce_int(raw.get("num_ctx", config.num_ctx), config.num_ctx),
            num_predict=_coerce_optional_int(raw.get("num_predict", config.num_predict), config.num_predict),
        )


@dataclass
class DiscordConsoleState:
    config: ChatConfig | None = None
    llm: LLMProvider | None = None
    provider_registry: ProviderRegistry | None = None
    assistant_services: dict[str, AssistantService] = field(default_factory=dict)
    _llm_providers: dict[tuple[str, str], LLMProvider] = field(default_factory=dict)
    llm_profiles: dict[str, DiscordLLMProfile] = field(default_factory=dict)
    channel_profile_map: dict[int, str] = field(default_factory=dict)
    tts: Any | None = None
    sessions: dict[str, ChatSession] = field(default_factory=dict)
    session_locks: dict[str, threading.Lock] = field(default_factory=dict)
    sessions_lock: threading.Lock = field(default_factory=threading.Lock)
    web_search: WebSearchContext | None = None
    allowed_user_ids: set[int] = field(default_factory=set)
    allowed_channel_ids: set[int] = field(default_factory=set)
    auto_reply_channel_ids: set[int] = field(default_factory=set)
    terminal_channel_ids: set[int] = field(default_factory=set)
    allow_service_control: bool = False
    allow_terminal: bool = False
    tts_lock: threading.Lock = field(default_factory=threading.Lock)
    training_log: DiscordTrainingLog | None = None
    growth_tracker: GrowthTracker | None = None
    terminal: DiscordTerminal | None = None
    diary_enabled: bool = False
    diary_channel_id: int | None = None
    diary_timezone: str = "Asia/Tokyo"
    diary_post_time: time = field(default_factory=lambda: time(hour=23, minute=50))
    diary_calendar_enabled: bool = True
    diary_calendar_id: str = "primary"
    diary_service: DailyDiaryService | None = None
    diary_task: asyncio.Task | None = None
    diary_personalization_enabled: bool = True
    diary_personalization_min_confidence: float = 0.72
    daily_personalizer: DailyPersonalizer | None = None
    voice_stt: DiscordVoiceSTT | None = None
    voice_tts: VoiceTTSPlayer | None = None
    voice_reply_debounce_ms: int = 3000
    voice_reply_debounce_max_ms: int = 10000
    voice_reply_debouncer: Any | None = None
    proactive_bridge: ProactiveBridge | None = None
    screen_context: ScreenContext | None = None
    task_store: TaskStore | None = None
    task_reminder_engine: TaskReminderEngine | None = None
    event_reminder_engine: Any | None = None
    event_remind_enabled: bool = True
    event_remind_lead_min: float = 15.0
    tasks_reminder_enabled: bool = True
    tasks_quiet_hours: tuple[int, int] = (1, 8)
    tasks_timezone: str = "Asia/Tokyo"
    tasks_board_enabled: bool = True
    task_board_channel_id: int | None = None
    task_board: Any | None = None
    priority_enabled: bool = True
    priority_controller: PriorityController | None = None

    def initialize(self) -> None:
        self.config = ChatConfig.load(PROJECT_ROOT / "config" / "chat_config.json")
        self.llm_profiles = self._load_llm_profiles()
        self.channel_profile_map = self._load_channel_profile_map()
        self.llm = self._llm_for_profile(self.llm_profiles["default"])
        self.allowed_user_ids = parse_id_set(os.environ.get("DISCORD_ALLOWED_USER_IDS"))
        self.allowed_channel_ids = parse_id_set(os.environ.get("DISCORD_ALLOWED_CHANNEL_IDS"))
        self.auto_reply_channel_ids = parse_id_set(os.environ.get("DISCORD_AUTO_REPLY_CHANNEL_IDS"))
        if not self.auto_reply_channel_ids:
            self.auto_reply_channel_ids = set(self.allowed_channel_ids)
        self.terminal_channel_ids = parse_id_set(os.environ.get("DISCORD_TERMINAL_CHANNEL_IDS"))
        self.allow_service_control = parse_bool(
            os.environ.get("DISCORD_ALLOW_SERVICE_CONTROL"),
            default=False,
        )
        self.allow_terminal = parse_bool(
            os.environ.get("DISCORD_ALLOW_TERMINAL"),
            default=False,
        )
        training_dir = Path(os.environ.get("DISCORD_TRAINING_DIR", "data/discord_training"))
        if not training_dir.is_absolute():
            training_dir = PROJECT_ROOT / training_dir
        try:
            self.growth_tracker = GrowthTracker(
                PROJECT_ROOT / "data" / "growth" / "growth.db",
                timezone_name=(
                    os.environ.get("DIARY_TIMEZONE", "Asia/Tokyo").strip()
                    or "Asia/Tokyo"
                ),
            )
        except Exception as e:
            logger.error("[Discord] growth tracker 初期化失敗: %s", e)
            self.growth_tracker = None
        self.training_log = DiscordTrainingLog(
            training_dir,
            enabled=parse_bool(os.environ.get("DISCORD_TRAINING_LOG_ENABLED"), default=True),
            system_prompt=self.config.system_prompt,
            growth_tracker=self.growth_tracker,
        )
        self.web_search = create_web_search_context(self.config)
        if parse_bool(os.environ.get("DISCORD_SCREEN_CONTEXT_ENABLED"), default=False):
            try:
                # SCREEN_CONTEXT_MODE (local|remote) でローカル/リモートを切替
                screen = create_screen_context()
                if screen.start():
                    self.screen_context = screen
                else:
                    logger.warning("[Discord] screen context unavailable; continuing without it")
            except Exception as e:
                logger.error("[Discord] screen context init failed: %s", e)
        terminal_workdir = Path(
            os.environ.get("DISCORD_TERMINAL_WORKDIR", str(PROJECT_ROOT))
        ).expanduser()
        self.terminal = DiscordTerminal(
            enabled=self.allow_terminal,
            workdir=terminal_workdir,
            session_prefix=os.environ.get("DISCORD_TERMINAL_SESSION_PREFIX", "subpc_discord"),
            capture_lines=int(os.environ.get("DISCORD_TERMINAL_CAPTURE_LINES", "80")),
            command_prefix=os.environ.get("DISCORD_TERMINAL_PREFIX", "!t"),
        )
        self.diary_enabled = parse_bool(os.environ.get("DIARY_ENABLED"), default=False)
        self.diary_channel_id = parse_optional_id(os.environ.get("DISCORD_DIARY_CHANNEL_ID"))
        self.diary_timezone = os.environ.get("DIARY_TIMEZONE", "Asia/Tokyo").strip() or "Asia/Tokyo"
        self.diary_post_time = parse_hhmm(os.environ.get("DIARY_POST_TIME"), default="23:50")
        self.diary_calendar_enabled = parse_bool(os.environ.get("DIARY_CALENDAR_ENABLED"), default=True)
        self.diary_calendar_id = os.environ.get("DIARY_CALENDAR_ID", "primary").strip() or "primary"
        self.diary_personalization_enabled = parse_bool(
            os.environ.get("DIARY_PERSONALIZATION_ENABLED"),
            default=True,
        )
        self.diary_personalization_min_confidence = parse_float(
            os.environ.get("DIARY_PERSONALIZATION_MIN_CONFIDENCE"),
            default=0.72,
        )
        self.voice_stt = DiscordVoiceSTT(
            VoiceSTTConfig.from_env(PROJECT_ROOT, timezone=self.diary_timezone)
        )
        self.voice_tts = VoiceTTSPlayer(
            config=VoiceTTSConfig.from_env(),
            synthesize=self.synthesize,
            get_voice_client=lambda: self.voice_stt.voice_client if self.voice_stt else None,
        )
        self.voice_reply_debounce_ms = _coerce_int(
            os.environ.get("DISCORD_VOICE_REPLY_DEBOUNCE_MS"), 3000
        )
        self.voice_reply_debounce_max_ms = _coerce_int(
            os.environ.get("DISCORD_VOICE_REPLY_DEBOUNCE_MAX_MS"), 10000
        )
        self.tasks_reminder_enabled = parse_bool(
            os.environ.get("TASKS_REMINDER_ENABLED"), default=True
        )
        self.tasks_quiet_hours = parse_quiet_hours(
            os.environ.get("TASKS_QUIET_HOURS"), default=(1, 8)
        )
        self.event_remind_enabled = parse_bool(
            os.environ.get("EVENT_REMIND_ENABLED"), default=True
        )
        self.event_remind_lead_min = parse_float(
            os.environ.get("EVENT_REMIND_LEAD_MIN"), default=15.0
        )
        self.tasks_timezone = self.diary_timezone
        self.tasks_board_enabled = parse_bool(
            os.environ.get("TASKS_BOARD_ENABLED"), default=True
        )
        self.task_board_channel_id = parse_optional_id(
            os.environ.get("DISCORD_TASK_BOARD_CHANNEL_ID")
        )
        try:
            self.task_store = TaskStore(
                db_path=str(PROJECT_ROOT / "data" / "tasks" / "tasks.db"),
                timezone_name=self.tasks_timezone,
            ).initialize()
        except Exception as e:
            logger.error("[Discord] task store 初期化失敗 (タスク機能なしで続行): %s", e)
            self.task_store = None
        self.priority_enabled = parse_bool(
            os.environ.get("PRIORITY_ENABLED"), default=True
        )
        self.priority_controller = None
        if self.priority_enabled and self.task_store is not None:
            try:
                state_path = Path(
                    os.environ.get("PRIORITY_STATE_PATH", "data/tasks/priority_state.json")
                )
                upcoming_path = Path(
                    os.environ.get("PRIORITY_UPCOMING_PATH", "data/calendar/upcoming.json")
                )
                if not state_path.is_absolute():
                    state_path = PROJECT_ROOT / state_path
                if not upcoming_path.is_absolute():
                    upcoming_path = PROJECT_ROOT / upcoming_path
                self.priority_controller = PriorityController(
                    self.task_store,
                    state_path=state_path,
                    upcoming_path=upcoming_path,
                    timezone_name=self.tasks_timezone,
                    skip_hours=parse_float(
                        os.environ.get("PRIORITY_SKIP_HOURS"), default=2.0
                    ),
                    calendar_buffer_min=int(parse_float(
                        os.environ.get("PRIORITY_CALENDAR_BUFFER_MIN"), default=10.0
                    )),
                )
            except Exception as e:
                logger.error("[Discord] priority controller 初期化失敗: %s", e)
                self.priority_controller = None
        # タスク ⇔ Google Calendar 同期
        self.tasks_calendar_sync_enabled = parse_bool(
            os.environ.get("TASKS_CALENDAR_SYNC_ENABLED"), default=False
        )
        self.tasks_calendar_id = (
            os.environ.get("TASKS_CALENDAR_ID", "").strip() or self.diary_calendar_id
        )
        self.calendar_sync_interval_min = parse_float(
            os.environ.get("CALENDAR_SYNC_INTERVAL_MIN"), default=20.0
        )
        self.calendar_sync_days_ahead = int(parse_float(
            os.environ.get("CALENDAR_SYNC_DAYS_AHEAD"), default=45.0
        ))
        self.calendar_pull_past_days = int(parse_float(
            os.environ.get("CALENDAR_PULL_PAST_DAYS"), default=14.0
        ))
        self.task_calendar_sync = None
        self.calendar_pull_worker = None
        # CalendarContext は upcoming.json を読むだけなので同期無効でも常に注入可
        # (ファイルが無ければ空文字を返す)。
        try:
            from src.tasks.calendar_sync import CalendarContext
            self.calendar_context = CalendarContext(
                upcoming_path=str(PROJECT_ROOT / "data" / "calendar" / "upcoming.json"),
                timezone=self.tasks_timezone,
            )
        except Exception as e:
            logger.warning("[Discord] calendar context 初期化スキップ: %s", e)
            self.calendar_context = None
        try:
            calendar_client = (
                GoogleCalendarMCPClient.from_env()
                if self.diary_calendar_enabled
                else None
            )
            collector = DiaryCollector(
                PROJECT_ROOT,
                calendar_client=calendar_client,
                timezone=self.diary_timezone,
            )
            self.diary_service = DailyDiaryService(
                project_root=PROJECT_ROOT,
                llm=self.llm,
                collector=collector,
                timezone=self.diary_timezone,
                temperature=0.4,
                num_ctx=self.config.num_ctx,
            )
            self.daily_personalizer = DailyPersonalizer(
                project_root=PROJECT_ROOT,
                llm=self.llm,
                min_confidence=self.diary_personalization_min_confidence,
                num_ctx=self.config.num_ctx,
            )
        except Exception as e:
            logger.error("[Discord] diary initialization failed: %s", e)
            self.diary_service = None
            self.daily_personalizer = None

    def start_calendar_sync(self, profile: Any = None) -> None:
        """タスク→カレンダー同期ワーカーとカレンダー→bot pull ワーカーを起動する。

        profile を渡すと (ProactiveEngine と共有する UserProfile)、当日+翌日の
        予定を profile.schedule に洗い替えし、既存のリマインド (15分前) が効く。
        """
        if not self.tasks_calendar_sync_enabled or self.task_store is None:
            return
        if self.task_calendar_sync is not None:
            return
        try:
            from src.tasks.calendar_sync import TaskCalendarSync, CalendarPullWorker

            client = GoogleCalendarMCPClient.from_env()
            sync = TaskCalendarSync(
                store=self.task_store,
                calendar_client=client,
                calendar_id=self.tasks_calendar_id,
                enabled=True,
                timezone=self.tasks_timezone,
            )
            sync.start()
            self.task_store.on_change = sync.enqueue
            self.task_calendar_sync = sync

            pull = CalendarPullWorker(
                calendar_client=client,
                calendar_id=self.tasks_calendar_id,
                profile=profile,
                store=self.task_store,
                timezone=self.tasks_timezone,
                interval_min=self.calendar_sync_interval_min,
                days_ahead=self.calendar_sync_days_ahead,
                past_days=self.calendar_pull_past_days,
                upcoming_path=str(PROJECT_ROOT / "data" / "calendar" / "upcoming.json"),
            )
            pull.start()
            self.calendar_pull_worker = pull
            logger.info(
                "[Discord] calendar sync started: "
                "calendar=%s pull_interval=%smin schedule_sync=%s",
                self.tasks_calendar_id,
                self.calendar_sync_interval_min,
                "on" if profile is not None else "off",
            )
        except Exception as e:
            logger.error("[Discord] calendar sync 起動失敗 (同期なしで続行): %s", e)
            self.task_calendar_sync = None
            self.calendar_pull_worker = None

    def close(self) -> None:
        if self.diary_task is not None:
            self.diary_task.cancel()
        if self.task_calendar_sync is not None:
            try:
                self.task_calendar_sync.stop()
            except Exception:
                pass
        if self.calendar_pull_worker is not None:
            try:
                self.calendar_pull_worker.stop()
            except Exception:
                pass
        if self.task_reminder_engine is not None:
            try:
                self.task_reminder_engine.stop()
            except Exception:
                pass
        if self.event_reminder_engine is not None:
            try:
                self.event_reminder_engine.stop()
            except Exception:
                pass
        if self.task_board is not None:
            try:
                self.task_board.stop()
            except Exception:
                pass
        if self.task_store is not None:
            try:
                self.task_store.close()
            except Exception:
                pass
        if self.proactive_bridge is not None:
            self.proactive_bridge.stop()
        if self.screen_context is not None:
            self.screen_context.stop()
        if self.voice_stt is not None:
            self.voice_stt.close()
        if self.provider_registry is not None:
            self.provider_registry.close()

    def is_allowed(self, interaction: discord.Interaction) -> bool:
        if self.allowed_user_ids and interaction.user.id not in self.allowed_user_ids:
            return False
        if self.allowed_channel_ids and interaction.channel_id not in self.allowed_channel_ids:
            return False
        return True

    def is_allowed_message(self, message: discord.Message) -> bool:
        if message.author.bot:
            return False
        if self.allowed_user_ids and message.author.id not in self.allowed_user_ids:
            return False
        if not (self._message_channel_ids(message) & self.auto_reply_channel_ids):
            return False
        return True

    def is_allowed_voice_transcript_message(
        self,
        message: discord.Message,
        *,
        bot_user_id: int | None,
    ) -> bool:
        if bot_user_id is None or message.author.id != bot_user_id:
            return False
        if self.voice_stt is None or self.voice_stt.transcript_channel_id is None:
            return False
        channel_ids = self._message_channel_ids(message)
        if self.voice_stt.transcript_channel_id not in channel_ids:
            return False
        if not (channel_ids & self.auto_reply_channel_ids):
            return False
        return self.extract_voice_transcript_text(message.content) is not None

    @staticmethod
    def extract_voice_transcript(content: str) -> tuple[str, str] | None:
        """Return (speaker, text) from a transcript line, or None if it isn't one."""
        match = VOICE_TRANSCRIPT_RE.match(content.strip())
        if match is None:
            return None
        speaker = match.group("speaker").strip()
        text = match.group("text").strip()
        if not text:
            return None
        return speaker, text

    @staticmethod
    def extract_voice_transcript_text(content: str) -> str | None:
        parsed = DiscordConsoleState.extract_voice_transcript(content)
        return parsed[1] if parsed is not None else None

    def is_terminal_message(self, message: discord.Message) -> bool:
        return self.terminal_reject_reason(message) is None

    def terminal_reject_reason(self, message: discord.Message) -> str | None:
        if not self.allow_terminal:
            return "terminal is disabled. Set DISCORD_ALLOW_TERMINAL=true."
        if not self.terminal_channel_ids:
            return "terminal channel is not configured. Set DISCORD_TERMINAL_CHANNEL_IDS."
        if message.author.bot:
            return "bot message"
        if self.allowed_user_ids and message.author.id not in self.allowed_user_ids:
            return f"user is not allowed: {message.author.id}"
        if not (self._message_channel_ids(message) & self.terminal_channel_ids):
            return (
                f"terminal is not enabled in this channel: {message.channel.id}. "
                f"configured: {sorted(self.terminal_channel_ids)}"
            )
        return None

    def is_allowed_feedback(self, user_id: int, channel_id: int) -> bool:
        if self.allowed_user_ids and user_id not in self.allowed_user_ids:
            return False
        if self.auto_reply_channel_ids and channel_id not in self.auto_reply_channel_ids:
            return False
        return True

    def auto_reply_enabled(self) -> bool:
        return bool(self.auto_reply_channel_ids)

    def terminal_enabled(self) -> bool:
        return self.allow_terminal and bool(self.terminal_channel_ids)

    def message_content_required(self) -> bool:
        return self.auto_reply_enabled() or self.terminal_enabled()

    def _load_llm_profiles(self) -> dict[str, DiscordLLMProfile]:
        assert self.config is not None
        profiles = {
            "default": DiscordLLMProfile.from_config(self.config, name="default"),
        }
        for name, raw_profile in self.config.discord_channel_profiles.items():
            if not isinstance(raw_profile, dict):
                logger.warning("[Discord] LLM profile を無視します: %s", name)
                continue
            profiles[name] = DiscordLLMProfile.from_config(
                self.config,
                name=name,
                overrides=raw_profile,
            )
        return profiles

    def _load_channel_profile_map(self) -> dict[int, str]:
        assert self.config is not None
        mapping: dict[int, str] = {}
        for raw_channel_id, profile_name in self.config.discord_channel_profile_map.items():
            try:
                channel_id = int(raw_channel_id)
            except (TypeError, ValueError):
                logger.warning("[Discord] channel profile ID を無視します: %s", raw_channel_id)
                continue
            if profile_name not in self.llm_profiles:
                logger.warning(
                    "[Discord] 未定義のLLM profile を無視します: channel=%s profile=%s",
                    channel_id,
                    profile_name,
                )
                continue
            mapping[channel_id] = profile_name
        return mapping

    def _llm_for_profile(self, profile: DiscordLLMProfile) -> LLMProvider:
        if self.provider_registry is None:
            self.provider_registry = ProviderRegistry()
        key = (profile.ollama_base_url, profile.model)
        if profile.name in self.provider_registry:
            provider = self.provider_registry.get(profile.name).provider
            self._llm_providers.setdefault(key, provider)
            return provider
        if key not in self._llm_providers:
            self._llm_providers[key] = OllamaProvider(
                base_url=profile.ollama_base_url,
                model=profile.model,
                provider_id=profile.name,
            )
        provider = self._llm_providers[key]
        self.provider_registry.register(profile.name, provider, local=True)
        return provider

    def _service_for_profile(self, profile: DiscordLLMProfile) -> AssistantService:
        if profile.name not in self.assistant_services:
            self._llm_for_profile(profile)
            assert self.provider_registry is not None
            self.assistant_services[profile.name] = AssistantService(
                self.provider_registry,
                StaticRouter(
                    self.provider_registry,
                    default_provider_id=profile.name,
                ),
                options=GenerationOptions(
                    temperature=profile.temperature,
                    top_p=profile.top_p,
                    top_k=profile.top_k,
                    repeat_penalty=profile.repeat_penalty,
                    num_ctx=profile.num_ctx,
                    num_predict=profile.num_predict,
                ),
            )
        return self.assistant_services[profile.name]

    def profile_for_channel_ids(self, channel_ids: set[int]) -> DiscordLLMProfile:
        for channel_id in channel_ids:
            profile_name = self.channel_profile_map.get(channel_id)
            if profile_name:
                return self.llm_profiles[profile_name]
        return self.llm_profiles["default"]

    @staticmethod
    def _message_channel_ids(message: discord.Message) -> set[int]:
        ids = {message.channel.id}
        parent_id = getattr(message.channel, "parent_id", None)
        if isinstance(parent_id, int):
            ids.add(parent_id)
        return ids

    @staticmethod
    def _shared_session_key(guild_id: int | None, channel_id: int | None) -> str:
        return f"discord:{guild_id or 0}:{channel_id or 0}"

    @staticmethod
    def _ephemeral_session_key(
        guild_id: int | None,
        channel_id: int | None,
        user_id: int,
    ) -> str:
        return f"discord-ephemeral:{guild_id or 0}:{channel_id or 0}:{user_id}"

    def _conversation_id_for_session(self, session: ChatSession) -> str:
        with self.sessions_lock:
            for key, current_session in self.sessions.items():
                if current_session is session:
                    return key
        return session.session_id

    def _session_for_key(
        self,
        key: str,
        profile: DiscordLLMProfile,
    ) -> tuple[ChatSession, threading.Lock]:
        assert self.config is not None
        with self.sessions_lock:
            if key not in self.sessions:
                self.sessions[key] = ChatSession(
                    system_prompt=profile.system_prompt,
                    max_history_turns=profile.max_history_turns,
                    history_dir=str(PROJECT_ROOT / self.config.history_dir),
                    web_search=self.web_search,
                    screen_context=self.screen_context,
                    task_store=self.task_store,
                    calendar_context=self.calendar_context,
                    growth_tracker=self.growth_tracker,
                    conversation_source="discord",
                    emotion_tags=self.config.emotion_tag_enabled,
                )
            else:
                self.sessions[key].system_prompt = profile.system_prompt
                self.sessions[key].max_history_turns = profile.max_history_turns
            if key not in self.session_locks:
                self.session_locks[key] = threading.Lock()
            return self.sessions[key], self.session_locks[key]

    def get_session(
        self,
        interaction: discord.Interaction,
        *,
        ephemeral: bool = False,
    ) -> tuple[ChatSession, threading.Lock, DiscordLLMProfile]:
        channel_ids = {interaction.channel_id} if interaction.channel_id is not None else set()
        profile = self.profile_for_channel_ids(channel_ids)
        if ephemeral:
            key = self._ephemeral_session_key(
                interaction.guild_id,
                interaction.channel_id,
                interaction.user.id,
            )
        else:
            key = self._shared_session_key(interaction.guild_id, interaction.channel_id)
        session, lock = self._session_for_key(key, profile)
        return session, lock, profile

    def get_message_session(
        self,
        message: discord.Message,
    ) -> tuple[ChatSession, threading.Lock, DiscordLLMProfile]:
        profile = self.profile_for_channel_ids(self._message_channel_ids(message))
        key = self._shared_session_key(
            message.guild.id if message.guild else None,
            message.channel.id,
        )
        session, lock = self._session_for_key(key, profile)
        return session, lock, profile

    def reset_session(self, interaction: discord.Interaction) -> None:
        key = self._shared_session_key(interaction.guild_id, interaction.channel_id)
        with self.sessions_lock:
            self.sessions.pop(key, None)
            self.session_locks.pop(key, None)

    def ask(
        self, interaction: discord.Interaction, prompt: str, *, ephemeral: bool = False
    ) -> tuple[str, str]:
        if self.llm is None or self.config is None:
            raise RuntimeError("LLM is not initialized")
        session, lock, profile = self.get_session(interaction, ephemeral=ephemeral)
        return self.ask_session(session, prompt, lock, profile)

    def ask_message(self, message: discord.Message) -> tuple[str, str]:
        if self.llm is None or self.config is None:
            raise RuntimeError("LLM is not initialized")
        return self.ask_message_text(message, message.content.strip())

    def ask_message_text(self, message: discord.Message, prompt: str) -> tuple[str, str]:
        if self.llm is None or self.config is None:
            raise RuntimeError("LLM is not initialized")
        session, lock, profile = self.get_message_session(message)
        return self.ask_session(session, prompt, lock, profile)

    def record_auto_reply(
        self,
        message: discord.Message,
        response: str,
        reply_messages: list[discord.Message],
        user_text: str | None = None,
        source: str = "discord_auto_reply",
    ) -> None:
        if self.training_log is None or self.config is None:
            return
        profile = self.profile_for_channel_ids(self._message_channel_ids(message))
        self.training_log.record_turn(
            guild_id=message.guild.id if message.guild else None,
            channel_id=message.channel.id,
            user_id=message.author.id,
            user_message_id=message.id,
            assistant_message_ids=[reply.id for reply in reply_messages],
            user_text=user_text if user_text is not None else message.content.strip(),
            assistant_text=response,
            model=profile.model,
            num_ctx=profile.num_ctx,
            source=source,
            profile=profile.name,
            num_predict=profile.num_predict,
            temperature=profile.temperature,
        )

    def record_direct_reply(
        self, message: discord.Message, user_text: str, assistant_text: str
    ) -> None:
        """LLMを介さない予定登録なども会話履歴・成長台帳へ1往復として残す。"""
        session, lock, _profile = self.get_message_session(message)
        with lock:
            session.add_user_message(user_text)
            session.add_assistant_message(assistant_text)

    def record_feedback(
        self,
        *,
        assistant_message_id: int,
        guild_id: int | None,
        channel_id: int,
        user_id: int,
        emoji: str,
    ) -> bool:
        if self.training_log is None:
            return False
        return self.training_log.record_feedback(
            assistant_message_id=assistant_message_id,
            guild_id=guild_id,
            channel_id=channel_id,
            user_id=user_id,
            emoji=emoji,
        )

    def generate_correction_candidate(self, turn: dict, temperature: float) -> str:
        """👎されたターンに対する言い直し案を1つ生成する (同期・スレッドで実行)。"""
        profile = (
            self.llm_profiles.get(turn.get("profile") or "default")
            or self.llm_profiles["default"]
        )
        llm = self._llm_for_profile(profile)
        messages = [
            {"role": "system", "content": profile.system_prompt},
            {"role": "user", "content": turn["user"]},
            {"role": "assistant", "content": turn["assistant"]},
            {
                "role": "user",
                "content": (
                    "今の返答は良くなかった。同じ発言への返答を、違う切り口でもう一度。"
                    "前置きや説明はなしで、返答本文だけを出して。"
                ),
            },
        ]
        # 修正候補は呼び出し側指定の temperature を使うため Provider を直接呼ぶ。
        return llm.generate(
            messages,
            temperature=temperature,
            top_p=profile.top_p,
            top_k=profile.top_k,
            repeat_penalty=profile.repeat_penalty,
            num_ctx=profile.num_ctx,
            num_predict=profile.num_predict,
        ).strip()

    def record_correction(self, message: discord.Message, corrected_text: str) -> str:
        if self.training_log is None:
            return "training log is disabled"
        assistant_message_id = message.reference.message_id if message.reference else None
        result = self.training_log.record_correction(
            assistant_message_id=assistant_message_id,
            channel_id=message.channel.id,
            guild_id=message.guild.id if message.guild else None,
            user_id=message.author.id,
            correction_message_id=message.id,
            corrected_text=corrected_text,
        )
        if result.ok:
            return "修正を学習候補に保存しました。"
        return f"修正を保存できませんでした: {result.reason}"

    def handle_terminal_message(self, message: discord.Message) -> str | None:
        if self.terminal is None:
            return None
        result = self.terminal.handle_message(message.channel.id, message.content)
        if result is None:
            return None
        prefix = "" if result.ok else "terminal error:\n"
        return prefix + result.text

    def ask_session(
        self,
        session: ChatSession,
        prompt: str,
        lock: threading.Lock,
        profile: DiscordLLMProfile,
    ) -> tuple[str, str]:
        """(clean_text, emotion) を返す。履歴・トレーニングにはタグ除去後のテキストを保存する。"""
        if self.llm is None or self.config is None:
            raise RuntimeError("LLM is not initialized")
        service = self._service_for_profile(profile)
        with lock:
            session.add_user_message(prompt)
            try:
                request = AssistantRequest(
                    text=prompt,
                    conversation_id=self._conversation_id_for_session(session),
                    channel="discord",
                    profile="chat_auto",
                    privacy="local_only",
                    requested_provider=profile.name,
                )
                response = service.generate(
                    request,
                    session.build_messages(),
                ).text
            except Exception:
                if session._messages and session._messages[-1]["role"] == "user":
                    session._messages.pop()
                raise
            # 感情タグを分離。履歴には clean text のみ格納する。
            if self.config.emotion_tag_enabled:
                emotion, response = parse_emotion_tag(response)
            else:
                emotion = "neutral"
            session.add_assistant_message(response)
            return response, emotion

    def synthesize(
        self,
        text: str,
        voice: str | None,
        speed: float,
        style: str | None = None,
    ) -> bytes:
        with self.tts_lock:
            if self.tts is None:
                self.tts = create_tts_backend(
                    env_prefix="DISCORD_VOICE",
                    models_dir=PROJECT_ROOT / "models" / "tts" / "kokoro",
                    voice=voice,
                    speed=speed,
                )
            elif voice:
                self.tts.set_voice(voice)
            self.tts.speed = speed
            # style は SBV2 backend で使われ、kokoro では無視される。
            return self.tts.synthesize(text, style=style)

    def resolve_task_channel_id(self) -> int | None:
        """リマインド配送先: DISCORD_PROACTIVE_CHANNEL_ID → auto-reply 先頭。"""
        pid = parse_optional_id(os.environ.get("DISCORD_PROACTIVE_CHANNEL_ID"))
        if pid is not None:
            return pid
        if self.auto_reply_channel_ids:
            return sorted(self.auto_reply_channel_ids)[0]
        return None

    def resolve_task_board_channel_id(self) -> int | None:
        """ボード設置先: DISCORD_TASK_BOARD_CHANNEL_ID → リマインド配送先と同じフォールバック。"""
        if self.task_board_channel_id is not None:
            return self.task_board_channel_id
        return self.resolve_task_channel_id()

    def refresh_task_board(self) -> None:
        """ボードが有効なら更新を要求する (無効・未初期化なら何もしない)。

        bot ループ上のコルーチンから呼ぶこと (View コールバック等)。
        """
        board = self.task_board
        if board is not None:
            board.request_refresh()

    def rewrite_task_text(self, message: str) -> str:
        """タスク通知の定型文をペルソナ口調へ言い換える (履歴を汚さない使い捨て呼び出し)。"""
        profile = self.llm_profiles.get("default")
        system_prompt = profile.system_prompt if profile is not None else ""
        gen_kwargs: dict[str, Any] = {}
        if profile is not None:
            gen_kwargs = dict(
                temperature=profile.temperature,
                top_p=profile.top_p,
                top_k=profile.top_k,
                repeat_penalty=profile.repeat_penalty,
                num_ctx=profile.num_ctx,
                num_predict=profile.num_predict,
            )
        # 通知言い換えは関数側が生成引数を受け取るため Provider を直接呼ぶ。
        return rewrite_message(
            self.llm, system_prompt=system_prompt, message=message, **gen_kwargs
        )

    def extract_task_from_text(self, user_text: str) -> dict | None:
        """ユーザー発言から strict JSON でタスクを抽出・検証する (同期・スレッド実行)。"""
        if self.llm is None or self.config is None:
            return None
        now_local = datetime.now(ZoneInfo(self.tasks_timezone))
        messages = [
            {"role": "system", "content": build_extraction_prompt(now_local)},
            {"role": "user", "content": user_text},
        ]
        try:
            # JSON抽出は固定の生成引数を使うため Provider を直接呼ぶ。
            raw = self.llm.generate(
                messages,
                temperature=0.0,
                num_ctx=self.config.num_ctx,
                num_predict=256,
            )
        except Exception as e:
            logger.error("[Discord] task extraction LLM error: %s", e)
            return None
        return validate_extraction(raw, assume_tz=ZoneInfo(self.tasks_timezone))

    def add_task_quick(self, title: str, source: str) -> int | None:
        """「タスク:」プレフィックスの即登録。確認なし。失敗時 None。"""
        if self.task_store is None:
            return None
        try:
            return self.task_store.add(title, source=source)
        except Exception as e:
            logger.error("[Discord] quick task add failed: %s", e)
            return None

    def try_register_event_text(self, text: str) -> str | None:
        """予定登録の意図があれば Google Calendar に登録し結果文を返す。

        意図が無ければ None (呼び出し側は通常の LLM 応答へ)。
        ブロッキング (MCP 呼び出しで数秒) なので to_thread 経由で呼ぶこと。
        """
        try:
            from src.tasks.event_intent import try_register_event

            client = (
                self.task_calendar_sync.client
                if self.task_calendar_sync is not None
                else None
            )
            return try_register_event(
                text,
                client=client,
                calendar_id=self.tasks_calendar_id,
                timezone_name=self.tasks_timezone,
                upcoming_path=str(PROJECT_ROOT / "data" / "calendar" / "upcoming.json"),
            )
        except Exception as e:
            logger.error("[Discord] event register failed: %s", e)
            return None

    def status_text(self) -> str:
        assert self.config is not None
        checker = HealthChecker(ollama_url=self.config.ollama_base_url)
        health = checker.check_all(include_web=False)
        llm_ok = self.llm.is_available() if self.llm is not None else False
        tts_loaded = self.tts is not None and self.tts.is_loaded()
        with self.sessions_lock:
            session_count = len(self.sessions)
        default_profile = self.llm_profiles.get("default")
        return (
            f"status: {health['status']}\n"
            f"ollama: {'ok' if llm_ok else 'ng'}\n"
            f"model: {default_profile.model if default_profile else self.config.model}\n"
            f"llm_profiles: {', '.join(sorted(self.llm_profiles))}\n"
            f"channel_profile_map: {len(self.channel_profile_map)}\n"
            f"sessions: {session_count}\n"
            f"tts_loaded: {tts_loaded}\n"
            f"tts_backend: {backend_name(self.tts)}\n"
            f"auto_reply_channels: {len(self.auto_reply_channel_ids)}\n"
            f"terminal_channels: {len(self.terminal_channel_ids)}\n"
            f"terminal_enabled: {self.allow_terminal}\n"
            f"terminal_tmux: {self.terminal.is_available() if self.terminal else False}\n"
            f"diary_enabled: {self.diary_enabled}\n"
            f"diary_channel_id: {self.diary_channel_id or '-'}\n"
            f"diary_post_time: {self.diary_post_time.strftime('%H:%M')} {self.diary_timezone}\n"
            f"diary_personalization: {self.diary_personalization_enabled}\n"
            f"{self.voice_stt.status_text() if self.voice_stt else 'voice_stt: unavailable'}\n"
            f"message_content_intent: {self.message_content_required()}\n"
            f"web_search: {self.web_search is not None}\n"
            f"tasks_store: {self.task_store is not None}\n"
            f"tasks_reminder: {self.task_reminder_engine.is_running if self.task_reminder_engine else False}\n"
            f"tasks_open: {len(self.task_store.list('open')) if self.task_store else 0}\n"
            f"tasks_quiet_hours: {self.tasks_quiet_hours[0]}-{self.tasks_quiet_hours[1]}\n"
            f"priority_controller: {self.priority_controller is not None}\n"
            f"proactive_conversation: "
            f"{self.proactive_bridge.conversation_status() if self.proactive_bridge else 'disabled'}\n"
            f"service_control: {self.allow_service_control}\n"
            f"{self.training_log.summary_text() if self.training_log else 'training_log: unavailable'}\n"
            f"checks: {health['checks']}"
        )

    def run_service_command(self, action: str, target: str) -> str:
        safe_actions = {"status", "health", "gpu", "logs", "start", "stop", "restart"}
        safe_targets = {"web", "voice", "discord", "all"}
        if action not in safe_actions:
            raise ValueError(f"unsupported action: {action}")
        if target not in safe_targets:
            raise ValueError(f"unsupported target: {target}")
        if action in {"start", "stop", "restart"} and not self.allow_service_control:
            return "service control is disabled. Set DISCORD_ALLOW_SERVICE_CONTROL=true to enable it."

        if action == "status":
            cmd = ["bash", "scripts/service_ctl.sh", "status"]
        elif action == "health":
            cmd = ["bash", "scripts/service_ctl.sh", "health"]
        elif action == "gpu":
            cmd = ["bash", "scripts/service_ctl.sh", "gpu"]
        elif action == "logs":
            cmd = ["bash", "scripts/service_ctl.sh", "logs", target, "-n", "80", "--no-pager"]
        else:
            cmd = ["bash", "scripts/service_ctl.sh", action, target]

        proc = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=60,
            check=False,
        )
        return clean_output(proc.stdout or f"exit={proc.returncode}")


async def require_allowed(interaction: discord.Interaction, state: DiscordConsoleState) -> bool:
    if state.is_allowed(interaction):
        return True
    await interaction.response.send_message("この Discord ユーザー/チャンネルは許可されていません。", ephemeral=True)
    return False


class CorrectionModal(discord.ui.Modal, title="返答の修正を学習候補に保存"):
    """bot返答を右クリック→「修正する」で開く入力フォーム。

    botの返答をプリフィルして表示するので、理想の返答に書き換えて送信する
    だけで `修正: ...` 返信と同じ preferred/rejected ペアが記録される。
    """

    def __init__(self, state: DiscordConsoleState, target: discord.Message):
        super().__init__()
        self.state = state
        self.target = target
        self.corrected: discord.ui.TextInput = discord.ui.TextInput(
            label="理想の返答に書き換えてください",
            style=discord.TextStyle.paragraph,
            default=(target.content or "")[:4000],
            max_length=4000,
            required=True,
        )
        self.add_item(self.corrected)

    async def on_submit(self, interaction: discord.Interaction) -> None:
        corrected_text = str(self.corrected.value).strip()
        if self.state.training_log is None:
            await interaction.response.send_message("training log is disabled", ephemeral=True)
            return
        if corrected_text == (self.target.content or "").strip():
            await interaction.response.send_message(
                "返答が書き換えられていません。理想の返答に直してから送信してください。",
                ephemeral=True,
            )
            return
        result = self.state.training_log.record_correction(
            assistant_message_id=self.target.id,
            channel_id=self.target.channel.id,
            guild_id=self.target.guild.id if self.target.guild else None,
            user_id=interaction.user.id,
            correction_message_id=interaction.id,
            corrected_text=corrected_text,
        )
        if result.ok:
            await interaction.response.send_message(
                f"修正を学習候補に保存しました。\n> {corrected_text[:200]}",
                ephemeral=True,
            )
        else:
            await interaction.response.send_message(
                f"修正を保存できませんでした: {result.reason}",
                ephemeral=True,
            )


class CorrectionPickView(discord.ui.View):
    """👎後に提示する言い直し案の選択UI。

    ボタンで案を選ぶと、その案を preferred、元のbot返答を rejected として
    学習候補に保存する。「自分で書く」は CorrectionModal に落ちる。
    """

    def __init__(
        self,
        state: DiscordConsoleState,
        target: discord.Message,
        candidates: list[str],
    ):
        super().__init__(timeout=600)
        self.state = state
        self.target = target
        self.candidates = candidates
        self.message: discord.Message | None = None  # 選択肢を載せたbotメッセージ

        for index in range(len(candidates)):
            button = discord.ui.Button(
                label=f"案{index + 1}",
                style=discord.ButtonStyle.primary,
                row=0,
            )
            button.callback = self._make_pick_callback(index)
            self.add_item(button)

        write_button = discord.ui.Button(
            label="自分で書く", style=discord.ButtonStyle.secondary, row=1
        )
        write_button.callback = self._write_callback
        self.add_item(write_button)

        cancel_button = discord.ui.Button(
            label="やめる", style=discord.ButtonStyle.secondary, row=1
        )
        cancel_button.callback = self._cancel_callback
        self.add_item(cancel_button)

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        channel_id = interaction.channel_id or 0
        if self.state.is_allowed_feedback(interaction.user.id, channel_id):
            return True
        await interaction.response.send_message(
            "この操作は許可されていません。", ephemeral=True
        )
        return False

    def _make_pick_callback(self, index: int):
        async def _pick(interaction: discord.Interaction) -> None:
            chosen = self.candidates[index]
            if self.state.training_log is None:
                await interaction.response.send_message("training log is disabled", ephemeral=True)
                return
            result = self.state.training_log.record_correction(
                assistant_message_id=self.target.id,
                channel_id=self.target.channel.id,
                guild_id=self.target.guild.id if self.target.guild else None,
                user_id=interaction.user.id,
                correction_message_id=interaction.id,
                corrected_text=chosen,
            )
            if result.ok:
                text = f"案{index + 1} を学習候補に保存しました。\n> {chosen[:300]}"
            else:
                text = f"保存できませんでした: {result.reason}"
            self.stop()
            await interaction.response.edit_message(content=text, view=None)

        return _pick

    async def _write_callback(self, interaction: discord.Interaction) -> None:
        await interaction.response.send_modal(CorrectionModal(self.state, self.target))

    async def _cancel_callback(self, interaction: discord.Interaction) -> None:
        self.stop()
        await interaction.response.edit_message(content="修正をキャンセルしました。", view=None)

    async def on_timeout(self) -> None:
        if self.message is not None:
            try:
                await self.message.edit(
                    content="修正案の選択は期限切れになりました。", view=None
                )
            except discord.HTTPException:
                pass


def render_correction_choices(candidates: list[str]) -> str:
    lines = ["どの返答が良い? ボタンで選ぶと学習候補に保存します。"]
    per_candidate = max(200, 1500 // max(1, len(candidates)))
    for index, candidate in enumerate(candidates):
        shown = candidate[:per_candidate] + ("…" if len(candidate) > per_candidate else "")
        lines.append(f"**案{index + 1}**: {shown}")
    return "\n".join(lines)


def parse_diary_date(value: str | None, timezone: str) -> date:
    if not value:
        return datetime.now(ZoneInfo(timezone)).date()
    return date.fromisoformat(value.strip())


async def send_diary_to_channel(
    bot: commands.Bot,
    state: DiscordConsoleState,
    target_date: date,
    *,
    overwrite: bool = False,
) -> DailyDiaryResult:
    if state.diary_service is None:
        raise RuntimeError("diary service is not initialized")
    if state.diary_channel_id is None:
        raise RuntimeError("DISCORD_DIARY_CHANNEL_ID is not configured")

    channel = bot.get_channel(state.diary_channel_id)
    if channel is None:
        channel = await bot.fetch_channel(state.diary_channel_id)
    if not hasattr(channel, "send"):
        raise RuntimeError(f"diary target is not a text channel: {state.diary_channel_id}")

    result = await asyncio.to_thread(
        state.diary_service.generate,
        target_date,
        save=True,
        overwrite=overwrite,
        include_calendar=state.diary_calendar_enabled,
        calendar_id=state.diary_calendar_id,
    )
    for chunk in split_message(result.markdown):
        await channel.send(chunk)
    state.diary_service.mark_posted(target_date, channel_id=state.diary_channel_id)
    await maybe_personalize_from_diary(state, target_date, result.markdown)
    if result.calendar_error:
        logger.warning("[Discord] diary calendar warning: %s", result.calendar_error[:500])
    return result


async def maybe_personalize_from_diary(
    state: DiscordConsoleState,
    target_date: date,
    diary_markdown: str,
) -> PersonalizationResult | None:
    if not state.diary_personalization_enabled or state.daily_personalizer is None:
        return None
    try:
        result = await asyncio.to_thread(
            state.daily_personalizer.run,
            target_date,
            diary_markdown=diary_markdown,
            dry_run=False,
        )
        logger.info(
            "[Discord] daily personalization: date=%s applied=%s audit=%s",
            result.target_date,
            result.applied_count,
            result.audit_path,
        )
        return result
    except Exception as e:
        logger.error("[Discord] daily personalization failed: %s", e)
        return None


async def diary_scheduler_loop(bot: commands.Bot, state: DiscordConsoleState) -> None:
    await bot.wait_until_ready()
    while not bot.is_closed():
        try:
            if not state.diary_enabled or state.diary_service is None or state.diary_channel_id is None:
                await asyncio.sleep(300)
                continue

            tz = ZoneInfo(state.diary_timezone)
            now = datetime.now(tz)
            run_at = datetime.combine(now.date(), state.diary_post_time, tzinfo=tz)

            if now >= run_at and not state.diary_service.was_posted(now.date()):
                try:
                    await send_diary_to_channel(bot, state, now.date())
                    logger.info("[Discord] diary posted for %s", now.date().isoformat())
                except Exception as e:
                    logger.error("[Discord] diary post failed: %s", e)
                await asyncio.sleep(60)
                continue

            if now >= run_at:
                run_at = run_at + timedelta(days=1)
            sleep_sec = max(30.0, min((run_at - now).total_seconds(), 3600.0))
            await asyncio.sleep(sleep_sec)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("[Discord] diary scheduler error: %s", e)
            await asyncio.sleep(300)


def build_bot(state: DiscordConsoleState) -> commands.Bot:
    intents = discord.Intents.default()
    intents.message_content = state.message_content_required()
    intents.reactions = True
    bot = commands.Bot(command_prefix="!", intents=intents)

    # 常設タスクボードの管理。永続View登録 (setup_hook) → ボード探索/作成 (on_ready)。
    board_manager = TaskBoardManager(
        bot,
        state,
        channel_id=state.resolve_task_board_channel_id(),
        enabled=state.tasks_board_enabled and state.task_store is not None,
        state_path=PROJECT_ROOT / "data" / "tasks" / "board_state.json",
    )
    state.task_board = board_manager
    board_setup_started = False

    @bot.event
    async def setup_hook() -> None:
        if board_manager.enabled:
            # 再起動後もボードのボタン/Selectを生かすため、永続Viewを登録する。
            bot.add_view(TaskBoardView(board_manager))
        guild_id = os.environ.get("DISCORD_GUILD_ID", "").strip()
        if guild_id:
            guild = discord.Object(id=int(guild_id))
            bot.tree.copy_global_to(guild=guild)
            await bot.tree.sync(guild=guild)
            logger.info("[Discord] slash commands synced to guild %s", guild_id)
        else:
            await bot.tree.sync()
            logger.info("[Discord] global slash commands synced")
        if state.diary_enabled and state.diary_task is None:
            state.diary_task = asyncio.create_task(diary_scheduler_loop(bot, state))
            logger.info("[Discord] diary scheduler started")

    # プロアクティブ発話の起動は1度だけ (on_ready は再接続で複数回発火しうる)
    proactive_started = False

    @bot.event
    async def on_ready() -> None:
        nonlocal proactive_started, board_setup_started
        logger.info("[Discord] logged in as %s (%s)", bot.user, bot.user.id if bot.user else "unknown")
        logger.info("[Discord] auto reply channels: %s", sorted(state.auto_reply_channel_ids))
        logger.info("[Discord] terminal channels: %s", sorted(state.terminal_channel_ids))
        logger.info(
            "[Discord] diary: enabled=%s channel=%s time=%s %s",
            state.diary_enabled,
            state.diary_channel_id or "-",
            state.diary_post_time.strftime("%H:%M"),
            state.diary_timezone,
        )
        if state.voice_stt is not None:
            logger.info(
                "[Discord] voice STT: enabled=%s available=%s transcript_channel=%s",
                state.voice_stt.config.enabled,
                state.voice_stt.available,
                state.voice_stt.transcript_channel_id or "-",
            )
        if not proactive_started:
            proactive_started = True
            try:
                state.proactive_bridge = create_proactive_bridge(state, bot)
            except Exception as e:
                logger.error("[Discord] proactive 初期化失敗 (proactiveなしで続行): %s", e)
                state.proactive_bridge = None
            if state.proactive_bridge is not None:
                state.proactive_bridge.start()
                cfg = state.proactive_bridge.config
                logger.info(
                    "[Discord] proactive started: channel=%s rewrite=%s interval=%ss "
                    "chat=%s chat_interval=%sh idle=%smin daily_limit=%s "
                    "max_backoff=%sh autoread=%s quiet=%s-%s",
                    cfg.channel_id,
                    cfg.llm_rewrite,
                    cfg.check_interval,
                    cfg.conversation_enabled,
                    cfg.conversation_interval_hours,
                    cfg.conversation_idle_minutes,
                    cfg.conversation_daily_limit,
                    cfg.conversation_max_backoff_hours,
                    cfg.conversation_autoread,
                    cfg.quiet_hours[0],
                    cfg.quiet_hours[1],
                )
            _start_task_reminder()
            # カレンダー同期: schedule 洗い替えは ProactiveEngine と同じ
            # UserProfile インスタンスに書く必要がある (エンジンは disk を再読込
            # しないため)。bridge が無ければ profile=None で pull は upcoming.json
            # の書き出しのみ行う。
            shared_profile = None
            if state.proactive_bridge is not None:
                shared_profile = getattr(state.proactive_bridge.engine, "profile", None)
            try:
                state.start_calendar_sync(profile=shared_profile)
            except Exception as e:
                logger.error("[Discord] calendar sync 起動失敗: %s", e)
            _start_event_reminder()
        if not board_setup_started and board_manager.enabled:
            board_setup_started = True
            asyncio.create_task(board_manager.setup())

    def _start_task_reminder() -> None:
        """タスクのリマインドエンジンを起動 (Discord bot プロセスでのみ、v1)。"""
        if not state.tasks_reminder_enabled or state.task_store is None:
            logger.info("[Discord] task reminder disabled (enabled=%s)", state.tasks_reminder_enabled)
            return
        if state.task_reminder_engine is not None:
            return
        try:
            engine = TaskReminderEngine(
                state.task_store,
                task_reminder_callback,
                owner="discord",
                timezone_name=state.tasks_timezone,
                quiet_hours=state.tasks_quiet_hours,
                check_interval=60.0,
            )
            engine.start()
            state.task_reminder_engine = engine
            logger.info(
                "[Discord] task reminder started: quiet_hours=%s-%s channel=%s",
                state.tasks_quiet_hours[0],
                state.tasks_quiet_hours[1],
                state.resolve_task_channel_id(),
            )
        except Exception as e:
            logger.error("[Discord] task reminder 起動失敗 (タスク通知なしで続行): %s", e)
            state.task_reminder_engine = None

    def _start_event_reminder() -> None:
        """予定 (Google Calendar イベント) の開始前リマインドエンジンを起動する。

        upcoming.json は calendar pull ワーカーが更新するため、カレンダー同期が
        有効なときだけ意味を持つ。二重通知を防ぐため、起動時に ProactiveEngine の
        gcal 由来 schedule リマインド (15分前) をスキップさせる。
        """
        if not state.event_remind_enabled or not state.tasks_calendar_sync_enabled:
            logger.info(
                "[Discord] event reminder disabled (enabled=%s, calendar_sync=%s)",
                state.event_remind_enabled,
                state.tasks_calendar_sync_enabled,
            )
            return
        if state.event_reminder_engine is not None:
            return
        try:
            from src.tasks.event_reminder import EventReminderEngine

            engine = EventReminderEngine(
                callback=event_reminder_callback,
                upcoming_path=str(PROJECT_ROOT / "data" / "calendar" / "upcoming.json"),
                db_path=str(PROJECT_ROOT / "data" / "tasks" / "tasks.db"),
                lead_min=state.event_remind_lead_min,
                timezone_name=state.tasks_timezone,
                quiet_hours=state.tasks_quiet_hours,
                check_interval=60.0,
            )
            engine.start()
            state.event_reminder_engine = engine
            if state.proactive_bridge is not None:
                try:
                    state.proactive_bridge.engine.skip_gcal_schedule = True
                except Exception:
                    pass
            logger.info(
                "[Discord] event reminder started: lead=%smin quiet_hours=%s-%s",
                state.event_remind_lead_min,
                state.tasks_quiet_hours[0],
                state.tasks_quiet_hours[1],
            )
        except Exception as e:
            logger.error("[Discord] event reminder 起動失敗 (予定通知なしで続行): %s", e)
            state.event_reminder_engine = None

    def event_reminder_callback(*, trigger_type: str, message: str, event_id: str, title: str) -> None:
        """EventReminderEngine のスレッドから呼ばれる。bot ループへブリッジする。"""
        loop = getattr(bot, "loop", None)
        if loop is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(deliver_event_reminder(message), loop)
        except Exception as e:
            logger.error("[Discord] event reminder bridge error: %s", e)

    async def deliver_event_reminder(message: str) -> None:
        channel_id = state.resolve_task_channel_id()
        if channel_id is None:
            logger.warning("[Discord] event reminder: 配送先チャンネルが未設定")
            return
        channel = bot.get_channel(channel_id)
        if channel is None:
            try:
                channel = await bot.fetch_channel(channel_id)
            except discord.HTTPException:
                logger.warning("[Discord] event reminder: チャンネル取得失敗 %s", channel_id)
                return
        # 定型文をペルソナ口調へ言い換え (失敗時は原文)
        try:
            text = await asyncio.to_thread(state.rewrite_task_text, message)
        except Exception:
            text = message
        emotion, text = parse_emotion_tag(text)
        if not text:
            text = message
        try:
            await channel.send(text)
        except Exception as e:
            logger.error("[Discord] event reminder send failed: %s", e)
            return
        # 通話接続中なら音声でも読み上げる (既存パターン)
        if state.voice_tts is not None:
            try:
                asyncio.create_task(state.voice_tts.autoread(text, emotion=emotion))
            except Exception:
                pass

    def task_reminder_callback(*, trigger_type: str, message: str, task_id: int, stage: str) -> None:
        """リマインドエンジンのスレッドから呼ばれる。bot ループへブリッジする。"""
        loop = getattr(bot, "loop", None)
        if loop is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(
                deliver_task_reminder(message, task_id, stage), loop
            )
        except Exception as e:
            logger.error("[Discord] task reminder bridge error: %s", e)

    async def deliver_task_reminder(message: str, task_id: int, stage: str) -> None:
        channel_id = state.resolve_task_channel_id()
        if channel_id is None:
            logger.warning("[Discord] task reminder: 配送先チャンネルが未設定")
            return
        channel = bot.get_channel(channel_id)
        if channel is None:
            try:
                channel = await bot.fetch_channel(channel_id)
            except discord.HTTPException:
                logger.warning("[Discord] task reminder: チャンネル取得失敗 %s", channel_id)
                return
        # 定型文をペルソナ口調へ言い換え (失敗時は原文)
        try:
            text = await asyncio.to_thread(state.rewrite_task_text, message)
        except Exception:
            text = message
        emotion, text = parse_emotion_tag(text)
        if not text:
            text = message
        view = TaskReminderView(state, task_id)
        try:
            await channel.send(text, view=view)
        except Exception as e:
            logger.error("[Discord] task reminder send failed: %s", e)
            return
        # 通話接続中なら音声でも読み上げる (既存パターン)
        if state.voice_tts is not None:
            try:
                asyncio.create_task(state.voice_tts.autoread(text, emotion=emotion))
            except Exception:
                pass

    task_extraction_enabled = parse_bool(
        os.environ.get("TASKS_CHAT_EXTRACTION_ENABLED"), default=True
    )

    async def register_quick_task(message: discord.Message, body: str, source: str) -> None:
        """「タスク: ...」を確認なしで即登録し、短く返信する。"""
        title = body.strip()
        if not title:
            return
        task_id = await asyncio.to_thread(state.add_task_quick, title, source)
        if task_id is None:
            await message.reply("タスクを登録できませんでした。", mention_author=False)
            return
        await message.reply(f"タスクを登録しました (#{task_id}): {title}", mention_author=False)
        state.refresh_task_board()

    async def maybe_offer_task_from_chat(message: discord.Message, user_text: str) -> None:
        """自然会話からタスク候補を抽出し、確認ボタン付きで提示する (自動登録しない)。"""
        if not task_extraction_enabled or state.task_store is None:
            return
        extracted = await asyncio.to_thread(state.extract_task_from_text, user_text)
        if extracted is None:
            return
        tz = ZoneInfo(state.tasks_timezone)
        due_at = extracted["due_at"]
        granularity = extracted.get("due_granularity")
        due_str = make_due_summary(due_at, tz)
        prio = extracted["priority"]
        view = TaskConfirmView(
            state,
            title=extracted["title"],
            due_at=due_at,
            due_granularity=granularity,
            priority=prio,
            source="chat",
        )
        content = (
            f"これ、タスクにする? 「{extracted['title']}」"
            f" (期限: {due_str}, 優先度: {prio})"
        )
        try:
            sent = await message.reply(content, view=view, mention_author=False)
            view.message = sent
        except discord.HTTPException as e:
            logger.error("[Discord] task confirm send failed: %s", e)

    async def handle_voice_reply(message: discord.Message, transcript_text: str) -> None:
        """通話 transcript (デバウンス後の結合テキスト) に1回だけLLM返信する。"""
        async with message.channel.typing():
            try:
                response, emotion = await asyncio.to_thread(
                    state.ask_message_text,
                    message,
                    transcript_text,
                )
            except Exception as e:
                await message.reply(f"LLM error: `{e}`", mention_author=False)
                return
        reply_messages = []
        for chunk in split_message(response):
            sent = await message.reply(chunk, mention_author=False)
            reply_messages.append(sent)
        state.record_auto_reply(
            message,
            response,
            reply_messages,
            user_text=transcript_text,
            source="discord_voice_transcript",
        )
        if state.voice_tts is not None:
            # 通話由来の質問には音声でも返す (接続中のみ、失敗しても無視)
            asyncio.create_task(state.voice_tts.autoread(response, emotion=emotion))
        if reply_messages:
            for reaction in (GOOD_REACTION, BAD_REACTION):
                try:
                    await reply_messages[0].add_reaction(reaction)
                except discord.HTTPException as e:
                    logger.warning("[Discord] feedback reaction failed: %s", e)

    # 断片ごとの即返信を避け、同じ話者の後続断片が落ち着いてから1回だけ返信する。
    voice_reply_debouncer = VoiceReplyDebouncer(
        debounce_ms=state.voice_reply_debounce_ms,
        max_ms=state.voice_reply_debounce_max_ms,
        on_flush=lambda msg, txt: asyncio.create_task(handle_voice_reply(msg, txt)),
    )
    state.voice_reply_debouncer = voice_reply_debouncer

    @bot.event
    async def on_message(message: discord.Message) -> None:
        bot_user_id = bot.user.id if bot.user is not None else None
        transcript = None
        if state.is_allowed_voice_transcript_message(message, bot_user_id=bot_user_id):
            transcript = state.extract_voice_transcript(message.content)
        elif message.author.bot:
            return

        if transcript is not None:
            speaker, transcript_text = transcript
            # 通話でも「タスク:」プレフィックスは確認なしで即登録する。
            task_match = TASK_PREFIX_RE.match(transcript_text.strip())
            if task_match and state.task_store is not None:
                await register_quick_task(message, task_match.group("body"), source="voice")
                return
            # 「予定〜入れて」も通話から直接 Google Calendar に登録する。
            event_reply = await asyncio.to_thread(
                state.try_register_event_text, transcript_text.strip()
            )
            if event_reply is not None:
                await message.reply(event_reply, mention_author=False)
                if state.voice_tts is not None:
                    try:
                        asyncio.create_task(state.voice_tts.autoread(event_reply))
                    except Exception:
                        pass
                return
            # 話者ごとにバッファし、デバウンス満了で結合テキストを1回返信する。
            voice_reply_debouncer.submit(speaker, message, transcript_text)
            return

        # 許可ユーザーの発言で休憩提案タイマーをリセット (proactive)
        if state.proactive_bridge is not None and (
            not state.allowed_user_ids or message.author.id in state.allowed_user_ids
        ):
            state.proactive_bridge.notify_user_activity()

        terminal_content = message.content.strip()
        terminal_prefix = state.terminal.command_prefix if state.terminal else "!t"
        terminal_reason = state.terminal_reject_reason(message)

        if terminal_reason is None and not terminal_content:
            await message.reply(
                "terminal error:\n"
                "message content is empty. Discord Developer Portal で Message Content Intent を有効にしてください。",
                mention_author=False,
            )
            return

        if terminal_reason is None and terminal_content.startswith(terminal_prefix):
            async with message.channel.typing():
                try:
                    output = await asyncio.to_thread(state.handle_terminal_message, message)
                except Exception as e:
                    output = f"terminal error: `{e}`"
            if output:
                for chunk in split_message(f"```text\n{clean_output(output, 1800)}\n```"):
                    await message.reply(chunk, mention_author=False)
            return

        if terminal_content.startswith(terminal_prefix):
            logger.info("[Discord] terminal message ignored: %s", terminal_reason)
            if terminal_reason and not terminal_reason.startswith("user is not allowed"):
                await message.reply(f"terminal error:\n{terminal_reason}", mention_author=False)
            return

        if not state.is_allowed_message(message):
            return
        content = message.content.strip()
        if not content:
            return

        if state.proactive_bridge is not None:
            control_reply = state.proactive_bridge.handle_conversation_control(
                message.channel.id, content
            )
            if control_reply is not None:
                await message.reply(control_reply, mention_author=False)
                return

        correction = parse_correction(content)
        if correction is not None:
            result_text = state.record_correction(message, correction)
            await message.reply(result_text, mention_author=False)
            return

        # 「タスク: ...」は明示的なので確認なしで即登録する。
        task_match = TASK_PREFIX_RE.match(content)
        if task_match and state.task_store is not None:
            await register_quick_task(message, task_match.group("body"), source="command")
            return

        # 「予定: ...」や「〜の予定入れて」は Google Calendar に登録する。
        event_reply = await asyncio.to_thread(state.try_register_event_text, content)
        if event_reply is not None:
            await message.reply(event_reply, mention_author=False)
            state.record_direct_reply(message, content, event_reply)
            return

        # 自発質問の直後なら質問文も一緒にLLM・学習ログへ渡す。
        # 「カレーかな」のような短い返答でも、何への回答かを失わない。
        llm_content = content
        reply_source = "discord_auto_reply"
        if state.proactive_bridge is not None:
            llm_content, was_proactive_reply = state.proactive_bridge.contextualize_reply(
                message.channel.id, content
            )
            if was_proactive_reply:
                reply_source = "discord_proactive_reply"

        async with message.channel.typing():
            try:
                response, _emotion = await asyncio.to_thread(
                    state.ask_message_text, message, llm_content
                )
            except Exception as e:
                await message.reply(f"LLM error: `{e}`", mention_author=False)
                return
        reply_messages = []
        for chunk in split_message(response):
            sent = await message.reply(chunk, mention_author=False)
            reply_messages.append(sent)
        state.record_auto_reply(
            message,
            response,
            reply_messages,
            user_text=llm_content,
            source=reply_source,
        )
        # 返信送信後に非同期でタスク抽出 (レイテンシに影響させない)。自動登録はしない。
        asyncio.create_task(maybe_offer_task_from_chat(message, content))
        if reply_messages:
            for reaction in (GOOD_REACTION, BAD_REACTION):
                try:
                    await reply_messages[0].add_reaction(reaction)
                except discord.HTTPException as e:
                    logger.warning("[Discord] feedback reaction failed: %s", e)

    # 👎で言い直し案を出したメッセージのID (多重生成を防ぐ)
    correction_offered_ids: set[int] = set()
    correction_suggest_enabled = parse_bool(
        os.environ.get("DISCORD_CORRECTION_SUGGESTIONS_ENABLED"), default=True
    )

    async def offer_correction_candidates(payload: discord.RawReactionActionEvent) -> None:
        """👎されたbot返答への言い直し案を生成し、選択ボタン付きで提示する。"""
        if state.training_log is None:
            return
        turn = state.training_log.get_turn_by_assistant_message_id(payload.message_id)
        if turn is None:
            return
        channel = bot.get_channel(payload.channel_id)
        if channel is None:
            try:
                channel = await bot.fetch_channel(payload.channel_id)
            except discord.HTTPException:
                return
        try:
            target = await channel.fetch_message(payload.message_id)
        except discord.HTTPException:
            return

        notice = await target.reply("言い直し案を生成中...", mention_author=False)
        try:
            raw = await asyncio.gather(
                *[
                    asyncio.to_thread(state.generate_correction_candidate, turn, temp)
                    for temp in (0.5, 0.85, 1.15)
                ],
                return_exceptions=True,
            )
        except Exception as e:
            await notice.edit(content=f"言い直し案の生成に失敗しました: `{e}`")
            return

        original = (turn.get("assistant") or "").strip()
        candidates: list[str] = []
        for item in raw:
            if isinstance(item, Exception) or not isinstance(item, str):
                continue
            text = item.strip()
            if not text or text == original or text in candidates:
                continue
            candidates.append(text)

        if not candidates:
            await notice.edit(content="言い直し案を生成できませんでした。「修正する」(右クリック)で直接書けます。")
            return

        view = CorrectionPickView(state, target, candidates)
        await notice.edit(content=render_correction_choices(candidates), view=view)
        view.message = notice

    @bot.event
    async def on_raw_reaction_add(payload: discord.RawReactionActionEvent) -> None:
        if bot.user is not None and payload.user_id == bot.user.id:
            return
        channel_id = payload.channel_id
        if channel_id is None or not state.is_allowed_feedback(payload.user_id, channel_id):
            return
        state.record_feedback(
            assistant_message_id=payload.message_id,
            guild_id=payload.guild_id,
            channel_id=channel_id,
            user_id=payload.user_id,
            emoji=str(payload.emoji),
        )
        if (
            correction_suggest_enabled
            and str(payload.emoji) == BAD_REACTION
            and payload.message_id not in correction_offered_ids
        ):
            correction_offered_ids.add(payload.message_id)
            asyncio.create_task(offer_correction_candidates(payload))

    @bot.tree.command(name="ask", description="subpc_living の LLM に質問します")
    @app_commands.describe(prompt="質問または指示", ephemeral="自分だけに表示する")
    async def ask(interaction: discord.Interaction, prompt: str, ephemeral: bool = False) -> None:
        if not await require_allowed(interaction, state):
            return
        await interaction.response.defer(thinking=True, ephemeral=ephemeral)
        try:
            response, _emotion = await asyncio.to_thread(
                state.ask, interaction, prompt, ephemeral=ephemeral
            )
        except Exception as e:
            await interaction.followup.send(f"LLM error: `{e}`", ephemeral=True)
            return
        for chunk in split_message(response):
            await interaction.followup.send(chunk, ephemeral=ephemeral)

    @bot.tree.command(name="reset", description="このチャンネルの会話履歴をリセットします")
    async def reset(interaction: discord.Interaction) -> None:
        if not await require_allowed(interaction, state):
            return
        state.reset_session(interaction)
        await interaction.response.send_message("会話履歴をリセットしました。", ephemeral=True)

    @bot.tree.command(name="tts", description="テキストをTTSで読み上げ WAV にします")
    @app_commands.describe(text="読み上げるテキスト", voice="TTS voice", speed="読み上げ速度")
    @app_commands.choices(
        voice=[
            app_commands.Choice(name=f"{voice} - {label}", value=voice)
            for voice, label in all_tts_voices().items()
        ]
    )
    async def tts(
        interaction: discord.Interaction,
        text: str,
        voice: str | None = None,
        speed: float = 1.0,
    ) -> None:
        if not await require_allowed(interaction, state):
            return
        speed = max(0.5, min(speed, 1.5))
        await interaction.response.defer(thinking=True)
        try:
            wav = await asyncio.to_thread(state.synthesize, text, voice, speed)
        except Exception as e:
            await interaction.followup.send(f"TTS error: `{e}`", ephemeral=True)
            return
        file_obj = io.BytesIO(wav)
        await interaction.followup.send(
            content=f"voice={voice} speed={speed}",
            file=discord.File(file_obj, filename="tts.wav"),
        )

    voice_group = app_commands.Group(name="voice", description="Discord通話のSTTを操作します")

    @voice_group.command(name="join", description="実行者がいる通話チャンネルへ参加します")
    async def voice_join(interaction: discord.Interaction) -> None:
        if not await require_allowed(interaction, state):
            return
        if state.voice_stt is None:
            await interaction.response.send_message("voice STT が初期化されていません。", ephemeral=True)
            return
        await interaction.response.defer(thinking=True)
        try:
            text = await state.voice_stt.join(interaction)
        except VoiceSTTError as e:
            await interaction.followup.send(f"voice STT error: {e}", ephemeral=True)
            return
        except Exception as e:
            await interaction.followup.send(f"voice STT error: `{e}`", ephemeral=True)
            return
        await interaction.followup.send(text)

    @voice_group.command(name="start", description="通話チャンネルのSTTを開始します")
    @app_commands.describe(transcript_channel="文字起こし投稿先。空なら設定値または実行チャンネル")
    async def voice_start(
        interaction: discord.Interaction,
        transcript_channel: discord.TextChannel | None = None,
    ) -> None:
        if not await require_allowed(interaction, state):
            return
        if state.voice_stt is None:
            await interaction.response.send_message("voice STT が初期化されていません。", ephemeral=True)
            return
        await interaction.response.defer(thinking=True)
        try:
            text = await state.voice_stt.start(
                interaction,
                transcript_channel=transcript_channel,
            )
        except VoiceSTTError as e:
            await interaction.followup.send(f"voice STT error: {e}", ephemeral=True)
            return
        except Exception as e:
            await interaction.followup.send(f"voice STT error: `{e}`", ephemeral=True)
            return
        await interaction.followup.send(text)

    @voice_group.command(name="stop", description="通話チャンネルのSTTを停止します")
    async def voice_stop(interaction: discord.Interaction) -> None:
        if not await require_allowed(interaction, state):
            return
        if state.voice_stt is None:
            await interaction.response.send_message("voice STT が初期化されていません。", ephemeral=True)
            return
        await interaction.response.defer(thinking=True)
        try:
            text = await state.voice_stt.stop()
        except Exception as e:
            await interaction.followup.send(f"voice STT error: `{e}`", ephemeral=True)
            return
        await interaction.followup.send(text)

    @voice_group.command(name="leave", description="通話チャンネルから退出します")
    async def voice_leave(interaction: discord.Interaction) -> None:
        if not await require_allowed(interaction, state):
            return
        if state.voice_stt is None:
            await interaction.response.send_message("voice STT が初期化されていません。", ephemeral=True)
            return
        await interaction.response.defer(thinking=True)
        try:
            text = await state.voice_stt.leave()
        except Exception as e:
            await interaction.followup.send(f"voice STT error: `{e}`", ephemeral=True)
            return
        await interaction.followup.send(text)

    @voice_group.command(name="say", description="接続中の通話チャンネルでテキストを読み上げます")
    @app_commands.describe(text="読み上げるテキスト", voice="TTS voice", speed="読み上げ速度")
    @app_commands.choices(
        voice=[
            app_commands.Choice(name=f"{voice} - {label}", value=voice)
            for voice, label in all_tts_voices().items()
        ]
    )
    async def voice_say(
        interaction: discord.Interaction,
        text: str,
        voice: str | None = None,
        speed: float | None = None,
    ) -> None:
        if not await require_allowed(interaction, state):
            return
        if state.voice_tts is None:
            await interaction.response.send_message("voice TTS が初期化されていません。", ephemeral=True)
            return
        await interaction.response.defer(thinking=True)
        try:
            duration = await state.voice_tts.say(text, voice=voice, speed=speed)
        except VoiceTTSError as e:
            await interaction.followup.send(f"voice TTS error: {e}", ephemeral=True)
            return
        except Exception as e:
            await interaction.followup.send(f"voice TTS error: `{e}`", ephemeral=True)
            return
        await interaction.followup.send(
            f"読み上げました ({duration:.1f}秒): {text[:100]}{'...' if len(text) > 100 else ''}"
        )

    @voice_group.command(name="tts", description="通話でのTTS自動読み上げをON/OFFします")
    @app_commands.describe(enabled="true=読み上げON / false=OFF。省略で現在の状態を表示")
    async def voice_tts_toggle(
        interaction: discord.Interaction,
        enabled: bool | None = None,
    ) -> None:
        if not await require_allowed(interaction, state):
            return
        if state.voice_tts is None:
            await interaction.response.send_message("voice TTS が初期化されていません。", ephemeral=True)
            return
        if enabled is None:
            current = "ON" if state.voice_tts.autoread_enabled else "OFF"
            await interaction.response.send_message(
                f"TTS自動読み上げは現在 {current} です。/voice tts enabled:true|false で切り替えられます。",
                ephemeral=True,
            )
            return
        state.voice_tts.autoread_enabled = enabled
        await interaction.response.send_message(
            f"TTS自動読み上げを {'ON' if enabled else 'OFF'} にしました。"
            + ("" if enabled else " (/voice say での手動読み上げは引き続き使えます)")
        )

    @voice_group.command(name="status", description="通話STTの状態を確認します")
    async def voice_status(interaction: discord.Interaction) -> None:
        if not await require_allowed(interaction, state):
            return
        if state.voice_stt is None:
            await interaction.response.send_message("voice STT が初期化されていません。", ephemeral=True)
            return
        status_lines = state.voice_stt.status_text()
        if state.voice_tts is not None:
            status_lines += (
                f"\nvoice_tts_autoread: {state.voice_tts.autoread_enabled}"
                f"\nvoice_tts_played: {state.voice_tts.played_count}"
                f"\nvoice_tts_last_error: {state.voice_tts.last_error or '-'}"
            )
        await interaction.response.send_message(
            f"```text\n{clean_output(status_lines, 1800)}\n```",
            ephemeral=True,
        )

    bot.tree.add_command(voice_group)

    task_group = app_commands.Group(name="task", description="タスク管理")

    def _tasks_tz() -> ZoneInfo:
        return ZoneInfo(state.tasks_timezone)

    def _format_task_line(task: dict) -> str:
        tz = _tasks_tz()
        due = task["due_at"]
        if due is None:
            due_str = "期限なし"
        else:
            local = due.astimezone(tz)
            due_str = (
                local.strftime("%-m/%-d")
                if task["due_granularity"] == "date"
                else local.strftime("%-m/%-d %H:%M")
            )
        prio = {"high": "[高]", "low": "[低]", "normal": ""}.get(task["priority"], "")
        line = f"#{task['id']} {prio}{task['title']} (期限: {due_str})"
        if task["action_hint"]:
            line += f" 次の一手: {task['action_hint']}"
        return line

    @task_group.command(name="add", description="タスクを追加します")
    @app_commands.describe(
        title="タスク名",
        due="期限。例: 明日 18時 / 金曜 / 来週水曜 / 7/10 15:00 (省略可)",
        priority="優先度",
        note="メモ (省略可)",
    )
    @app_commands.choices(
        priority=[
            app_commands.Choice(name="high", value="high"),
            app_commands.Choice(name="normal", value="normal"),
            app_commands.Choice(name="low", value="low"),
        ]
    )
    async def task_add(
        interaction: discord.Interaction,
        title: str,
        due: str = "",
        priority: str = "normal",
        note: str = "",
    ) -> None:
        if not await require_allowed(interaction, state):
            return
        if state.task_store is None:
            await interaction.response.send_message("タスクストアが無効です。", ephemeral=True)
            return
        now = datetime.now(timezone.utc)
        due_at = None
        granularity = None
        if due.strip():
            due_at, granularity = parse_due(due, now, _tasks_tz())
            if due_at is None:
                await interaction.response.send_message(
                    "期限を解釈できませんでした。例: 明日 18時 / 金曜 / 来週水曜 / 7/10 15:00", ephemeral=True
                )
                return
        try:
            task_id = await asyncio.to_thread(
                state.task_store.add,
                title,
                note=note or None,
                due_at=due_at,
                due_granularity=granularity,
                priority=priority,
                source="command",
            )
        except Exception as e:
            await interaction.response.send_message(f"追加に失敗しました: `{e}`", ephemeral=True)
            return
        due_str = make_due_summary(due_at, _tasks_tz())
        await interaction.response.send_message(
            f"タスクを追加しました (#{task_id}): {title} (期限: {due_str}, 優先度: {priority})"
        )
        state.refresh_task_board()

    @task_group.command(name="list", description="未完了タスクを一覧表示します")
    async def task_list(interaction: discord.Interaction) -> None:
        if not await require_allowed(interaction, state):
            return
        if state.task_store is None:
            await interaction.response.send_message("タスクストアが無効です。", ephemeral=True)
            return
        await interaction.response.defer(thinking=True, ephemeral=True)
        tasks = await asyncio.to_thread(state.task_store.list, "open", 50)
        if not tasks:
            await interaction.followup.send("未完了のタスクはありません。", ephemeral=True)
            return
        lines = [_format_task_line(t) for t in tasks]
        text = "未完了タスク:\n" + "\n".join(lines)
        for chunk in split_message(text):
            await interaction.followup.send(chunk, ephemeral=True)

    @task_group.command(name="done", description="タスクを完了にします")
    @app_commands.describe(task_id="タスクID (/task list で確認)")
    async def task_done(interaction: discord.Interaction, task_id: int) -> None:
        if not await require_allowed(interaction, state):
            return
        if state.task_store is None:
            await interaction.response.send_message("タスクストアが無効です。", ephemeral=True)
            return
        ok = await asyncio.to_thread(state.task_store.done, task_id)
        msg = f"タスク #{task_id} を完了にしました。" if ok else f"タスク #{task_id} が見つからないか、すでに完了/削除済みです。"
        await interaction.response.send_message(msg)
        if ok:
            state.refresh_task_board()

    @task_group.command(name="snooze", description="タスクのリマインドを先送りします")
    @app_commands.describe(task_id="タスクID", when="例: 30m / 2h / 明日")
    async def task_snooze(interaction: discord.Interaction, task_id: int, when: str) -> None:
        if not await require_allowed(interaction, state):
            return
        if state.task_store is None:
            await interaction.response.send_message("タスクストアが無効です。", ephemeral=True)
            return
        now = datetime.now(timezone.utc)
        until = parse_snooze(when, now, _tasks_tz())
        if until is None:
            await interaction.response.send_message(
                "先送り時間を解釈できませんでした。例: 30m / 2h / 明日", ephemeral=True
            )
            return
        ok = await asyncio.to_thread(state.task_store.snooze, task_id, until)
        if not ok:
            await interaction.response.send_message(
                f"タスク #{task_id} が見つからないか、すでに完了/削除済みです。", ephemeral=True
            )
            return
        until_str = until.astimezone(_tasks_tz()).strftime("%-m/%-d %H:%M")
        await interaction.response.send_message(f"タスク #{task_id} を {until_str} まで先送りしました。")
        state.refresh_task_board()

    @task_group.command(name="del", description="タスクを削除 (dropped) します")
    @app_commands.describe(task_id="タスクID")
    async def task_del(interaction: discord.Interaction, task_id: int) -> None:
        if not await require_allowed(interaction, state):
            return
        if state.task_store is None:
            await interaction.response.send_message("タスクストアが無効です。", ephemeral=True)
            return
        ok = await asyncio.to_thread(state.task_store.drop, task_id)
        msg = f"タスク #{task_id} を削除しました。" if ok else f"タスク #{task_id} が見つからないか、すでに完了/削除済みです。"
        await interaction.response.send_message(msg)
        if ok:
            state.refresh_task_board()

    bot.tree.add_command(task_group)

    focus_group = app_commands.Group(
        name="focus",
        description="システムに次の1件を決めてもらいます",
    )

    async def _require_priority(interaction: discord.Interaction) -> PriorityController | None:
        if not await require_allowed(interaction, state):
            return None
        if state.priority_controller is None:
            await interaction.response.send_message(
                "優先順位オーケストレーターが無効です。", ephemeral=True
            )
            return None
        return state.priority_controller

    @focus_group.command(name="now", description="今やる1件と、その決定理由を表示します")
    async def focus_now(interaction: discord.Interaction) -> None:
        controller = await _require_priority(interaction)
        if controller is None:
            return
        decision = await asyncio.to_thread(controller.recommend)
        await interaction.response.send_message(
            format_focus_decision(decision, _tasks_tz()), ephemeral=True
        )

    @focus_group.command(name="start", description="今の1件を開始し、選択を固定します")
    async def focus_start(interaction: discord.Interaction) -> None:
        controller = await _require_priority(interaction)
        if controller is None:
            return
        decision = await asyncio.to_thread(controller.start)
        await interaction.response.send_message(
            format_focus_decision(decision, _tasks_tz()), ephemeral=True
        )

    @focus_group.command(name="done", description="今の1件を完了し、次の1件を選びます")
    async def focus_done(interaction: discord.Interaction) -> None:
        controller = await _require_priority(interaction)
        if controller is None:
            return
        completed, decision = await asyncio.to_thread(controller.complete)
        prefix = "✅ 完了を記録しました。\n" if completed else "完了対象がなかったため、現在の推奨を表示します。\n"
        await interaction.response.send_message(
            prefix + format_focus_decision(decision, _tasks_tz()), ephemeral=True
        )
        if completed:
            state.refresh_task_board()

    @focus_group.command(name="next", description="今の1件を一時的に見送り、次を選びます")
    async def focus_next(interaction: discord.Interaction) -> None:
        controller = await _require_priority(interaction)
        if controller is None:
            return
        decision = await asyncio.to_thread(controller.next)
        await interaction.response.send_message(
            "今の候補を一時的に見送りました。\n"
            + format_focus_decision(decision, _tasks_tz()),
            ephemeral=True,
        )

    @focus_group.command(name="pick", description="タスクIDを指定して今やる1件を上書きします")
    @app_commands.describe(task_id="タスクID (/task list で確認)")
    async def focus_pick(interaction: discord.Interaction, task_id: int) -> None:
        controller = await _require_priority(interaction)
        if controller is None:
            return
        decision = await asyncio.to_thread(controller.pick, task_id)
        if decision is None:
            await interaction.response.send_message(
                f"未完了のタスク #{task_id} が見つかりません。", ephemeral=True
            )
            return
        await interaction.response.send_message(
            "手動指定を優先しました。\n" + format_focus_decision(decision, _tasks_tz()),
            ephemeral=True,
        )

    @focus_group.command(name="status", description="委任状態と継続の記録を表示します")
    async def focus_status(interaction: discord.Interaction) -> None:
        controller = await _require_priority(interaction)
        if controller is None:
            return
        status = await asyncio.to_thread(controller.status)
        active = status["active_task_id"] or "-"
        text = (
            f"active_task: {active}\n"
            f"started: {status['started']}\n"
            f"completed_today: {status['completed_today']}\n"
            f"streak_days: {status['streak_days']}\n"
            f"decisions_delegated: {status['decision_count']}\n"
            f"state_error: {status['last_error'] or '-'}"
        )
        await interaction.response.send_message(f"```text\n{text}\n```", ephemeral=True)

    bot.tree.add_command(focus_group)

    async def correction_menu_callback(
        interaction: discord.Interaction,
        message: discord.Message,
    ) -> None:
        if not await require_allowed(interaction, state):
            return
        if bot.user is None or message.author.id != bot.user.id:
            await interaction.response.send_message(
                "botの返答メッセージに対して実行してください。", ephemeral=True
            )
            return
        await interaction.response.send_modal(CorrectionModal(state, message))

    correction_menu = app_commands.ContextMenu(
        name="修正する",
        callback=correction_menu_callback,
    )
    bot.tree.add_command(correction_menu)

    async def add_task_menu_callback(
        interaction: discord.Interaction,
        message: discord.Message,
    ) -> None:
        if not await require_allowed(interaction, state):
            return
        if state.task_store is None:
            await interaction.response.send_message("タスクストアが無効です。", ephemeral=True)
            return
        # モーダルは3秒以内に開く必要があるため、ここでは重い処理をしない。
        prefill = (message.content or "").strip()
        await interaction.response.send_modal(
            TaskModal(state, prefill_title=prefill, source="context_menu")
        )

    add_task_menu = app_commands.ContextMenu(
        name="タスクに登録",
        callback=add_task_menu_callback,
    )
    bot.tree.add_command(add_task_menu)

    @bot.tree.command(name="status", description="subpc_living の状態を確認します")
    async def status(interaction: discord.Interaction) -> None:
        if not await require_allowed(interaction, state):
            return
        await interaction.response.defer(thinking=True, ephemeral=True)
        text = await asyncio.to_thread(state.status_text)
        await interaction.followup.send(f"```text\n{clean_output(text, 1800)}\n```", ephemeral=True)

    @bot.tree.command(name="diary", description="日記を生成します。投稿先は DISCORD_DIARY_CHANNEL_ID です")
    @app_commands.describe(
        target_date="YYYY-MM-DD。空なら今日",
        post="trueなら日記チャンネルに投稿。falseなら自分だけにプレビュー",
        overwrite="保存済みの日記を作り直す",
    )
    async def diary(
        interaction: discord.Interaction,
        target_date: str = "",
        post: bool = False,
        overwrite: bool = False,
    ) -> None:
        if not await require_allowed(interaction, state):
            return
        if state.diary_service is None:
            await interaction.response.send_message("日記サービスが初期化されていません。", ephemeral=True)
            return
        try:
            diary_date = parse_diary_date(target_date, state.diary_timezone)
        except ValueError:
            await interaction.response.send_message("target_date は YYYY-MM-DD で指定してください。", ephemeral=True)
            return

        await interaction.response.defer(thinking=True, ephemeral=True)
        try:
            if post:
                result = await send_diary_to_channel(
                    bot,
                    state,
                    diary_date,
                    overwrite=overwrite,
                )
                await interaction.followup.send(
                    f"{result.target_date} の日記を投稿しました。",
                    ephemeral=True,
                )
                return

            result = await asyncio.to_thread(
                state.diary_service.generate,
                diary_date,
                save=False,
                overwrite=True,
                include_calendar=state.diary_calendar_enabled,
                calendar_id=state.diary_calendar_id,
            )
        except Exception as e:
            await interaction.followup.send(f"diary error: `{e}`", ephemeral=True)
            return

        for chunk in split_message(result.markdown):
            await interaction.followup.send(chunk, ephemeral=True)

    @bot.tree.command(name="personalize", description="保存済み日記からプロフィール更新候補を抽出します")
    @app_commands.describe(
        target_date="YYYY-MM-DD。空なら今日",
        dry_run="trueならプロフィールを更新せず監査ログだけ作成",
    )
    async def personalize(
        interaction: discord.Interaction,
        target_date: str = "",
        dry_run: bool = True,
    ) -> None:
        if not await require_allowed(interaction, state):
            return
        if state.daily_personalizer is None:
            await interaction.response.send_message("パーソナライズサービスが初期化されていません。", ephemeral=True)
            return
        try:
            diary_date = parse_diary_date(target_date, state.diary_timezone)
        except ValueError:
            await interaction.response.send_message("target_date は YYYY-MM-DD で指定してください。", ephemeral=True)
            return

        await interaction.response.defer(thinking=True, ephemeral=True)
        try:
            result = await asyncio.to_thread(
                state.daily_personalizer.run,
                diary_date,
                dry_run=dry_run,
            )
        except Exception as e:
            await interaction.followup.send(f"personalize error: `{e}`", ephemeral=True)
            return

        applied = "\n".join(
            f"- {key}: {len(values)}"
            for key, values in result.applied.items()
        )
        text = (
            f"target_date: {result.target_date}\n"
            f"dry_run: {result.dry_run}\n"
            f"applied_count: {result.applied_count}\n"
            f"audit_path: {result.audit_path}\n"
            f"applied:\n{applied}"
        )
        await interaction.followup.send(f"```text\n{clean_output(text, 1800)}\n```", ephemeral=True)

    @bot.tree.command(name="service", description="限定されたサービス管理コマンドを実行します")
    @app_commands.choices(
        action=[
            app_commands.Choice(name="status", value="status"),
            app_commands.Choice(name="health", value="health"),
            app_commands.Choice(name="gpu", value="gpu"),
            app_commands.Choice(name="logs", value="logs"),
            app_commands.Choice(name="start", value="start"),
            app_commands.Choice(name="stop", value="stop"),
            app_commands.Choice(name="restart", value="restart"),
        ],
        target=[
            app_commands.Choice(name="web", value="web"),
            app_commands.Choice(name="voice", value="voice"),
            app_commands.Choice(name="discord", value="discord"),
            app_commands.Choice(name="all", value="all"),
        ],
    )
    async def service(interaction: discord.Interaction, action: str, target: str = "web") -> None:
        if not await require_allowed(interaction, state):
            return
        await interaction.response.defer(thinking=True, ephemeral=True)
        try:
            output = await asyncio.to_thread(state.run_service_command, action, target)
        except Exception as e:
            output = f"service command error: {e}"
        await interaction.followup.send(f"```text\n{clean_output(output, 1800)}\n```", ephemeral=True)

    return bot


def main() -> None:
    load_env_file(PROJECT_ROOT / ".env")
    load_env_file(PROJECT_ROOT / "config" / "discord.env")
    token = os.environ.get("DISCORD_BOT_TOKEN", "").strip()
    if not token:
        raise SystemExit("DISCORD_BOT_TOKEN is required. See config/discord.env.example")

    state = DiscordConsoleState()
    state.initialize()
    bot = build_bot(state)
    try:
        bot.run(token)
    finally:
        state.close()


if __name__ == "__main__":
    main()
