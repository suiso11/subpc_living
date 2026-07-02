"""
Discord 操作コンソール。

Slash commands:
- /ask: Ollama に質問
- /tts: kokoro-onnx で WAV を生成して添付
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
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import discord
from discord import app_commands
from discord.ext import commands

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.audio.tts import KokoroTTS
from src.chat.client import OllamaClient
from src.chat.config import ChatConfig
from src.chat.session import ChatSession
from src.chat.web_search import WebSearchContext, create_web_search_context
from src.discord_bot.training import (
    BAD_REACTION,
    GOOD_REACTION,
    DiscordTrainingLog,
    parse_correction,
)
from src.discord_bot.terminal import DiscordTerminal
from src.discord_bot.voice_stt import DiscordVoiceSTT, VoiceSTTConfig, VoiceSTTError
from src.discord_bot.voice_tts import VoiceTTSConfig, VoiceTTSError, VoiceTTSPlayer
from src.diary.collector import DiaryCollector
from src.diary.service import DailyDiaryResult, DailyDiaryService
from src.integrations.google_calendar import GoogleCalendarMCPClient
from src.persona.daily_personalizer import DailyPersonalizer, PersonalizationResult
from src.service.healthcheck import HealthChecker


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
VOICE_TRANSCRIPT_RE = re.compile(r"^\[\d{2}:\d{2}:\d{2}\]\s+.+?:\s*(?P<text>.+)$", re.DOTALL)
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
            print(f"[Discord] ID を無視します: {item}")
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
        print(f"[Discord] ID を無視します: {value}")
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
        print(f"[Discord] DIARY_POST_TIME を無視します: {raw}")
        hour_text, minute_text = default.split(":", 1)
        return time(hour=int(hour_text), minute=int(minute_text))


def parse_float(value: str | None, default: float) -> float:
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError:
        print(f"[Discord] float 設定値を無視します: {value}")
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
        system_prompt = str(raw.get("system_prompt", config.system_prompt))
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
    llm: OllamaClient | None = None
    llm_clients: dict[tuple[str, str], OllamaClient] = field(default_factory=dict)
    llm_profiles: dict[str, DiscordLLMProfile] = field(default_factory=dict)
    channel_profile_map: dict[int, str] = field(default_factory=dict)
    tts: KokoroTTS | None = None
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
        self.training_log = DiscordTrainingLog(
            training_dir,
            enabled=parse_bool(os.environ.get("DISCORD_TRAINING_LOG_ENABLED"), default=True),
            system_prompt=self.config.system_prompt,
        )
        self.web_search = create_web_search_context(self.config)
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
            print(f"[Discord] diary initialization failed: {e}")
            self.diary_service = None
            self.daily_personalizer = None

    def close(self) -> None:
        if self.diary_task is not None:
            self.diary_task.cancel()
        if self.voice_stt is not None:
            self.voice_stt.close()
        for client in set(self.llm_clients.values()):
            client.close()

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
    def extract_voice_transcript_text(content: str) -> str | None:
        match = VOICE_TRANSCRIPT_RE.match(content.strip())
        if match is None:
            return None
        text = match.group("text").strip()
        return text or None

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
                print(f"[Discord] LLM profile を無視します: {name}")
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
                print(f"[Discord] channel profile ID を無視します: {raw_channel_id}")
                continue
            if profile_name not in self.llm_profiles:
                print(
                    "[Discord] 未定義のLLM profile を無視します: "
                    f"channel={channel_id} profile={profile_name}"
                )
                continue
            mapping[channel_id] = profile_name
        return mapping

    def _llm_for_profile(self, profile: DiscordLLMProfile) -> OllamaClient:
        key = (profile.ollama_base_url, profile.model)
        if key not in self.llm_clients:
            self.llm_clients[key] = OllamaClient(
                base_url=profile.ollama_base_url,
                model=profile.model,
            )
        return self.llm_clients[key]

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

    def ask(self, interaction: discord.Interaction, prompt: str, *, ephemeral: bool = False) -> str:
        if self.llm is None or self.config is None:
            raise RuntimeError("LLM is not initialized")
        session, lock, profile = self.get_session(interaction, ephemeral=ephemeral)
        return self.ask_session(session, prompt, lock, profile)

    def ask_message(self, message: discord.Message) -> str:
        if self.llm is None or self.config is None:
            raise RuntimeError("LLM is not initialized")
        return self.ask_message_text(message, message.content.strip())

    def ask_message_text(self, message: discord.Message, prompt: str) -> str:
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
    ) -> str:
        if self.llm is None or self.config is None:
            raise RuntimeError("LLM is not initialized")
        llm = self._llm_for_profile(profile)
        with lock:
            session.add_user_message(prompt)
            try:
                response = llm.generate(
                    session.build_messages(),
                    temperature=profile.temperature,
                    top_p=profile.top_p,
                    top_k=profile.top_k,
                    repeat_penalty=profile.repeat_penalty,
                    num_ctx=profile.num_ctx,
                    num_predict=profile.num_predict,
                )
            except Exception:
                if session._messages and session._messages[-1]["role"] == "user":
                    session._messages.pop()
                raise
            session.add_assistant_message(response)
            return response

    def synthesize(self, text: str, voice: str, speed: float) -> bytes:
        with self.tts_lock:
            if self.tts is None:
                self.tts = KokoroTTS(models_dir=PROJECT_ROOT / "models" / "tts" / "kokoro")
            self.tts.voice = voice
            self.tts.speed = speed
            return self.tts.synthesize(text)

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
        print(f"[Discord] diary calendar warning: {result.calendar_error[:500]}")
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
        print(
            "[Discord] daily personalization: "
            f"date={result.target_date} applied={result.applied_count} audit={result.audit_path}"
        )
        return result
    except Exception as e:
        print(f"[Discord] daily personalization failed: {e}")
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
                    print(f"[Discord] diary posted for {now.date().isoformat()}")
                except Exception as e:
                    print(f"[Discord] diary post failed: {e}")
                await asyncio.sleep(60)
                continue

            if now >= run_at:
                run_at = run_at + timedelta(days=1)
            sleep_sec = max(30.0, min((run_at - now).total_seconds(), 3600.0))
            await asyncio.sleep(sleep_sec)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"[Discord] diary scheduler error: {e}")
            await asyncio.sleep(300)


def build_bot(state: DiscordConsoleState) -> commands.Bot:
    intents = discord.Intents.default()
    intents.message_content = state.message_content_required()
    intents.reactions = True
    bot = commands.Bot(command_prefix="!", intents=intents)

    @bot.event
    async def setup_hook() -> None:
        guild_id = os.environ.get("DISCORD_GUILD_ID", "").strip()
        if guild_id:
            guild = discord.Object(id=int(guild_id))
            bot.tree.copy_global_to(guild=guild)
            await bot.tree.sync(guild=guild)
            print(f"[Discord] slash commands synced to guild {guild_id}")
        else:
            await bot.tree.sync()
            print("[Discord] global slash commands synced")
        if state.diary_enabled and state.diary_task is None:
            state.diary_task = asyncio.create_task(diary_scheduler_loop(bot, state))
            print("[Discord] diary scheduler started")

    @bot.event
    async def on_ready() -> None:
        print(f"[Discord] logged in as {bot.user} ({bot.user.id if bot.user else 'unknown'})")
        print(f"[Discord] auto reply channels: {sorted(state.auto_reply_channel_ids)}")
        print(f"[Discord] terminal channels: {sorted(state.terminal_channel_ids)}")
        print(
            "[Discord] diary: "
            f"enabled={state.diary_enabled} "
            f"channel={state.diary_channel_id or '-'} "
            f"time={state.diary_post_time.strftime('%H:%M')} {state.diary_timezone}"
        )
        if state.voice_stt is not None:
            print(
                "[Discord] voice STT: "
                f"enabled={state.voice_stt.config.enabled} "
                f"available={state.voice_stt.available} "
                f"transcript_channel={state.voice_stt.transcript_channel_id or '-'}"
            )

    @bot.event
    async def on_message(message: discord.Message) -> None:
        bot_user_id = bot.user.id if bot.user is not None else None
        transcript_text = None
        if state.is_allowed_voice_transcript_message(message, bot_user_id=bot_user_id):
            transcript_text = state.extract_voice_transcript_text(message.content)
        elif message.author.bot:
            return

        if transcript_text is not None:
            async with message.channel.typing():
                try:
                    response = await asyncio.to_thread(
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
                asyncio.create_task(state.voice_tts.autoread(response))
            if reply_messages:
                for reaction in (GOOD_REACTION, BAD_REACTION):
                    try:
                        await reply_messages[0].add_reaction(reaction)
                    except discord.HTTPException as e:
                        print(f"[Discord] feedback reaction failed: {e}")
            return

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
            print(f"[Discord] terminal message ignored: {terminal_reason}")
            if terminal_reason and not terminal_reason.startswith("user is not allowed"):
                await message.reply(f"terminal error:\n{terminal_reason}", mention_author=False)
            return

        if not state.is_allowed_message(message):
            return
        content = message.content.strip()
        if not content:
            return

        correction = parse_correction(content)
        if correction is not None:
            result_text = state.record_correction(message, correction)
            await message.reply(result_text, mention_author=False)
            return

        async with message.channel.typing():
            try:
                response = await asyncio.to_thread(state.ask_message, message)
            except Exception as e:
                await message.reply(f"LLM error: `{e}`", mention_author=False)
                return
        reply_messages = []
        for chunk in split_message(response):
            sent = await message.reply(chunk, mention_author=False)
            reply_messages.append(sent)
        state.record_auto_reply(message, response, reply_messages)
        if reply_messages:
            for reaction in (GOOD_REACTION, BAD_REACTION):
                try:
                    await reply_messages[0].add_reaction(reaction)
                except discord.HTTPException as e:
                    print(f"[Discord] feedback reaction failed: {e}")

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

    @bot.tree.command(name="ask", description="subpc_living の LLM に質問します")
    @app_commands.describe(prompt="質問または指示", ephemeral="自分だけに表示する")
    async def ask(interaction: discord.Interaction, prompt: str, ephemeral: bool = False) -> None:
        if not await require_allowed(interaction, state):
            return
        await interaction.response.defer(thinking=True, ephemeral=ephemeral)
        try:
            response = await asyncio.to_thread(state.ask, interaction, prompt, ephemeral=ephemeral)
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

    @bot.tree.command(name="tts", description="テキストを kokoro-onnx で読み上げ WAV にします")
    @app_commands.describe(text="読み上げるテキスト", voice="Kokoro voice", speed="読み上げ速度")
    @app_commands.choices(
        voice=[
            app_commands.Choice(name=f"{voice} - {label}", value=voice)
            for voice, label in KokoroTTS.JA_VOICES.items()
        ]
    )
    async def tts(
        interaction: discord.Interaction,
        text: str,
        voice: str = "jf_alpha",
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
    @app_commands.describe(text="読み上げるテキスト", voice="Kokoro voice", speed="読み上げ速度")
    @app_commands.choices(
        voice=[
            app_commands.Choice(name=f"{voice} - {label}", value=voice)
            for voice, label in KokoroTTS.JA_VOICES.items()
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
