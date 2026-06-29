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
from pathlib import Path

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
from src.service.healthcheck import HealthChecker


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
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


@dataclass
class DiscordConsoleState:
    config: ChatConfig | None = None
    llm: OllamaClient | None = None
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

    def initialize(self) -> None:
        self.config = ChatConfig.load(PROJECT_ROOT / "config" / "chat_config.json")
        self.llm = OllamaClient(
            base_url=self.config.ollama_base_url,
            model=self.config.model,
        )
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

    def close(self) -> None:
        if self.llm is not None:
            self.llm.close()

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

    def _session_for_key(self, key: str) -> tuple[ChatSession, threading.Lock]:
        assert self.config is not None
        with self.sessions_lock:
            if key not in self.sessions:
                self.sessions[key] = ChatSession(
                    system_prompt=self.config.system_prompt,
                    max_history_turns=self.config.max_history_turns,
                    history_dir=str(PROJECT_ROOT / self.config.history_dir),
                    web_search=self.web_search,
                )
            if key not in self.session_locks:
                self.session_locks[key] = threading.Lock()
            return self.sessions[key], self.session_locks[key]

    def get_session(
        self,
        interaction: discord.Interaction,
        *,
        ephemeral: bool = False,
    ) -> tuple[ChatSession, threading.Lock]:
        if ephemeral:
            key = self._ephemeral_session_key(
                interaction.guild_id,
                interaction.channel_id,
                interaction.user.id,
            )
        else:
            key = self._shared_session_key(interaction.guild_id, interaction.channel_id)
        return self._session_for_key(key)

    def get_message_session(self, message: discord.Message) -> tuple[ChatSession, threading.Lock]:
        key = self._shared_session_key(
            message.guild.id if message.guild else None,
            message.channel.id,
        )
        return self._session_for_key(key)

    def reset_session(self, interaction: discord.Interaction) -> None:
        key = self._shared_session_key(interaction.guild_id, interaction.channel_id)
        with self.sessions_lock:
            self.sessions.pop(key, None)
            self.session_locks.pop(key, None)

    def ask(self, interaction: discord.Interaction, prompt: str, *, ephemeral: bool = False) -> str:
        if self.llm is None or self.config is None:
            raise RuntimeError("LLM is not initialized")
        session, lock = self.get_session(interaction, ephemeral=ephemeral)
        return self.ask_session(session, prompt, lock)

    def ask_message(self, message: discord.Message) -> str:
        if self.llm is None or self.config is None:
            raise RuntimeError("LLM is not initialized")
        session, lock = self.get_message_session(message)
        return self.ask_session(session, message.content.strip(), lock)

    def record_auto_reply(
        self,
        message: discord.Message,
        response: str,
        reply_messages: list[discord.Message],
    ) -> None:
        if self.training_log is None or self.config is None:
            return
        self.training_log.record_turn(
            guild_id=message.guild.id if message.guild else None,
            channel_id=message.channel.id,
            user_id=message.author.id,
            user_message_id=message.id,
            assistant_message_ids=[reply.id for reply in reply_messages],
            user_text=message.content.strip(),
            assistant_text=response,
            model=self.config.model,
            num_ctx=self.config.num_ctx,
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

    def ask_session(self, session: ChatSession, prompt: str, lock: threading.Lock) -> str:
        if self.llm is None or self.config is None:
            raise RuntimeError("LLM is not initialized")
        with lock:
            session.add_user_message(prompt)
            try:
                response = self.llm.generate(
                    session.build_messages(),
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k,
                    repeat_penalty=self.config.repeat_penalty,
                    num_ctx=self.config.num_ctx,
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
        return (
            f"status: {health['status']}\n"
            f"ollama: {'ok' if llm_ok else 'ng'}\n"
            f"model: {self.config.model}\n"
            f"sessions: {session_count}\n"
            f"tts_loaded: {tts_loaded}\n"
            f"auto_reply_channels: {len(self.auto_reply_channel_ids)}\n"
            f"terminal_channels: {len(self.terminal_channel_ids)}\n"
            f"terminal_enabled: {self.allow_terminal}\n"
            f"terminal_tmux: {self.terminal.is_available() if self.terminal else False}\n"
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

    @bot.event
    async def on_ready() -> None:
        print(f"[Discord] logged in as {bot.user} ({bot.user.id if bot.user else 'unknown'})")
        print(f"[Discord] auto reply channels: {sorted(state.auto_reply_channel_ids)}")
        print(f"[Discord] terminal channels: {sorted(state.terminal_channel_ids)}")

    @bot.event
    async def on_message(message: discord.Message) -> None:
        if message.author.bot:
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

    @bot.tree.command(name="status", description="subpc_living の状態を確認します")
    async def status(interaction: discord.Interaction) -> None:
        if not await require_allowed(interaction, state):
            return
        await interaction.response.defer(thinking=True, ephemeral=True)
        text = await asyncio.to_thread(state.status_text)
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
