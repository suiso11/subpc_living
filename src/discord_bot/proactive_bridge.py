"""
ProactiveEngine を Discord bot に配線するブリッジ。

ProactiveEngine (src/persona/proactive.py) はバックグラウンドスレッドで
条件をチェックし、トリガー発火時に callback(trigger_type, message) を
「別スレッドから」呼ぶ。Discord の送信は bot のイベントループ上で行う必要が
あるため、asyncio.run_coroutine_threadsafe でループへブリッジする。

音声パイプライン (src/audio/pipeline.py) の _on_proactive_trigger 相当を
テキストチャンネル向けに実装したもの。
"""
from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Optional

from src.chat.emotion import parse_emotion_tag
from src.persona.conversation_loop import ConversationLoopStore
from src.persona.profile import UserProfile
from src.persona.proactive import ProactiveEngine

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# 定型メッセージをペルソナ口調に言い換えるための指示テンプレート
REWRITE_INSTRUCTION = (
    "次の通知文を、あなたのキャラクターの口調で1〜2文に言い換えてください。"
    "事実・数字は変えないでください: {message}"
)
CONVERSATION_REWRITE_INSTRUCTION = (
    "次の会話のきっかけを、あなたのキャラクターの口調で自然な一問に"
    "言い換えてください。質問の意図は変えず、前置きを長くしないでください。"
    "返答を義務のように迫る、無視を責める、不安を煽る表現は使わないでください: {message}"
)

CONVERSATION_CONTROL_HINT = (
    "\n-# 今は話さないときは「あとで」「今日は静かに」、"
    "停止は「自発会話を停止」と送ってください。"
)

_PAUSE_COMMANDS = {
    "自発会話を停止",
    "自発的な会話を停止",
    "会話のきっかけを停止",
}
_RESUME_COMMANDS = {
    "自発会話を再開",
    "自発的な会話を再開",
    "会話のきっかけを再開",
}
_LATER_COMMANDS = {"あとで", "またあとで", "今はいい", "今は大丈夫"}
_QUIET_TODAY_COMMANDS = {"今日は静かに", "今日は話しかけないで"}


def _parse_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_optional_id(value: Optional[str]) -> Optional[int]:
    if not value or not value.strip():
        return None
    try:
        return int(value.strip())
    except ValueError:
        print(f"[Discord] proactive channel ID を無視します: {value}")
        return None


def _parse_float(value: Optional[str], default: float) -> float:
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError:
        print(f"[Discord] proactive 数値設定を無視します: {value}")
        return default


def _parse_quiet_hours(
    value: Optional[str], default: tuple[int, int] = (1, 9)
) -> tuple[int, int]:
    if value is None or not value.strip():
        return default
    try:
        start_text, end_text = value.split("-", 1)
        start, end = int(start_text), int(end_text)
        if not (0 <= start <= 23 and 0 <= end <= 23):
            raise ValueError
        return start, end
    except ValueError:
        print(f"[Discord] proactive quiet hours を無視します: {value}")
        return default


@dataclass(frozen=True)
class ProactiveConfig:
    """プロアクティブ発話の環境変数設定。"""

    enabled: bool = False
    channel_id: Optional[int] = None
    llm_rewrite: bool = True
    check_interval: float = 60.0
    conversation_enabled: bool = False
    conversation_interval_hours: float = 6.0
    conversation_idle_minutes: float = 60.0
    conversation_reply_timeout_hours: float = 12.0
    conversation_daily_limit: int = 1
    conversation_max_backoff_hours: float = 72.0
    conversation_autoread: bool = False
    conversation_state_path: str = "data/proactive/conversation_state.json"
    quiet_hours: tuple[int, int] = (1, 9)

    @classmethod
    def from_env(cls, env: Optional[dict] = None) -> "ProactiveConfig":
        env = env if env is not None else os.environ
        return cls(
            enabled=_parse_bool(env.get("DISCORD_PROACTIVE_ENABLED"), default=False),
            channel_id=_parse_optional_id(env.get("DISCORD_PROACTIVE_CHANNEL_ID")),
            llm_rewrite=_parse_bool(env.get("DISCORD_PROACTIVE_LLM_REWRITE"), default=True),
            check_interval=_parse_float(env.get("DISCORD_PROACTIVE_CHECK_INTERVAL"), default=60.0),
            conversation_enabled=_parse_bool(
                env.get("DISCORD_PROACTIVE_CHAT_ENABLED"), default=False
            ),
            conversation_interval_hours=max(
                1 / 60,
                _parse_float(
                    env.get("DISCORD_PROACTIVE_CHAT_INTERVAL_HOURS"), default=6.0
                ),
            ),
            conversation_idle_minutes=max(
                0.0,
                _parse_float(
                    env.get("DISCORD_PROACTIVE_CHAT_IDLE_MINUTES"), default=60.0
                ),
            ),
            conversation_reply_timeout_hours=max(
                1 / 60,
                _parse_float(
                    env.get("DISCORD_PROACTIVE_CHAT_REPLY_TIMEOUT_HOURS"), default=12.0
                ),
            ),
            conversation_daily_limit=max(
                1,
                int(
                    _parse_float(
                        env.get("DISCORD_PROACTIVE_CHAT_DAILY_LIMIT"), default=1.0
                    )
                ),
            ),
            conversation_max_backoff_hours=max(
                1.0,
                _parse_float(
                    env.get("DISCORD_PROACTIVE_CHAT_MAX_BACKOFF_HOURS"),
                    default=72.0,
                ),
            ),
            conversation_autoread=_parse_bool(
                env.get("DISCORD_PROACTIVE_CHAT_AUTOREAD"), default=False
            ),
            conversation_state_path=(
                (env.get("DISCORD_PROACTIVE_CHAT_STATE_PATH") or "").strip()
                or "data/proactive/conversation_state.json"
            ),
            quiet_hours=_parse_quiet_hours(
                env.get("DISCORD_PROACTIVE_QUIET_HOURS"), default=(1, 9)
            ),
        )


def rewrite_message(
    llm: Any,
    *,
    system_prompt: str,
    message: str,
    instruction: str = REWRITE_INSTRUCTION,
    **gen_kwargs: Any,
) -> str:
    """定型通知文をペルソナ口調に言い換える (使い捨て呼び出し・履歴は汚さない)。

    LLM が無い / 失敗 / 空応答の場合は元メッセージをそのまま返す。
    """
    if llm is None:
        return message
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction.format(message=message)},
        ]
        result = llm.generate(messages, **gen_kwargs)
        text = (result or "").strip()
        return text or message
    except Exception:
        return message


class ProactiveBridge:
    """ProactiveEngine を Discord bot のテキストチャンネルへ配線する。"""

    def __init__(
        self,
        *,
        bot: Any,
        state: Any,
        config: ProactiveConfig,
        profile: UserProfile,
        monitor_context: Any = None,
        engine: Optional[ProactiveEngine] = None,
        conversation_store: Optional[ConversationLoopStore] = None,
        companion_getter: Optional[Callable[[], Any]] = None,
    ) -> None:
        self.bot = bot
        self.state = state
        self.config = config
        self.monitor_context = monitor_context
        state_path = Path(config.conversation_state_path)
        if not state_path.is_absolute():
            state_path = PROJECT_ROOT / state_path
        self.conversation_store = conversation_store or ConversationLoopStore(
            state_path,
            base_interval_sec=config.conversation_interval_hours * 3600,
            reply_timeout_sec=config.conversation_reply_timeout_hours * 3600,
            daily_limit=config.conversation_daily_limit,
            max_backoff_sec=config.conversation_max_backoff_hours * 3600,
        )
        self.engine = engine or ProactiveEngine(
            profile=profile,
            check_interval=config.check_interval,
            monitor_context=monitor_context,
            conversation_enabled=config.conversation_enabled,
            conversation_interval=config.conversation_interval_hours * 3600,
            conversation_idle=config.conversation_idle_minutes * 60,
            quiet_hours=config.quiet_hours,
            conversation_gate=self._can_start_conversation,
            companion_getter=companion_getter,
        )

    # --- ライフサイクル ---

    def start(self) -> None:
        self.engine.start(callback=self._on_trigger)

    def stop(self) -> None:
        try:
            self.engine.stop()
        except Exception:
            pass
        if self.monitor_context is not None:
            try:
                self.monitor_context.stop()
            except Exception:
                pass

    def notify_user_activity(self) -> None:
        self.engine.notify_user_activity()

    def _can_start_conversation(self) -> bool:
        channel_id = self.config.channel_id
        if channel_id is None:
            return False
        return self.conversation_store.can_prompt(channel_id)

    def handle_conversation_control(self, channel_id: int, user_text: str) -> str | None:
        """自発会話の後回し・今日は停止・無期限停止を自然文で受け付ける。

        短い一般発話の誤検出を避けるため、「あとで」系は保留中の
        自発質問があるときだけ制御文として扱う。
        """
        normalized = " ".join(user_text.strip().split())
        if normalized in _RESUME_COMMANDS:
            self.conversation_store.resume(channel_id)
            return "自発会話を再開しました。次のきっかけは通知上限と静穏時間に従います。"
        if normalized in _PAUSE_COMMANDS:
            self.conversation_store.pause(channel_id)
            return "自発会話を停止しました。「自発会話を再開」で戻せます。"
        if not self.conversation_store.has_pending(channel_id):
            return None
        now = time.time()
        if normalized in _LATER_COMMANDS:
            self.conversation_store.snooze(channel_id, until=now + 3 * 3600)
            return "了解です。少なくとも3時間は、会話のきっかけを送りません。"
        if normalized in _QUIET_TODAY_COMMANDS:
            local_now = datetime.fromtimestamp(now).astimezone()
            tomorrow = (local_now + timedelta(days=1)).replace(
                hour=self.config.quiet_hours[1] % 24,
                minute=0,
                second=0,
                microsecond=0,
            )
            self.conversation_store.snooze(channel_id, until=tomorrow.timestamp())
            return "了解です。今日はこのまま静かにしておきます。"
        return None

    def contextualize_reply(self, channel_id: int, user_text: str) -> tuple[str, bool]:
        """保留中の自発質問を1度だけ消費し、対話を1段深める指示を付ける。"""
        prompt = self.conversation_store.consume_reply(channel_id)
        if prompt is None:
            return user_text, False
        contextualized = (
            "これはアシスタントからの任意の会話のきっかけに対する返答です。\n"
            "まずユーザーの内容を受け止め、具体的に返答してください。"
            "ユーザーに話す意欲が見える場合だけ、関連する掘り下げ質問を1つまで添えてください。"
            "断り、話題変更、短く終えたい意図があれば質問せず終了し、"
            "センシティブな属性を推測しないでください。\n"
            f"アシスタントからの質問: {prompt}\n"
            f"ユーザーの返答: {user_text}"
        )
        return contextualized, True

    def conversation_status(self) -> dict[str, Any]:
        if self.config.channel_id is None:
            return {"enabled": False}
        return {
            "enabled": self.config.conversation_enabled,
            **self.conversation_store.status(self.config.channel_id),
            "state_error": self.conversation_store.last_error,
        }

    # --- コールバック (エンジンのスレッドから呼ばれる) ---

    def _on_trigger(self, trigger_type: str, message: str) -> None:
        """エンジンのバックグラウンドスレッドから呼ばれる。

        Discord 送信は bot のイベントループ上で行う必要があるため、
        run_coroutine_threadsafe でループへブリッジする。
        """
        loop = getattr(self.bot, "loop", None)
        if loop is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(self._deliver(trigger_type, message), loop)
        except Exception:
            pass

    async def _deliver(self, trigger_type: str, message: str) -> None:
        text = message
        if self.config.llm_rewrite:
            try:
                text = await asyncio.to_thread(self.rewrite, message, trigger_type)
            except Exception:
                text = message
        if not text:
            text = message
        # 言い換え結果に感情タグが混入しうるので防御的に除去する。
        emotion, text = parse_emotion_tag(text)
        if not text:
            text = message
        print(f"[Discord] proactive/{trigger_type}: {text}")
        await self._send(text, emotion=emotion, trigger_type=trigger_type)

    def rewrite(self, message: str, trigger_type: str = "") -> str:
        """state.llm を使って言い換え (同期・スレッドで実行)。失敗時は元文。"""
        profiles = getattr(self.state, "llm_profiles", None) or {}
        profile = profiles.get("default")
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
        return rewrite_message(
            getattr(self.state, "llm", None),
            system_prompt=system_prompt,
            message=message,
            instruction=(
                CONVERSATION_REWRITE_INSTRUCTION
                if trigger_type == "conversation_start"
                else REWRITE_INSTRUCTION
            ),
            **gen_kwargs,
        )

    async def _resolve_channel(self) -> Any:
        channel = self.bot.get_channel(self.config.channel_id)
        if channel is None:
            try:
                channel = await self.bot.fetch_channel(self.config.channel_id)
            except Exception:
                return None
        return channel

    async def _send(
        self,
        text: str,
        *,
        emotion: str | None = None,
        trigger_type: str = "",
    ) -> None:
        channel = await self._resolve_channel()
        if channel is None:
            print(f"[Discord] proactive channel が見つかりません: {self.config.channel_id}")
            return
        channel_id = getattr(channel, "id", self.config.channel_id)
        display_text = text
        if (
            trigger_type == "conversation_start"
            and isinstance(channel_id, int)
            and not self.conversation_store.controls_shown(channel_id)
        ):
            display_text += CONVERSATION_CONTROL_HINT
        try:
            await channel.send(display_text)
            if trigger_type == "conversation_start" and isinstance(channel_id, int):
                self.conversation_store.record_prompt(channel_id, text)
        except Exception as e:
            print(f"[Discord] proactive send failed: {e}")
            return
        # ボイス接続中で autoread 有効なら音声にも流す (on_message と同じパターン)
        voice_tts = getattr(self.state, "voice_tts", None)
        should_autoread = (
            trigger_type != "conversation_start" or self.config.conversation_autoread
        )
        if voice_tts is not None and should_autoread:
            try:
                await voice_tts.autoread(text, emotion=emotion)
            except Exception:
                pass


def _build_monitor_context() -> Any:
    """MonitorContext を pipeline.py と同様に構築 (失敗時 None)。"""
    try:
        from src.monitor.context import MonitorContext

        ctx = MonitorContext(
            db_path=str(PROJECT_ROOT / "data" / "metrics" / "system_metrics.db"),
            collect_interval=30.0,
        )
        if not ctx.start():
            return None
        return ctx
    except Exception as e:
        print(f"[Discord] proactive monitor 初期化スキップ: {e}")
        return None


def create_proactive_bridge(
    state: Any, bot: Any, companion_getter: Optional[Callable[[], Any]] = None
) -> Optional[ProactiveBridge]:
    """環境変数から ProactiveBridge を構築する。無効なら None。

    クラッシュ禁止: 構築に失敗しても None を返して bot を継続させる。
    """
    config = ProactiveConfig.from_env()
    if not config.enabled:
        return None
    if config.channel_id is None:
        auto_reply_channels = sorted(getattr(state, "auto_reply_channel_ids", set()))
        if not auto_reply_channels:
            print(
                "[Discord] proactive 有効だが配送先が未設定のため無効化 "
                "(DISCORD_PROACTIVE_CHANNEL_ID または AUTO_REPLY_CHANNEL_IDS)"
            )
            return None
        config = replace(config, channel_id=auto_reply_channels[0])
    try:
        profile = UserProfile(
            profile_path=str(PROJECT_ROOT / "data" / "profile" / "user_profile.json"),
        )
        profile.load()
    except Exception as e:
        print(f"[Discord] proactive profile 初期化失敗のため無効化: {e}")
        return None

    monitor_context = _build_monitor_context()
    return ProactiveBridge(
        bot=bot,
        state=state,
        config=config,
        profile=profile,
        monitor_context=monitor_context,
        companion_getter=companion_getter,
    )
