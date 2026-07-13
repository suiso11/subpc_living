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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from src.chat.emotion import parse_emotion_tag
from src.persona.profile import UserProfile
from src.persona.proactive import ProactiveEngine

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# 定型メッセージをペルソナ口調に言い換えるための指示テンプレート
REWRITE_INSTRUCTION = (
    "次の通知文を、あなたのキャラクターの口調で1〜2文に言い換えてください。"
    "事実・数字は変えないでください: {message}"
)


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
        print(f"[Discord] proactive check interval を無視します: {value}")
        return default


@dataclass(frozen=True)
class ProactiveConfig:
    """プロアクティブ発話の環境変数設定。"""

    enabled: bool = False
    channel_id: Optional[int] = None
    llm_rewrite: bool = True
    check_interval: float = 60.0

    @classmethod
    def from_env(cls, env: Optional[dict] = None) -> "ProactiveConfig":
        env = env if env is not None else os.environ
        return cls(
            enabled=_parse_bool(env.get("DISCORD_PROACTIVE_ENABLED"), default=False),
            channel_id=_parse_optional_id(env.get("DISCORD_PROACTIVE_CHANNEL_ID")),
            llm_rewrite=_parse_bool(env.get("DISCORD_PROACTIVE_LLM_REWRITE"), default=True),
            check_interval=_parse_float(env.get("DISCORD_PROACTIVE_CHECK_INTERVAL"), default=60.0),
        )


def rewrite_message(llm: Any, *, system_prompt: str, message: str, **gen_kwargs: Any) -> str:
    """定型通知文をペルソナ口調に言い換える (使い捨て呼び出し・履歴は汚さない)。

    LLM が無い / 失敗 / 空応答の場合は元メッセージをそのまま返す。
    """
    if llm is None:
        return message
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": REWRITE_INSTRUCTION.format(message=message)},
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
    ) -> None:
        self.bot = bot
        self.state = state
        self.config = config
        self.monitor_context = monitor_context
        self.engine = engine or ProactiveEngine(
            profile=profile,
            check_interval=config.check_interval,
            monitor_context=monitor_context,
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
                text = await asyncio.to_thread(self.rewrite, message)
            except Exception:
                text = message
        if not text:
            text = message
        # 言い換え結果に感情タグが混入しうるので防御的に除去する。
        emotion, text = parse_emotion_tag(text)
        if not text:
            text = message
        print(f"[Discord] proactive/{trigger_type}: {text}")
        await self._send(text, emotion=emotion)

    def rewrite(self, message: str) -> str:
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

    async def _send(self, text: str, *, emotion: str | None = None) -> None:
        channel = await self._resolve_channel()
        if channel is None:
            print(f"[Discord] proactive channel が見つかりません: {self.config.channel_id}")
            return
        try:
            await channel.send(text)
        except Exception as e:
            print(f"[Discord] proactive send failed: {e}")
            return
        # ボイス接続中で autoread 有効なら音声にも流す (on_message と同じパターン)
        voice_tts = getattr(self.state, "voice_tts", None)
        if voice_tts is not None:
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


def create_proactive_bridge(state: Any, bot: Any) -> Optional[ProactiveBridge]:
    """環境変数から ProactiveBridge を構築する。無効なら None。

    クラッシュ禁止: 構築に失敗しても None を返して bot を継続させる。
    """
    config = ProactiveConfig.from_env()
    if not config.enabled:
        return None
    if config.channel_id is None:
        print("[Discord] proactive 有効だが DISCORD_PROACTIVE_CHANNEL_ID 未設定のため無効化")
        return None
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
    )
