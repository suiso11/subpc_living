"""
Web UIサーバー
スマホ・PC からLAN経由でアクセス可能なチャットインターフェース
FastAPI + WebSocket によるストリーミング対話
"""
import sys
import os
import json
import asyncio
import base64
import hashlib
import queue
import socket
import secrets
import subprocess
import threading
import time
from collections import deque
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional
from zoneinfo import ZoneInfo

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, Response, FileResponse

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.assistant.contracts import AssistantRequest
from src.assistant.factory import build_local_service
from src.assistant.nodes import build_node_service, NodeInventory, ProviderSpec
from src.assistant.service import AssistantService
from src.assistant.stream_queue import stream_to_queue
from src.chat.session import ChatSession
from src.chat.config import ChatConfig, validate_local_provider_kind
from src.chat.emotion import EmotionTagStreamFilter, emotion_to_sbv2_style
from src.chat.web_search import WebSearchContext, create_web_search_context
from src.audio.tts_factory import backend_name, create_tts_backend
from src.audio.stt import WhisperSTT
from src.memory.vectorstore import VectorStore
from src.memory.rag import RAGRetriever
from src.vision.context import VisionContext
from src.screen.context import ScreenContext
from src.screen import create_screen_context
from src.perception import (
    ActivityRuntime,
    SensorPolicy,
    create_activity_source,
    create_activity_runtime_from_env,
    companion_state_payload,
    resolve_sensor_policy,
)
from src.perception.bootstrap import sensor_error_code, sensor_error_code_from_name
from src.monitor.context import MonitorContext
from src.persona.profile import UserProfile
from src.persona.summarizer import ConversationSummarizer
from src.persona.preloader import SessionPreloader
from src.service.healthcheck import HealthChecker
from src.service.idle import IdleManager, create_idle_manager
from src.discord_bot.task_ui import parse_due, parse_snooze, split_quick_input
from src.tasks.store import TaskStore
from src.tasks.formatting import format_short_due
from src.tasks.chat_editor import TaskChatEditor
from src.tasks import extractor as task_extractor
from src.growth.tracker import GrowthTracker
from src.llm.provider import LLMProvider
from src.llm.registry import ProviderRegistry
from src.service.log_setup import setup_logging, DEFAULT_LOG_DIR
from src.chat import history_admin

logger = setup_logging("subpc-web")


# --- グローバル状態 ---
config: ChatConfig = None
llm: LLMProvider = None
assistant_service: AssistantService = None
provider_registry: ProviderRegistry = None
tts = None
stt: WhisperSTT = None
rag: RAGRetriever = None
vision: VisionContext = None
screen: Optional[ScreenContext] = None
monitor: MonitorContext = None
# 共有 SensorPolicy (P0-3)。lifespan で一度だけ解決し、env 名・値・token は
# 公開しない。各センサーの構築・start はこの policy の boolean でのみ gate する。
sensor_policy: Optional[SensorPolicy] = None
profile: UserProfile = None
summarizer: ConversationSummarizer = None
preloader: SessionPreloader = None
web_search: Optional[WebSearchContext] = None
sessions: dict[str, ChatSession] = {}
idle_manager: Optional[IdleManager] = None
activity_runtime: Optional[ActivityRuntime] = None
task_store: Optional[TaskStore] = None
task_chat_editor = TaskChatEditor()
growth_tracker: Optional[GrowthTracker] = None
tasks_timezone: str = "Asia/Tokyo"
task_calendar_sync = None  # TaskCalendarSync | None (Webで作ったタスクをカレンダーへ push)
calendar_client = None  # GoogleCalendarMCPClient | None (イベント CRUD 用)
tasks_calendar_id: str = "primary"
# 選択中ローカルProviderの追跡 (P0-2.4)。ChatConfig の resolved id /
# NodeInventory の default_provider_id を起動時に保持し、status / health は
# ハードコードされた "ollama" ではなくこの id 経由で判定する。
primary_provider_id: Optional[str] = None
primary_provider_kind: str = "ollama"
# 選択中ローカルProviderの実 base URL。lifespan で解決し、外部へ出力・ログしない。
# None のときは未設定 (reachability = "unconfigured") としてプローブしない。
primary_provider_base_url: Optional[str] = None
# 選択中ローカルProviderの API キー環境変数名のみ。キー値は保持しない。
# lifespan で ChatConfig / NodeInventory default spec から解決し、各 health/status
# プローブ時に os.environ から実行時解決する。env 名自体も外部へ出力・ログしない。
primary_provider_api_key_env: Optional[str] = None
UPCOMING_PATH = PROJECT_ROOT / "data" / "calendar" / "upcoming.json"

# 低温度 Web 抽出の資源上限。本流の num_ctx/num_predict とは独立に小さく抑える。
_EXTRACTION_NUM_CTX = 2048
_EXTRACTION_NUM_PREDICT = 256
_candidate_offer_tasks: set[asyncio.Task] = set()


def get_local_ip() -> str:
    """LAN内のIPアドレスを取得"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def _inventory_provider_spec(
    inventory, provider_id: str
) -> Optional[ProviderSpec]:
    """NodeInventory から provider_id に対応する ProviderSpec を返す。

    見つからない場合は ``None``。
    """
    for spec in inventory.providers():
        if spec.provider_id == provider_id:
            return spec
    return None


def _inventory_provider_kind(inventory, provider_id: str) -> str:
    """NodeInventory の provider_id に対応する ``provider_kind`` を返す。

    見つからない場合は従来既定の ``"ollama"`` を返す (安全側)。
    """
    spec = _inventory_provider_spec(inventory, provider_id)
    return spec.provider_kind if spec is not None else "ollama"


def _selected_provider_entry() -> tuple[Optional[str], Optional[LLMProvider]]:
    """選択中ローカルProviderの (provider_id, provider) を解決する。

    追跡中の ``primary_provider_id`` を優先し、未設定の単一Provider構成では
    後方互換のため ``"ollama"``、最後に登録済みの最初のProviderへフォールバック
    する。registry が無ければ ``(None, None)`` を返す。
    """
    if provider_registry is None:
        return None, None
    if primary_provider_id is not None and primary_provider_id in provider_registry:
        entry = provider_registry.get(primary_provider_id)
        return entry.provider_id, entry.provider
    if "ollama" in provider_registry:
        entry = provider_registry.get("ollama")
        return entry.provider_id, entry.provider
    for entry in provider_registry.entries():
        return entry.provider_id, entry.provider
    return None, None


def _reachability_status(check: Optional[dict]) -> str:
    """Providerヘルスチェック辞書から到達可能性を正規化する。

    ``ok`` / ``error`` / ``unknown`` はそのまま返し、チェックが無い・
    ``skip`` など到達判定できない場合は ``"unconfigured"`` を返す。
    """
    if check is None:
        return "unconfigured"
    status = check.get("status")
    if status in ("ok", "error", "unknown"):
        return status
    return "unconfigured"


def _provider_reachability_from_checks(result: dict) -> str:
    """``check_all`` の結果から選択中Providerの到達可能性を抽出する。

    選択backend (``primary_provider_kind``) に応じて ``checks["ollama"]`` か
    ``checks["local_provider"]`` を使う。未設定 (base_url 未解決) のときは
    ネットワークプローブせず ``"unconfigured"`` を返す。
    """
    if not primary_provider_kind or not primary_provider_base_url:
        return "unconfigured"
    check_key = "ollama" if primary_provider_kind == "ollama" else "local_provider"
    return _reachability_status(result["checks"].get(check_key))


def _default_provider_api_key_env(config, spec=None) -> Optional[str]:
    """NodeInventory の default spec / ChatConfig から API キー env 名を選ぶ。

    spec (inventory の default spec) を優先し、無ければ ``config.local_api_key_env``
    を使う。env 名のみを返し、キー値は解決しない。空なら ``None``。
    """
    if spec is not None:
        env_name = getattr(spec, "api_key_env", "") or ""
    else:
        env_name = getattr(config, "local_api_key_env", "") or ""
    return env_name.strip() or None


def _resolve_primary_provider_api_key() -> Optional[str]:
    """選択中ローカルProviderの API キーを実行時に解決する。

    追跡中の env 名のみを参照し、キー値はグローバル・cache・status に保持しない。
    Ollama / env 名未指定 / 環境変数が未設定・空のときは ``None`` (keyless)。
    """
    if primary_provider_kind != "openai_compatible":
        return None
    env_name = primary_provider_api_key_env
    if not env_name:
        return None
    return os.environ.get(env_name) or None


def _log_llm_startup_status(llm, *, provider_id, provider_kind, model) -> None:
    """起動時のLLM availabilityを真実に沿ってログする。

    openai_compatible の ``is_available()`` は lifecycle-only (closed でない限り
    True) で到達性を検証しない。そのため ``"LLM OK"`` とは断言せず、設定済みで
    あることと、到達性は ``/api/health`` と generation で確認する旨だけをログ
    する。Ollama は probe-based ``is_available()`` の従来動作を維持する。
    """
    if llm is None:
        logger.warning("Providerが1つも登録されていません。")
        return
    if provider_kind == "openai_compatible":
        if not llm.is_available():
            logger.warning(
                "LLM (provider: %s) が利用可能な状態にありません。", provider_id
            )
            return
        logger.info(
            "LLM configured (provider: %s, model: %s); reachability checked by health/generation",
            provider_id,
            model,
        )
        return
    if not llm.is_available():
        logger.warning(
            "LLM (provider: %s) に接続できません。チャット機能は使用不可です。",
            provider_id,
        )
        return
    logger.info("✅ LLM OK (provider: %s, model: %s)", provider_id, model)


@lru_cache(maxsize=1)
def get_secure_web_url() -> str:
    """マイクAPIを使えるHTTPS公開URLを返す。

    明示設定を優先し、未設定ならTailscale Serveの現在設定から検出する。
    """
    configured = os.environ.get("WEB_PUBLIC_HTTPS_URL", "").strip().rstrip("/")
    if configured.startswith("https://"):
        return configured
    try:
        result = subprocess.run(
            ["tailscale", "serve", "status", "--json"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if result.returncode != 0:
            return ""
        payload = json.loads(result.stdout or "{}")
        web = payload.get("Web") if isinstance(payload, dict) else None
        if not isinstance(web, dict):
            return ""
        for authority, route in web.items():
            handlers = route.get("Handlers") if isinstance(route, dict) else None
            root = handlers.get("/") if isinstance(handlers, dict) else None
            proxy = str(root.get("Proxy") or "") if isinstance(root, dict) else ""
            if proxy.endswith(":8000"):
                host = str(authority).removesuffix(":443")
                return f"https://{host}"
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return ""
    return ""


def _sensor_error_response(message: str, exc: Exception, status_code: int = 500) -> JSONResponse:
    """センサー関連エラーの固定レスポンス。

    例外の本文・パス・URL・token・デバイス名・画像/テキスト内容は露出せず、
    allowlist の固定コード (timeout / invalid_input / unavailable / internal_error)
    だけを ``error_type`` に載せる。例外クラス名そのものは外部 JSON へ載せない
    (HTTP 意味は維持)。
    """
    return JSONResponse(
        {"error": message, "error_type": sensor_error_code(exc)},
        status_code=status_code,
    )


def _safe_get_status(ctx) -> Optional[dict]:
    """センサー status 取得の安全ラッパー。

    例外時は型名のみログし None を返す。例外本文・内部状態は外部へ露出しない。
    """
    try:
        return ctx.get_status()
    except Exception as e:
        logger.warning("sensor status unavailable: %s", type(e).__name__)
        return None


# センサー status の allowlist。bool / タイムスタンプ / ソース種別のみを許容し、
# カメラデバイスID・VLM モデル名・描写テキスト・生の context 文字列・パス・
# last_error など、未認証 Web API から露出させない派生/生情報を除外する。
VISION_STATUS_ALLOWLIST = (
    "running",
    "paused",
    "stop_pending",
    "thread_alive",
    "user_present",
    "emotion_detection",
)

SCREEN_STATUS_ALLOWLIST = (
    "running",
    "paused",
    "stop_pending",
    "thread_alive",
    "captured_at",
    "age_seconds",
    "source",
)

# センサー status の allowlist (Monitor)。bool / タイムスタンプのみを許容し、
# メトリクス集計値 (CPU/メモリ/GPU/ディスク)・プロセス数・レコード数・設定値
# (collect_interval)・DB パス・エラー・本文テキストなど、未認証 Web API から
# 露出させない派生/生情報を除外する。source は常に固定 "monitor"。
MONITOR_STATUS_ALLOWLIST = (
    "running",
    "last_collected",
)


def _sensor_disabled_response() -> JSONResponse:
    """廃止した生/デバッグセンサー出力の固定レスポンス。

    404 (未公開) としてデータを一切返さない。403 は「認証/許可」を暗示するため
    同意・認証と混同させる 403 は使わず、単に非公開であることを返す。
    ボディは固定文言のみで、センサー内容は含めない。
    """
    return JSONResponse(
        {"error": "deprecated", "detail": "このエンドポイントは廃止されました。"},
        status_code=404,
    )


def _microphone_enabled() -> bool:
    """共有 SensorPolicy.microphone が明示 true のときだけマイク入力を許可する。

    未解決 (None)・false・不正値は False (fail closed)。env 名・値は直接読まず、
    lifespan で resolve 済みの frozen な policy の boolean のみを使う。
    """
    return sensor_policy is not None and sensor_policy.microphone


def _microphone_denied_response() -> JSONResponse:
    """マイク入力が許可されていないときの固定レスポンス (permission-denied)。

    センサー内容・policy・env 情報は一切含めず、固定文言のみを返す (403)。
    """
    return JSONResponse(
        {"error": "forbidden", "detail": "マイク入力は許可されていません。"},
        status_code=403,
    )


def _stt_usable() -> bool:
    """STT が利用可能か (engine ロード済み かつ マイク policy 有効)。

    マイク入力は engine の有無と policy の両方が必要で、どちらか不足なら False
    (fail closed)。env 名・値は露出しない。
    """
    return stt is not None and stt.is_loaded() and _microphone_enabled()


def _filter_sensor_status(status: Optional[dict], allowlist: tuple[str, ...]) -> dict:
    """センサー status を allowlist のキーのみに絞る。

    未公開のキーが status に存在しても、外部へは一切載せない (fail closed)。
    """
    if not status:
        return {}
    return {key: status[key] for key in allowlist if key in status}


def _minimized_status(ctx, allowlist: tuple[str, ...]) -> Optional[dict]:
    """status 取得を安全ラップし、allowlist のみに絞って返す。

    例外時は None。生例外・内部状態は外部へ露出しない。
    """
    status = _safe_get_status(ctx)
    if not status:
        return None
    return _filter_sensor_status(status, allowlist)


def _minimized_monitor_status(ctx) -> Optional[dict]:
    """Monitor status を allowlist のみに絞り、source を固定 "monitor" で返す。

    例外時は None。生例外・内部状態は外部へ露出しない。
    """
    status = _minimized_status(ctx, MONITOR_STATUS_ALLOWLIST)
    if status is None:
        return None
    status["source"] = "monitor"
    return status


def _start_companion_activity_runtime() -> Optional[ActivityRuntime]:
    """オプトインの活動収集ランタイムを起動する。

    env 読取と ActivityRuntime 生成は src.perception.bootstrap の
    create_activity_runtime_from_env へ委譲する。数値設定不正や起動失敗は
    companion 機能だけを無効化し、Web 起動は続行する。
    """
    global activity_runtime
    activity_runtime = create_activity_runtime_from_env(os.environ, logger=logger)
    return activity_runtime


def _init_vision_from_policy(policy: Optional[SensorPolicy]) -> Optional[VisionContext]:
    """policy.camera が有効のときだけ VisionContext を構築・start して返す。

    無効時は構築も start もせず None (fail closed)。カメラが開けなくても例外は
    伝播せず None にして Web 起動を続行する。
    """
    if policy is None or not policy.camera:
        logger.info("Vision disabled (set SENSOR_CAMERA_ENABLED=true)")
        return None
    logger.info("[5/7] Vision init...")
    try:
        emotion_model = str(PROJECT_ROOT / "models" / "vision" / "emotion-ferplus-8.onnx")
        vctx = VisionContext(
            camera_id=0,
            analysis_interval=2.0,
            emotion_model_path=emotion_model,
        )
        try:
            started = vctx.start()
        except Exception as e:
            logger.warning("Vision start failed (continue): %s", type(e).__name__)
            started = False
        if started:
            time.sleep(1.0)
            status = vctx.get_status()
            emotion_str = "enabled" if status["emotion_detection"] else "face-only"
            logger.info("Vision OK (camera, emotion: %s)", emotion_str)
            return vctx
        # start が false / 例外で失敗した場合のみ、破棄前に best-effort で stop する。
        try:
            vctx.stop()
        except Exception as e:
            logger.warning("Vision cleanup failed: %s", type(e).__name__)
        logger.warning("Vision: camera open failed (continue without Vision)")
        return None
    except Exception as e:
        logger.warning("Vision init failed (continue): %s", type(e).__name__)
        return None


def _init_screen_from_policy(
    policy: Optional[SensorPolicy],
    *,
    config,
    primary_provider_kind: str,
) -> Optional[ScreenContext]:
    """policy.screen_capture が有効のときだけ ScreenContext を構築・start して返す。

    従来の legacy env (WEB_SCREEN_CONTEXT_ENABLED) はここでは読まず、共有
    SensorPolicy 解決器が既に screen_capture へ反映済みの boolean だけを使う。
    ScreenDescriber は Ollama /api/chat 前提のため Ollama backend 時のみ作成する。
    """
    if policy is None or not policy.screen_capture:
        logger.info("Screen disabled (set SENSOR_SCREEN_CAPTURE_ENABLED=true)")
        return None
    logger.info("[+] Screen init...")
    if primary_provider_kind != "ollama":
        logger.warning("Screen: only available with Ollama backend; skipping")
        return None
    try:
        sctx = create_screen_context(
            analysis_interval=90.0,
            base_url=config.ollama_base_url,
            model=config.model,
        )
        try:
            started = sctx.start()
        except Exception as e:
            logger.warning("Screen start failed (continue): %s", type(e).__name__)
            started = False
        if started:
            status = sctx.get_status()
            mode = status.get("mode", "local")
            detail = (
                f"VLM: {status['model']}, interval: {status['analysis_interval']:.0f}s"
                if mode == "local"
                else "remote: reading data/screen/latest.json"
            )
            logger.info("Screen OK (%s)", detail)
            return sctx
        # start が false / 例外で失敗した場合のみ、破棄前に best-effort で stop する。
        try:
            sctx.stop()
        except Exception as e:
            logger.warning("Screen cleanup failed: %s", type(e).__name__)
        logger.warning("Screen: capture failed (DISPLAY unset? continue without Screen)")
        return None
    except Exception as e:
        logger.warning("Screen init failed (continue): %s", type(e).__name__)
        return None


def _init_monitor_from_policy(policy: Optional[SensorPolicy]) -> Optional[MonitorContext]:
    """policy.monitor が有効のときだけ MonitorContext を構築・start して返す。

    既定はオフ。無効時は構築も start もせず None。
    """
    if policy is None or not policy.monitor:
        logger.info("Monitor disabled (set SENSOR_MONITOR_ENABLED=true)")
        return None
    logger.info("[6/7] Monitor init...")
    try:
        mctx = MonitorContext(
            db_path=str(PROJECT_ROOT / "data" / "metrics" / "system_metrics.db"),
            collect_interval=30.0,
        )
        try:
            started = mctx.start()
        except Exception as e:
            logger.warning("Monitor start failed (continue): %s", type(e).__name__)
            started = False
        if started:
            logger.info("Monitor OK (metrics collection started)")
            return mctx
        # start が false / 例外で失敗した場合のみ、破棄前に best-effort で stop する。
        try:
            mctx.stop()
        except Exception as e:
            logger.warning("Monitor cleanup failed: %s", type(e).__name__)
        logger.warning("Monitor start failed (continue without Monitor)")
        return None
    except Exception as e:
        logger.warning("Monitor init failed (continue): %s", type(e).__name__)
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """サーバー起動/終了時の処理"""
    global config, llm, assistant_service, provider_registry, tts, stt, rag, vision, screen, monitor, profile, summarizer, preloader, web_search, idle_manager, task_store, growth_tracker, tasks_timezone, task_calendar_sync, calendar_client, tasks_calendar_id, activity_runtime, primary_provider_id, primary_provider_kind, primary_provider_base_url, primary_provider_api_key_env, sensor_policy

    logger.info("Web UI サーバー起動中...")

    # GPU構成表示 (Phase 10)
    try:
        from src.service.gpu_config import get_device_config
        gpu_cfg = get_device_config()
        if gpu_cfg.gpus:
            logger.info("GPU構成: %s", gpu_cfg.profile)
            for g in gpu_cfg.gpus:
                logger.info("  GPU%s: %s (%sGB)", g.index, g.name, g.vram_gb)
            if gpu_cfg.profile == "dual_gpu":
                logger.info("  LLM → GPU%s / 推論 → GPU%s", gpu_cfg.llm_gpu_index, gpu_cfg.inference_gpu_index)
    except Exception:
        pass

    # 設定ロード
    config_path = PROJECT_ROOT / "config" / "chat_config.json"
    config = ChatConfig.load(config_path)
    web_search = create_web_search_context(config)

    # 共有 SensorPolicy を一度だけ解決する (P0-3)。以降のセンサー gate は
    # この frozen な policy の boolean のみを使い、env を直接読まない。
    sensor_policy = resolve_sensor_policy()
    # 生JPEGは保持しない運用のため、レガシー latest.jpg があれば best-effort で消す。
    _remove_legacy_latest_jpg()
    tasks_timezone = os.environ.get("DIARY_TIMEZONE", "Asia/Tokyo").strip() or "Asia/Tokyo"
    try:
        growth_tracker = GrowthTracker(
            PROJECT_ROOT / "data" / "growth" / "growth.db",
            timezone_name=tasks_timezone,
        )
        logger.info("✅ Growth tracker OK")
    except Exception as e:
        logger.warning("Growth tracker 初期化失敗 (計測なしで続行): %s", e)
        growth_tracker = None

    try:
        task_store = TaskStore(
            db_path=str(PROJECT_ROOT / "data" / "tasks" / "tasks.db"),
            timezone_name=tasks_timezone,
        ).initialize()
        logger.info("✅ Tasks OK (Webタスク管理有効)")
    except Exception as e:
        logger.warning("Tasks 初期化失敗 (タスク管理なしで続行): %s", e)
        task_store = None

    # Google Calendar 連携 (イベント CRUD + Webで作ったタスクの push 同期)
    # 同期ワーカーは Discord 側にもあるが、on_change はプロセス内でしか発火しない
    # ため、Web で作成・更新したタスクをカレンダーへ反映するには Web 側にも
    # push ワーカーが必要 (pull は Discord 側のみが行う)。
    calendar_sync_enabled = os.environ.get("TASKS_CALENDAR_SYNC_ENABLED", "").strip().lower() == "true"
    tasks_calendar_id = (
        os.environ.get("TASKS_CALENDAR_ID", "").strip()
        or os.environ.get("DIARY_CALENDAR_ID", "").strip()
        or "primary"
    )
    if calendar_sync_enabled:
        try:
            from src.integrations.google_calendar import GoogleCalendarMCPClient
            from src.tasks.calendar_sync import TaskCalendarSync

            calendar_client = GoogleCalendarMCPClient.from_env()
            if task_store is not None:
                sync = TaskCalendarSync(
                    store=task_store,
                    calendar_client=calendar_client,
                    calendar_id=tasks_calendar_id,
                    enabled=True,
                    timezone=tasks_timezone,
                )
                sync.start()
                task_store.on_change = sync.enqueue
                task_calendar_sync = sync
            logger.info("✅ Calendar OK (calendar=%s, task push同期有効)", tasks_calendar_id)
        except Exception as e:
            logger.warning("Calendar 初期化失敗 (カレンダー連携なしで続行): %s", e)
            calendar_client = None
            task_calendar_sync = None

    # LLM 初期化
    logger.info("[1/6] LLM 接続確認...")
    # 選択backend の実 base URL を毎回の起動で安全に再解決する (未設定へ戻す)。
    primary_provider_base_url = None
    # API キーは env 名のみを追跡し、キー値は保持しない。env 名も起動時に再解決する。
    primary_provider_api_key_env = None
    # Opt-in multi-node inventory wiring
    _subpc_inventory_path = os.environ.get("SUBPC_NODE_INVENTORY", "").strip()
    if _subpc_inventory_path:
        try:
            _inventory = NodeInventory.load(_subpc_inventory_path)
            assistant_service, provider_registry = build_node_service(_inventory)
            primary_provider_id = _inventory.default_provider_id
            _default_spec = _inventory_provider_spec(
                _inventory, primary_provider_id
            )
            if _default_spec is not None:
                primary_provider_kind = _default_spec.provider_kind
                primary_provider_base_url = _default_spec.base_url
                primary_provider_api_key_env = _default_provider_api_key_env(
                    config, _default_spec
                )
            else:
                primary_provider_kind = "ollama"
                primary_provider_base_url = None
                primary_provider_api_key_env = _default_provider_api_key_env(
                    config, None
                )
            logger.info("✅ Multi-node inventory loaded from %s", _subpc_inventory_path)
        except Exception as e:
            logger.warning(
                "SUBPC_NODE_INVENTORY load failed, falling back to single-provider: %s", e
            )
            assistant_service, provider_registry = build_local_service(config)
            primary_provider_id = config.resolved_local_provider_id()
            primary_provider_kind = validate_local_provider_kind(config)
            primary_provider_base_url = config.resolved_local_base_url()
            primary_provider_api_key_env = _default_provider_api_key_env(config, None)
    else:
        assistant_service, provider_registry = build_local_service(config)
        primary_provider_id = config.resolved_local_provider_id()
        primary_provider_kind = validate_local_provider_kind(config)
        primary_provider_base_url = config.resolved_local_base_url()
        primary_provider_api_key_env = _default_provider_api_key_env(config, None)
    def _primary_llm(reg, primary_provider_id=None):
        """既定のLLMプロバイダを安全に解決する。

        追跡中の primary_provider_id を優先し、'ollama' が無いInventory構成
        (local-strong等) では登録済みの最初のローカルProviderへフォールバックする。
        """
        if primary_provider_id and primary_provider_id in reg:
            return reg.get(primary_provider_id).provider
        if "ollama" in reg:
            return reg.get("ollama").provider
        for entry in reg.entries():
            return entry.provider
        return None

    llm = _primary_llm(provider_registry, primary_provider_id)
    _log_llm_startup_status(
        llm,
        provider_id=primary_provider_id,
        provider_kind=primary_provider_kind,
        model=getattr(llm, "model", config.model),
    )
    if web_search is not None:
        logger.info("✅ Web検索 ON (auto=%s, max_results=%s)", config.web_search_auto, config.web_search_max_results)

    # STT 初期化
    logger.info("[2/7] STT init...")
    try:
        stt = WhisperSTT(model_size="auto", language="ja", device="auto")
        stt.load()
        logger.info("STT OK (model: %s, device: %s)", stt.model_size, stt.device)
    except Exception as e:
        logger.warning("STT load failed: %s", type(e).__name__)
        stt = None

    # TTS 初期化
    logger.info("[3/7] TTS 初期化...")
    tts = create_tts_backend(models_dir=PROJECT_ROOT / "models" / "tts" / "kokoro")
    try:
        tts.load()
        logger.info("✅ TTS OK (%s)", backend_name(tts))
    except Exception as e:
        logger.warning("TTS ロード失敗: %s", e)
        tts = None

    # RAG 初期化 (Phase 4)。ネイティブ依存の起動障害時にもWeb UIだけは復旧できるよう、
    # 明示的なfalseで安全にバイパスできる運用スイッチを持つ。
    rag_enabled = os.environ.get("WEB_RAG_ENABLED", "true").strip().lower() not in {
        "0", "false", "no", "off",
    }
    if rag_enabled:
        logger.info("[4/7] RAG (長期記憶) 初期化...")
        try:
            vector_store = VectorStore(
                persist_dir=str(PROJECT_ROOT / "data" / "vectordb"),
            )
            vector_store.initialize()
            rag = RAGRetriever(vector_store=vector_store)
            stats = rag.get_stats()
            logger.info("✅ RAG OK (会話: %s件, 知識: %s件)", stats['conversations'], stats['knowledge'])
        except Exception as e:
            logger.warning("RAG 初期化失敗 (RAGなしで続行): %s", e)
            rag = None
    else:
        rag = None
        logger.warning("RAG 無効 (WEB_RAG_ENABLED=false)")

    # Vision 初期化 (Phase 5)。policy.camera が有効のときだけ構築・startする。
    vision = _init_vision_from_policy(sensor_policy)

    # Screen 初期化 (画面認識: スクリーンショット → VLM描写)。
    # 既定無効。共有 SensorPolicy.screen_capture でのみ有効化 (legacy の
    # WEB_SCREEN_CONTEXT_ENABLED は解決器経由でのみ効く)。
    screen = _init_screen_from_policy(
        sensor_policy,
        config=config,
        primary_provider_kind=primary_provider_kind,
    )

    # Monitor 初期化 (Phase 6)。既定オフ。policy.monitor が有効のときだけ
    # 構築・startする。
    monitor = _init_monitor_from_policy(sensor_policy)

    # Persona 初期化 (Phase 7)
    logger.info("[7/7] Persona (パーソナライズ) 初期化...")
    try:
        profile = UserProfile(
            profile_path=str(PROJECT_ROOT / "data" / "profile" / "user_profile.json"),
        )
        profile.load()
        summarizer = ConversationSummarizer(
            summaries_dir=str(PROJECT_ROOT / "data" / "profile" / "summaries"),
        )
        preloader = SessionPreloader(
            profile=profile,
            summarizer=summarizer,
        )
        profile_name = profile.name or "(未設定)"
        facts_count = len(profile.extracted_facts)
        today_count = len(profile.get_today_schedule())
        logger.info("✅ Persona OK (名前: %s, 事実: %s件, 今日の予定: %s件)", profile_name, facts_count, today_count)
    except Exception as e:
        logger.warning("Persona 初期化失敗 (Personaなしで続行): %s", e)
        profile = None
        summarizer = None
        preloader = None

    # IdleManager は明示的に opt-in した場合のみ起動する。
    # Web/Voice の複数プロセスが同じ GPU 制限を競合更新するため、既定は無効。
    idle_manager = create_idle_manager()
    if idle_manager is not None:
        idle_manager.start(monitor_context=monitor, vision_context=vision)
        if idle_manager.gpu_power_control_enabled:
            logger.info("✅ IdleManager OK (GPU電力の動的切替有効)")
        else:
            logger.info("✅ IdleManager OK (GPU電力制御は無効: %s)", idle_manager.gpu_power_control_reason)
    else:
        idle_manager = None
        logger.info("IdleManager 無効 (IDLE_MANAGER_ENABLED=true で明示的に有効化)")

    # Companion 活動収集 (オプトイン)。false または設定不正・起動失敗なら
    # companion 機能だけ無効化して Web 起動は続行する。
    activity_runtime = _start_companion_activity_runtime()
    if activity_runtime is not None:
        logger.info("Companion activity OK (companion state API enabled)")
    else:
        logger.info("Companion activity disabled (set COMPANION_ACTIVITY_ENABLED=true)")

    # Screen ingest の受付世代を有効化する。シャットダウン側で revoke される。
    _start_ingest_generation()

    local_ip = get_local_ip()
    logger.info("✅ サーバー起動完了! PC: http://localhost:8000 / スマホ: http://%s:8000", local_ip)

    # systemd sd_notify: READY=1 (Type=notify 用)
    _sd_notify("READY=1")

    # Watchdog 定期通知タスク
    watchdog_task = asyncio.create_task(_watchdog_loop())

    yield

    # Watchdog タスク停止
    watchdog_task.cancel()
    try:
        await watchdog_task
    except asyncio.CancelledError:
        pass

    # Screen ingest: 受付 revoke (cancel より先) → 保持 Future の cancel + bounded await →
    # safe なときだけ ownership 解除。実行中 worker はシャットダウン後 latest.json を書けない。
    try:
        await _stop_ingest_describe()
    except Exception as e:
        logger.warning("screen ingest stop failed: %s", type(e).__name__)

    # 応答後の候補抽出タスクを停止してから共有クライアント/DBを閉じる。
    pending_candidates = list(_candidate_offer_tasks)
    if pending_candidates:
        await asyncio.gather(*pending_candidates, return_exceptions=True)

    # Companion 活動ランタイムを共有リソースを閉じる前に停止する。
    if activity_runtime is not None:
        try:
            activity_runtime.stop()
        except Exception as e:
            logger.warning("activity runtime stop failed: %s", type(e).__name__)

    # 終了処理
    # IdleManager 停止
    if idle_manager is not None:
        idle_manager.stop()
    # セッション要約 (Phase 7)
    if summarizer is not None and llm is not None:
        for sid, sess in sessions.items():
            if sess.turn_count >= 2:
                try:
                    summarizer.process_session_end(
                        llm=llm,
                        messages=sess.messages,
                        session_id=sess.session_id,
                        profile=profile,
                    )
                except Exception:
                    pass
    if monitor is not None:
        try:
            monitor.stop()
        except Exception as e:
            logger.warning("monitor stop failed: %s", type(e).__name__)
    if vision is not None:
        try:
            vision.stop()
        except Exception as e:
            logger.warning("vision stop failed: %s", type(e).__name__)
    if screen is not None:
        try:
            screen.stop()
        except Exception as e:
            logger.warning("screen stop failed: %s", type(e).__name__)
    if task_calendar_sync is not None:
        task_calendar_sync.stop()
    if task_store is not None:
        task_store.close()
    if provider_registry is not None:
        provider_registry.close()
    # レガシー latest.jpg は保持しない運用のため、終了時に best-effort で削除する。
    _remove_legacy_latest_jpg()
    # グローバル policy を安全に戻す (次回起動時に再解決する)。
    sensor_policy = None
    logger.info("サーバーを終了しました。")


app = FastAPI(title="subpc_living Web UI", lifespan=lifespan)

# 静的ファイル
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# --- ページ ---

@app.get("/favicon.ico")
async def favicon():
    svg_path = STATIC_DIR / "favicon.svg"
    return FileResponse(svg_path, media_type="image/svg+xml")


@app.get("/service-worker.js")
async def service_worker():
    """PWA workerをルートscopeで配信する。"""
    worker_path = STATIC_DIR / "service-worker.js"
    return FileResponse(
        worker_path,
        media_type="application/javascript",
        headers={
            "Service-Worker-Allowed": "/",
            "Cache-Control": "no-cache",
        },
    )


@app.get("/", response_class=HTMLResponse)
async def index():
    """メインページ"""
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/tasks", response_class=HTMLResponse)
async def tasks_page():
    """タスク管理ページ"""
    html_path = STATIC_DIR / "tasks.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/logs", response_class=HTMLResponse)
async def logs_page():
    """ログ管理ページ"""
    html_path = STATIC_DIR / "logs.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/achievements", response_class=HTMLResponse)
async def achievements_page():
    """相棒との実績ページ"""
    html_path = STATIC_DIR / "achievements.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


# --- sd_notify ヘルパー ---

def _sd_notify(state: str) -> None:
    """systemd sd_notify プロトコルで状態を通知する"""
    addr = os.environ.get("NOTIFY_SOCKET")
    if not addr:
        return
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        if addr[0] == "@":
            addr = "\0" + addr[1:]
        sock.sendto(state.encode(), addr)
        sock.close()
    except Exception:
        pass


async def _watchdog_loop() -> None:
    """WatchdogSec に合わせて定期的に WATCHDOG=1 を送信する"""
    usec = os.environ.get("WATCHDOG_USEC")
    if not usec:
        return
    interval = int(usec) / 1_000_000 / 2  # 半分の間隔で通知
    if interval < 1:
        interval = 1
    while True:
        await asyncio.sleep(interval)
        _sd_notify("WATCHDOG=1")


# --- REST API ---

@app.get("/api/health")
async def health():
    """ヘルスチェック (systemd watchdog / 外部監視用)"""
    checker = HealthChecker(
        ollama_url=(
            primary_provider_base_url
            or (config.ollama_base_url if config else "http://localhost:11434")
        ),
    )
    if primary_provider_kind and primary_provider_base_url:
        # 選択中ローカルProviderの実 backend kind/base を selected-provider モードで
        # プローブする。Ollama は checks["ollama"]、openai_compatible は
        # checks["local_provider"] に出力される。URL・キーは出力しない。
        result = checker.check_all(
            provider_kind=primary_provider_kind,
            provider_url=primary_provider_base_url,
            provider_api_key=_resolve_primary_provider_api_key(),
            include_web=False,
        )
        reachability = _provider_reachability_from_checks(result)
    else:
        # 未設定 (base_url 未解決) はプローブせず unconfigured とする。
        result = {"status": "unconfigured", "checks": {}}
        reachability = "unconfigured"

    _pid, _selected = _selected_provider_entry()

    # モジュール稼働状況を追加
    result["modules"] = {
        # 後方互換: 選択中ローカルProviderの到達性を表すエイリアス。
        # reachability が ok のときだけ True にする (is_available は使わない)。
        "ollama": reachability == "ok",
        "local_provider": reachability == "ok",
        "provider_id": _pid,
        "provider_kind": primary_provider_kind,
        "provider_reachability": reachability,
        "tts": tts is not None and tts.is_loaded(),
        "stt": stt is not None and stt.is_loaded(),
        "rag": rag is not None,
        "vision": vision is not None and vision.is_running,
        "monitor": monitor is not None and monitor.is_running,
        "persona": profile is not None,
        "growth": growth_tracker is not None,
        "idle_manager": idle_manager is not None and idle_manager.is_running,
        "companion": activity_runtime is not None and activity_runtime.is_running,
    }

    # プロバイダレベルの最小情報 (URL・キー・モデル・内部統計は含まない)
    try:
        providers_list = []
        check_key = "ollama" if primary_provider_kind == "ollama" else "local_provider"
        selected_check = result.get("checks", {}).get(check_key)
        if provider_registry is not None:
            for entry in provider_registry.entries():
                pid = entry.provider_id
                try:
                    lifecycle_available = bool(entry.provider.is_available())
                except Exception:
                    lifecycle_available = False
                entry_payload = {
                    "provider_id": pid,
                    "provider_kind": primary_provider_kind,
                    "local": entry.local,
                    "available": lifecycle_available,
                    "error": None,
                }
                if pid == _pid:
                    # 選択中Providerの available は lifecycle (is_available) ではなく
                    # 到達性 (reachability) で決める。ok 以外は False。
                    entry_payload["available"] = reachability == "ok"
                    if isinstance(selected_check, dict) and selected_check.get("status") == "error":
                        entry_payload["error"] = selected_check.get("message")
                providers_list.append(entry_payload)
        result["providers"] = providers_list
    except Exception:
        result["providers"] = []

    status_code = 200 if result["status"] == "ok" else 503
    return JSONResponse(content=result, status_code=status_code)


def _count_nonempty_lines(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except OSError:
        return 0


def _growth_asset_counts() -> dict[str, int]:
    try:
        rag_stats = rag.get_stats() if rag is not None else {}
    except Exception:
        rag_stats = {}
    training_dir = Path(os.environ.get("DISCORD_TRAINING_DIR", "data/discord_training"))
    if not training_dir.is_absolute():
        training_dir = PROJECT_ROOT / training_dir
    summaries_dir = PROJECT_ROOT / "data" / "profile" / "summaries"
    return {
        "retrievable_memories": int(rag_stats.get("conversations", 0)),
        "knowledge_items": int(rag_stats.get("knowledge", 0)),
        "training_turns": _count_nonempty_lines(training_dir / "conversations.jsonl"),
        "feedback_items": _count_nonempty_lines(training_dir / "feedback.jsonl"),
        "correction_candidates": _count_nonempty_lines(
            training_dir / "training_candidates.jsonl"
        ),
        "profile_facts": len(profile.extracted_facts) if profile is not None else 0,
        "conversation_summaries": len(list(summaries_dir.glob("summary_*.json"))),
    }


GAME_MISSIONS = (
    {
        "id": "first_turn",
        "name": "まずはひとこと",
        "detail": "今日1往復する",
        "metric": "today_turns",
        "target": 1,
        "reward": 10,
    },
    {
        "id": "three_turns",
        "name": "話をつなげる",
        "detail": "今日3往復する",
        "metric": "today_turns",
        "target": 3,
        "reward": 20,
    },
    {
        "id": "deep_talk",
        "name": "じっくり話す",
        "detail": "今日の会話を600文字まで育てる",
        "metric": "today_chars",
        "target": 600,
        "reward": 25,
    },
)

GAME_RANKS = (
    (1, "はじめまして"),
    (3, "話し相手"),
    (5, "相棒"),
    (8, "名コンビ"),
    (12, "長年の相棒"),
)

GAME_STARTERS = (
    {"id": "continue", "label": "続きを話す", "prompt": "前の話の続きをしよう"},
    {"id": "reflect", "label": "今日を整理", "prompt": "今日あったことを一緒に整理したい"},
    {"id": "decide", "label": "次を決める", "prompt": "いま一番優先すべきことを一緒に決めて"},
    {"id": "tasks", "label": "タスクを編集", "prompt": "タスクを見せて"},
    {"id": "refresh", "label": "気分転換", "prompt": "気分転換になる話題をひとつ出して"},
)


def _local_game_now(now: datetime | None = None) -> datetime:
    tz = ZoneInfo(tasks_timezone or "Asia/Tokyo")
    if now is None:
        return datetime.now(tz)
    if now.tzinfo is None:
        return now.replace(tzinfo=tz)
    return now.astimezone(tz)


def _game_rank(level: int) -> dict:
    current_name = GAME_RANKS[0][1]
    next_rank = None
    for minimum, name in GAME_RANKS:
        if level >= minimum:
            current_name = name
        elif next_rank is None:
            next_rank = {"level": minimum, "name": name}
    return {"name": current_name, "level": level, "next": next_rank}


def _game_state(
    *,
    now: datetime | None = None,
    asset_counts: dict[str, int] | None = None,
) -> dict:
    """成長台帳から、決定論的で課金圧のないゲーム状態を組み立てる。"""
    if growth_tracker is None:
        return {"enabled": False}
    local_now = _local_game_now(now)
    counts = _growth_asset_counts() if asset_counts is None else asset_counts
    summary = growth_tracker.summary(now=local_now, asset_counts=counts)
    local_date = local_now.date().isoformat()
    event_keys = [f"quest:{local_date}:{mission['id']}" for mission in GAME_MISSIONS]
    claimed_keys = growth_tracker.existing_event_keys(event_keys)

    missions = []
    for mission in GAME_MISSIONS:
        current = int(summary.get(mission["metric"], 0))
        event_key = f"quest:{local_date}:{mission['id']}"
        missions.append({
            "id": mission["id"],
            "name": mission["name"],
            "detail": mission["detail"],
            "reward": mission["reward"],
            "current": min(current, mission["target"]),
            "target": mission["target"],
            "complete": current >= mission["target"],
            "claimed": event_key in claimed_keys,
        })

    asset = summary["asset_counts"]
    badge_specs = (
        ("first", "●", "はじめの一歩", "最初の会話を記録", summary["total_turns"], 1, "往復"),
        ("talk10", "◆", "話し好き", "10往復を記録", summary["total_turns"], 10, "往復"),
        ("memory10", "■", "思い出係", "記憶を10回保存", summary["memory_turns"], 10, "回"),
        ("streak3", "▲", "三日仲間", "3日続けて話す", summary["streak_days"], 3, "日"),
        ("level10", "✦", "ベテランコンビ", "レベル10に到達", summary["level"], 10, "Lv"),
        (
            "correction",
            "＋",
            "学び上手",
            "学び直しを1件保存",
            int(asset.get("correction_candidates", 0)),
            1,
            "件",
        ),
    )
    badges = [
        {
            "id": bid,
            "mark": mark,
            "name": name,
            "detail": detail,
            "current": int(current),
            "target": int(target),
            "unit": unit,
            "unlocked": int(current) >= int(target),
        }
        for bid, mark, name, detail, current, target, unit in badge_specs
    ]
    return {
        "enabled": True,
        "date": local_date,
        "rank": _game_rank(int(summary["level"])),
        "points": int(summary["growth_points"]),
        "missions": missions,
        "completed_missions": sum(1 for mission in missions if mission["complete"]),
        "claimed_missions": sum(1 for mission in missions if mission["claimed"]),
        "claimable_missions": sum(
            1 for mission in missions if mission["complete"] and not mission["claimed"]
        ),
        "badges": badges,
        "unlocked_badges": sum(1 for badge in badges if badge["unlocked"]),
        "starters": list(GAME_STARTERS),
    }


def _claim_game_mission(
    mission_id: str,
    *,
    now: datetime | None = None,
    asset_counts: dict[str, int] | None = None,
) -> dict:
    """達成済みクエストの固定報酬を、一意キーで一度だけ受け取る。"""
    state = _game_state(now=now, asset_counts=asset_counts)
    if not state.get("enabled"):
        return {"ok": False, "status": 503, "error": "game is unavailable"}
    mission = next((m for m in state["missions"] if m["id"] == mission_id), None)
    if mission is None:
        return {"ok": False, "status": 404, "error": "unknown mission"}
    if not mission["complete"]:
        return {"ok": False, "status": 409, "error": "mission is not complete"}
    local_now = _local_game_now(now)
    event_key = f"quest:{state['date']}:{mission_id}"
    claimed_now = growth_tracker.record_signal(
        kind="quest_reward",
        source="web_game",
        event_key=event_key,
        points=int(mission["reward"]),
        metadata={"mission_id": mission_id, "date": state["date"]},
        now=local_now,
    )
    return {
        "ok": True,
        "status": 200,
        "claimed_now": claimed_now,
        "reward": int(mission["reward"]) if claimed_now else 0,
        "state": _game_state(now=local_now, asset_counts=asset_counts),
    }


@app.get("/api/growth")
async def growth_summary(days: int = 14):
    """実際に増えた適応資産と、観測開始後の会話成長を返す。"""
    if growth_tracker is None:
        return JSONResponse({"enabled": False}, status_code=503)
    summary = growth_tracker.summary(days=days, asset_counts=_growth_asset_counts())
    return {
        "enabled": True,
        "metric_note": (
            "GPは会話例・検索可能記憶・評価・修正候補・個人化事実の蓄積量です。"
            "モデル重みや知能指数ではありません。"
        ),
        **summary,
    }


@app.get("/api/game")
async def game_state():
    if growth_tracker is None:
        return JSONResponse({"enabled": False}, status_code=503)
    return _game_state()


@app.post("/api/game/claim")
async def game_claim(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    result = _claim_game_mission(str(body.get("mission_id") or ""))
    status_code = int(result.get("status", 200))
    if status_code != 200:
        return JSONResponse(result, status_code=status_code)
    return result


@app.get("/api/status")
async def status():
    """システム状態"""
    _pid, _selected = _selected_provider_entry()
    if primary_provider_kind and primary_provider_base_url:
        checker = HealthChecker(ollama_url=primary_provider_base_url)
        result = checker.check_all(
            provider_kind=primary_provider_kind,
            provider_url=primary_provider_base_url,
            provider_api_key=_resolve_primary_provider_api_key(),
            include_web=False,
        )
        reachability = _provider_reachability_from_checks(result)
    else:
        reachability = "unconfigured"
    return {
        # 後方互換: 選択中ローカルProviderの到達性を表すエイリアス。
        # reachability が ok のときだけ True (is_available は使わない)。
        "ollama": reachability == "ok",
        "local_provider": reachability == "ok",
        "provider_id": _pid,
        "provider_kind": primary_provider_kind,
        "provider_reachability": reachability,
        "tts": tts is not None and tts.is_loaded(),
        "tts_backend": backend_name(tts),
        "tts_voice": tts.voice if tts else None,
        "tts_voices": tts.list_ja_voices() if tts else {},
        "stt": _stt_usable(),
        "secure_web_url": get_secure_web_url(),
        "rag": rag is not None,
        "rag_stats": rag.get_stats() if rag else None,
        "vision": vision is not None and vision.is_running,
        "vision_status": _minimized_status(vision, VISION_STATUS_ALLOWLIST),
        "monitor": monitor is not None and monitor.is_running,
        "monitor_status": _minimized_monitor_status(monitor) if monitor else None,
        "persona": profile is not None,
        "persona_status": preloader.get_status() if preloader else None,
        "growth": growth_tracker is not None,
        "idle_manager": idle_manager.get_status() if idle_manager else None,
        "companion": activity_runtime is not None,
        "tasks": task_store is not None,
        "tasks_timezone": tasks_timezone,
    }


def _tasks_tz() -> ZoneInfo:
    return ZoneInfo(tasks_timezone)


def _task_to_json(task: dict) -> dict:
    due_at = task.get("due_at")
    created_at = task.get("created_at")
    completed_at = task.get("completed_at")
    return {
        "id": task["id"],
        "title": task.get("title") or "",
        "note": task.get("note") or "",
        "action_hint": task.get("action_hint") or "",
        "steps": task.get("steps") or [],
        "step_done": task.get("step_done") or [],
        "due_at": due_at.isoformat() if due_at else None,
        "due_granularity": task.get("due_granularity"),
        "priority": task.get("priority") or "normal",
        "status": task.get("status") or "open",
        "source": task.get("source") or "",
        "created_at": created_at.isoformat() if created_at else None,
        "completed_at": completed_at.isoformat() if completed_at else None,
        "calendar_event_id": task.get("calendar_event_id"),
    }


def _require_task_store() -> Optional[JSONResponse]:
    if task_store is None:
        return JSONResponse({"error": "Task store not available"}, status_code=503)
    return None


@app.get("/api/tasks")
async def tasks_list(status: str = "open", limit: int = 100):
    """タスク一覧を返す。既定は未完了のみ。"""
    unavailable = _require_task_store()
    if unavailable is not None:
        return unavailable
    safe_status = status if status in ("open", "done", "dropped") else "open"
    safe_limit = max(1, min(int(limit), 200))
    rows = await asyncio.to_thread(task_store.list, safe_status, safe_limit)
    return {
        "tasks": [_task_to_json(t) for t in rows],
        "status": safe_status,
        "timezone": tasks_timezone,
    }


@app.post("/api/tasks/preview")
async def tasks_preview(request: Request):
    """1行テキストをパースして preview を返す。
    
    body: {"text": "..."}
    response: {"title": str, "due_at": iso|null, "due_granularity": str|null, "due_display": "M/D HH:MM"|"M/D"|null, "priority": str}
    """
    unavailable = _require_task_store()
    if unavailable is not None:
        return unavailable
    body = await request.json()
    text = str(body.get("text") or "").strip()
    if not text:
        return JSONResponse({"error": "text is required"}, status_code=400)
    
    now = datetime.now(timezone.utc)
    tz = _tasks_tz()
    result = split_quick_input(text, now, tz)
    
    # due_display の計算
    due_display = None
    if result["due_at"] is not None:
        local_due = result["due_at"].astimezone(tz)
        due_display = format_short_due(local_due, with_time=result["due_granularity"] != "date")
    
    return JSONResponse({
        "title": result["title"],
        "due_at": result["due_at"].isoformat() if result["due_at"] else None,
        "due_granularity": result["due_granularity"],
        "due_display": due_display,
        "priority": result["priority"],
    }, status_code=200)


@app.post("/api/tasks")
async def tasks_add(request: Request):
    """タスクを追加する。
    
    body: title, due, priority, note (従来形式) または text (クイック入力)
    text キーがあれば split_quick_input で分解し、明示的な priority/note があればそちらを優先。
    """
    unavailable = _require_task_store()
    if unavailable is not None:
        return unavailable
    body = await request.json()
    
    now = datetime.now(timezone.utc)
    tz = _tasks_tz()
    
    # --- Quick input mode (text キーがある) ---
    if "text" in body:
        text = str(body.get("text") or "").strip()
        if not text:
            return JSONResponse({"error": "text is required"}, status_code=400)
        
        result = split_quick_input(text, now, tz)
        title = result["title"]
        if not title:
            return JSONResponse({"error": "title is required"}, status_code=400)
        
        due_at = result["due_at"]
        granularity = result["due_granularity"]
        priority = result["priority"]
        note = None
        
        # 明示的な priority/note があればそちらを優先
        if "priority" in body:
            p = str(body.get("priority") or "normal").strip().lower()
            if p in ("high", "normal", "low"):
                priority = p
        if "note" in body:
            n = str(body.get("note") or "").strip()
            if n:
                note = n
    
    # --- Traditional mode (title + due/priority/note) ---
    else:
        title = str(body.get("title") or "").strip()
        if not title:
            return JSONResponse({"error": "title is required"}, status_code=400)
        
        due_at = None
        granularity = None
        due = str(body.get("due") or "").strip()
        if due:
            due_at, granularity = parse_due(due, now, tz)
            if due_at is None:
                return JSONResponse(
                    {"error": "due could not be parsed", "hint": "例: 明日 18時 / 金曜 / 来週水曜 / 7/10 15:00"},
                    status_code=400,
                )
        priority = str(body.get("priority") or "normal").strip().lower()
        if priority not in ("high", "normal", "low"):
            priority = "normal"
        note = str(body.get("note") or "").strip() or None
    
    task_id = await asyncio.to_thread(
        task_store.add,
        title[:200],
        note=note,
        due_at=due_at,
        due_granularity=granularity,
        priority=priority,
        source="web",
    )
    task = await asyncio.to_thread(task_store.get, task_id)
    return JSONResponse({"task": _task_to_json(task)}, status_code=201)


@app.patch("/api/tasks/{task_id}")
async def tasks_update(task_id: int, request: Request):
    """未完了タスクを更新する。空の due は期限削除。"""
    unavailable = _require_task_store()
    if unavailable is not None:
        return unavailable
    body = await request.json()
    kwargs: dict = {}

    if "title" in body:
        title = str(body.get("title") or "").strip()
        if not title:
            return JSONResponse({"error": "title is required"}, status_code=400)
        kwargs["title"] = title[:200]
    if "note" in body:
        kwargs["note"] = str(body.get("note") or "").strip()
    if "action_hint" in body:
        kwargs["action_hint"] = str(body.get("action_hint") or "").strip()
    if "priority" in body:
        priority = str(body.get("priority") or "normal").strip().lower()
        if priority not in ("high", "normal", "low"):
            return JSONResponse({"error": "priority must be high, normal, or low"}, status_code=400)
        kwargs["priority"] = priority
    if "due" in body:
        due = str(body.get("due") or "").strip()
        if not due:
            kwargs["clear_due"] = True
        else:
            due_at, granularity = parse_due(due, datetime.now(timezone.utc), _tasks_tz())
            if due_at is None:
                return JSONResponse(
                    {"error": "due could not be parsed", "hint": "例: 明日 18時 / 金曜 / 来週水曜 / 7/10 15:00"},
                    status_code=400,
                )
            kwargs["due_at"] = due_at
            kwargs["due_granularity"] = granularity

    if not kwargs:
        return JSONResponse({"error": "no fields to update"}, status_code=400)
    ok = await asyncio.to_thread(task_store.update, task_id, **kwargs)
    if not ok:
        return JSONResponse({"error": "task not found or not open"}, status_code=404)
    task = await asyncio.to_thread(task_store.get, task_id)
    return {"task": _task_to_json(task)}


@app.post("/api/tasks/{task_id}/done")
async def tasks_done(task_id: int):
    unavailable = _require_task_store()
    if unavailable is not None:
        return unavailable
    ok = await asyncio.to_thread(task_store.done, task_id)
    if not ok:
        return JSONResponse({"error": "task not found or not open"}, status_code=404)
    return {"ok": True}


@app.post("/api/tasks/{task_id}/steps/{step_index}")
async def tasks_step_done(task_id: int, step_index: int, request: Request):
    """分割された手順のチェック状態を更新する。body: {"done": bool}"""
    unavailable = _require_task_store()
    if unavailable is not None:
        return unavailable
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "JSON body is required"}, status_code=400)
    if not isinstance(body.get("done"), bool):
        return JSONResponse({"error": "done must be boolean"}, status_code=400)
    ok = await asyncio.to_thread(
        task_store.set_step_done,
        task_id,
        step_index,
        body["done"],
    )
    if not ok:
        return JSONResponse(
            {"error": "task or step not found, or task is not open"},
            status_code=404,
        )
    task = await asyncio.to_thread(task_store.get, task_id)
    return {"task": _task_to_json(task)}


@app.post("/api/tasks/{task_id}/breakdown")
async def tasks_breakdown(task_id: int):
    """最初の一歩と小さな手順を、現在のタイトル・メモから作り直す。"""
    unavailable = _require_task_store()
    if unavailable is not None:
        return unavailable
    ok = await asyncio.to_thread(task_store.regenerate_breakdown, task_id)
    if not ok:
        return JSONResponse({"error": "task not found or not open"}, status_code=404)
    task = await asyncio.to_thread(task_store.get, task_id)
    return {"task": _task_to_json(task)}


@app.post("/api/tasks/{task_id}/drop")
async def tasks_drop(task_id: int):
    unavailable = _require_task_store()
    if unavailable is not None:
        return unavailable
    ok = await asyncio.to_thread(task_store.drop, task_id)
    if not ok:
        return JSONResponse({"error": "task not found or not open"}, status_code=404)
    return {"ok": True}


@app.post("/api/tasks/{task_id}/snooze")
async def tasks_snooze(task_id: int, request: Request):
    unavailable = _require_task_store()
    if unavailable is not None:
        return unavailable
    body = await request.json()
    when = str(body.get("when") or "").strip()
    until = parse_snooze(when, datetime.now(timezone.utc), _tasks_tz())
    if until is None:
        return JSONResponse(
            {"error": "snooze could not be parsed", "hint": "例: 30m / 2h / 明日"},
            status_code=400,
        )
    ok = await asyncio.to_thread(task_store.snooze, task_id, until)
    if not ok:
        return JSONResponse({"error": "task not found or not open"}, status_code=404)
    return {"ok": True, "until": until.isoformat()}


# --- 会話由来タスク候補 Inbox API ---

@app.get("/api/tasks/candidates")
async def task_candidates_list(status: str = "pending", limit: int = 100):
    unavailable = _require_task_store()
    if unavailable is not None:
        return unavailable
    safe_status = status if status in ("pending", "accepted", "dismissed") else "pending"
    safe_limit = max(1, min(int(limit), 200))
    rows = await asyncio.to_thread(task_store.list_candidates, safe_status, safe_limit)
    return {"candidates": [_candidate_to_json(row) for row in rows], "status": safe_status}


@app.post("/api/tasks/candidates/{candidate_id}/accept")
async def task_candidate_accept(candidate_id: int):
    unavailable = _require_task_store()
    if unavailable is not None:
        return unavailable
    candidate = await asyncio.to_thread(task_store.get_candidate, candidate_id)
    if candidate is None:
        return JSONResponse({"error": "candidate not found"}, status_code=404)
    try:
        task_id, created = await asyncio.to_thread(task_store.accept_candidate, candidate_id)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=409)
    if task_id is None:
        return JSONResponse({"error": "candidate has no task"}, status_code=409)
    task = await asyncio.to_thread(task_store.get, task_id)
    resolved = await asyncio.to_thread(task_store.get_candidate, candidate_id)
    return {
        "ok": True,
        "created": created,
        "candidate": _candidate_to_json(resolved),
        "task": _task_to_json(task),
    }


@app.post("/api/tasks/candidates/{candidate_id}/dismiss")
async def task_candidate_dismiss(candidate_id: int):
    unavailable = _require_task_store()
    if unavailable is not None:
        return unavailable
    candidate = await asyncio.to_thread(task_store.get_candidate, candidate_id)
    if candidate is None:
        return JSONResponse({"error": "candidate not found"}, status_code=404)
    try:
        changed = await asyncio.to_thread(task_store.dismiss_candidate, candidate_id)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=409)
    resolved = await asyncio.to_thread(task_store.get_candidate, candidate_id)
    return {"ok": True, "changed": changed, "candidate": _candidate_to_json(resolved)}


# --- Google Calendar イベント API ---
# 読み取りは upcoming.json (Discord 側 pull ワーカーが定期更新するキャッシュ) から。
# 書き込みは MCP で Google に直接反映し、成功したらキャッシュも楽観更新する
# (次回 pull で正となる値に洗い替えされる)。


def _read_upcoming_payload() -> dict:
    try:
        with open(UPCOMING_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _patch_upcoming_cache(mutate) -> None:
    """upcoming.json を読み、mutate(events_list) で変更してアトミックに書き戻す。

    キャッシュ更新は best-effort。失敗しても API 自体は成功扱い
    (Google には反映済みで、次回 pull で追いつくため)。
    """
    try:
        payload = _read_upcoming_payload()
        events = payload.get("events")
        if not isinstance(events, list):
            events = []
        payload["events"] = events
        mutate(events)
        events.sort(key=lambda e: str(e.get("start") or ""))
        UPCOMING_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = UPCOMING_PATH.with_suffix(UPCOMING_PATH.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, UPCOMING_PATH)
    except Exception as e:
        logger.warning("upcoming.json 楽観更新失敗 (pull で追いつくため無視): %s", e)


def _require_calendar_client() -> Optional[JSONResponse]:
    if calendar_client is None:
        return JSONResponse(
            {"error": "Calendar not available", "hint": "TASKS_CALENDAR_SYNC_ENABLED=true と OAuth 設定が必要です"},
            status_code=503,
        )
    return None


def _parse_event_window(body: dict, base: Optional[dict] = None) -> tuple[str, str, Optional[str]]:
    """body の date/time/duration_min から (start, end, error) を作る。

    date: "YYYY-MM-DD" 必須 (PATCH 時は base の start から補完可)。
    time: "HH:MM" (空なら終日イベント)。duration_min: 時刻付きの長さ (既定60分)。
    """
    date_str = str(body.get("date") or "").strip()
    time_str = str(body.get("time") or "").strip()
    if base is not None:
        base_start = str(base.get("start") or "")
        if not date_str:
            date_str = base_start[:10]
        # time キー自体が無いときは元の時刻を維持する (date だけの変更で
        # 時刻付き予定が終日に化けないように)。time="" は明示的な終日化。
        if "time" not in body and "T" in base_start:
            time_str = base_start[11:16]
    try:
        d = date.fromisoformat(date_str)
    except ValueError:
        return "", "", "date は YYYY-MM-DD 形式で指定してください"
    if not time_str:
        return d.isoformat(), (d + timedelta(days=1)).isoformat(), None
    try:
        hh, mm = time_str.split(":", 1)
        start_dt = datetime(d.year, d.month, d.day, int(hh), int(mm))
    except (ValueError, TypeError):
        return "", "", "time は HH:MM 形式で指定してください"
    try:
        duration = int(body.get("duration_min") or 60)
    except (ValueError, TypeError):
        duration = 60
    duration = max(5, min(duration, 24 * 60))
    end_dt = start_dt + timedelta(minutes=duration)
    return (
        start_dt.isoformat(timespec="seconds"),
        end_dt.isoformat(timespec="seconds"),
        None,
    )


@app.get("/api/calendar/events")
async def calendar_events_list(start: str = "", end: str = ""):
    """upcoming.json のイベントを日付範囲 (start/end, YYYY-MM-DD, 両端含む) で返す。"""
    payload = _read_upcoming_payload()
    events = [e for e in payload.get("events", []) if isinstance(e, dict)]
    if start:
        events = [e for e in events if str(e.get("start") or "")[:10] >= start]
    if end:
        events = [e for e in events if str(e.get("start") or "")[:10] <= end]
    return {
        "events": events,
        "generated_at": payload.get("generated_at", ""),
        "range": payload.get("range", {}),
        "timezone": payload.get("timezone", tasks_timezone),
        "writable": calendar_client is not None,
    }


@app.post("/api/calendar/events")
async def calendar_event_create(request: Request):
    """Google Calendar にイベントを作成する。

    body: {"title": str, "date": "YYYY-MM-DD", "time": "HH:MM"?, "duration_min": int?,
           "location": str?, "description": str?}
    """
    unavailable = _require_calendar_client()
    if unavailable is not None:
        return unavailable
    body = await request.json()
    title = str(body.get("title") or "").strip()
    if not title:
        return JSONResponse({"error": "title is required"}, status_code=400)
    start_str, end_str, err = _parse_event_window(body)
    if err:
        return JSONResponse({"error": err}, status_code=400)
    location = str(body.get("location") or "").strip()
    description = str(body.get("description") or "").strip()

    res = await asyncio.to_thread(
        lambda: calendar_client.create_event(
            calendar_id=tasks_calendar_id,
            summary=title,
            start=start_str,
            end=end_str,
            description=description,
            location=location,
            timezone=tasks_timezone,
        )
    )
    if not res.ok:
        return JSONResponse({"error": f"作成に失敗しました: {res.error}"}, status_code=502)

    event = {
        "title": title,
        "start": start_str,
        "end": end_str,
        "location": location,
        "description": description,
        "event_id": res.event_id,
    }
    _patch_upcoming_cache(lambda events: events.append(dict(event)))
    return JSONResponse({"event": event}, status_code=201)


@app.patch("/api/calendar/events/{event_id}")
async def calendar_event_update(event_id: str, request: Request):
    """既存イベントを更新する。渡したフィールドのみ変更。"""
    unavailable = _require_calendar_client()
    if unavailable is not None:
        return unavailable
    body = await request.json()

    cached = next(
        (
            e
            for e in _read_upcoming_payload().get("events", [])
            if isinstance(e, dict) and e.get("event_id") == event_id
        ),
        None,
    )

    kwargs: dict = {}
    if "title" in body:
        title = str(body.get("title") or "").strip()
        if not title:
            return JSONResponse({"error": "title is required"}, status_code=400)
        kwargs["summary"] = title
    if "date" in body or "time" in body or "duration_min" in body:
        start_str, end_str, err = _parse_event_window(body, base=cached)
        if err:
            return JSONResponse({"error": err}, status_code=400)
        kwargs["start"] = start_str
        kwargs["end"] = end_str
    if "location" in body:
        kwargs["location"] = str(body.get("location") or "").strip()
    if "description" in body:
        kwargs["description"] = str(body.get("description") or "").strip()
    if not kwargs:
        return JSONResponse({"error": "no fields to update"}, status_code=400)

    res = await asyncio.to_thread(
        lambda: calendar_client.update_event(
            event_id,
            calendar_id=tasks_calendar_id,
            timezone=tasks_timezone,
            **kwargs,
        )
    )
    if not res.ok:
        return JSONResponse({"error": f"更新に失敗しました: {res.error}"}, status_code=502)

    def _apply(events: list) -> None:
        for e in events:
            if isinstance(e, dict) and e.get("event_id") == event_id:
                if "summary" in kwargs:
                    e["title"] = kwargs["summary"]
                if "start" in kwargs:
                    e["start"] = kwargs["start"]
                    e["end"] = kwargs["end"]
                if "location" in kwargs:
                    e["location"] = kwargs["location"]
                if "description" in kwargs:
                    e["description"] = kwargs["description"]

    _patch_upcoming_cache(_apply)
    return {"ok": True, "event_id": event_id}


@app.delete("/api/calendar/events/{event_id}")
async def calendar_event_delete(event_id: str):
    unavailable = _require_calendar_client()
    if unavailable is not None:
        return unavailable
    res = await asyncio.to_thread(
        lambda: calendar_client.delete_event(event_id, calendar_id=tasks_calendar_id)
    )
    if not res.ok:
        return JSONResponse({"error": f"削除に失敗しました: {res.error}"}, status_code=502)
    def _apply(events: list) -> None:
        events[:] = [
            e for e in events
            if not (isinstance(e, dict) and e.get("event_id") == event_id)
        ]

    _patch_upcoming_cache(_apply)
    return {"ok": True}


@app.post("/api/tts")
async def synthesize(request: Request):
    """テキストを音声合成してWAVを返す"""
    if tts is None:
        return JSONResponse({"error": "TTS not available"}, status_code=503)

    body = await request.json()
    text = body.get("text", "")
    if not text:
        return JSONResponse({"error": "text is required"}, status_code=400)

    # 同期的なTTS処理をスレッドで実行
    loop = asyncio.get_event_loop()
    try:
        wav_data = await loop.run_in_executor(None, tts.synthesize, text)
    except Exception as e:
        logger.warning("TTS synthesis failed: %s", type(e).__name__)
        return JSONResponse(
            {"error": _TTS_ERROR_MESSAGE, "error_type": _INTERNAL_ERROR_CODE},
            status_code=500,
        )

    return Response(
        content=wav_data,
        media_type="audio/wav",
        headers={"Content-Disposition": "inline"},
    )


@app.post("/api/tts/voice")
async def set_voice(request: Request):
    """TTSボイスを変更"""
    if tts is None:
        return JSONResponse({"error": "TTS not available"}, status_code=503)

    body = await request.json()
    voice = body.get("voice", "")
    voices = tts.list_ja_voices()
    if voice not in voices:
        return JSONResponse({"error": f"Unknown voice: {voice}"}, status_code=400)

    tts.set_voice(voice)
    return {"voice": voice, "description": voices[voice]}


# --- Vision API ---


@app.post("/api/stt")
async def stt_transcribe(request: Request):
    """音声データを受け取ってSTTでテキストに変換"""
    if not _microphone_enabled():
        return _microphone_denied_response()
    if stt is None:
        return JSONResponse({"error": "STT not available"}, status_code=503)

    body = await request.json()
    audio_b64 = body.get("audio", "")
    if not audio_b64:
        return JSONResponse({"error": "audio is required (base64 WAV)"}, status_code=400)

    try:
        import numpy as np
        audio_bytes = base64.b64decode(audio_b64)
        audio_array = _decode_wav_bytes(audio_bytes)

        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(None, stt.transcribe, audio_array, 16000)

        return {"text": text}
    except Exception as e:
        return _sensor_error_response("STT error", e)


def _decode_wav_bytes(wav_bytes: bytes) -> "np.ndarray":
    """生のWAVバイトをfloat32 numpy配列に変換 (16kHzモノラル)"""
    import numpy as np
    import wave
    import io

    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    # int16 で読み込み
    if sampwidth == 2:
        audio = np.frombuffer(raw, dtype=np.int16)
    elif sampwidth == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.int16)
    else:
        audio = np.frombuffer(raw, dtype=np.int16)

    # ステレオ→モノラル
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels)[:, 0]

    # float32に変換
    audio = audio.astype(np.float32) / 32768.0

    # 16kHzにリサンプリング
    if framerate != 16000:
        duration = len(audio) / framerate
        target_len = int(duration * 16000)
        indices = np.linspace(0, len(audio) - 1, target_len).astype(int)
        audio = audio[indices]

    return audio


def _decode_webm_bytes(webm_bytes: bytes) -> "np.ndarray":
    """WebM/Oggバイトをfloat32 numpy配列に変換 (16kHzモノラル) — ffmpeg使用"""
    import numpy as np
    import subprocess
    import tempfile
    import os

    # 一時ファイルに書き出し
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        tmp.write(webm_bytes)
        tmp_path = tmp.name

    try:
        # ffmpegで 16kHz, mono, s16le PCM に変換
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", tmp_path,
                "-ar", "16000", "-ac", "1", "-f", "s16le",
                "-acodec", "pcm_s16le", "-"
            ],
            capture_output=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError("ffmpeg decode failed")

        audio = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
        return audio
    finally:
        os.unlink(tmp_path)


@app.get("/api/vision/status")
async def vision_status():
    """映像入力の状態 (privacy-safe, allowlist のみ)。

    返すのは bool / タイムスタンプ類のみ。カメラデバイス情報・感情ラベル・
    解析カウントなどの生/派生情報は含めない。
    """
    if vision is None:
        return {"enabled": False}
    try:
        return {
            "enabled": True,
            **_filter_sensor_status(vision.get_status(), VISION_STATUS_ALLOWLIST),
        }
    except Exception as e:
        return _sensor_error_response("Vision status unavailable", e)


@app.get("/api/vision/snapshot")
async def vision_snapshot():
    """廃止: 生カメラ画像 (JPEG) は未認証公開しない (固定404, データなし)。"""
    return _sensor_disabled_response()


@app.get("/api/vision/context")
async def vision_context_text():
    """廃止: デバッグ用映像コンテキストテキストは未認証公開しない (固定404, データなし)。"""
    return _sensor_disabled_response()


# --- Screen API (画面認識) ---

# リモート画面 push (メインPC の scripts/screen_agent.py → ingest) の保存先。
SCREEN_DIR = PROJECT_ROOT / "data" / "screen"
SCREEN_LATEST_JPG = SCREEN_DIR / "latest.jpg"
SCREEN_LATEST_JSON = SCREEN_DIR / "latest.json"
MAX_INGEST_BYTES = 8 * 1024 * 1024  # 8MB

# ingest 用 VLM describer (遅延生成 / テストで差し替え可能)。
screen_ingest_describer = None
# 描写の単一飛行 + シャットダウン世代ゲート (すべて _ingest_describe_lock で保護)。
_ingest_describe_lock = threading.Lock()
# 現在実行中 worker の世代。None のとき実行中 worker なし (単一飛行)。
_ingest_active_generation: Optional[int] = None
# run_in_executor で提出した Future の保持。シャットダウンで cancel / bounded await する。
_ingest_future: Optional[asyncio.Future] = None
# 現在の受付世代。lifespan 起動で +1 して有効化、シャットダウンで +1 して revoke する。
_ingest_generation: int = 0
# 結果の受け付け可否。シャットダウンで False にし、実行中 worker の書き込みを revoke する。
_ingest_accepting: bool = True
# 実行中 worker の完了通知 (世代ごと)。_describe_ingested の finally でのみ set される。
# asyncio の run_in_executor Future が done (cancel 済み含む) でも下位 worker が継続中
# のことがあるため、シャットダウンはこの Event を bounded wait して完了を判定する。
# 提出より先に登録され、提出失敗時は除去される (_stop は常に Event を見つけられる)。
_ingest_done_events: dict[int, threading.Event] = {}
# シャットダウン時の bounded await 上限。
_INGEST_STOP_TIMEOUT = 5.0


def _ingest_tmp_path() -> Path:
    """latest.json と同じディレクトリの tmp パスを返す (アトミック書き込み用)。"""
    return SCREEN_LATEST_JSON.with_name(SCREEN_LATEST_JSON.name + ".tmp")


def _unlink_ingest_tmp(path: Path) -> None:
    """revoke / 準備失敗時に tmp を best-effort で破棄する (失敗は黙って無視)。"""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _remove_legacy_latest_jpg() -> None:
    """レガシー latest.jpg を best-effort で削除する。

    生JPEGは永続化しない運用のため、過去に保存された latest.jpg があれば
    起動・停止・無効状態で削除を試みる。失敗してもパス・エラーを外部へ露出せず
    黙って無視する。
    """
    try:
        if SCREEN_LATEST_JPG.exists():
            SCREEN_LATEST_JPG.unlink()
    except OSError:
        pass


def _ingest_results_accepted(generation: int) -> bool:
    """``generation`` の worker が結果を受け付け可能か。

    シャットダウン (受付 revoke) 後は False になり、実行中 worker が latest.json を
    書き込めなくなる。fail closed。
    """
    with _ingest_describe_lock:
        return _ingest_accepting and generation == _ingest_generation


def _start_ingest_generation() -> None:
    """受付世代を新しく有効化する (lifespan 起動時に呼ぶ)。"""
    global _ingest_generation, _ingest_accepting
    with _ingest_describe_lock:
        _ingest_generation += 1
        _ingest_accepting = True


def _submit_ingest_describe(loop, jpeg: bytes, received_at: float) -> bool:
    """単一飛行で executor へ描写を提出し、Future を保持する。

    受付 revoke (シャットダウン) 後は、Event/Future・単一飛行状態を一切登録する
    前に固定の内部エラー (外部へは 503 + ``unavailable`` 固定) で拒否する。これに
    より revoke 後の受信で ``described`` が true になることはなく、終了中の
    ownership を汚さない。実行中 worker があるときは False (この受信では描写を
    スキップ)。完了 Event は提出より先に世代へ登録し、提出 (``run_in_executor``)
    が失敗したときは登録と単一飛行状態を原子的に解除して例外を再送出する
    (固定 503 化は呼び出し側)。
    """
    global _ingest_active_generation, _ingest_future, _ingest_done_events
    with _ingest_describe_lock:
        # 受付 revoke 済み: 何も登録せずに拒否する (fail closed)。ConnectionError は
        # 外部向け固定コード "unavailable" へ写像され、本文・型名は露出しない。
        if not _ingest_accepting:
            raise ConnectionError("screen ingest not accepting")
        if _ingest_active_generation is not None:
            return False
        generation = _ingest_generation
        _ingest_active_generation = generation
        # worker は提出と同時に走り出すため、完了 Event は提出より先に登録する。
        # 提出失敗時はここで除去する (worker 未生成なので set されない)。
        _ingest_done_events[generation] = threading.Event()
        try:
            fut = loop.run_in_executor(
                None, _describe_ingested, jpeg, received_at, generation
            )
        except BaseException:
            _ingest_done_events.pop(generation, None)
            _ingest_active_generation = None
            raise
        _ingest_future = fut
    return True


async def _stop_ingest_describe(timeout: float = _INGEST_STOP_TIMEOUT) -> None:
    """シャットダウン時の ingest 描写停止。

    1. 受付世代を revoke して、実行中 worker の latest.json 書き込みを cancel より先に止める
    2. 保持済み Future を条件付きでキャンセル (Future が無くても完了判定は Event が権威)
    3. 実行中 worker 本体の完了 Event を bounded wait (オフループ) する
    4. Event が set されたときだけ ownership (単一飛行 / 保持 Future / 完了 Event) を解除する

    完了の権威は active generation とその Event であり、Future は補助情報に過ぎない。
    Future が None (提出直後の窓や barrier 相当) でも Event を bounded wait して
    完了を待つ。``run_in_executor`` Future の done (cancel 済み含む) は下位 worker の
    完了を保証しないため、完了判定には worker 本体の finally でのみ set される Event を
    使う。タイムアウト時は ownership を保持し、restart / 新規 ingest が実行中の旧
    worker と重ならないようにする。Event 待ちは lock の外 (オフループ) で行い
    deadlock しない。
    """
    global _ingest_generation, _ingest_accepting, _ingest_active_generation, _ingest_future
    global _ingest_done_events
    with _ingest_describe_lock:
        _ingest_accepting = False
        _ingest_generation += 1
        fut = _ingest_future
        generation = _ingest_active_generation
        event = _ingest_done_events.get(generation) if generation is not None else None
    if fut is not None and not fut.done():
        fut.cancel()
    if generation is None:
        # 実行中 worker なし → 何も待たず完了。
        return
    if event is not None:
        try:
            loop = asyncio.get_running_loop()
            completed = await loop.run_in_executor(None, event.wait, timeout)
        except Exception:
            completed = False
    else:
        # active 世代に完了 Event が無い → 完了を Event で確認できないため ownership は
        # 解除しない (fail safe)。worker の finally は Event set と active 解除を同一の
        # lock 区間で行うため、この状態は本来到達しない防御分岐。
        completed = False
    if completed:
        with _ingest_describe_lock:
            if _ingest_active_generation == generation:
                _ingest_active_generation = None
                _ingest_future = None
            _ingest_done_events.pop(generation, None)
    # 未完了 (タイムアウト) なら ownership を保持したまま返る。


def _get_ingest_describer():
    """ingest 用の ScreenDescriber を遅延生成 (テストで screen_ingest_describer に差し替え可)。"""
    global screen_ingest_describer
    if screen_ingest_describer is None:
        from src.screen.describer import ScreenDescriber
        base = config.ollama_base_url if config else "http://localhost:11434"
        model = config.model if config else None
        screen_ingest_describer = ScreenDescriber(base_url=base, model=model)
    return screen_ingest_describer


def _commit_ingest_result(generation: int, payload: dict) -> bool:
    """最新結果をアトミックにコミットする。

    latest.json.tmp への書き込み (open/write/flush/fsync) は lock の外で行い、
    lock 区間では受付確認と os.replace だけを実行する。これにより lock をブロッキング
    なファイル I/O の間保持しない (revoke / ``_stop`` が詰まらない)。

    strict ordering: revoke (``_ingest_accepting=False`` + 世代前進) が replace より先に
    lock を取れば最終受付確認が失敗して replace は抑止され tmp は破棄される。replace が
    先に完了すれば revoke はコミット完了後に続き、書き込み済み結果は残る。どちらの順序
    でも revoke 済み世代のコミットは起こらない (fail closed)。
    """
    tmp = _ingest_tmp_path()
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False))
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        _unlink_ingest_tmp(tmp)
        raise
    with _ingest_describe_lock:
        if not (_ingest_accepting and generation == _ingest_generation):
            _unlink_ingest_tmp(tmp)
            return False
        try:
            os.replace(tmp, SCREEN_LATEST_JSON)
        except Exception:
            _unlink_ingest_tmp(tmp)
            raise
        return True


def _describe_ingested(jpeg: bytes, received_at: float, generation: int) -> None:
    """受信画像を VLM で 1 回描写し latest.json をアトミック書き込み。失敗はログのみ。

    generation gate: 提出時点の世代が現在の受付世代と一致しない (シャットダウンで
    revoke 済み) ときは描写・書き込みを行わない。実行中 worker は自分の世代と一致する
    ときだけ単一飛行 / 保持 Future を後始末する (古い worker が新しい worker の
    ownership を消さない)。最終 acceptance check とコミットは ``_commit_ingest_result``
    が lock 下で原子的に行い、revoke と競合しない。
    """
    global _ingest_active_generation, _ingest_future, _ingest_done_events
    try:
        if not _ingest_results_accepted(generation):
            logger.warning("screen ingest: シャットダウンにより描写を破棄")
            return
        describer = _get_ingest_describer()
        description = describer.describe(jpeg)
        if not description:
            logger.warning("screen ingest: 描写が空でした (次の ingest で再試行)")
            return
        payload = {
            "description": description,
            "captured_at": received_at,
            "described_at": time.time(),
            "source": "remote",
        }
        if not _commit_ingest_result(generation, payload):
            logger.warning("screen ingest: シャットダウンにより描写結果を破棄")
    except Exception as e:
        logger.warning("screen ingest describe failed: %s", type(e).__name__)
    finally:
        with _ingest_describe_lock:
            ev = _ingest_done_events.get(generation)
            if ev is not None:
                ev.set()
            if _ingest_active_generation == generation:
                _ingest_active_generation = None
                _ingest_future = None
                _ingest_done_events.pop(generation, None)


def _read_ingest_status() -> dict:
    """ingest 受信状況を返す (privacy-safe)。

    screen_ingest が無効のときは {"enabled": False} を返し、latest.json の古い
    描写を available として露出しない。有効時のみ latest.json の鮮度と出所を返す。
    env 名・token 値は含めず、token_configured boolean は ingest 有効時のみ残す。
    VLM 描写テキスト (description) ・生JPEG (latest.jpg) は含めない・保持しない。
    """
    if sensor_policy is None or not sensor_policy.screen_ingest:
        _remove_legacy_latest_jpg()
        return {"enabled": False}
    token_configured = bool(os.environ.get("SCREEN_INGEST_TOKEN"))
    info = {
        "enabled": True,
        "token_configured": token_configured,
        "available": False,
    }
    try:
        if SCREEN_LATEST_JSON.exists():
            data = json.loads(SCREEN_LATEST_JSON.read_text(encoding="utf-8"))
            captured_at = float(data.get("captured_at") or 0.0)
            info.update({
                "available": True,
                "captured_at": captured_at,
                "described_at": data.get("described_at"),
                "source": "remote",
                "age_seconds": (time.time() - captured_at) if captured_at > 0 else None,
            })
    except Exception:
        pass
    return info


@app.get("/api/screen/status")
async def screen_status():
    """画面認識の状態 (privacy-safe, allowlist のみ)。

    local (自機キャプチャ) は bool / タイムスタンプ / ソース種別のみを返し、
    VLM 描写テキスト・モデル名は含めない。remote (ingest) の鮮度も同様。
    """
    result: dict = {"enabled": screen is not None}
    if screen is not None:
        try:
            result["source"] = "local"
            result.update(
                _filter_sensor_status(screen.get_status(), SCREEN_STATUS_ALLOWLIST)
            )
        except Exception as e:
            return _sensor_error_response("Screen status unavailable", e)
    # リモート push (ingest) の受信状況は screen コンテキストの有無に関わらず返す
    result["ingest"] = _read_ingest_status()
    return result


@app.get("/api/screen/context")
async def screen_context_text():
    """廃止: デバッグ用画面コンテキストテキストは未認証公開しない (固定404, データなし)。"""
    return _sensor_disabled_response()


@app.post("/api/screen/ingest")
async def screen_ingest(request: Request):
    """メインPC のキャプチャエージェントから生 JPEG を受信して描写する。

    認証: 共有 SensorPolicy.screen_ingest が有効で、かつ env SCREEN_INGEST_TOKEN と
          X-Screen-Token ヘッダが一致すること (compare_digest)。policy が無効のときは
          token があっても body を読まずに 403 (安全側デフォルト)。
    ボディ: 生 JPEG バイト (Content-Type: image/jpeg)。上限 8MB。
    レスポンス: 200 {"ok": true, "described": <この受信で描写を開始したか>}
    """
    if sensor_policy is None or not sensor_policy.screen_ingest:
        return JSONResponse({"error": "forbidden"}, status_code=403)
    token = os.environ.get("SCREEN_INGEST_TOKEN")
    if not token:
        return JSONResponse({"error": "forbidden"}, status_code=403)
    provided = request.headers.get("X-Screen-Token", "")
    if not secrets.compare_digest(provided, token):
        return JSONResponse({"error": "forbidden"}, status_code=403)

    # Content-Length で早期拒否 (可能なら)
    clen = request.headers.get("content-length")
    if clen and clen.isdigit() and int(clen) > MAX_INGEST_BYTES:
        return JSONResponse({"error": "payload too large"}, status_code=413)

    body = await request.body()
    if len(body) > MAX_INGEST_BYTES:
        return JSONResponse({"error": "payload too large"}, status_code=413)
    # JPEG マジックバイト (FF D8 FF)
    if len(body) < 3 or body[:3] != b"\xff\xd8\xff":
        return JSONResponse({"error": "not a JPEG"}, status_code=400)

    received_at = time.time()
    try:
        # 生JPEGは保存せず、describer へ bytes を渡す。latest.json 出力用の
        # ディレクトリだけ用意する。
        SCREEN_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return _sensor_error_response("screen ingest save failed", e)

    # 単一飛行で executor へ提出し Future を保持する (実行中なら描写はスキップ)。
    # 提出が失敗したら単一飛行状態は原子的に解除され、固定 503 を返す (生例外は露出しない)。
    loop = asyncio.get_event_loop()
    try:
        started = _submit_ingest_describe(loop, body, received_at)
    except Exception as e:
        return _sensor_error_response(
            "screen ingest describe submit failed", e, status_code=503
        )

    return {"ok": True, "described": started}


# --- Monitor API (Phase 6) ---

@app.get("/api/monitor/status")
async def monitor_status():
    """PCモニターの状態 (privacy-safe, allowlist のみ)。

    bool / タイムスタンプ / source (固定 "monitor") のみを返し、CPU/メモリ/GPU/
    ディスク等のメトリクス集計値・プロセス数・レコード数・DB パス・エラー・
    本文テキストは含めない。
    """
    if monitor is None:
        return {"enabled": False}
    try:
        return {
            "enabled": True,
            "source": "monitor",
            **_filter_sensor_status(monitor.get_status(), MONITOR_STATUS_ALLOWLIST),
        }
    except Exception as e:
        return _sensor_error_response("Monitor status unavailable", e)


@app.get("/api/monitor/context")
async def monitor_context_text():
    """廃止: デバッグ用PCモニターコンテキストテキストは未認証公開しない (固定404, データなし)。"""
    return _sensor_disabled_response()


@app.get("/api/monitor/summary")
async def monitor_summary(minutes: int = 60):
    """廃止: 直近N分のメトリクスサマリーは未認証公開しない (固定404, データなし)。"""
    return _sensor_disabled_response()


# --- Persona API (Phase 7) ---

@app.get("/api/persona/status")
async def persona_status():
    """パーソナライズの状態"""
    if profile is None:
        return {"enabled": False}
    return {
        "enabled": True,
        "profile": profile.get_status(),
        "preloader": preloader.get_status() if preloader else None,
    }


@app.get("/api/persona/profile")
async def persona_profile():
    """ユーザープロフィール取得"""
    if profile is None:
        return JSONResponse({"error": "Persona not available"}, status_code=503)
    return profile.data


@app.post("/api/persona/profile")
async def update_persona_profile(request: Request):
    """ユーザープロフィール更新"""
    if profile is None:
        return JSONResponse({"error": "Persona not available"}, status_code=503)

    body = await request.json()

    if "name" in body:
        profile.name = body["name"]
    if "nickname" in body:
        profile.data["nickname"] = body["nickname"]
        profile.save()
    if "preferences" in body and isinstance(body["preferences"], dict):
        for k, v in body["preferences"].items():
            profile.set_preference(k, v)
    if "habits" in body and isinstance(body["habits"], dict):
        for k, v in body["habits"].items():
            profile.set_habit(k, v)
    if "note" in body:
        profile.add_note(body["note"])
    if "schedule" in body and isinstance(body["schedule"], dict):
        s = body["schedule"]
        profile.add_schedule(
            title=s.get("title", ""),
            date_str=s.get("date", ""),
            time_str=s.get("time", ""),
            note=s.get("note", ""),
        )

    return {"status": "updated", "profile": profile.get_status()}


@app.get("/api/persona/summaries")
async def persona_summaries(count: int = 5):
    """直近の会話要約を取得"""
    if summarizer is None:
        return JSONResponse({"error": "Persona not available"}, status_code=503)
    return {"summaries": summarizer.get_recent_summaries(count=count)}


@app.get("/api/persona/context")
async def persona_context():
    """現在のプリロードコンテキスト（デバッグ用）"""
    if preloader is None:
        return {"context": "", "enabled": False}
    return {"context": preloader.build_preload_context(), "enabled": True}


# --- Idle API ---

@app.get("/api/idle/status")
async def idle_status():
    """アイドル管理の状態"""
    if idle_manager is None:
        return {"enabled": False}
    return {"enabled": True, **idle_manager.get_status()}


# --- Companion API ---

def _companion_state_payload() -> dict:
    """GET /api/companion/state のレスポンス (読み取り専用・privacy-safe)。

    src.perception.bootstrap の共通 helper へ委譲する。
    """
    return companion_state_payload(activity_runtime)


@app.get("/api/companion/state")
async def companion_state():
    """コンパニオン活動状態 (読み取り専用)。未オプトイン時は enabled=false。"""
    return _companion_state_payload()


# --- ログ管理 API ---

JOURNAL_UNITS = ("subpc-web", "subpc-discord", "subpc-sbv2-tts", "subpc-gpu-powersave")


def _history_dir() -> Path:
    rel = config.history_dir if config is not None else "data/chat_history"
    return PROJECT_ROOT / rel


def _history_max_files() -> int:
    try:
        return int(os.environ.get("HISTORY_MAX_FILES", "200"))
    except ValueError:
        return 200


@app.get("/api/logs/journal")
async def logs_journal(unit: str = "subpc-web", lines: int = 200):
    """systemd ユーザーサービスのログを返す (unit はホワイトリスト制)"""
    if unit not in JOURNAL_UNITS:
        return JSONResponse(
            {"error": f"Unknown unit: {unit}", "units": list(JOURNAL_UNITS)},
            status_code=400,
        )
    lines = max(10, min(lines, 1000))

    def run() -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                "journalctl", "--user", "-u", f"{unit}.service",
                "-n", str(lines), "--no-pager", "-o", "short-iso",
            ],
            capture_output=True, text=True, timeout=15,
        )

    try:
        proc = await asyncio.get_event_loop().run_in_executor(None, run)
    except Exception as e:
        return JSONResponse({"error": f"journalctl 実行失敗: {e}"}, status_code=500)
    if proc.returncode != 0:
        return JSONResponse(
            {"error": proc.stderr.strip() or "journalctl エラー"}, status_code=500
        )
    return {"unit": unit, "units": list(JOURNAL_UNITS), "lines": proc.stdout.splitlines()}


@app.get("/api/logs/files")
async def logs_files():
    """logs/ ディレクトリのアプリログファイル一覧"""
    files = []
    if DEFAULT_LOG_DIR.is_dir():
        for path in sorted(DEFAULT_LOG_DIR.iterdir()):
            if not path.is_file() or ".log" not in path.name:
                continue
            st = path.stat()
            files.append({
                "name": path.name,
                "size_bytes": st.st_size,
                "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
            })
    files.sort(key=lambda f: f["mtime"], reverse=True)
    return {"files": files}


@app.get("/api/logs/files/{name}")
async def logs_file_tail(name: str, lines: int = 300):
    """アプリログファイルの末尾を返す"""
    if "/" in name or "\\" in name or ".." in name or ".log" not in name:
        return JSONResponse({"error": "不正なファイル名"}, status_code=400)
    path = DEFAULT_LOG_DIR / name
    if not path.is_file():
        return JSONResponse({"error": "ファイルが見つかりません"}, status_code=404)
    lines = max(10, min(lines, 2000))

    def read_tail() -> list[str]:
        with open(path, encoding="utf-8", errors="replace") as f:
            return [line.rstrip("\n") for line in deque(f, maxlen=lines)]

    content = await asyncio.get_event_loop().run_in_executor(None, read_tail)
    return {"name": name, "lines": content}


# --- 会話履歴 API ---

@app.get("/api/history/sessions")
async def history_sessions():
    """会話履歴ファイルの一覧"""
    return {"sessions": history_admin.list_sessions(_history_dir())}


@app.get("/api/history/sessions/{filename}")
async def history_session_detail(filename: str):
    """会話履歴ファイルの中身"""
    data = history_admin.read_session(_history_dir(), filename)
    if data is None:
        return JSONResponse({"error": "履歴が見つかりません"}, status_code=404)
    return data


@app.delete("/api/history/sessions/{filename}")
async def history_session_delete(filename: str):
    """会話履歴ファイルを削除"""
    if not history_admin.delete_session(_history_dir(), filename):
        return JSONResponse({"error": "履歴が見つかりません"}, status_code=404)
    logger.info("会話履歴を削除: %s", filename)
    return {"deleted": filename}


# --- WebSocket チャット ---

# ストリーミング失敗の中立なエラーメッセージ。URL・APIキーなど秘密情報を
# クライアントへ露出しない (詳細はサーバーログのみに残す)。
_STREAM_TIMEOUT_MESSAGE = "応答の生成がタイムアウトしました。もう一度お試しください。"
_STREAM_EMPTY_MESSAGE = "応答を生成できませんでした。もう一度お試しください。"
_STREAM_ERROR_MESSAGE = "応答の生成に失敗しました。もう一度お試しください。"
_TTS_ERROR_MESSAGE = "TTS error"
_INTERNAL_ERROR_CODE = "internal_error"


class _StreamError(Exception):
    """ストリーミング失敗。中立なユーザー向けメッセージと内部詳細を保持する。"""

    def __init__(self, message: str, *, detail: BaseException | None = None) -> None:
        super().__init__(message)
        self.detail = detail


def _extraction_enabled() -> bool:
    """TASKS_CHAT_EXTRACTION_ENABLED は既定 true。無効化は "false" のみ。"""
    val = os.environ.get("TASKS_CHAT_EXTRACTION_ENABLED", "").strip().lower()
    if val in ("false", "0", "no", "off"):
        return False
    return True


def _candidate_to_json(cand: dict) -> dict:
    """候補1件を API/WS 配信用の JSON 形式へ直列化する。"""
    due_at = cand.get("due_at")
    created_at = cand.get("created_at")
    decided_at = cand.get("decided_at")
    return {
        "id": cand["id"],
        "title": cand.get("title") or "",
        "due_at": due_at.isoformat() if isinstance(due_at, datetime) else due_at,
        "due_granularity": cand.get("due_granularity"),
        "priority": cand.get("priority") or "normal",
        "status": cand.get("status") or "pending",
        "task_id": cand.get("task_id"),
        "created_at": created_at.isoformat() if isinstance(created_at, datetime) else created_at,
        "decided_at": decided_at.isoformat() if isinstance(decided_at, datetime) else decided_at,
    }


def _make_source_ref(session_id: str, turn: int, utterance: str) -> str:
    """安全な session_id・番号・発言ダイジェストから source_ref を作る。

    生発言は一切保存しない。session_id は WebSocket 入口で安全判定済みだが、
    ここでも念のため sanitize して異常値は "unknown" に丸める。
    """
    safe_sid = session_id if history_admin.is_safe_session_id(session_id) else "unknown"
    digest = hashlib.sha256(utterance.encode("utf-8")).hexdigest()[:16]
    return f"web:{safe_sid}:t{int(turn)}:{digest}"


def _extraction_timeout_seconds() -> float:
    raw = os.environ.get("TASKS_CHAT_EXTRACTION_TIMEOUT_SECONDS", "15").strip()
    try:
        return max(3.0, min(float(raw), 60.0))
    except ValueError:
        return 15.0


def _extract_task_candidates(user_text: str) -> list[dict]:
    """人格会話とは分離した低温度JSON抽出。生成設定を保つためServiceを通さない。"""
    if not _extraction_enabled() or task_store is None or llm is None or config is None:
        return []
    # 秘密らしい文字列をモデルへ渡さない。抽出後タイトルもvalidatorで再検査する。
    if task_extractor.is_sensitive_text(user_text):
        logger.info("task candidate extraction skipped: sensitive text detected")
        return []
    now_local = datetime.now(_tasks_tz())
    messages = [
        {"role": "system", "content": task_extractor.build_multi_extraction_prompt(now_local)},
        {"role": "user", "content": user_text},
    ]
    try:
        raw = llm.generate(
            messages,
            temperature=0.0,
            num_ctx=min(int(config.num_ctx), _EXTRACTION_NUM_CTX),
            num_predict=_EXTRACTION_NUM_PREDICT,
            timeout=_extraction_timeout_seconds(),
        )
    except Exception as exc:
        logger.warning("task candidate extraction failed: %s", exc)
        return []
    validated = task_extractor.validate_multi_extraction(raw, assume_tz=_tasks_tz())
    return list(validated.get("tasks") or []) if validated else []


async def _offer_task_candidates(
    websocket: WebSocket,
    *,
    user_text: str,
    session_id: str,
    turn: int,
) -> None:
    """候補を抽出・永続化し、新規候補だけWebSocketへ送る。"""
    if not _extraction_enabled() or task_store is None:
        return
    extracted = await asyncio.to_thread(_extract_task_candidates, user_text)
    if not extracted:
        return
    source_ref = _make_source_ref(session_id, turn, user_text)
    for index, item in enumerate(extracted[:3]):
        title = item.get("title") or ""
        if task_extractor.is_sensitive_text(title):
            logger.warning("sensitive task candidate rejected after extraction")
            continue
        candidate_now = datetime.now(timezone.utc)
        candidate_id = await asyncio.to_thread(
            task_store.create_candidate,
            title=title,
            due_at=item.get("due_at"),
            due_granularity=item.get("due_granularity"),
            priority=item.get("priority") or "normal",
            source="chat",
            now=candidate_now,
        )
        if candidate_id is None:
            continue
        candidate = await asyncio.to_thread(task_store.get_candidate, candidate_id)
        # create_candidate が既存pendingを返した場合は再提示しない。
        if candidate is None or candidate.get("created_at") != candidate_now:
            continue
        logger.info("task candidate created: %s:%s id=%s", source_ref, index, candidate_id)
        await websocket.send_json({
            "type": "task_candidate",
            "candidate": _candidate_to_json(candidate),
        })


async def _run_task_candidate_offer(
    websocket: WebSocket,
    *,
    user_text: str,
    session_id: str,
    turn: int,
) -> None:
    """候補処理専用の例外境界。通常応答・TTSへ失敗を伝播させない。"""
    try:
        await asyncio.wait_for(
            _offer_task_candidates(
                websocket,
                user_text=user_text,
                session_id=session_id,
                turn=turn,
            ),
            timeout=_extraction_timeout_seconds() + 2.0,
        )
    except asyncio.TimeoutError:
        logger.warning("task candidate extraction timed out")
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning("task candidate offer failed: %s", exc)


def _launch_task_candidate_offer(
    websocket: WebSocket,
    *,
    user_text: str,
    session_id: str,
    turn: int,
) -> None:
    """done後にbest-effort候補処理を起動し、受信ループを止めない。"""
    if not _extraction_enabled() or task_store is None:
        return
    task = asyncio.create_task(_run_task_candidate_offer(
        websocket,
        user_text=user_text,
        session_id=session_id,
        turn=turn,
    ))
    _candidate_offer_tasks.add(task)
    task.add_done_callback(_candidate_offer_tasks.discard)


def _try_register_event_text(text: str) -> Optional[str]:
    """予定登録の意図があれば Google Calendar に登録して結果文を返す (無ければ None)。

    ブロッキング (MCP 呼び出しで数秒) なので to_thread 経由で呼ぶこと。
    """
    try:
        from src.tasks.event_intent import try_register_event

        return try_register_event(
            text,
            client=calendar_client,
            calendar_id=tasks_calendar_id,
            timezone_name=tasks_timezone,
            upcoming_path=UPCOMING_PATH,
        )
    except Exception as e:
        logger.error("event register failed: %s", e)
        return None


def _try_edit_task_text(text: str, session_id: str) -> Optional[str]:
    """明示的なタスク操作だけをルールベースで処理する。"""
    try:
        return task_chat_editor.handle(
            text,
            store=task_store,
            session_id=session_id,
        )
    except Exception as e:
        logger.error("task chat edit failed: %s", e)
        return "タスクの編集中にエラーが起きました。まだ変更は反映していません。"


async def _send_direct_chat_reply(
    websocket: WebSocket,
    *,
    session_id: str,
    user_text: str,
    reply: str,
    want_tts: bool,
    store_memory: bool = True,
) -> None:
    """LLMを介さないタスク/予定操作も、通常の会話と同じ形で保存・配信する。

    store_memory=False のときは RAG 長期記憶への保存をスキップする
    (セッション履歴・save() は常に行う)。既定は通常ターンと同じく保存する。
    """
    session = get_or_create_session(session_id)
    session.add_user_message(user_text)
    session.add_assistant_message(reply, store_memory=store_memory)
    try:
        session.save()
    except Exception as e:
        logger.warning("session save failed: %s", type(e).__name__)
    await websocket.send_json({"type": "token", "content": reply})
    await websocket.send_json({"type": "done", "full_text": reply})
    if want_tts and tts is not None:
        try:
            wav_data = await asyncio.get_event_loop().run_in_executor(
                None, lambda: tts.synthesize(reply)
            )
            await websocket.send_json({
                "type": "audio",
                "data": base64.b64encode(wav_data).decode("ascii"),
            })
        except Exception as e:
            logger.warning("TTS synthesis failed: %s", type(e).__name__)
            await websocket.send_json({
                "type": "error",
                "message": _TTS_ERROR_MESSAGE,
                "error_type": _INTERNAL_ERROR_CODE,
            })


def _effective_system_prompt(cfg) -> str:
    """実行時に使用する system_prompt を返す。

    ChatConfig なら model_prompt_overrides を考慮した effective_system_prompt を使う。
    テスト用の duck-typed モック (SimpleNamespace 等) には system_prompt だけ使う。
    """
    fn = getattr(cfg, "effective_system_prompt", None)
    if callable(fn):
        return fn()
    return cfg.system_prompt


def _start_assistant_stream(request, blocks, *, base_system):
    """経路選択からQueue worker開始までを同期的に行う。"""
    stream = assistant_service.respond_stream(request, blocks, base_system=base_system)
    return stream_to_queue(stream)


def _new_chat_session() -> ChatSession:
    """現在の依存コンポーネントを注入した ChatSession を新規作成する。"""
    return ChatSession(
        system_prompt=_effective_system_prompt(config),
        max_history_turns=config.max_history_turns,
        history_dir=str(PROJECT_ROOT / config.history_dir),
        rag=rag,
        vision_context=vision,
        screen_context=screen,
        monitor_context=monitor,
        task_store=task_store,
        preloader=preloader,
        web_search=web_search,
        growth_tracker=growth_tracker,
        conversation_source="web",
        emotion_tags=config.emotion_tag_enabled,
    )


def _history_dir_path() -> Path:
    return _history_dir()


def _messages_for_resume(session: ChatSession) -> list[dict]:
    """resume API 用に user/assistant の文字列 content だけ抽出する。"""
    out = []
    for m in session.messages:
        role = m.get("role")
        content = m.get("content")
        if role in ("user", "assistant") and isinstance(content, str):
            out.append({"role": role, "content": content})
    return out


def _new_web_session_id() -> str:
    """衝突耐性のある新規WebチャットセッションIDを返す。

    可読性のため ``web_`` プレフィックスとミリ秒時刻を残しつつ、同一ミリ秒でも
    衝突しないよう stdlib ``secrets`` の乱数サフィックスを付与する。ユーザー/IP等
    からは導出しない (nonsecret)。
    """
    return f"web_{int(time.time() * 1000)}_{secrets.token_hex(4)}"


@app.get("/api/chat/resume")
async def chat_resume(session_id: str | None = None):
    """直近の会話を引き継ぐためのセッション情報を返す。

    - session_id 指定かつ安全: そのIDで復元 (不在なら空セッション、新IDは発行しない)
      不正ID: 400
    - 指定なし: 最新の有効履歴を引き継ぐ。無ければ新しい web_<ms>_<rand> ID を発行
    """
    history_dir = _history_dir_path()

    if session_id is not None:
        session_id = session_id.strip()
        if not history_admin.is_safe_session_id(session_id):
            return JSONResponse({"error": "invalid session_id"}, status_code=400)
        session = get_or_create_session(session_id)
        return {"session_id": session.session_id, "messages": _messages_for_resume(session)}

    latest = history_admin.read_latest_valid_session(history_dir)
    if latest is not None:
        latest_id = latest.get("session_id")
        if latest_id and history_admin.is_safe_session_id(str(latest_id)):
            session = get_or_create_session(str(latest_id))
            return {"session_id": session.session_id, "messages": _messages_for_resume(session)}

    # 保存IDも履歴も無い初回 → 新ID発行 (衝突耐性)
    new_id = _new_web_session_id()
    return {"session_id": new_id, "messages": []}


def get_or_create_session(session_id: str) -> ChatSession:
    """セッションを取得または新規作成。

    安全なIDに対応する session_<id>.json が存在すれば ChatSession.load で復元し、
    無ければ新規作成後に session.session_id を要求IDに設定して以降同じファイルへ save する。
    """
    if not history_admin.is_safe_session_id(session_id):
        raise ValueError("invalid session_id")
    if session_id in sessions:
        return sessions[session_id]

    history_dir = _history_dir_path()
    session_file = history_admin.session_file_for(history_dir, session_id)
    if session_file is not None and history_admin.read_session_by_id(history_dir, session_id):
        try:
            session = ChatSession.load(
                session_file,
                max_history_turns=config.max_history_turns,
                history_dir=str(history_dir),
                rag=rag,
                vision_context=vision,
                screen_context=screen,
                monitor_context=monitor,
                task_store=task_store,
                preloader=preloader,
                web_search=web_search,
                growth_tracker=growth_tracker,
                conversation_source="web",
                emotion_tags=config.emotion_tag_enabled,
            )
            # 復元時も現在設定の system_prompt を優先
            session.system_prompt = _effective_system_prompt(config)
            session.emotion_tags = config.emotion_tag_enabled
            session.max_history_turns = config.max_history_turns
            session.session_id = session_id
            sessions[session_id] = session
            return session
        except Exception as e:
            logger.warning("session load failed: %s", type(e).__name__)

    session = _new_chat_session()
    session.session_id = session_id
    sessions[session_id] = session
    return session


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocketチャット（ストリーミング応答）

    クライアント → サーバー:
        {"type": "message", "text": "...", "session_id": "...", "tts": true/false}
        {"type": "audio_message", "data": "base64...", "format": "webm", "session_id": "...", "tts": true/false}

    サーバー → クライアント:
        {"type": "token", "content": "..."}       # ストリーミングトークン
        {"type": "done", "full_text": "..."}       # 応答完了
        {"type": "audio", "data": "base64..."}     # TTS音声 (base64 WAV)
        {"type": "stt_result", "text": "..."}      # STT認識結果
        {"type": "task_candidate", "candidate": {}} # 会話から見つけた候補
        {"type": "error", "message": "..."}        # エラー
    """
    await websocket.accept()

    try:
        while True:
            queue_stream = None
            raw = await websocket.receive_text()
            data = json.loads(raw)
            inference_started = False

            msg_type = data.get("type", "")
            if msg_type not in ("message", "audio_message"):
                continue

            user_text = data.get("text", "").strip()
            session_id = data.get("session_id", "default")
            want_tts = data.get("tts", False)

            if not history_admin.is_safe_session_id(session_id):
                await websocket.send_json({
                    "type": "error",
                    "message": "invalid session_id",
                })
                continue

            # --- 音声メッセージ処理 ---
            if msg_type == "audio_message":
                audio_b64 = data.get("data", "")
                audio_format = data.get("format", "wav")
                if not _microphone_enabled():
                    await websocket.send_json({
                        "type": "error",
                        "message": "マイク入力は許可されていません。",
                    })
                    continue
                if not audio_b64 or stt is None:
                    await websocket.send_json({
                        "type": "error",
                        "message": "STT not available" if stt is None else "No audio data",
                    })
                    continue

                try:
                    import numpy as np
                    audio_bytes = base64.b64decode(audio_b64)

                    # フォーマットに応じてデコード
                    loop = asyncio.get_event_loop()
                    if audio_format in ("webm", "ogg", "mp4", "m4a"):
                        audio_array = await loop.run_in_executor(
                            None, _decode_webm_bytes, audio_bytes
                        )
                    else:
                        audio_array = await loop.run_in_executor(
                            None, _decode_wav_bytes, audio_bytes
                        )

                    # アイドル管理: STT前にGPUをアクティブ化
                    if idle_manager is not None and not inference_started:
                        idle_manager.notify_inference_start(wait_for_gpu=True)
                        inference_started = True

                    # STT実行
                    text = await loop.run_in_executor(
                        None, stt.transcribe, audio_array, 16000
                    )

                    if not text:
                        if idle_manager is not None and inference_started:
                            idle_manager.notify_inference_end()
                            inference_started = False
                        await websocket.send_json({
                            "type": "stt_result",
                            "text": "",
                            "message": "音声を認識できませんでした",
                        })
                        continue

                    # 認識テキストをクライアントに通知
                    await websocket.send_json({
                        "type": "stt_result",
                        "text": text,
                    })

                    # 認識テキストをそのままLLMに流す
                    user_text = text
                    # 以下のチャット処理にフォールスルー

                except Exception as e:
                    if idle_manager is not None and inference_started:
                        idle_manager.notify_inference_end()
                        inference_started = False
                    await websocket.send_json({
                        "type": "error",
                        "message": "STT error",
                        "error_type": sensor_error_code(e),
                    })
                    continue

            if not user_text:
                continue

            # 「タスクを見せて」「タスク13を明日に変更」などは
            # LLMを介さず、曖昧性検査と削除確認を持つ編集器で処理する。
            task_reply = await asyncio.to_thread(
                _try_edit_task_text, user_text, session_id
            )
            if task_reply is not None:
                if idle_manager is not None and inference_started:
                    idle_manager.notify_inference_end()
                    inference_started = False
                # タスク状態の返答はRAG長期記憶に残さない
                # (履歴・save() は通常通り行う)。
                await _send_direct_chat_reply(
                    websocket,
                    session_id=session_id,
                    user_text=user_text,
                    reply=task_reply,
                    want_tts=want_tts,
                    store_memory=False,
                )
                continue

            # 「予定: ...」「〜の予定入れて」は LLM を介さず Google Calendar に
            # 登録し、定型文で応答する (日時解釈はルールベース)。
            event_reply = await asyncio.to_thread(_try_register_event_text, user_text)
            if event_reply is not None:
                if idle_manager is not None and inference_started:
                    idle_manager.notify_inference_end()
                    inference_started = False
                await _send_direct_chat_reply(
                    websocket,
                    session_id=session_id,
                    user_text=user_text,
                    reply=event_reply,
                    want_tts=want_tts,
                )
                continue

            # アイドル管理: ユーザー操作通知
            if idle_manager is not None and not inference_started:
                idle_manager.notify_inference_start()
                inference_started = True

            session = get_or_create_session(session_id)
            session.add_user_message(user_text)
            blocks = session.build_blocks()

            # ストリーミング応答生成
            loop = asyncio.get_event_loop()
            full_response = ""
            # 感情タグフィルタ (有効時のみ)。クライアントにはタグを見せない。
            emo_filter = (
                EmotionTagStreamFilter() if config.emotion_tag_enabled else None
            )

            try:
                # queue ベースのリアルタイムストリーミング
                # _start_assistant_stream内でassistant_service.respond_stream(
                # request, blocks)とQueue worker開始を同じthread上で行う。
                request = AssistantRequest(
                    text=user_text,
                    conversation_id=session_id,
                    channel="web",
                    privacy="local_only",
                )
                queue_stream = await asyncio.to_thread(
                    _start_assistant_stream, request, blocks,
                    base_system=session.system_prompt,
                )
                token_queue = queue_stream.queue

                while True:
                    try:
                        token = await asyncio.get_event_loop().run_in_executor(
                            None, token_queue.get, True, 300.0
                        )
                    except queue.Empty:
                        # ブロッキングgetのタイムアウト。中立なタイムアウトエラーとして
                        # 扱い、外側のハンドラにtype=error送出・cancel・巻き戻しを任せる。
                        raise _StreamError(_STREAM_TIMEOUT_MESSAGE)
                    # queue.Empty 以外の get/executor 例外は握り潰さず伝播させる。

                    if token is None:
                        # ストリーム終了。1tokenも得られずに終了した場合は、空応答の
                        # 正常契約が無いためエラーとして扱う。
                        if not full_response:
                            raise _StreamError(_STREAM_EMPTY_MESSAGE)
                        break
                    if isinstance(token, Exception):
                        # Provider例外 (sentinel)。詳細はログへ、文言は中立にする。
                        raise _StreamError(_STREAM_ERROR_MESSAGE, detail=token)

                    piece = emo_filter.feed(token) if emo_filter is not None else token
                    if not piece:
                        continue
                    full_response += piece
                    await websocket.send_json({"type": "token", "content": piece})

                session.add_assistant_message(full_response)

                # 会話履歴を保存 (古いファイルは HISTORY_MAX_FILES 件まで整理)
                try:
                    session.save()
                    history_admin.prune_sessions(_history_dir(), _history_max_files())
                except Exception as e:
                    logger.warning("会話履歴の保存に失敗: %s", type(e).__name__)

                await websocket.send_json({
                    "type": "done",
                    "full_text": full_response,
                })

                # 返答を先に見せてから、人格モデルと分離した抽出器で候補を提示する。
                _launch_task_candidate_offer(
                    websocket,
                    user_text=user_text,
                    session_id=session_id,
                    turn=len(session.messages),
                )

                # TTS
                if want_tts and tts is not None and full_response:
                    try:
                        style = (
                            emotion_to_sbv2_style(emo_filter.emotion)
                            if emo_filter is not None
                            else None
                        )
                        wav_data = await loop.run_in_executor(
                            None, lambda: tts.synthesize(full_response, style=style)
                        )
                        audio_b64 = base64.b64encode(wav_data).decode("ascii")
                        await websocket.send_json({
                            "type": "audio",
                            "data": audio_b64,
                        })
                    except Exception as e:
                        logger.warning("TTS synthesis failed: %s", type(e).__name__)
                        await websocket.send_json({
                            "type": "error",
                            "message": _TTS_ERROR_MESSAGE,
                            "error_type": _INTERNAL_ERROR_CODE,
                        })

            except _StreamError as e:
                logger.warning(
                    "chat stream failed: %s",
                    type(e.detail).__name__ if e.detail is not None else type(e).__name__,
                )
                await websocket.send_json({
                    "type": "error",
                    "message": str(e),
                })
                # ユーザーメッセージを巻き戻す
                session.rollback_last_user_message()
            except Exception as e:
                logger.warning("chat stream failed: %s", type(e).__name__)
                await websocket.send_json({
                    "type": "error",
                    "message": _STREAM_ERROR_MESSAGE,
                })
                # ユーザーメッセージを巻き戻す
                session.rollback_last_user_message()
            finally:
                if queue_stream is not None:
                    queue_stream.cancel()
                if idle_manager is not None and inference_started:
                    idle_manager.notify_inference_end()
                    inference_started = False

    except WebSocketDisconnect:
        pass


# --- エントリポイント ---

def main():
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="subpc_living Web UI")
    parser.add_argument("--host", default="0.0.0.0", help="バインドアドレス (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="ポート番号 (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="開発用ホットリロード")
    args = parser.parse_args()

    uvicorn.run(
        "src.web.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
