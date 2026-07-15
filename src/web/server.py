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

from src.chat.client import OllamaClient
from src.chat.session import ChatSession
from src.chat.config import ChatConfig
from src.chat.emotion import EmotionTagStreamFilter, emotion_to_sbv2_style
from src.chat.web_search import WebSearchContext, create_web_search_context
from src.audio.tts_factory import backend_name, create_tts_backend
from src.audio.stt import WhisperSTT
from src.memory.vectorstore import VectorStore
from src.memory.rag import RAGRetriever
from src.vision.context import VisionContext
from src.screen.context import ScreenContext
from src.screen import create_screen_context
from src.monitor.context import MonitorContext
from src.persona.profile import UserProfile
from src.persona.summarizer import ConversationSummarizer
from src.persona.preloader import SessionPreloader
from src.service.healthcheck import HealthChecker
from src.service.idle import IdleManager
from src.discord_bot.task_ui import parse_due, parse_snooze, split_quick_input
from src.tasks.store import TaskStore
from src.tasks.chat_editor import TaskChatEditor
from src.growth.tracker import GrowthTracker
from src.service.log_setup import setup_logging, DEFAULT_LOG_DIR
from src.chat import history_admin

logger = setup_logging("subpc-web")


# --- グローバル状態 ---
config: ChatConfig = None
llm: OllamaClient = None
tts = None
stt: WhisperSTT = None
rag: RAGRetriever = None
vision: VisionContext = None
screen: Optional[ScreenContext] = None
monitor: MonitorContext = None
profile: UserProfile = None
summarizer: ConversationSummarizer = None
preloader: SessionPreloader = None
web_search: Optional[WebSearchContext] = None
sessions: dict[str, ChatSession] = {}
idle_manager: Optional[IdleManager] = None
task_store: Optional[TaskStore] = None
task_chat_editor = TaskChatEditor()
growth_tracker: Optional[GrowthTracker] = None
tasks_timezone: str = "Asia/Tokyo"
task_calendar_sync = None  # TaskCalendarSync | None (Webで作ったタスクをカレンダーへ push)
calendar_client = None  # GoogleCalendarMCPClient | None (イベント CRUD 用)
tasks_calendar_id: str = "primary"
UPCOMING_PATH = PROJECT_ROOT / "data" / "calendar" / "upcoming.json"


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """サーバー起動/終了時の処理"""
    global config, llm, tts, stt, rag, vision, screen, monitor, profile, summarizer, preloader, web_search, idle_manager, task_store, growth_tracker, tasks_timezone, task_calendar_sync, calendar_client, tasks_calendar_id

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
    logger.info("[1/6] Ollama 接続確認...")
    llm = OllamaClient(base_url=config.ollama_base_url, model=config.model)
    if not llm.is_available():
        logger.warning("Ollamaに接続できません。チャット機能は使用不可です。")
    else:
        logger.info("✅ Ollama OK (model: %s)", config.model)
    if web_search is not None:
        logger.info("✅ Web検索 ON (auto=%s, max_results=%s)", config.web_search_auto, config.web_search_max_results)

    # STT 初期化
    logger.info("[2/7] STT 初期化...")
    try:
        stt = WhisperSTT(model_size="auto", language="ja", device="auto")
        stt.load()
        logger.info("✅ STT OK (model: %s, device: %s)", stt.model_size, stt.device)
    except Exception as e:
        logger.warning("STT ロード失敗: %s", e)
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

    # RAG 初期化 (Phase 4)
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

    # Vision 初期化 (Phase 5)
    logger.info("[5/7] Vision (映像入力) 初期化...")
    try:
        emotion_model = str(PROJECT_ROOT / "models" / "vision" / "emotion-ferplus-8.onnx")
        vision = VisionContext(
            camera_id=0,
            analysis_interval=2.0,
            emotion_model_path=emotion_model,
        )
        if vision.start():
            import time
            time.sleep(1.0)
            status = vision.get_status()
            emotion_str = "有効" if status["emotion_detection"] else "顔検出のみ"
            logger.info("✅ Vision OK (カメラ起動, 感情推定: %s)", emotion_str)
        else:
            logger.warning("カメラを開けません (Visionなしで続行)")
            vision = None
    except Exception as e:
        logger.warning("Vision 初期化失敗 (Visionなしで続行): %s", e)
        vision = None

    # Screen 初期化 (画面認識: スクリーンショット → VLM描写)
    # デフォルト無効。WEB_SCREEN_CONTEXT_ENABLED=true のときだけ起動する。
    if os.environ.get("WEB_SCREEN_CONTEXT_ENABLED", "").lower() == "true":
        logger.info("[+] Screen (画面認識) 初期化...")
        try:
            # SCREEN_CONTEXT_MODE (local|remote) でローカル/リモートを切替
            screen = create_screen_context(
                analysis_interval=90.0,
                base_url=config.ollama_base_url,
                model=config.model,
            )
            if screen.start():
                status = screen.get_status()
                mode = status.get("mode", "local")
                detail = (
                    f"VLM: {status['model']}, 解析間隔: {status['analysis_interval']:.0f}秒"
                    if mode == "local"
                    else "remote: data/screen/latest.json を読取"
                )
                logger.info("✅ Screen OK (%s)", detail)
            else:
                logger.warning("画面をキャプチャできません (DISPLAY未設定? Screenなしで続行)")
                screen = None
        except Exception as e:
            logger.warning("Screen 初期化失敗 (Screenなしで続行): %s", e)
            screen = None
    else:
        screen = None

    # Monitor 初期化 (Phase 6)
    logger.info("[6/7] Monitor (PCログ収集) 初期化...")
    try:
        monitor = MonitorContext(
            db_path=str(PROJECT_ROOT / "data" / "metrics" / "system_metrics.db"),
            collect_interval=30.0,
        )
        if monitor.start():
            logger.info("✅ Monitor OK (メトリクス収集開始)")
        else:
            logger.warning("Monitor 起動失敗 (Monitorなしで続行)")
            monitor = None
    except Exception as e:
        logger.warning("Monitor 初期化失敗 (Monitorなしで続行): %s", e)
        monitor = None

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

    # IdleManager 初期化
    idle_manager = IdleManager()
    idle_manager.start(monitor_context=monitor, vision_context=vision)
    if idle_manager.gpu_power_control_enabled:
        logger.info("✅ IdleManager OK (GPU電力の動的切替有効)")
    else:
        logger.info("✅ IdleManager OK (GPU電力制御は無効: %s)", idle_manager.gpu_power_control_reason)

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
        monitor.stop()
    if vision is not None:
        vision.stop()
    if screen is not None:
        screen.stop()
    if task_calendar_sync is not None:
        task_calendar_sync.stop()
    if task_store is not None:
        task_store.close()
    llm.close()
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
        ollama_url=config.ollama_base_url if config else "http://localhost:11434",
    )
    result = checker.check_all(include_web=False)

    # モジュール稼働状況を追加
    result["modules"] = {
        "ollama": llm is not None and llm.is_available() if llm else False,
        "tts": tts is not None and tts.is_loaded(),
        "stt": stt is not None and stt.is_loaded(),
        "rag": rag is not None,
        "vision": vision is not None and vision.is_running,
        "monitor": monitor is not None and monitor.is_running,
        "persona": profile is not None,
        "growth": growth_tracker is not None,
        "idle_manager": idle_manager is not None and idle_manager.is_running,
    }

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
    return {
        "ollama": llm.is_available() if llm else False,
        "model": config.model if config else None,
        "tts": tts is not None and tts.is_loaded(),
        "tts_backend": backend_name(tts),
        "tts_voice": tts.voice if tts else None,
        "tts_voices": tts.list_ja_voices() if tts else {},
        "stt": stt is not None and stt.is_loaded(),
        "stt_model": stt.model_size if stt else None,
        "secure_web_url": get_secure_web_url(),
        "rag": rag is not None,
        "rag_stats": rag.get_stats() if rag else None,
        "vision": vision is not None and vision.is_running,
        "vision_status": vision.get_status() if vision else None,
        "monitor": monitor is not None and monitor.is_running,
        "monitor_status": monitor.get_status() if monitor else None,
        "persona": profile is not None,
        "persona_status": preloader.get_status() if preloader else None,
        "growth": growth_tracker is not None,
        "idle_manager": idle_manager.get_status() if idle_manager else None,
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
        if result["due_granularity"] == "date":
            due_display = local_due.strftime("%-m/%-d")
        else:  # datetime
            due_display = local_due.strftime("%-m/%-d %H:%M")
    
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
    wav_data = await loop.run_in_executor(None, tts.synthesize, text)

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
        return JSONResponse({"error": f"STT error: {e}"}, status_code=500)


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
            raise RuntimeError(f"ffmpeg error: {result.stderr.decode()[:200]}")

        audio = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
        return audio
    finally:
        os.unlink(tmp_path)


@app.get("/api/vision/status")
async def vision_status():
    """映像入力の状態"""
    if vision is None:
        return {"enabled": False}
    return {"enabled": True, **vision.get_status()}


@app.get("/api/vision/snapshot")
async def vision_snapshot():
    """現在のカメラ画像をJPEGで取得"""
    if vision is None or not vision.is_running:
        return JSONResponse({"error": "Vision not available"}, status_code=503)

    jpeg = vision.camera.get_jpeg(quality=75)
    if jpeg is None:
        return JSONResponse({"error": "No frame available"}, status_code=503)

    return Response(content=jpeg, media_type="image/jpeg")


@app.get("/api/vision/context")
async def vision_context_text():
    """現在の映像コンテキストテキスト（デバッグ用）"""
    if vision is None:
        return {"context": "", "enabled": False}
    return {"context": vision.get_context_text(), "enabled": True, **vision.get_status()}


# --- Screen API (画面認識) ---

# リモート画面 push (メインPC の scripts/screen_agent.py → ingest) の保存先。
SCREEN_DIR = PROJECT_ROOT / "data" / "screen"
SCREEN_LATEST_JPG = SCREEN_DIR / "latest.jpg"
SCREEN_LATEST_JSON = SCREEN_DIR / "latest.json"
MAX_INGEST_BYTES = 8 * 1024 * 1024  # 8MB

# ingest 用 VLM describer (遅延生成 / テストで差し替え可能)。
screen_ingest_describer = None
# 描写の単一飛行制御: 描写実行中に来た ingest は画像保存のみで描写はスキップ。
_ingest_describe_lock = threading.Lock()
_ingest_describing = False


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _get_ingest_describer():
    """ingest 用の ScreenDescriber を遅延生成 (テストで screen_ingest_describer に差し替え可)。"""
    global screen_ingest_describer
    if screen_ingest_describer is None:
        from src.screen.describer import ScreenDescriber
        base = config.ollama_base_url if config else "http://localhost:11434"
        model = config.model if config else None
        screen_ingest_describer = ScreenDescriber(base_url=base, model=model)
    return screen_ingest_describer


def _describe_ingested(jpeg: bytes, received_at: float) -> None:
    """受信画像を VLM で 1 回描写し latest.json をアトミック書き込み。失敗はログのみ。"""
    global _ingest_describing
    try:
        describer = _get_ingest_describer()
        description = describer.describe(jpeg)
        if description:
            payload = {
                "description": description,
                "captured_at": received_at,
                "described_at": time.time(),
                "source": "remote",
            }
            _atomic_write_text(SCREEN_LATEST_JSON, json.dumps(payload, ensure_ascii=False))
        else:
            logger.warning("screen ingest: 描写が空でした (次の ingest で再試行)")
    except Exception as e:
        logger.warning("screen ingest describe failed: %s", e)
    finally:
        with _ingest_describe_lock:
            _ingest_describing = False


def _read_ingest_status() -> dict:
    """latest.json の内容と鮮度を返す (ファイル無し/壊れは available=False)。"""
    token_configured = bool(os.environ.get("SCREEN_INGEST_TOKEN"))
    info = {
        "token_configured": token_configured,
        "jpg_exists": SCREEN_LATEST_JPG.exists(),
        "available": False,
    }
    try:
        if SCREEN_LATEST_JSON.exists():
            data = json.loads(SCREEN_LATEST_JSON.read_text(encoding="utf-8"))
            captured_at = float(data.get("captured_at") or 0.0)
            info.update({
                "available": True,
                "description": data.get("description", ""),
                "captured_at": captured_at,
                "described_at": data.get("described_at"),
                "source": data.get("source", "remote"),
                "age_seconds": (time.time() - captured_at) if captured_at > 0 else None,
            })
    except Exception:
        pass
    return info


@app.get("/api/screen/status")
async def screen_status():
    """画面認識の状態 (local/remote いずれも)。remote の latest.json 鮮度も返す。"""
    result: dict = {"enabled": screen is not None}
    if screen is not None:
        result["context"] = screen.get_context_text()
        result.update(screen.get_status())
    # リモート push (ingest) の受信状況は screen コンテキストの有無に関わらず返す
    result["ingest"] = _read_ingest_status()
    return result


@app.post("/api/screen/ingest")
async def screen_ingest(request: Request):
    """メインPC のキャプチャエージェントから生 JPEG を受信して保存・描写する。

    認証: env SCREEN_INGEST_TOKEN と X-Screen-Token ヘッダの一致 (compare_digest)。
          env 未設定なら常に 403 (安全側デフォルト)。
    ボディ: 生 JPEG バイト (Content-Type: image/jpeg)。上限 8MB。
    レスポンス: 200 {"ok": true, "described": <この受信で描写を開始したか>}
    """
    token = os.environ.get("SCREEN_INGEST_TOKEN")
    if not token:
        return JSONResponse({"error": "ingest disabled (SCREEN_INGEST_TOKEN unset)"}, status_code=403)
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
        SCREEN_DIR.mkdir(parents=True, exist_ok=True)
        _atomic_write_bytes(SCREEN_LATEST_JPG, body)
    except Exception as e:
        return JSONResponse({"error": f"save failed: {e}"}, status_code=500)

    # 単一飛行: 描写実行中でなければ開始 (実行中なら画像保存のみでスキップ)
    global _ingest_describing
    started = False
    with _ingest_describe_lock:
        if not _ingest_describing:
            _ingest_describing = True
            started = True
    if started:
        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, _describe_ingested, body, received_at)

    return {"ok": True, "described": started}


# --- Monitor API (Phase 6) ---

@app.get("/api/monitor/status")
async def monitor_status():
    """PCモニターの状態"""
    if monitor is None:
        return {"enabled": False}
    return {"enabled": True, **monitor.get_status()}


@app.get("/api/monitor/context")
async def monitor_context_text():
    """現在のPCモニターコンテキストテキスト（デバッグ用）"""
    if monitor is None:
        return {"context": "", "enabled": False}
    return {"context": monitor.get_context_text(), "enabled": True}


@app.get("/api/monitor/summary")
async def monitor_summary(minutes: int = 60):
    """直近N分のメトリクスサマリー"""
    if monitor is None:
        return JSONResponse({"error": "Monitor not available"}, status_code=503)
    return monitor.get_recent_summary(minutes=minutes)


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
) -> None:
    """LLMを介さないタスク/予定操作も、通常の会話と同じ形で保存・配信する。"""
    session = get_or_create_session(session_id)
    session.add_user_message(user_text)
    session.add_assistant_message(reply)
    try:
        session.save()
    except Exception:
        pass
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
            await websocket.send_json({
                "type": "error",
                "message": f"TTS error: {e}",
            })


def _new_chat_session() -> ChatSession:
    """現在の依存コンポーネントを注入した ChatSession を新規作成する。"""
    return ChatSession(
        system_prompt=config.system_prompt,
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


@app.get("/api/chat/resume")
async def chat_resume(session_id: str | None = None):
    """直近の会話を引き継ぐためのセッション情報を返す。

    - session_id 指定かつ安全: そのIDで復元 (不在なら空セッション、新IDは発行しない)
      不正ID: 400
    - 指定なし: 最新の有効履歴を引き継ぐ。無ければ新しい web_<ms> ID を発行
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

    # 保存IDも履歴も無い初回 → 新ID発行
    new_id = f"web_{int(time.time() * 1000)}"
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
            session.system_prompt = config.system_prompt
            session.emotion_tags = config.emotion_tag_enabled
            session.max_history_turns = config.max_history_turns
            session.session_id = session_id
            sessions[session_id] = session
            return session
        except Exception as e:
            logger.warning("session load failed for %s: %s", session_id, e)

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
        {"type": "error", "message": "..."}        # エラー
    """
    await websocket.accept()

    try:
        while True:
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
                        "message": f"STT error: {e}",
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
                await _send_direct_chat_reply(
                    websocket,
                    session_id=session_id,
                    user_text=user_text,
                    reply=task_reply,
                    want_tts=want_tts,
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
            messages = session.build_messages()

            # ストリーミング応答生成
            loop = asyncio.get_event_loop()
            full_response = ""
            # 感情タグフィルタ (有効時のみ)。クライアントにはタグを見せない。
            emo_filter = (
                EmotionTagStreamFilter() if config.emotion_tag_enabled else None
            )

            try:
                # queue ベースのリアルタイムストリーミング
                token_queue = llm.generate_stream_queue(
                    messages,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    top_k=config.top_k,
                    repeat_penalty=config.repeat_penalty,
                    num_ctx=config.num_ctx,
                    num_predict=config.num_predict,
                )

                while True:
                    try:
                        token = await asyncio.get_event_loop().run_in_executor(
                            None, token_queue.get, True, 300.0
                        )
                    except Exception:
                        break

                    if token is None:
                        # ストリーム終了
                        break
                    if isinstance(token, Exception):
                        raise token

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
                    logger.warning("会話履歴の保存に失敗: %s", e)

                await websocket.send_json({
                    "type": "done",
                    "full_text": full_response,
                })

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
                        await websocket.send_json({
                            "type": "error",
                            "message": f"TTS error: {e}",
                        })

            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "message": str(e),
                })
                # ユーザーメッセージを巻き戻す
                if session._messages and session._messages[-1]["role"] == "user":
                    session._messages.pop()
            finally:
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
