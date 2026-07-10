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
from datetime import datetime, timezone
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
tasks_timezone: str = "Asia/Tokyo"


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """サーバー起動/終了時の処理"""
    global config, llm, tts, stt, rag, vision, screen, monitor, profile, summarizer, preloader, web_search, idle_manager, task_store, tasks_timezone

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
        task_store = TaskStore(
            db_path=str(PROJECT_ROOT / "data" / "tasks" / "tasks.db"),
            timezone_name=tasks_timezone,
        ).initialize()
        logger.info("✅ Tasks OK (Webタスク管理有効)")
    except Exception as e:
        logger.warning("Tasks 初期化失敗 (タスク管理なしで続行): %s", e)
        task_store = None

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
        "idle_manager": idle_manager is not None and idle_manager.is_running,
    }

    status_code = 200 if result["status"] == "ok" else 503
    return JSONResponse(content=result, status_code=status_code)


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
        "rag": rag is not None,
        "rag_stats": rag.get_stats() if rag else None,
        "vision": vision is not None and vision.is_running,
        "vision_status": vision.get_status() if vision else None,
        "monitor": monitor is not None and monitor.is_running,
        "monitor_status": monitor.get_status() if monitor else None,
        "persona": profile is not None,
        "persona_status": preloader.get_status() if preloader else None,
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

def get_or_create_session(session_id: str) -> ChatSession:
    """セッションを取得または新規作成"""
    if session_id not in sessions:
        sessions[session_id] = ChatSession(
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
            emotion_tags=config.emotion_tag_enabled,
        )
    return sessions[session_id]


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
