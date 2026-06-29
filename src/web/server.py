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
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, Response, FileResponse

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.chat.client import OllamaClient
from src.chat.session import ChatSession
from src.chat.config import ChatConfig
from src.chat.web_search import WebSearchContext, create_web_search_context
from src.audio.tts import KokoroTTS
from src.audio.stt import WhisperSTT
from src.memory.vectorstore import VectorStore
from src.memory.rag import RAGRetriever
from src.vision.context import VisionContext
from src.monitor.context import MonitorContext
from src.persona.profile import UserProfile
from src.persona.summarizer import ConversationSummarizer
from src.persona.preloader import SessionPreloader
from src.service.healthcheck import HealthChecker
from src.service.idle import IdleManager


# --- グローバル状態 ---
config: ChatConfig = None
llm: OllamaClient = None
tts: KokoroTTS = None
stt: WhisperSTT = None
rag: RAGRetriever = None
vision: VisionContext = None
monitor: MonitorContext = None
profile: UserProfile = None
summarizer: ConversationSummarizer = None
preloader: SessionPreloader = None
web_search: Optional[WebSearchContext] = None
sessions: dict[str, ChatSession] = {}
idle_manager: Optional[IdleManager] = None


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
    global config, llm, tts, stt, rag, vision, monitor, profile, summarizer, preloader, web_search, idle_manager

    print("=" * 50)
    print(" Web UI サーバー起動中...")
    print("=" * 50)

    # GPU構成表示 (Phase 10)
    try:
        from src.service.gpu_config import get_device_config
        gpu_cfg = get_device_config()
        if gpu_cfg.gpus:
            print(f"  GPU構成: {gpu_cfg.profile}")
            for g in gpu_cfg.gpus:
                print(f"    GPU{g.index}: {g.name} ({g.vram_gb}GB)")
            if gpu_cfg.profile == "dual_gpu":
                print(f"    LLM → GPU{gpu_cfg.llm_gpu_index} / 推論 → GPU{gpu_cfg.inference_gpu_index}")
    except Exception:
        pass

    # 設定ロード
    config_path = PROJECT_ROOT / "config" / "chat_config.json"
    config = ChatConfig.load(config_path)
    web_search = create_web_search_context(config)

    # LLM 初期化
    print("[1/6] Ollama 接続確認...")
    llm = OllamaClient(base_url=config.ollama_base_url, model=config.model)
    if not llm.is_available():
        print("⚠️  Ollamaに接続できません。チャット機能は使用不可です。")
    else:
        print(f"✅ Ollama OK (model: {config.model})")
    if web_search is not None:
        print(f"✅ Web検索 ON (auto={config.web_search_auto}, max_results={config.web_search_max_results})")

    # STT 初期化
    print("[2/7] STT 初期化...")
    try:
        stt = WhisperSTT(model_size="auto", language="ja", device="auto")
        stt.load()
        print(f"✅ STT OK (model: {stt.model_size}, device: {stt.device})")
    except Exception as e:
        print(f"⚠️  STT ロード失敗: {e}")
        stt = None

    # TTS 初期化
    print("[3/7] TTS 初期化...")
    tts = KokoroTTS(models_dir=PROJECT_ROOT / "models" / "tts" / "kokoro")
    try:
        tts.load()
        print("✅ TTS OK (kokoro-onnx)")
    except Exception as e:
        print(f"⚠️  TTS ロード失敗: {e}")
        tts = None

    # RAG 初期化 (Phase 4)
    print("[4/7] RAG (長期記憶) 初期化...")
    try:
        vector_store = VectorStore(
            persist_dir=str(PROJECT_ROOT / "data" / "vectordb"),
        )
        vector_store.initialize()
        rag = RAGRetriever(vector_store=vector_store)
        stats = rag.get_stats()
        print(f"✅ RAG OK (会話: {stats['conversations']}件, 知識: {stats['knowledge']}件)")
    except Exception as e:
        print(f"⚠️  RAG 初期化失敗 (RAGなしで続行): {e}")
        rag = None

    # Vision 初期化 (Phase 5)
    print("[5/7] Vision (映像入力) 初期化...")
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
            print(f"✅ Vision OK (カメラ起動, 感情推定: {emotion_str})")
        else:
            print("⚠️  カメラを開けません (Visionなしで続行)")
            vision = None
    except Exception as e:
        print(f"⚠️  Vision 初期化失敗 (Visionなしで続行): {e}")
        vision = None

    # Monitor 初期化 (Phase 6)
    print("[6/7] Monitor (PCログ収集) 初期化...")
    try:
        monitor = MonitorContext(
            db_path=str(PROJECT_ROOT / "data" / "metrics" / "system_metrics.db"),
            collect_interval=30.0,
        )
        if monitor.start():
            print("✅ Monitor OK (メトリクス収集開始)")
        else:
            print("⚠️  Monitor 起動失敗 (Monitorなしで続行)")
            monitor = None
    except Exception as e:
        print(f"⚠️  Monitor 初期化失敗 (Monitorなしで続行): {e}")
        monitor = None

    # Persona 初期化 (Phase 7)
    print("[7/7] Persona (パーソナライズ) 初期化...")
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
        print(f"✅ Persona OK (名前: {profile_name}, 事実: {facts_count}件, 今日の予定: {today_count}件)")
    except Exception as e:
        print(f"⚠️  Persona 初期化失敗 (Personaなしで続行): {e}")
        profile = None
        summarizer = None
        preloader = None

    # IdleManager 初期化
    idle_manager = IdleManager()
    idle_manager.start(monitor_context=monitor, vision_context=vision)
    if idle_manager.gpu_power_control_enabled:
        print("✅ IdleManager OK (GPU電力の動的切替有効)")
    else:
        print(f"✅ IdleManager OK (GPU電力制御は無効: {idle_manager.gpu_power_control_reason})")

    local_ip = get_local_ip()
    print()
    print("=" * 50)
    print(f" ✅ サーバー起動完了!")
    print(f" PC:     http://localhost:8000")
    print(f" スマホ:  http://{local_ip}:8000")
    print("=" * 50)
    print()

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
    llm.close()
    print("サーバーを終了しました。")


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
        "tts_voice": tts.voice if tts else None,
        "tts_voices": KokoroTTS.list_ja_voices(),
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
    }


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
    if voice not in KokoroTTS.JA_VOICES:
        return JSONResponse({"error": f"Unknown voice: {voice}"}, status_code=400)

    tts.set_voice(voice)
    return {"voice": voice, "description": KokoroTTS.JA_VOICES[voice]}


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
            monitor_context=monitor,
            preloader=preloader,
            web_search=web_search,
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

            try:
                # queue ベースのリアルタイムストリーミング
                token_queue = llm.generate_stream_queue(
                    messages,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    top_k=config.top_k,
                    repeat_penalty=config.repeat_penalty,
                    num_ctx=config.num_ctx,
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

                    full_response += token
                    await websocket.send_json({"type": "token", "content": token})

                session.add_assistant_message(full_response)

                await websocket.send_json({
                    "type": "done",
                    "full_text": full_response,
                })

                # TTS
                if want_tts and tts is not None and full_response:
                    try:
                        wav_data = await loop.run_in_executor(
                            None, tts.synthesize, full_response
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
