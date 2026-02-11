"""
Web UIサーバー
スマホ・PC からLAN経由でアクセス可能なチャットインターフェース
FastAPI + WebSocket によるストリーミング対話
"""
import sys
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
from src.audio.tts import KokoroTTS
from src.memory.vectorstore import VectorStore
from src.memory.rag import RAGRetriever
from src.vision.context import VisionContext
from src.monitor.context import MonitorContext


# --- グローバル状態 ---
config: ChatConfig = None
llm: OllamaClient = None
tts: KokoroTTS = None
rag: RAGRetriever = None
vision: VisionContext = None
monitor: MonitorContext = None
sessions: dict[str, ChatSession] = {}


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
    global config, llm, tts, rag, vision, monitor

    print("=" * 50)
    print(" Web UI サーバー起動中...")
    print("=" * 50)

    # 設定ロード
    config_path = PROJECT_ROOT / "config" / "chat_config.json"
    config = ChatConfig.load(config_path)

    # LLM 初期化
    print("[1/5] Ollama 接続確認...")
    llm = OllamaClient(base_url=config.ollama_base_url, model=config.model)
    if not llm.is_available():
        print("⚠️  Ollamaに接続できません。チャット機能は使用不可です。")
    else:
        print(f"✅ Ollama OK (model: {config.model})")

    # TTS 初期化
    print("[2/5] TTS 初期化...")
    tts = KokoroTTS(models_dir=PROJECT_ROOT / "models" / "tts" / "kokoro")
    try:
        tts.load()
        print("✅ TTS OK (kokoro-onnx)")
    except Exception as e:
        print(f"⚠️  TTS ロード失敗: {e}")
        tts = None

    # RAG 初期化 (Phase 4)
    print("[3/5] RAG (長期記憶) 初期化...")
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
    print("[4/5] Vision (映像入力) 初期化...")
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
    print("[5/5] Monitor (PCログ収集) 初期化...")
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

    local_ip = get_local_ip()
    print()
    print("=" * 50)
    print(f" ✅ サーバー起動完了!")
    print(f" PC:     http://localhost:8000")
    print(f" スマホ:  http://{local_ip}:8000")
    print("=" * 50)
    print()

    yield

    # 終了処理
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


# --- REST API ---

@app.get("/api/status")
async def status():
    """システム状態"""
    return {
        "ollama": llm.is_available() if llm else False,
        "model": config.model if config else None,
        "tts": tts is not None and tts.is_loaded(),
        "tts_voice": tts.voice if tts else None,
        "tts_voices": KokoroTTS.list_ja_voices(),
        "rag": rag is not None,
        "rag_stats": rag.get_stats() if rag else None,
        "vision": vision is not None and vision.is_running,
        "vision_status": vision.get_status() if vision else None,
        "monitor": monitor is not None and monitor.is_running,
        "monitor_status": monitor.get_status() if monitor else None,
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
        )
    return sessions[session_id]


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocketチャット（ストリーミング応答）

    クライアント → サーバー:
        {"type": "message", "text": "...", "session_id": "...", "tts": true/false}

    サーバー → クライアント:
        {"type": "token", "content": "..."}       # ストリーミングトークン
        {"type": "done", "full_text": "..."}       # 応答完了
        {"type": "audio", "data": "base64..."}     # TTS音声 (base64 WAV)
        {"type": "error", "message": "..."}        # エラー
    """
    await websocket.accept()

    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)

            if data.get("type") != "message":
                continue

            user_text = data.get("text", "").strip()
            session_id = data.get("session_id", "default")
            want_tts = data.get("tts", False)

            if not user_text:
                continue

            session = get_or_create_session(session_id)
            session.add_user_message(user_text)
            messages = session.build_messages()

            # ストリーミング応答生成
            loop = asyncio.get_event_loop()
            full_response = ""

            try:
                # generate_stream は同期ジェネレータなので、スレッドで回す
                def collect_tokens():
                    tokens = []
                    for token in llm.generate_stream(
                        messages,
                        temperature=config.temperature,
                        top_p=config.top_p,
                        top_k=config.top_k,
                        num_ctx=config.num_ctx,
                        repeat_penalty=config.repeat_penalty,
                    ):
                        tokens.append(token)
                    return tokens

                tokens = await loop.run_in_executor(None, collect_tokens)

                for token in tokens:
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
