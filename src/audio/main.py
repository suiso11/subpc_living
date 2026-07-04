"""
Phase 3: 音声対話 CLIメインエントリポイント
STT + LLM + TTS のパイプラインを統合した音声対話
"""
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# --- ANSI カラーコード ---
class Color:
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def main():
    parser = argparse.ArgumentParser(description="Phase 3: 音声対話")
    parser.add_argument("--stt-model", default="small", help="Whisperモデルサイズ (tiny/base/small/medium)")
    parser.add_argument("--tts-voice", default="jf_alpha", help="kokoro-onnx 音声名 (jf_alpha/jm_kumo等)")
    parser.add_argument("--text-mode", action="store_true", help="テキスト入力モード (マイクなし)")
    parser.add_argument("--vad", default="auto", choices=["auto", "silero", "energy"],
                        help="VAD方式: auto(Silero優先), silero, energy (default: auto)")
    parser.add_argument("--no-streaming-tts", action="store_true",
                        help="ストリーミングTTSを無効化 (全文完了後に音声合成)")
    parser.add_argument("--no-rag", action="store_true",
                        help="RAG (長期記憶) を無効化")
    parser.add_argument("--no-vision", action="store_true",
                        help="Vision (映像入力) を無効化")
    parser.add_argument("--camera-id", type=int, default=0,
                        help="カメラデバイスID (default: 0)")
    parser.add_argument("--screen", action="store_true",
                        help="Screen (画面認識: スクリーンショット→VLM描写) を有効化")
    parser.add_argument("--no-monitor", action="store_true",
                        help="Monitor (PCログ収集) を無効化")
    parser.add_argument("--no-persona", action="store_true",
                        help="Persona (パーソナライズ) を無効化")
    parser.add_argument("--wakeword", action="store_true",
                        help="ウェイクワードモードを有効化 (呼びかけで起動)")
    parser.add_argument("--wakeword-model", default="hey_jarvis",
                        help="ウェイクワードモデル名 (default: hey_jarvis)")
    parser.add_argument("--wakeword-threshold", type=float, default=0.5,
                        help="ウェイクワード検知の閾値 (0.0〜1.0, default: 0.5)")
    args = parser.parse_args()

    print(f"""
{Color.CYAN}{Color.BOLD}╔══════════════════════════════════════════╗
║   subpc_living — 音声対話 (Phase 3)       ║
╚══════════════════════════════════════════╝{Color.RESET}
""")

    if args.text_mode:
        # テキスト入力 → TTS再生モード（マイクなしでTTSをテスト可能）
        run_text_to_speech_mode(args)
    else:
        # フル音声対話モード
        run_voice_mode(args)


def run_voice_mode(args):
    """フル音声対話モード: マイク → STT → LLM → TTS → スピーカー"""
    from src.audio.pipeline import VoicePipeline
    from src.chat.config import ChatConfig

    config = ChatConfig.load(PROJECT_ROOT / "config" / "chat_config.json")

    # ウェイクワードモデル名をリストに変換
    wakeword_models = [args.wakeword_model] if args.wakeword else None

    pipeline = VoicePipeline(
        chat_config=config,
        stt_model=args.stt_model,
        tts_voice=args.tts_voice,
        vad_type=args.vad,
        streaming_tts=not args.no_streaming_tts,
        enable_rag=not args.no_rag,
        enable_vision=not args.no_vision,
        camera_id=args.camera_id,
        enable_screen=args.screen,
        enable_monitor=not args.no_monitor,
        enable_persona=not args.no_persona,
        enable_wakeword=args.wakeword,
        wakeword_models=wakeword_models,
        wakeword_threshold=args.wakeword_threshold,
    )

    if not pipeline.initialize():
        print(f"\n{Color.RED}初期化に失敗しました。{Color.RESET}")
        sys.exit(1)

    pipeline.run_interactive()


def run_text_to_speech_mode(args):
    """テキスト入力 → LLM応答 → TTS再生モード"""
    from src.audio.audio_io import AudioPlayer
    from src.audio.tts_factory import backend_name, create_tts_backend
    from src.chat.client import OllamaClient
    from src.chat.session import ChatSession
    from src.chat.config import ChatConfig
    from src.chat.web_search import create_web_search_context

    config = ChatConfig.load(PROJECT_ROOT / "config" / "chat_config.json")
    web_search = create_web_search_context(config)

    # TTS初期化
    tts = create_tts_backend(
        models_dir=PROJECT_ROOT / "models" / "tts" / "kokoro",
        voice=args.tts_voice,
    )
    tts.load()

    player = AudioPlayer(sample_rate=24000)

    # LLM初期化
    client = OllamaClient(base_url=config.ollama_base_url, model=config.model)
    if not client.is_available():
        print(f"{Color.RED}Ollamaに接続できません{Color.RESET}")
        sys.exit(1)

    session = ChatSession(
        system_prompt=config.system_prompt,
        max_history_turns=config.max_history_turns,
        history_dir=str(PROJECT_ROOT / config.history_dir),
        web_search=web_search,
    )

    print(f"{Color.DIM}テキスト入力 → LLM応答 → 音声再生モード{Color.RESET}")
    print(f"{Color.DIM}モデル: {config.model}{Color.RESET}")
    print(f"{Color.DIM}TTS: {backend_name(tts)} / {tts.voice}{Color.RESET}")
    print(f"{Color.YELLOW}Ctrl+C で終了{Color.RESET}\n")

    try:
        while True:
            user_input = input(f"{Color.GREEN}{Color.BOLD}あなた> {Color.RESET}").strip()
            if not user_input:
                continue
            if user_input in ("/quit", "/exit"):
                break

            session.add_user_message(user_input)
            messages = session.build_messages()

            # LLM応答生成
            print(f"{Color.CYAN}{Color.BOLD}AI> {Color.RESET}", end="", flush=True)
            response = ""
            for token in client.generate_stream(
                messages,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                num_ctx=config.num_ctx,
                repeat_penalty=config.repeat_penalty,
                num_predict=config.num_predict,
            ):
                print(token, end="", flush=True)
                response += token
            print()

            session.add_assistant_message(response)

            # TTS再生
            print(f"{Color.DIM}🔊 読み上げ中...{Color.RESET}")
            try:
                wav_data = tts.synthesize(response)
                player.play_wav(wav_data, blocking=True)
            except Exception as e:
                print(f"{Color.RED}TTS再生エラー: {e}{Color.RESET}")

    except KeyboardInterrupt:
        print(f"\n{Color.YELLOW}終了します...{Color.RESET}")

    if session.turn_count > 0:
        saved = session.save()
        print(f"{Color.DIM}会話を保存しました: {saved}{Color.RESET}")
    client.close()


if __name__ == "__main__":
    main()
