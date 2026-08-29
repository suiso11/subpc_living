"""
Phase 3: 音声対話 CLIメインエントリポイント
STT + LLM + TTS のパイプラインを統合した音声対話
"""
import sys
import argparse
import logging
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.perception import resolve_sensor_policy

logger = logging.getLogger("audio.main")
_activity_runtime = None


def _start_companion_activity_runtime(env=None):
    """COMPANION_ACTIVITY_ENABLED=true のときだけ活動収集ランタイムを起動して保持する。"""
    from src.perception import create_activity_runtime_from_env

    global _activity_runtime
    _activity_runtime = create_activity_runtime_from_env(
        os.environ if env is None else env, logger=logger
    )
    return _activity_runtime


def _stop_companion_activity_runtime():
    """保持中の活動収集ランタイムを停止し、参照を破棄する。"""
    global _activity_runtime
    runtime = _activity_runtime
    _activity_runtime = None
    if runtime is not None:
        runtime.stop()


# --- ANSI カラーコード ---
class Color:
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def build_parser():
    """音声対話CLIの argparse パーサーを構築する。"""
    parser = argparse.ArgumentParser(description="Phase 3: 音声対話")
    parser.add_argument("--stt-model", default="small", help="Whisperモデルサイズ (tiny/base/small/medium)")
    parser.add_argument("--tts-voice", default="jf_alpha", help="kokoro-onnx 音声名 (jf_alpha/jm_kumo等)")
    parser.add_argument("--text-mode", action="store_true",
                        help="テキスト入力モード (マイクなし・センサー不使用)")
    parser.add_argument("--vad", default="auto", choices=["auto", "silero", "energy"],
                        help="VAD方式: auto(Silero優先), silero, energy (default: auto)")
    parser.add_argument("--no-streaming-tts", action="store_true",
                        help="ストリーミングTTSを無効化 (全文完了後に音声合成)")
    parser.add_argument("--no-rag", action="store_true",
                        help="RAG (長期記憶) を無効化")
    parser.add_argument("--microphone", action="store_true",
                        help="マイク入力を有効化 (音声対話の必須同意。既定: 無効)")
    parser.add_argument("--camera", action="store_true",
                        help="カメラ (Vision) を有効化 (既定: 無効)")
    parser.add_argument("--monitor", action="store_true",
                        help="Monitor (PCログ収集) を有効化 (既定: 無効)")
    parser.add_argument("--no-vision", action="store_true",
                        help="(非推奨) Vision (映像入力) を明示的に無効化。--camera や SENSOR_CAMERA_ENABLED より優先")
    parser.add_argument("--camera-id", type=int, default=0,
                        help="カメラデバイスID (default: 0)")
    parser.add_argument("--screen", action="store_true",
                        help="Screen (画面認識: スクリーンショット→VLM描写) を有効化 (既定: 無効)")
    parser.add_argument("--no-monitor", action="store_true",
                        help="(非推奨) Monitor (PCログ収集) を明示的に無効化。--monitor や SENSOR_MONITOR_ENABLED より優先")
    parser.add_argument("--no-persona", action="store_true",
                        help="Persona (パーソナライズ) を無効化")
    parser.add_argument("--wakeword", action="store_true",
                        help="ウェイクワードモードを有効化 (呼びかけで起動。マイクの同意が必要)")
    parser.add_argument("--wakeword-model", default="hey_jarvis",
                        help="ウェイクワードモデル名 (default: hey_jarvis)")
    parser.add_argument("--wakeword-threshold", type=float, default=0.5,
                        help="ウェイクワード検知の閾値 (0.0〜1.0, default: 0.5)")
    return parser


_SENSOR_ORDER = ("microphone", "camera", "screen_capture", "monitor", "activity")


def _resolve_sensor_flags(args, env=None):
    """CLIフラグと SensorPolicy (env) を合成して起動時のセンサー有効状態を返す。

    - CLI の明示フラグ (--microphone/--camera/--screen/--monitor) は、その起動の
      一回限りの明示同意として有効化する (永続化しない)。
    - canonical の SENSOR_*_ENABLED=true も env 側の同意として有効化する。
    - 非推奨の --no-vision / --no-monitor は明示的な無効上書きとして常に優先する。
    - 返り値は boolean と sensor source 名のみ (env 名・env 値は含めない)。
    """
    policy = resolve_sensor_policy(env)
    return {
        "microphone": getattr(args, "microphone", False) or policy.microphone,
        "camera": (getattr(args, "camera", False) or policy.camera)
        and not getattr(args, "no_vision", False),
        "screen_capture": getattr(args, "screen", False) or policy.screen_capture,
        "monitor": (getattr(args, "monitor", False) or policy.monitor)
        and not getattr(args, "no_monitor", False),
        "activity": policy.activity,
    }


def _print_sensor_summary(sensor_flags, text_mode=False):
    """起動時に有効化されたセンサー名だけを表示する privacy-safe サマリー。"""
    if text_mode:
        print(f"{Color.DIM}🔒 センサー: テキストモード (センサー不使用){Color.RESET}")
        return
    enabled = [name for name in _SENSOR_ORDER if sensor_flags.get(name)]
    if enabled:
        print(f"{Color.DIM}🔒 有効センサー: {', '.join(enabled)}{Color.RESET}")
    else:
        print(f"{Color.DIM}🔒 有効センサー: なし{Color.RESET}")


def main():
    parser = build_parser()
    args = parser.parse_args()

    print(f"""
{Color.CYAN}{Color.BOLD}╔══════════════════════════════════════════╗
║   subpc_living — 音声対話 (Phase 3)       ║
╚══════════════════════════════════════════╝{Color.RESET}
""")

    sensor_flags = _resolve_sensor_flags(args)

    # 音声対話モードはマイクの明示同意 (フラグ or env) がない限り、活動収集
    # ランタイムや VoicePipeline・STT・音声デバイス作成より前に失敗させる。
    # ウェイクワードもマイクが必要なため同じゲートに従う。テキストモードは
    # マイクを要求・開放せず、マイク同意も無視する。
    if not args.text_mode and not sensor_flags["microphone"]:
        print(f"\n{Color.RED}マイクが有効化されていません。{Color.RESET}")
        print(f"{Color.DIM}音声対話にはマイクの明示的な同意が必要です。{Color.RESET}")
        print(f"{Color.DIM}  起動ごとに同意する: --microphone フラグ{Color.RESET}")
        print(f"{Color.DIM}  常時同意する:       SENSOR_MICROPHONE_ENABLED=true を設定{Color.RESET}")
        print(f"{Color.DIM}マイクを使わない:     --text-mode を指定{Color.RESET}")
        sys.exit(1)

    _print_sensor_summary(sensor_flags, text_mode=args.text_mode)

    if args.text_mode:
        # テキスト入力 → TTS再生モード（マイクなしでTTSをテスト可能）
        # テキストモードは SENSOR_ACTIVITY_ENABLED=true でも活動収集を構築・開始しない。
        run_text_to_speech_mode(args)
        return

    # 音声対話モードでのみ活動収集ランタイムを起動・停止する。
    _start_companion_activity_runtime()
    try:
        # フル音声対話モード
        run_voice_mode(args, sensor_flags=sensor_flags)
    finally:
        _stop_companion_activity_runtime()


def run_voice_mode(args, sensor_flags=None):
    """フル音声対話モード: マイク → STT → LLM → TTS → スピーカー"""
    from src.audio.pipeline import VoicePipeline
    from src.chat.config import ChatConfig

    config = ChatConfig.load(PROJECT_ROOT / "config" / "chat_config.json")

    # ウェイクワードモデル名をリストに変換
    wakeword_models = [args.wakeword_model] if args.wakeword else None

    if sensor_flags is None:
        sensor_flags = _resolve_sensor_flags(args)

    pipeline = VoicePipeline(
        chat_config=config,
        stt_model=args.stt_model,
        tts_voice=args.tts_voice,
        vad_type=args.vad,
        streaming_tts=not args.no_streaming_tts,
        enable_rag=not args.no_rag,
        enable_vision=sensor_flags["camera"],
        camera_id=args.camera_id,
        enable_screen=sensor_flags["screen_capture"],
        enable_monitor=sensor_flags["monitor"],
        enable_persona=not args.no_persona,
        enable_wakeword=args.wakeword,
        wakeword_models=wakeword_models,
        wakeword_threshold=args.wakeword_threshold,
        activity_runtime=_activity_runtime,
    )

    if not pipeline.initialize():
        print(f"\n{Color.RED}初期化に失敗しました。{Color.RESET}")
        # セーフティネット: 初期化失敗時も作成済みリソースを確実に解放してから
        # 終了する。initialize() 内で既に解放済みでも冪等なため二重解放しない。
        pipeline.cleanup()
        sys.exit(1)

    pipeline.run_interactive()


def _stream_assistant_response(service, request, blocks, *, base_system: str, on_token=None) -> str:
    """LLM応答をストリームで受け取り全文を返す。

    StreamResult は正常終了・空応答・生成例外のすべての経路で finally で必ず
    1回だけ close する (StreamResult.close は冪等)。close の失敗は主経路の例外
    (生成例外など) を上書きしないよう握りつぶして続行する。
    respond_stream 自体が失敗した場合は stream が生成されないため close しない。
    on_token が指定されていればトークン到着ごとに呼ぶ (描画など)。
    """
    stream = None
    try:
        stream = service.respond_stream(request, blocks, base_system=base_system)
        response = ""
        for token in stream:
            response += token
            if on_token is not None:
                on_token(token)
        return response
    finally:
        if stream is not None:
            try:
                stream.close()
            except Exception:
                pass


def run_text_to_speech_mode(args):
    """テキスト入力 → LLM応答 → TTS再生モード"""
    from src.assistant.factory import build_local_service
    from src.assistant.requests import create_request
    from src.audio.audio_io import AudioPlayer
    from src.audio.tts_factory import backend_name, create_tts_backend
    from src.chat.session import ChatSession
    from src.chat.config import ChatConfig, validate_local_provider_kind
    from src.chat.web_search import create_web_search_context
    from src.growth.tracker import GrowthTracker

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
    service, registry = build_local_service(config)
    client = registry.get(config.resolved_local_provider_id()).provider

    if not client.is_available():
        print(f"{Color.RED}ローカル推論サーバーに接続できません。サービスが起動しているか確認してください。{Color.RESET}")
        registry.close()
        sys.exit(1)

    if validate_local_provider_kind(config) == "ollama":
        # Ollama: /api/tags に基づく厳格なモデル存在確認 (has_model 1回で判定)
        if not client.has_model():
            print(f"{Color.RED}モデル '{config.model}' が見つかりません。{Color.RESET}")
            registry.close()
            sys.exit(1)
    else:
        # openai_compatible: /models はオプション。list_models() を1回だけ呼び、
        # 非空で設定モデルが含まれないときだけ失敗し、空なら生成時に検証して続行する。
        discovered = client.list_models()
        if discovered and config.model not in discovered:
            print(f"{Color.RED}モデル '{config.model}' が見つかりません。{Color.RESET}")
            print(f"{Color.DIM}利用可能なモデル: {', '.join(discovered)}{Color.RESET}")
            registry.close()
            sys.exit(1)
        if not discovered:
            print(f"{Color.YELLOW}⚠ モデル情報の取得に失敗しました。生成時に確認します。{Color.RESET}")

    try:
        growth_tracker = GrowthTracker(PROJECT_ROOT / "data" / "growth" / "growth.db")
    except Exception:
        growth_tracker = None
    session = ChatSession(
        system_prompt=config.effective_system_prompt(),
        max_history_turns=config.max_history_turns,
        history_dir=str(PROJECT_ROOT / config.history_dir),
        web_search=web_search,
        growth_tracker=growth_tracker,
        conversation_source="audio_text",
        emotion_tags=config.emotion_tag_enabled,
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
            # LLM応答生成
            print(f"{Color.CYAN}{Color.BOLD}AI> {Color.RESET}", end="", flush=True)
            try:
                blocks = session.build_blocks()
                request = create_request(
                    text=user_input,
                    conversation_id=session.session_id,
                    channel="voice",
                    profile="voice_fast",
                    privacy="local_only",
                )
                response = _stream_assistant_response(
                    service,
                    request,
                    blocks,
                    base_system=session.system_prompt,
                    on_token=lambda token: print(token, end="", flush=True),
                )
            except Exception as e:
                print()
                print(f"{Color.RED}AI応答の生成に失敗しました: {type(e).__name__}{Color.RESET}")
                session.rollback_last_user_message()
                continue
            print()

            session.add_assistant_message(response)

            # TTS再生
            print(f"{Color.DIM}🔊 読み上げ中...{Color.RESET}")
            try:
                wav_data = tts.synthesize(response)
                player.play_wav(wav_data, blocking=True)
            except Exception as e:
                print(f"{Color.RED}TTS再生エラー: {type(e).__name__}{Color.RESET}")

    except KeyboardInterrupt:
        print(f"\n{Color.YELLOW}終了します...{Color.RESET}")
    finally:
        # 成功ターンのみベストエフォートで保存し、保存失敗で主経路の失敗を隠さない。
        # registry は /quit / Ctrl+C / ループ・入力・保存の異常にかかわらず一度だけ閉じる。
        try:
            if session.turn_count > 0:
                saved = session.save()
                print(f"{Color.DIM}会話を保存しました: {saved}{Color.RESET}")
        except Exception as e:
            print(f"{Color.DIM}会話の保存に失敗しました: {type(e).__name__}{Color.RESET}")
        registry.close()


if __name__ == "__main__":
    main()
