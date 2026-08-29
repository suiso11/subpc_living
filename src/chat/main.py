"""
Phase 2: テキスト対話 CLIメインエントリポイント
ローカル推論backend (Ollama または OpenAI互換) を使ったインタラクティブなテキスト対話を実現する
"""
import sys
import signal
from pathlib import Path

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.assistant import AssistantService
from src.assistant.factory import build_local_service
from src.assistant.requests import create_request
from src.chat.config import ChatConfig, validate_local_provider_kind
from src.chat.session import ChatSession
from src.chat.web_search import create_web_search_context
from src.growth.tracker import GrowthTracker
from src.llm import ProviderRegistry


# --- ANSI カラーコード ---
class Color:
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def print_banner():
    """起動バナーを表示"""
    print(f"""
{Color.CYAN}{Color.BOLD}╔══════════════════════════════════════════╗
║   subpc_living — テキスト対話 (Phase 2)  ║
╚══════════════════════════════════════════╝{Color.RESET}
""")


def print_help():
    """コマンド一覧を表示"""
    print(f"""
{Color.YELLOW}コマンド一覧:{Color.RESET}
  /help     このヘルプを表示
  /info     セッション情報を表示
  /clear    会話履歴をクリア
  /system   システムプロンプトを表示
  /save     会話を保存
  /model    現在のモデル情報を表示
  /quit     終了 (Ctrl+C でも可)
""")


def format_stats(stats: dict) -> str:
    """生成統計をフォーマット"""
    if not stats:
        return ""
    total_ms = stats.get("total_duration", 0) / 1_000_000  # ns → ms
    eval_count = stats.get("eval_count", 0)
    eval_ms = stats.get("eval_duration", 0) / 1_000_000
    tokens_per_sec = (eval_count / (eval_ms / 1000)) if eval_ms > 0 else 0
    return f"{Color.DIM}[{eval_count}tokens, {total_ms:.0f}ms, {tokens_per_sec:.1f}tok/s]{Color.RESET}"


def build_cli_service(config, *, provider=None) -> tuple[AssistantService, ProviderRegistry]:
    """CLI用のAssistantサービスとProvider Registryを構築する。"""
    return build_local_service(config, provider=provider)


def run_chat_loop(config, session, service, *, read_input=input) -> None:
    """CLIの入力、コマンド処理、Assistant呼び出しを実行する。"""
    while True:
        try:
            user_input = read_input(
                f"\n{Color.GREEN}{Color.BOLD}あなた> {Color.RESET}"
            ).strip()
        except EOFError:
            return

        if not user_input:
            continue

        # コマンド処理
        if user_input.startswith("/"):
            cmd = user_input.lower().split()[0]
            if cmd == "/quit" or cmd == "/exit":
                return
            elif cmd == "/help":
                print_help()
                continue
            elif cmd == "/info":
                print(f"\n{Color.DIM}{session.get_summary()}{Color.RESET}")
                continue
            elif cmd == "/clear":
                session.clear()
                print(f"{Color.YELLOW}会話履歴をクリアしました。{Color.RESET}")
                continue
            elif cmd == "/system":
                print(f"\n{Color.DIM}[System Prompt]{Color.RESET}")
                print(f"{Color.DIM}{config.effective_system_prompt()}{Color.RESET}")
                continue
            elif cmd == "/save":
                saved_path = session.save()
                print(f"{Color.GREEN}保存しました: {saved_path}{Color.RESET}")
                continue
            elif cmd == "/model":
                print(f"\n{Color.DIM}モデル: {config.model}")
                print(f"Temperature: {config.temperature}")
                print(f"コンテキスト長: {config.num_ctx}")
                print(f"最大履歴ターン: {config.max_history_turns}{Color.RESET}")
                continue
            else:
                print(f"{Color.RED}不明なコマンド: {cmd}  (/help でコマンド一覧){Color.RESET}")
                continue

        # メッセージ送信
        session.add_user_message(user_input)
        blocks = session.build_blocks()
        request = create_request(
            text=user_input,
            conversation_id=session.session_id,
            channel="cli",
            profile="chat_auto",
            privacy="local_only",
        )

        print(f"\n{Color.CYAN}{Color.BOLD}AI> {Color.RESET}", end="", flush=True)

        try:
            if config.stream:
                # ストリーミング出力
                stream = None
                try:
                    stream = service.respond_stream(request, blocks, base_system=session.system_prompt)
                    for token in stream:
                        print(token, end="", flush=True)
                    print()  # 改行
                    # 統計表示
                    stats_str = format_stats(stream.response.stats)
                    if stats_str:
                        print(stats_str)
                    session.add_assistant_message(stream.response.text)
                finally:
                    if stream is not None:
                        try:
                            stream.close()
                        except Exception:
                            pass
            else:
                # 非ストリーミング
                response, _preview = service.respond(request, blocks, base_system=session.system_prompt)
                print(response.text)
                session.add_assistant_message(response.text)
        except Exception as e:
            print(f"{Color.RED}エラー: {e}{Color.RESET}")
            # エラー時はユーザーメッセージを巻き戻す
            session.rollback_last_user_message()


def main():
    # 設定のロード
    config_path = PROJECT_ROOT / "config" / "chat_config.json"
    config = ChatConfig.load(config_path)

    print_banner()
    print(f"{Color.DIM}モデル: {config.model}{Color.RESET}")
    print(f"{Color.DIM}コンテキスト長: {config.num_ctx}{Color.RESET}")
    web_search = create_web_search_context(config)
    if web_search is not None:
        print(f"{Color.DIM}Web検索: auto={config.web_search_auto}, max_results={config.web_search_max_results}{Color.RESET}")

    # Assistantサービスの初期化
    service, registry = build_cli_service(config)
    provider = registry.get(config.resolved_local_provider_id()).provider

    try:
        # 接続チェック (backend中性: Ollama / OpenAI互換のどちらでも同じ文言)
        print(f"\n{Color.DIM}ローカル推論サーバー接続確認中... ({config.resolved_local_base_url()}){Color.RESET}", end=" ", flush=True)
        if not provider.is_available():
            print(f"{Color.RED}❌ ローカル推論サーバーに接続できません。サービスが起動しているか確認してください。{Color.RESET}")
            sys.exit(1)

        if validate_local_provider_kind(config) == "ollama":
            # Ollama: /api/tags に基づく厳格なモデル存在確認 (has_model 1回で判定)
            print(f"{Color.GREEN}✅ 接続OK{Color.RESET}")
            if not provider.has_model():
                print(f"{Color.RED}❌ モデル '{config.model}' が見つかりません。{Color.RESET}")
                sys.exit(1)
            print(f"{Color.GREEN}✅ モデル確認OK{Color.RESET}")
        else:
            # openai_compatible: is_available() はライフサイクルのみで接続成功を意味しない。
            # /models はオプション。list_models() を1回だけ呼び、非空で設定モデルが
            # 含まれないときだけ失敗し、空 (未実装) なら生成時に検証する警告で続行する。
            discovered = provider.list_models()
            if discovered and config.model not in discovered:
                print(f"{Color.RED}❌ モデル '{config.model}' が見つかりません。{Color.RESET}")
                print(f"{Color.DIM}利用可能なモデル: {', '.join(discovered)}{Color.RESET}")
                sys.exit(1)
            if not discovered:
                print(f"{Color.YELLOW}⚠ モデル情報の取得に失敗しました。生成時に確認します。{Color.RESET}")
            else:
                print(f"{Color.GREEN}✅ モデル確認OK{Color.RESET}")
    except BaseException:
        registry.close()
        raise

    # セッションの初期化
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
        conversation_source="cli",
        emotion_tags=config.emotion_tag_enabled,
    )

    print_help()

    # Ctrl+C でグレースフル終了
    def signal_handler(sig, frame):
        print(f"\n\n{Color.YELLOW}セッションを保存して終了します...{Color.RESET}")
        if session.turn_count > 0:
            saved_path = session.save()
            print(f"{Color.DIM}保存先: {saved_path}{Color.RESET}")
        registry.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    run_chat_loop(config, session, service)

    # 終了処理
    print(f"\n{Color.YELLOW}終了します...{Color.RESET}")
    if session.turn_count > 0:
        saved_path = session.save()
        print(f"{Color.DIM}会話を保存しました: {saved_path}{Color.RESET}")
    registry.close()
    print(f"{Color.GREEN}お疲れ様でした！{Color.RESET}")


if __name__ == "__main__":
    main()
