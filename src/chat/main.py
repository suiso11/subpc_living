"""
Phase 2: テキスト対話 CLIメインエントリポイント
Ollamaの7Bモデル(Q4)を使ったインタラクティブなテキスト対話を実現する
"""
import sys
import signal
from pathlib import Path

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.chat.config import ChatConfig
from src.chat.client import OllamaClient
from src.chat.session import ChatSession


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


def main():
    # 設定のロード
    config_path = PROJECT_ROOT / "config" / "chat_config.json"
    config = ChatConfig.load(config_path)

    print_banner()
    print(f"{Color.DIM}モデル: {config.model}{Color.RESET}")
    print(f"{Color.DIM}コンテキスト長: {config.num_ctx}{Color.RESET}")

    # Ollamaクライアントの初期化
    client = OllamaClient(base_url=config.ollama_base_url, model=config.model)

    # 接続チェック
    print(f"\n{Color.DIM}Ollama接続確認中...{Color.RESET}", end=" ", flush=True)
    if not client.is_available():
        print(f"{Color.RED}❌ Ollamaに接続できません。サービスが起動しているか確認してください。{Color.RESET}")
        print(f"{Color.DIM}  sudo systemctl start ollama{Color.RESET}")
        sys.exit(1)
    print(f"{Color.GREEN}✅ 接続OK{Color.RESET}")

    # モデル存在チェック
    if not client.has_model():
        print(f"{Color.RED}❌ モデル '{config.model}' が見つかりません。{Color.RESET}")
        print(f"{Color.DIM}利用可能なモデル: {', '.join(client.list_models())}{Color.RESET}")
        sys.exit(1)
    print(f"{Color.GREEN}✅ モデル確認OK{Color.RESET}")

    # セッションの初期化
    session = ChatSession(
        system_prompt=config.system_prompt,
        max_history_turns=config.max_history_turns,
        history_dir=str(PROJECT_ROOT / config.history_dir),
    )

    print_help()

    # Ctrl+C でグレースフル終了
    def signal_handler(sig, frame):
        print(f"\n\n{Color.YELLOW}セッションを保存して終了します...{Color.RESET}")
        if session.turn_count > 0:
            saved_path = session.save()
            print(f"{Color.DIM}保存先: {saved_path}{Color.RESET}")
        client.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # --- メインループ ---
    while True:
        try:
            user_input = input(f"\n{Color.GREEN}{Color.BOLD}あなた> {Color.RESET}").strip()
        except EOFError:
            break

        if not user_input:
            continue

        # コマンド処理
        if user_input.startswith("/"):
            cmd = user_input.lower().split()[0]
            if cmd == "/quit" or cmd == "/exit":
                break
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
                print(f"{Color.DIM}{config.system_prompt}{Color.RESET}")
                continue
            elif cmd == "/save":
                saved_path = session.save()
                print(f"{Color.GREEN}保存しました: {saved_path}{Color.RESET}")
                continue
            elif cmd == "/model":
                print(f"\n{Color.DIM}モデル: {config.model}")
                print(f"Temperature: {config.temperature}")
                print(f"Top-P: {config.top_p}")
                print(f"コンテキスト長: {config.num_ctx}")
                print(f"最大履歴ターン: {config.max_history_turns}{Color.RESET}")
                continue
            else:
                print(f"{Color.RED}不明なコマンド: {cmd}  (/help でコマンド一覧){Color.RESET}")
                continue

        # メッセージ送信
        session.add_user_message(user_input)
        messages = session.build_messages()

        print(f"\n{Color.CYAN}{Color.BOLD}AI> {Color.RESET}", end="", flush=True)

        try:
            if config.stream:
                # ストリーミング出力
                full_response = ""
                for token in client.generate_stream(
                    messages,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    top_k=config.top_k,
                    num_ctx=config.num_ctx,
                    repeat_penalty=config.repeat_penalty,
                ):
                    print(token, end="", flush=True)
                    full_response += token
                print()  # 改行
                # 統計表示
                stats_str = format_stats(client.last_stats)
                if stats_str:
                    print(stats_str)
                session.add_assistant_message(full_response)
            else:
                # 非ストリーミング
                response = client.generate(
                    messages,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    top_k=config.top_k,
                    num_ctx=config.num_ctx,
                    repeat_penalty=config.repeat_penalty,
                )
                print(response)
                session.add_assistant_message(response)
        except Exception as e:
            print(f"{Color.RED}エラー: {e}{Color.RESET}")
            # エラー時はユーザーメッセージを巻き戻す
            if session._messages and session._messages[-1]["role"] == "user":
                session._messages.pop()

    # 終了処理
    print(f"\n{Color.YELLOW}終了します...{Color.RESET}")
    if session.turn_count > 0:
        saved_path = session.save()
        print(f"{Color.DIM}会話を保存しました: {saved_path}{Color.RESET}")
    client.close()
    print(f"{Color.GREEN}お疲れ様でした！{Color.RESET}")


if __name__ == "__main__":
    main()
