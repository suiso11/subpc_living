"""
チャット設定モジュール
Phase 2: テキスト対話用の設定を管理する
"""
from dataclasses import dataclass, field
from pathlib import Path
import json


@dataclass
class ChatConfig:
    """チャットシステムの設定"""

    # --- Ollama接続設定 ---
    ollama_base_url: str = "http://localhost:11434"
    model: str = "qwen2.5:7b-instruct-q4_K_M"

    # --- 生成パラメータ ---
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    num_ctx: int = 8192  # コンテキスト長
    num_predict: int | None = None  # 最大生成トークン数。NoneならOllamaデフォルト

    # --- システムプロンプト ---
    system_prompt: str = (
        "あなたはユーザー専属のパーソナルAIアシスタントです。\n"
        "以下のルールに従って応答してください:\n"
        "- 日本語で自然に会話してください\n"
        "- 簡潔で的確な応答を心がけてください\n"
        "- ユーザーの文脈や意図を汲み取って応答してください\n"
        "- 分からないことは正直に伝えてください"
    )

    # --- 会話履歴設定 ---
    max_history_turns: int = 20  # 保持する会話ターン数の上限
    history_dir: str = "data/chat_history"  # 履歴保存ディレクトリ

    # --- Web検索設定 ---
    web_search_enabled: bool = False  # 最新情報が必要そうな時にWeb検索する
    web_search_auto: bool = True  # True: 必要そうな発話だけ自動検索
    web_search_max_results: int = 4
    web_search_timeout_sec: float = 8.0
    web_search_cache_path: str = ""  # 空なら永続キャッシュしない

    # --- 表示設定 ---
    stream: bool = True  # ストリーミング出力を使用するか

    # --- Discord チャンネル別LLMプロファイル ---
    # discord_channel_profile_map: {"channel_id": "profile_name"}
    # discord_channel_profiles: {"profile_name": {"temperature": 0.4, ...}}
    discord_channel_profile_map: dict[str, str] = field(default_factory=dict)
    discord_channel_profiles: dict[str, dict] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path = "config/chat_config.json") -> "ChatConfig":
        """JSONファイルから設定をロード"""
        path = Path(path)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        return cls()

    def save(self, path: str | Path = "config/chat_config.json") -> None:
        """設定をJSONファイルに保存"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        from dataclasses import asdict
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)
