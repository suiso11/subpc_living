"""
チャット設定モジュール
Phase 2: テキスト対話用の設定を管理する
"""
import os
import tempfile
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

    # --- 感情タグ ---
    # True: 応答冒頭に [emo:happy] 形式のタグを出力させ、TTSスタイルを動的に切り替える
    emotion_tag_enabled: bool = False

    # --- Discord チャンネル別LLMプロファイル ---
    # discord_channel_profile_map: {"channel_id": "profile_name"}
    # discord_channel_profiles: {"profile_name": {"temperature": 0.4, ...}}
    discord_channel_profile_map: dict[str, str] = field(default_factory=dict)
    discord_channel_profiles: dict[str, dict] = field(default_factory=dict)

    # --- モデル別 system_prompt 上書き ---
    # model_prompt_overrides: {"model_name": "短いプロンプト"}
    # SFT/DPO 等の人格定着済みモデルでは長い system_prompt を付けず、
    # ここで指定した短い人格契約だけを差し込む。base の system_prompt は
    # 上書きされず保持され、モデル名を base に戻せば元の長いプロンプトに戻る。
    model_prompt_overrides: dict[str, str] = field(default_factory=dict)

    def effective_system_prompt(self, model: str | None = None) -> str:
        """指定モデル (省略時 self.model) で実際に使用する system_prompt を返す。

        model_prompt_overrides に該当モデル名があればその短いプロンプトを、
        無ければ base の self.system_prompt を返す。本フィールドは読み取り専用で、
        self.system_prompt を書き換えない。
        """
        target = model if model is not None else self.model
        if target and target in self.model_prompt_overrides:
            return self.model_prompt_overrides[target]
        return self.system_prompt

    @staticmethod
    def _atomic_write(path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_name, path)
        except Exception:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass
            raise

    def create_backup(self, path: str | Path = "config/chat_config.json") -> Path | None:
        """既存設定を同一ディレクトリの <path>.bak へアトミックに保存する。"""
        path = Path(path)
        if not path.exists():
            return None
        backup = path.with_name(path.name + ".bak")
        self._atomic_write(backup, path.read_bytes())
        return backup

    def rollback_from_backup(self, path: str | Path = "config/chat_config.json") -> bool:
        """バックアップをアトミックに復元する。"""
        path = Path(path)
        backup = path.with_name(path.name + ".bak")
        if not backup.exists():
            return False
        self._atomic_write(path, backup.read_bytes())
        return True

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
        """設定を同一ディレクトリの一時ファイル経由でアトミックに保存。"""
        from dataclasses import asdict
        path = Path(path)
        payload = (json.dumps(asdict(self), ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        self._atomic_write(path, payload)
