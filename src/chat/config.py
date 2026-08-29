"""
チャット設定モジュール
Phase 2: テキスト対話用の設定を管理する
"""
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
import json

from src.llm.local_endpoint import validate_loopback_openai_base_url


# ローカル推論 backend (P0-2) の取り得る kind。
LOCAL_PROVIDER_KINDS: tuple[str, ...] = ("ollama", "openai_compatible")

# openai_compatible 選択時に ``local_base_url`` が空だった場合の慣用既定エンドポイント。
# llama.cpp 系サーバーの慣用値であり、検証済みの実サーバー既定ではない。
LOCAL_OPENAI_DEFAULT_BASE_URL = "http://localhost:8080/v1"

_PROVIDER_ID_BY_KIND = {"ollama": "ollama", "openai_compatible": "local-openai"}


def validate_local_provider_kind(config) -> str:
    """``local_provider_kind`` を正規化して返す。

    ``None`` と見なされる欠落は後方互換のため ``"ollama"`` へ正規化する
    (``usage.md``: 無設定時は従来どおり Ollama)。空または空白のみの文字列も
    ``"ollama"`` へ正規化する。``str`` / ``None`` 以外の型 (int / list / bool など)
    は設定ミスとして ``ValueError`` で明示的に拒否する。未知の文字列値は
    ``ValueError``。
    """
    raw = getattr(config, "local_provider_kind", "ollama")
    if raw is None:
        raw = ""
    elif not isinstance(raw, str):
        raise ValueError(
            "invalid local_provider_kind: "
            f"{raw!r}; expected a string (or None for the default)"
        )
    kind = raw.strip()
    if not kind:
        kind = "ollama"
    if kind not in LOCAL_PROVIDER_KINDS:
        raise ValueError(
            "unknown local_provider_kind: "
            f"{kind!r}; expected one of {', '.join(LOCAL_PROVIDER_KINDS)}"
        )
    return kind


def resolve_local_provider_id(config) -> str:
    """ローカルbackendの実 provider_id を返す。

    ``local_provider_id`` が明示されていればそれを、空なら kind 別既定
    (``"ollama"`` / ``"local-openai"``) を返す。
    """
    kind = validate_local_provider_kind(config)
    explicit = (getattr(config, "local_provider_id", "") or "").strip()
    if explicit:
        return explicit
    return _PROVIDER_ID_BY_KIND[kind]


def validate_local_base_url(url: str) -> str:
    """openai_compatible の ``local_base_url`` を厳格に検証し、検証済みURLを返す。

    このマイルストーンでは送信先を**同一マシンのloopback**に限定する。``local=True``
    で登録されたproviderはcloudの承認・redactionセマンティクスを受けないため、
    任意のLAN / 公開 / 曖昧なhost名を許すと、クラウド経路の保護を迂回して
    機密情報を送信できる境界を生む。

    検証ロジックは ``src.llm.local_endpoint`` の共通validator
    (``validate_loopback_openai_base_url``) へ委譲し、provider側と同一の
    ルールを共有する。違反時は ``ValueError``。リモートの信頼済みノード
    (LAN / VPN / 公開) 対応は別途の明示的な信頼設計が必要なため、
    本マイルストーンでは deferred。
    """
    return validate_loopback_openai_base_url(url)


def resolve_local_base_url(config) -> str:
    """ローカルbackendの実 base URL を返す。

    Ollama は従来の ``ollama_base_url`` を常に尊重する (後方互換)。
    openai_compatible は ``local_base_url`` を優先し、空なら llama.cpp 慣用の
    ``LOCAL_OPENAI_DEFAULT_BASE_URL`` を使う。これは実サーバーの検証済み既定では
    なく、設定未指定時の慣用値である。openai_compatible は返す前に
    ``validate_local_base_url`` でloopback限定の厳格検証を受ける。
    """
    kind = validate_local_provider_kind(config)
    if kind == "ollama":
        return getattr(config, "ollama_base_url", "http://localhost:11434")
    local_base_url = (getattr(config, "local_base_url", "") or "").strip()
    resolved = local_base_url or LOCAL_OPENAI_DEFAULT_BASE_URL
    return validate_local_base_url(resolved)


def resolve_local_api_key(config) -> str | None:
    """openai_compatible かつ ``local_api_key_env`` 指定時のみ、実行時に環境変数からキーを解決する。

    キー自体はコード・設定・ログへ保持しない。Ollama 時、env 名未指定、または
    環境変数が未設定のときは ``None`` を返す。
    """
    kind = validate_local_provider_kind(config)
    if kind != "openai_compatible":
        return None
    env_name = (getattr(config, "local_api_key_env", "") or "").strip()
    if not env_name:
        return None
    return os.environ.get(env_name) or None


@dataclass
class ChatConfig:
    """チャットシステムの設定"""

    # --- Ollama接続設定 ---
    ollama_base_url: str = "http://localhost:11434"
    model: str = "qwen2.5:7b-instruct-q4_K_M"

    # --- ローカル推論 backend 選択 (P0-2) ---
    # local_provider_kind: "ollama" (既定) または "openai_compatible"
    local_provider_kind: str = "ollama"
    # 空なら backend 既定。Ollama は従来の ollama_base_url を引き続き尊重する
    local_base_url: str = ""
    # 空なら kind 別既定 ("ollama" / "local-openai")
    local_provider_id: str = ""
    # 環境変数名のみ。キー自体は保存しない
    local_api_key_env: str = ""

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

    def validate_local_provider(self) -> None:
        """ローカルbackend設定を検証する。

        未知 kind は ``ValueError``。openai_compatible は加えて
        ``local_base_url`` (既定含む) がloopback限定のURLであることを検証する。
        """
        kind = validate_local_provider_kind(self)
        if kind == "openai_compatible":
            validate_local_base_url(self.resolved_local_base_url())

    def resolved_local_provider_id(self) -> str:
        """実 provider_id。未指定なら kind 別既定 ("ollama" / "local-openai")。"""
        return resolve_local_provider_id(self)

    def resolved_local_base_url(self) -> str:
        """実 base URL。Ollama は ``ollama_base_url`` (後方互換)、openai_compatible は ``local_base_url`` か慣用既定。"""
        return resolve_local_base_url(self)

    def resolve_local_api_key(self) -> str | None:
        """openai_compatible かつ ``local_api_key_env`` 指定時のみ、実行時に環境変数からキーを返す。"""
        return resolve_local_api_key(self)

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
