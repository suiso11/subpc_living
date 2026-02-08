"""
チャットセッション管理
Phase 2: 会話履歴の管理・永続化を担当するモジュール
"""
from datetime import datetime
from pathlib import Path
import json
from typing import Optional


class ChatSession:
    """1つの対話セッションを管理するクラス"""

    def __init__(
        self,
        system_prompt: str = "",
        max_history_turns: int = 20,
        history_dir: str = "data/chat_history",
    ):
        self.system_prompt = system_prompt
        self.max_history_turns = max_history_turns
        self.history_dir = Path(history_dir)
        self.history_dir.mkdir(parents=True, exist_ok=True)

        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._messages: list[dict] = []  # {"role": "user"|"assistant", "content": "..."}
        self._created_at = datetime.now()

    def add_user_message(self, content: str) -> None:
        """ユーザーのメッセージを追加"""
        self._messages.append({"role": "user", "content": content})
        self._trim_history()

    def add_assistant_message(self, content: str) -> None:
        """アシスタントの応答を追加"""
        self._messages.append({"role": "assistant", "content": content})

    def build_messages(self) -> list[dict]:
        """Ollama APIに渡すメッセージリストを構築"""
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend(self._messages)
        return messages

    def _trim_history(self) -> None:
        """履歴をmax_history_turnsに収める（古いものから削除）"""
        # 1ターン = user + assistant の2メッセージ
        max_messages = self.max_history_turns * 2
        if len(self._messages) > max_messages:
            self._messages = self._messages[-max_messages:]

    @property
    def turn_count(self) -> int:
        """現在の会話ターン数"""
        return sum(1 for m in self._messages if m["role"] == "user")

    @property
    def messages(self) -> list[dict]:
        """会話履歴のコピーを返す"""
        return list(self._messages)

    def save(self, filepath: Optional[str] = None) -> Path:
        """セッションをJSONファイルに保存"""
        if filepath is None:
            filepath = self.history_dir / f"session_{self.session_id}.json"
        else:
            filepath = Path(filepath)

        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "session_id": self.session_id,
            "created_at": self._created_at.isoformat(),
            "saved_at": datetime.now().isoformat(),
            "system_prompt": self.system_prompt,
            "turn_count": self.turn_count,
            "messages": self._messages,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return filepath

    @classmethod
    def load(cls, filepath: str | Path, **kwargs) -> "ChatSession":
        """保存されたセッションをロード"""
        filepath = Path(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        session = cls(
            system_prompt=data.get("system_prompt", ""),
            **kwargs,
        )
        session.session_id = data["session_id"]
        session._messages = data.get("messages", [])
        session._created_at = datetime.fromisoformat(data["created_at"])
        return session

    def clear(self) -> None:
        """会話履歴をクリア"""
        self._messages.clear()

    def get_summary(self) -> str:
        """セッションのサマリーを返す"""
        return (
            f"セッション: {self.session_id}\n"
            f"開始時刻: {self._created_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"ターン数: {self.turn_count}\n"
            f"メッセージ数: {len(self._messages)}"
        )
