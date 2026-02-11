"""
チャットセッション管理
Phase 2: 会話履歴の管理・永続化を担当するモジュール
Phase 4: RAG統合 — ベクトルDBから関連文脈をプロンプトに注入
Phase 5: Vision統合 — カメラ映像の解析結果をプロンプトに注入
"""
from datetime import datetime
from pathlib import Path
import json
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.memory.rag import RAGRetriever
    from src.vision.context import VisionContext
    from src.monitor.context import MonitorContext


class ChatSession:
    """1つの対話セッションを管理するクラス"""

    def __init__(
        self,
        system_prompt: str = "",
        max_history_turns: int = 20,
        history_dir: str = "data/chat_history",
        rag: Optional["RAGRetriever"] = None,
        vision_context: Optional["VisionContext"] = None,
        monitor_context: Optional["MonitorContext"] = None,
    ):
        self.system_prompt = system_prompt
        self.max_history_turns = max_history_turns
        self.history_dir = Path(history_dir)
        self.history_dir.mkdir(parents=True, exist_ok=True)

        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._messages: list[dict] = []  # {"role": "user"|"assistant", "content": "..."}
        self._created_at = datetime.now()
        self.rag = rag
        self.vision_context = vision_context
        self.monitor_context = monitor_context

    def add_user_message(self, content: str) -> None:
        """ユーザーのメッセージを追加"""
        self._messages.append({"role": "user", "content": content})
        self._trim_history()

    def add_assistant_message(self, content: str) -> None:
        """アシスタントの応答を追加（RAG有効時はベクトルDBにも保存）"""
        self._messages.append({"role": "assistant", "content": content})

        # RAG: 直前のuser+assistantをベクトルDBに保存
        if self.rag is not None and len(self._messages) >= 2:
            user_msg = self._messages[-2]
            if user_msg.get("role") == "user":
                self.rag.store_turn(
                    user_message=user_msg["content"],
                    assistant_message=content,
                    session_id=self.session_id,
                )

    def build_messages(self) -> list[dict]:
        """
        Ollama APIに渡すメッセージリストを構築

        RAGが有効な場合、最新のユーザーメッセージで長期記憶を検索し、
        関連する過去の文脈をシステムプロンプトに注入する。
        """
        messages = []

        # システムプロンプト + RAGコンテキスト + Visionコンテキスト
        system_content = self.system_prompt or ""

        if self.rag is not None and self._messages:
            # 最新のユーザーメッセージで検索
            last_user = None
            for msg in reversed(self._messages):
                if msg["role"] == "user":
                    last_user = msg["content"]
                    break
            if last_user:
                rag_context = self.rag.build_context_prompt(last_user)
                if rag_context:
                    system_content = system_content + rag_context

        # Vision: カメラ映像の現在の状態を注入
        if self.vision_context is not None:
            vision_text = self.vision_context.get_context_text()
            if vision_text:
                system_content = system_content + vision_text

        # Monitor: サブPCの状態を注入 (Phase 6)
        if self.monitor_context is not None:
            monitor_text = self.monitor_context.get_context_text()
            if monitor_text:
                system_content = system_content + monitor_text

        if system_content:
            messages.append({"role": "system", "content": system_content})

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
