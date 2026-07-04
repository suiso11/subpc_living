"""
チャットセッション管理
Phase 2: 会話履歴の管理・永続化を担当するモジュール
Phase 4: RAG統合 — ベクトルDBから関連文脈をプロンプトに注入
Phase 5: Vision統合 — カメラ映像の解析結果をプロンプトに注入
Phase 7: パーソナライズ統合 — プリロードコンテキストをプロンプトに注入
"""
from datetime import datetime
from pathlib import Path
import json
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.memory.rag import RAGRetriever
    from src.vision.context import VisionContext
    from src.screen.context import ScreenContext
    from src.monitor.context import MonitorContext
    from src.persona.preloader import SessionPreloader
    from src.chat.web_search import WebSearchContext
    from src.tasks.store import TaskStore
    from src.tasks.calendar_sync import CalendarContext


class ChatSession:
    """1つの対話セッションを管理するクラス"""

    def __init__(
        self,
        system_prompt: str = "",
        max_history_turns: int = 20,
        history_dir: str = "data/chat_history",
        rag: Optional["RAGRetriever"] = None,
        vision_context: Optional["VisionContext"] = None,
        screen_context: Optional["ScreenContext"] = None,
        monitor_context: Optional["MonitorContext"] = None,
        preloader: Optional["SessionPreloader"] = None,
        web_search: Optional["WebSearchContext"] = None,
        task_store: Optional["TaskStore"] = None,
        calendar_context: Optional["CalendarContext"] = None,
        emotion_tags: bool = False,
    ):
        self.system_prompt = system_prompt
        self.max_history_turns = max_history_turns
        self.emotion_tags = emotion_tags
        self.history_dir = Path(history_dir)
        self.history_dir.mkdir(parents=True, exist_ok=True)

        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._messages: list[dict] = []  # {"role": "user"|"assistant", "content": "..."}
        self._created_at = datetime.now()
        self.rag = rag
        self.vision_context = vision_context
        self.screen_context = screen_context
        self.monitor_context = monitor_context
        self.preloader = preloader
        self.web_search = web_search
        self.task_store = task_store
        self.calendar_context = calendar_context

    def add_user_message(self, content: str) -> None:
        """ユーザーのメッセージを追加"""
        self._messages.append({"role": "user", "content": content})
        self._trim_history()

    def add_assistant_message(self, content: str) -> None:
        """アシスタントの応答を追加（RAG有効時はベクトルDBにも保存）"""
        self._messages.append({"role": "assistant", "content": content})
        self._trim_history()

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

        # システムプロンプト + プリロード + RAGコンテキスト + Visionコンテキスト
        system_content = self.system_prompt or ""

        # Preload: プロフィール・スケジュール・最近の会話要約・時刻 (Phase 7)
        if self.preloader is not None:
            preload_text = self.preloader.build_preload_context()
            if preload_text:
                system_content = system_content + preload_text

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

        if self.web_search is not None and self._messages:
            last_user = None
            for msg in reversed(self._messages):
                if msg["role"] == "user":
                    last_user = msg["content"]
                    break
            if last_user:
                web_context = self.web_search.build_context_prompt(last_user)
                if web_context:
                    system_content = system_content + web_context

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

        # Screen: ユーザーの画面で何をしているかを注入 (VLM描写)
        if self.screen_context is not None:
            screen_text = self.screen_context.get_context_text()
            if screen_text:
                system_content = system_content + screen_text

        # Calendar: Google Calendar の今日〜明日の予定を注入 (ファイル読取のみ)
        if self.calendar_context is not None:
            try:
                cal_text = self.calendar_context.get_context_text()
                if cal_text:
                    system_content = system_content + cal_text
            except Exception:
                pass

        # Tasks: 未完了タスクを注入 (0件なら注入しない)
        if self.task_store is not None:
            try:
                from src.tasks.store import build_task_context
                task_text = build_task_context(self.task_store)
                if task_text:
                    system_content = system_content + task_text
            except Exception:
                pass

        # 感情タグの指示 (有効時のみ、system content 末尾へ一元的に追加)
        if self.emotion_tags:
            from src.chat.emotion import EMOTION_TAG_INSTRUCTION
            separator = "\n\n" if system_content else ""
            system_content = system_content + separator + EMOTION_TAG_INSTRUCTION

        if system_content:
            messages.append({"role": "system", "content": system_content})

        messages.extend(self._messages)
        return messages

    def _trim_history(self) -> None:
        """履歴をターン単位で max_history_turns に収める"""
        if self.max_history_turns <= 0:
            self._messages.clear()
            return

        # 以前の不整合で先頭に assistant が来ている履歴を補正する。
        while self._messages and self._messages[0]["role"] != "user":
            self._messages.pop(0)

        while sum(1 for m in self._messages if m["role"] == "user") > self.max_history_turns:
            if not self._messages:
                break

            if self._messages[0]["role"] == "user":
                self._messages.pop(0)

            if self._messages and self._messages[0]["role"] == "assistant":
                self._messages.pop(0)

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
