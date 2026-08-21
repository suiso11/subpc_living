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
    from src.growth.tracker import GrowthTracker


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
        growth_tracker: Optional["GrowthTracker"] = None,
        conversation_source: str = "chat",
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
        self.growth_tracker = growth_tracker
        self.conversation_source = conversation_source

    def add_user_message(self, content: str) -> None:
        """ユーザーのメッセージを追加"""
        self._messages.append({"role": "user", "content": content})
        self._trim_history()

    def add_assistant_message(self, content: str, *, store_memory: bool = True) -> None:
        """アシスタントの応答を追加（RAG有効時はベクトルDBにも保存）

        store_memory=False のときは RAG への長期記憶保存をスキップする。
        セッション履歴の追加・トリムは常に行い、成長台帳には
        memory_saved=False で記録する。テキスト内容に基づく分類は行わない。
        save() は呼び出さないためファイル保存が必要なら別途呼ぶこと。
        """
        self._messages.append({"role": "assistant", "content": content})
        self._trim_history()

        # RAG: 直前のuser+assistantをベクトルDBに保存
        # store_memory=False のときは保存せず、memory_id を None のままにする。
        memory_id = None
        user_msg = None
        if store_memory and self.rag is not None and len(self._messages) >= 2:
            user_msg = self._messages[-2]
            if user_msg.get("role") == "user":
                memory_id = self.rag.store_turn(
                    user_message=user_msg["content"],
                    assistant_message=content,
                    session_id=self.session_id,
                )

        # 成長台帳: 本文は保存せず、成功した会話例とRAG保存成否だけを記録する。
        # store_memory=False のときは memory_id が None のままなので
        # memory_saved=False として記録される。
        if self.growth_tracker is not None and len(self._messages) >= 2:
            user_msg = user_msg or self._messages[-2]
            if user_msg.get("role") == "user":
                try:
                    self.growth_tracker.record_conversation(
                        source=self.conversation_source,
                        session_id=self.session_id,
                        user_chars=len(str(user_msg.get("content") or "")),
                        assistant_chars=len(str(content or "")),
                        memory_saved=memory_id is not None,
                    )
                except Exception:
                    # 計測失敗で会話自体を失敗させない。
                    pass

    def build_blocks(self) -> tuple:
        """build_messages が描画する ContextBlock を返す（描画前）。

        プリロード・RAG・Web検索・Vision・Monitor・Screen・Calendar・Tasks・History の
        各 ContextProvider から収集した ContextBlock を tuple で返す。
        """
        from src.context.contracts import ContextBlock as _CB
        from src.context.providers.preload import PreloadContextProvider
        from src.context.providers.rag import RAGContextProvider
        from src.context.providers.web_search import WebSearchContextProvider
        from src.context.providers.vision import VisionContextProvider
        from src.context.providers.monitor import MonitorContextProvider
        from src.context.providers.screen import ScreenContextProvider
        from src.context.providers.calendar import CalendarContextProvider
        from src.context.providers.tasks import TasksContextProvider
        from src.context.providers.history import HistoryContextProvider

        blocks: list[_CB] = []

        if self.preloader is not None:
            block = PreloadContextProvider.collect(self.preloader)
            if block is not None:
                blocks.append(block)

        if self.rag is not None and self._messages:
            last_user = None
            for msg in reversed(self._messages):
                if msg["role"] == "user":
                    last_user = msg["content"]
                    break
            if last_user:
                block = RAGContextProvider.collect(self.rag, last_user)
                if block is not None:
                    blocks.append(block)

        if self.web_search is not None and self._messages:
            last_user = None
            for msg in reversed(self._messages):
                if msg["role"] == "user":
                    last_user = msg["content"]
                    break
            if last_user:
                block = WebSearchContextProvider.collect(self.web_search, last_user)
                if block is not None:
                    blocks.append(block)

        if self.vision_context is not None:
            block = VisionContextProvider.collect(self.vision_context)
            if block is not None:
                blocks.append(block)

        if self.monitor_context is not None:
            block = MonitorContextProvider.collect(self.monitor_context)
            if block is not None:
                blocks.append(block)

        if self.screen_context is not None:
            block = ScreenContextProvider.collect(self.screen_context)
            if block is not None:
                blocks.append(block)

        if self.calendar_context is not None:
            block = CalendarContextProvider.collect(self.calendar_context)
            if block is not None:
                blocks.append(block)

        tasks_block = (
            TasksContextProvider.collect(self.task_store)
            if self.task_store is not None
            else None
        )
        history_block = HistoryContextProvider.collect(self._messages)

        for block in (tasks_block, history_block):
            if block is not None:
                blocks.append(block)

        return tuple(blocks)

    def build_messages(self) -> list[dict]:
        """
        Ollama APIに渡すメッセージリストを構築

        RAGが有効な場合、最新のユーザーメッセージで長期記憶を検索し、
        関連する過去の文脈をシステムプロンプトに注入する。
        History は ContextBlock 化して ContextBuilder 経由で描画する (Phase J)。
        """
        # システムプロンプト + プリロード + RAGコンテキスト + Visionコンテキスト
        system_content = self.system_prompt or ""

        # Preload: プロフィール・スケジュール・最近の会話要約・時刻 (Phase 7)
        # Phase J: PreloadContextProvider + ContextBuilder 経由で描画する。
        # system_prompt 直後・RAG 前の現行位置を維持し、local_only / local target で
        # ContextPolicy に通した str block だけを base system へ連結する。
        if self.preloader is not None:
            from src.context.builder import ContextBuilder
            from src.context.providers.preload import PreloadContextProvider

            preload_block = PreloadContextProvider.collect(self.preloader)
            builder = ContextBuilder(system_content)
            system_content = builder.build_system_content(
                [preload_block] if preload_block is not None else [],
                privacy="local_only",
                target_local=True,
            )

        if self.rag is not None and self._messages:
            # 最新のユーザーメッセージで検索
            last_user = None
            for msg in reversed(self._messages):
                if msg["role"] == "user":
                    last_user = msg["content"]
                    break
            # Phase J: RAGContextProvider + ContextBuilder 経由で描画する。
            # Preload 直後・Web search 前の現行位置を維持し、local_only / local target
            # で ContextPolicy に通した str block だけを base system へ連結する。
            if last_user:
                from src.context.builder import ContextBuilder
                from src.context.providers.rag import RAGContextProvider

                rag_block = RAGContextProvider.collect(self.rag, last_user)
                builder = ContextBuilder(system_content)
                system_content = builder.build_system_content(
                    [rag_block] if rag_block is not None else [],
                    privacy="local_only",
                    target_local=True,
                )

        if self.web_search is not None and self._messages:
            # 最新のユーザーメッセージで検索
            last_user = None
            for msg in reversed(self._messages):
                if msg["role"] == "user":
                    last_user = msg["content"]
                    break
            # Phase J: WebSearchContextProvider + ContextBuilder 経由で描画する。
            # RAG 直後・Vision 前の現行位置を維持し、local_only / local target
            # で ContextPolicy に通した str block だけを base system へ連結する。
            if last_user:
                from src.context.builder import ContextBuilder
                from src.context.providers.web_search import WebSearchContextProvider

                web_block = WebSearchContextProvider.collect(self.web_search, last_user)
                builder = ContextBuilder(system_content)
                system_content = builder.build_system_content(
                    [web_block] if web_block is not None else [],
                    privacy="local_only",
                    target_local=True,
                )

        # Vision: カメラ映像の現在の状態を注入 (Phase J)
        # Phase J: VisionContextProvider + ContextBuilder 経由で描画する。
        # Web search 直後・Monitor 前の現行位置を維持し、local_only / local target
        # で ContextPolicy に通した str block だけを base system へ連結する。
        if self.vision_context is not None:
            from src.context.builder import ContextBuilder
            from src.context.providers.vision import VisionContextProvider

            vision_block = VisionContextProvider.collect(self.vision_context)
            builder = ContextBuilder(system_content)
            system_content = builder.build_system_content(
                [vision_block] if vision_block is not None else [],
                privacy="local_only",
                target_local=True,
            )

        # Monitor: サブPCの状態を注入 (Phase J)
        # Phase J: MonitorContextProvider + ContextBuilder 経由で描画する。
        # Vision 直後・Screen 前の現行位置を維持し、local_only / local target
        # で ContextPolicy に通した str block だけを base system へ連結する。
        if self.monitor_context is not None:
            from src.context.builder import ContextBuilder
            from src.context.providers.monitor import MonitorContextProvider

            monitor_block = MonitorContextProvider.collect(self.monitor_context)
            builder = ContextBuilder(system_content)
            system_content = builder.build_system_content(
                [monitor_block] if monitor_block is not None else [],
                privacy="local_only",
                target_local=True,
            )

        # Screen: ユーザーの画面で何をしているかを注入 (VLM描写) (Phase J)
        # Phase J: ScreenContextProvider + ContextBuilder 経由で描画する。
        # Monitor 直後・Calendar 前の現行位置を維持し、local_only / local target
        # で ContextPolicy に通した str block だけを base system へ連結する。
        if self.screen_context is not None:
            from src.context.builder import ContextBuilder
            from src.context.providers.screen import ScreenContextProvider

            screen_block = ScreenContextProvider.collect(self.screen_context)
            builder = ContextBuilder(system_content)
            system_content = builder.build_system_content(
                [screen_block] if screen_block is not None else [],
                privacy="local_only",
                target_local=True,
            )

        # Calendar: Google Calendar の今日〜明日の予定を注入 (ファイル読取のみ) (Phase J)
        # Phase J: CalendarContextProvider + ContextBuilder 経由で描画する。
        # Screen 直後・Emotion 前の現行位置を維持し、local_only / local target
        # で ContextPolicy に通した str block だけを base system へ連結する。
        if self.calendar_context is not None:
            from src.context.builder import ContextBuilder
            from src.context.providers.calendar import CalendarContextProvider

            calendar_block = CalendarContextProvider.collect(self.calendar_context)
            builder = ContextBuilder(system_content)
            system_content = builder.build_system_content(
                [calendar_block] if calendar_block is not None else [],
                privacy="local_only",
                target_local=True,
            )

        # 感情タグの指示 (有効時のみ)。タスク状態の権威ブロックよりも前に置くことで、
        # タスク状態がシステムプロンプトの最終権威となる。
        if self.emotion_tags:
            from src.chat.emotion import EMOTION_TAG_INSTRUCTION
            separator = "\n\n" if system_content else ""
            system_content = system_content + separator + EMOTION_TAG_INSTRUCTION

        # Tasks: 未完了タスク + 現在状態の権威ブロックを末尾に置く (0件でも必ず注入)。
        # 会話履歴・RAG・訓練データを上書きする最終権威なので、他の動的コンテキスト・
        # 感情タグ指示の後に配置する。TasksContextProvider 経由で ContextBlock 化し、
        # History と同時に最終 ContextBuilder.build_messages へ渡す。
        # ContextPolicy の tasks-last と Builder の system-first により、system 本文の
        # 最終文字列 block が tasks authority になり、History の role messages は
        # system message の後ろに配置される。
        from src.context.providers.tasks import TasksContextProvider

        tasks_block = (
            TasksContextProvider.collect(self.task_store)
            if self.task_store is not None
            else None
        )

        # History: 現在の履歴を ContextBlock 化し、ContextBuilder 経由で描画する。
        # build_messages() は引数なしで呼ばれるため local_only / local target を既定とし、
        # ContextPolicy で選択された block だけを描画する。
        from src.context.builder import ContextBuilder
        from src.context.providers.history import HistoryContextProvider

        history_block = HistoryContextProvider.collect(self._messages)
        blocks = [
            block
            for block in (tasks_block, history_block)
            if block is not None
        ]
        builder = ContextBuilder(system_content)
        return builder.build_messages(
            blocks,
            privacy="local_only",
            target_local=True,
        )

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
