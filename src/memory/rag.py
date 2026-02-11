"""
RAGリトリーバー モジュール
Phase 4: ベクトルDB から関連文脈を検索してLLMプロンプトに注入する
"""
from typing import Optional
from src.memory.vectorstore import VectorStore


class RAGRetriever:
    """
    RAG (Retrieval-Augmented Generation) リトリーバー

    ユーザーの発言に対して、過去の会話・知識から関連する文脈を検索し、
    LLMのシステムプロンプトに注入する。
    """

    def __init__(
        self,
        vector_store: VectorStore,
        max_context_items: int = 5,
        max_context_chars: int = 2000,
        relevance_threshold: float = 1.5,
    ):
        """
        Args:
            vector_store: ベクトルストアインスタンス
            max_context_items: 検索結果の最大数
            max_context_chars: コンテキストの最大文字数 (プロンプト肥大化防止)
            relevance_threshold: 類似度の閾値 (コサイン距離、小さいほど類似)
        """
        self.vector_store = vector_store
        self.max_context_items = max_context_items
        self.max_context_chars = max_context_chars
        self.relevance_threshold = relevance_threshold

    def retrieve(self, query: str) -> list[dict]:
        """
        クエリに関連する過去の文脈を検索する

        Args:
            query: ユーザーの発言テキスト

        Returns:
            関連するドキュメントのリスト
        """
        if not self.vector_store.is_initialized():
            return []

        results = self.vector_store.search_all(
            query=query,
            n_results=self.max_context_items,
        )

        # 関連度で足切り
        filtered = [
            r for r in results
            if r.get("distance", 2.0) < self.relevance_threshold
        ]

        return filtered

    def build_context_prompt(self, query: str) -> str:
        """
        検索結果をLLMプロンプトに注入するための文字列を構築する

        Args:
            query: ユーザーの発言テキスト

        Returns:
            プロンプトに追加する文脈文字列（検索結果なしの場合は空文字列）
        """
        results = self.retrieve(query)
        if not results:
            return ""

        context_parts = []
        total_chars = 0

        for r in results:
            doc_type = r.get("metadata", {}).get("type", "unknown")
            doc = r.get("document", "")
            date = r.get("metadata", {}).get("date", "")

            if doc_type == "conversation":
                entry = f"[過去の会話 {date}] {doc}"
            elif doc_type == "knowledge":
                category = r.get("metadata", {}).get("category", "")
                entry = f"[記憶 ({category}) {date}] {doc}"
            else:
                entry = f"[{date}] {doc}"

            # 文字数制限
            if total_chars + len(entry) > self.max_context_chars:
                # 残り文字数分だけ追加
                remaining = self.max_context_chars - total_chars
                if remaining > 50:
                    entry = entry[:remaining] + "..."
                    context_parts.append(entry)
                break

            context_parts.append(entry)
            total_chars += len(entry)

        if not context_parts:
            return ""

        context = "\n".join(context_parts)
        return (
            "\n\n--- 関連する過去の記憶 ---\n"
            f"{context}\n"
            "--- 記憶ここまで ---\n"
            "上記の過去の記憶を参考にしつつ、自然に応答してください。"
            "記憶の内容を不自然に持ち出す必要はありません。"
        )

    def store_turn(
        self,
        user_message: str,
        assistant_message: str,
        session_id: str = "",
    ) -> Optional[str]:
        """
        会話ターンをベクトルDBに保存する

        Args:
            user_message: ユーザーの発言
            assistant_message: AIの応答
            session_id: セッションID

        Returns:
            保存したドキュメントのID
        """
        if not self.vector_store.is_initialized():
            return None

        try:
            doc_id = self.vector_store.store_conversation_turn(
                user_message=user_message,
                assistant_message=assistant_message,
                session_id=session_id,
            )
            return doc_id
        except Exception as e:
            print(f"[RAG] 会話保存エラー: {e}")
            return None

    def store_knowledge(
        self,
        text: str,
        category: str = "general",
        source: str = "user_input",
    ) -> Optional[str]:
        """知識をベクトルDBに保存する"""
        if not self.vector_store.is_initialized():
            return None
        try:
            return self.vector_store.store_knowledge(
                text=text, category=category, source=source,
            )
        except Exception as e:
            print(f"[RAG] 知識保存エラー: {e}")
            return None

    def get_stats(self) -> dict:
        """RAGの統計情報"""
        return self.vector_store.get_stats()
