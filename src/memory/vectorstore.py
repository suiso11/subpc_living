"""
ベクトルストア (Vector Store) モジュール
Phase 4: ChromaDB を使用した会話履歴・知識のベクトル保存・検索
"""
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional


class VectorStore:
    """
    ChromaDB ベースのベクトルストア

    会話の各ターン（user + assistant）をチャンク化してベクトル保存し、
    セマンティック検索で過去の文脈を引き出す。
    """

    COLLECTION_CONVERSATIONS = "conversations"
    COLLECTION_KNOWLEDGE = "knowledge"

    def __init__(
        self,
        persist_dir: str = "data/vectordb",
        embedding_model: str = "intfloat/multilingual-e5-small",
        embedding_device: str = "cpu",
    ):
        """
        Args:
            persist_dir: ChromaDB の永続化ディレクトリ
            embedding_model: 埋め込みモデル名
            embedding_device: 実行デバイス
        """
        self.persist_dir = Path(persist_dir)
        self.embedding_model_name = embedding_model
        self.embedding_device = embedding_device

        self._client = None
        self._conversations = None
        self._knowledge = None
        self._embedding_fn = None

    def initialize(self) -> None:
        """ChromaDB クライアントとコレクションを初期化"""
        import chromadb

        self.persist_dir.mkdir(parents=True, exist_ok=True)

        print(f"[VectorStore] ChromaDB 初期化中 (persist_dir={self.persist_dir})...")
        start = time.time()

        self._client = chromadb.PersistentClient(path=str(self.persist_dir))

        # 埋め込み関数の準備
        self._embedding_fn = self._create_embedding_function()

        # コレクション作成/取得
        self._conversations = self._client.get_or_create_collection(
            name=self.COLLECTION_CONVERSATIONS,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        self._knowledge = self._client.get_or_create_collection(
            name=self.COLLECTION_KNOWLEDGE,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

        elapsed = time.time() - start
        conv_count = self._conversations.count()
        know_count = self._knowledge.count()
        print(f"[VectorStore] 初期化完了 ({elapsed:.1f}秒)")
        print(f"  conversations: {conv_count} 件")
        print(f"  knowledge: {know_count} 件")

    def _create_embedding_function(self):
        """ChromaDB 用の埋め込み関数を作成"""
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        print(f"[VectorStore] 埋め込みモデル '{self.embedding_model_name}' をロード中...")
        start = time.time()
        fn = SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_model_name,
            device=self.embedding_device,
        )
        elapsed = time.time() - start
        print(f"[VectorStore] 埋め込みモデルロード完了 ({elapsed:.1f}秒)")
        return fn

    # --- 会話履歴の保存 ---

    def store_conversation_turn(
        self,
        user_message: str,
        assistant_message: str,
        session_id: str = "",
        metadata: Optional[dict] = None,
    ) -> str:
        """
        1ターンの会話をベクトルDBに保存する

        user と assistant のメッセージをまとめて1ドキュメントとして保存。
        検索時は会話のペアが返される。

        Args:
            user_message: ユーザーの発言
            assistant_message: AIの応答
            session_id: セッションID
            metadata: 追加メタデータ

        Returns:
            保存したドキュメントのID
        """
        if self._conversations is None:
            raise RuntimeError("VectorStore が初期化されていません。initialize() を呼んでください。")

        doc_id = str(uuid.uuid4())
        now = datetime.now()

        # 検索しやすい形式でテキストを構成
        document = f"ユーザー: {user_message}\nAI: {assistant_message}"

        meta = {
            "session_id": session_id,
            "user_message": user_message,
            "assistant_message": assistant_message,
            "timestamp": now.isoformat(),
            "date": now.strftime("%Y-%m-%d"),
            "type": "conversation",
        }
        if metadata:
            meta.update(metadata)

        self._conversations.add(
            ids=[doc_id],
            documents=[document],
            metadatas=[meta],
        )
        return doc_id

    def store_conversation_batch(
        self,
        turns: list[dict],
        session_id: str = "",
    ) -> list[str]:
        """
        複数ターンの会話を一括保存する

        Args:
            turns: [{"user": "...", "assistant": "..."}, ...]
            session_id: セッションID

        Returns:
            保存したドキュメントIDのリスト
        """
        if self._conversations is None:
            raise RuntimeError("VectorStore が初期化されていません。")

        ids = []
        documents = []
        metadatas = []
        now = datetime.now()

        for turn in turns:
            user_msg = turn.get("user", "")
            asst_msg = turn.get("assistant", "")
            if not user_msg:
                continue

            doc_id = str(uuid.uuid4())
            ids.append(doc_id)
            documents.append(f"ユーザー: {user_msg}\nAI: {asst_msg}")
            metadatas.append({
                "session_id": session_id,
                "user_message": user_msg,
                "assistant_message": asst_msg,
                "timestamp": now.isoformat(),
                "date": now.strftime("%Y-%m-%d"),
                "type": "conversation",
            })

        if ids:
            self._conversations.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )

        return ids

    # --- 知識の保存 ---

    def store_knowledge(
        self,
        text: str,
        category: str = "general",
        source: str = "",
        metadata: Optional[dict] = None,
    ) -> str:
        """
        知識・メモをベクトルDBに保存する

        会話から抽出した要約、ユーザーの嗜好、重要な情報等を保存。

        Args:
            text: 保存するテキスト
            category: カテゴリ (general, preference, schedule, fact, etc.)
            source: 情報源 (conversation, user_input, extraction, etc.)
            metadata: 追加メタデータ

        Returns:
            保存したドキュメントのID
        """
        if self._knowledge is None:
            raise RuntimeError("VectorStore が初期化されていません。")

        doc_id = str(uuid.uuid4())
        now = datetime.now()

        meta = {
            "category": category,
            "source": source,
            "timestamp": now.isoformat(),
            "date": now.strftime("%Y-%m-%d"),
            "type": "knowledge",
        }
        if metadata:
            meta.update(metadata)

        self._knowledge.add(
            ids=[doc_id],
            documents=[text],
            metadatas=[meta],
        )
        return doc_id

    # --- 検索 ---

    def search_conversations(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """
        過去の会話をセマンティック検索する

        Args:
            query: 検索クエリ
            n_results: 返す結果の最大数
            where: メタデータフィルタ (ChromaDB where句)

        Returns:
            [{"id": ..., "document": ..., "metadata": ..., "distance": ...}, ...]
        """
        if self._conversations is None:
            raise RuntimeError("VectorStore が初期化されていません。")

        kwargs = {
            "query_texts": [query],
            "n_results": min(n_results, self._conversations.count() or 1),
        }
        if where:
            kwargs["where"] = where

        results = self._conversations.query(**kwargs)
        return self._format_results(results)

    def search_knowledge(
        self,
        query: str,
        n_results: int = 5,
        category: Optional[str] = None,
    ) -> list[dict]:
        """
        知識ベースをセマンティック検索する

        Args:
            query: 検索クエリ
            n_results: 返す結果の最大数
            category: カテゴリでフィルタ

        Returns:
            [{"id": ..., "document": ..., "metadata": ..., "distance": ...}, ...]
        """
        if self._knowledge is None:
            raise RuntimeError("VectorStore が初期化されていません。")

        kwargs = {
            "query_texts": [query],
            "n_results": min(n_results, self._knowledge.count() or 1),
        }
        if category:
            kwargs["where"] = {"category": category}

        results = self._knowledge.query(**kwargs)
        return self._format_results(results)

    def search_all(
        self,
        query: str,
        n_results: int = 5,
    ) -> list[dict]:
        """会話と知識の両方を検索し、関連度順にマージして返す"""
        conv_results = []
        know_results = []

        if self._conversations.count() > 0:
            conv_results = self.search_conversations(query, n_results=n_results)
        if self._knowledge.count() > 0:
            know_results = self.search_knowledge(query, n_results=n_results)

        # 距離でソート（小さいほど類似）してマージ
        merged = conv_results + know_results
        merged.sort(key=lambda x: x.get("distance", 1.0))
        return merged[:n_results]

    # --- ユーティリティ ---

    @staticmethod
    def _format_results(results: dict) -> list[dict]:
        """ChromaDB query結果を使いやすい形式に変換"""
        formatted = []
        if not results or not results.get("ids"):
            return formatted

        ids = results["ids"][0]
        docs = results["documents"][0] if results.get("documents") else [None] * len(ids)
        metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(ids)
        dists = results["distances"][0] if results.get("distances") else [0.0] * len(ids)

        for i, doc_id in enumerate(ids):
            formatted.append({
                "id": doc_id,
                "document": docs[i],
                "metadata": metas[i],
                "distance": dists[i],
            })
        return formatted

    @property
    def conversation_count(self) -> int:
        """保存されている会話数"""
        return self._conversations.count() if self._conversations else 0

    @property
    def knowledge_count(self) -> int:
        """保存されている知識数"""
        return self._knowledge.count() if self._knowledge else 0

    def get_stats(self) -> dict:
        """統計情報を返す"""
        return {
            "conversations": self.conversation_count,
            "knowledge": self.knowledge_count,
            "persist_dir": str(self.persist_dir),
            "embedding_model": self.embedding_model_name,
        }

    def is_initialized(self) -> bool:
        return self._client is not None
