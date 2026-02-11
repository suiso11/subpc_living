"""
埋め込み (Embedding) モジュール
Phase 4: テキストをベクトルに変換する
multilingual-e5-small を使用（日本語対応、ローカル実行）
"""
import time
import numpy as np
from typing import Optional
from pathlib import Path


class EmbeddingModel:
    """
    sentence-transformers ベースの埋め込みモデル

    multilingual-e5-small: 多言語対応、384次元、~120MB
    完全ローカル実行、API不要
    """

    # 推奨モデル: 高精度かつ軽量
    DEFAULT_MODEL = "intfloat/multilingual-e5-small"
    # 上位モデル候補:
    #   intfloat/multilingual-e5-base  (278M, 768次元)
    #   intfloat/multilingual-e5-large (560M, 1024次元)

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cpu",
        cache_dir: Optional[str] = None,
    ):
        """
        Args:
            model_name: HuggingFace モデル名
            device: 実行デバイス ("cpu" or "cuda")
            cache_dir: モデルキャッシュディレクトリ
        """
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self._model = None

    def load(self) -> None:
        """モデルをロード（初回は自動ダウンロード）"""
        if self._model is not None:
            return

        from sentence_transformers import SentenceTransformer

        print(f"[Embedding] モデル '{self.model_name}' をロード中...")
        start = time.time()
        kwargs = {"device": self.device}
        if self.cache_dir:
            kwargs["cache_folder"] = self.cache_dir
        self._model = SentenceTransformer(self.model_name, **kwargs)
        elapsed = time.time() - start
        dim = self._model.get_sentence_embedding_dimension()
        print(f"[Embedding] モデルロード完了 ({elapsed:.1f}秒, {dim}次元)")

    def encode(self, texts: str | list[str], prefix: str = "query: ") -> np.ndarray:
        """
        テキストをベクトルに変換する

        E5モデルはクエリに "query: "、文書に "passage: " プレフィックスが必要。

        Args:
            texts: テキスト or テキストのリスト
            prefix: プレフィックス ("query: " or "passage: ")

        Returns:
            ベクトル (shape: [n, dim])
        """
        self.load()

        if isinstance(texts, str):
            texts = [texts]

        # E5モデル用プレフィックス付与
        prefixed = [f"{prefix}{t}" for t in texts]

        embeddings = self._model.encode(
            prefixed,
            normalize_embeddings=True,  # コサイン類似度用に正規化
            show_progress_bar=False,
        )
        return embeddings

    def encode_query(self, text: str) -> np.ndarray:
        """検索クエリをベクトル化"""
        return self.encode(text, prefix="query: ")[0]

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        """文書（保存対象のテキスト）をベクトル化"""
        return self.encode(texts, prefix="passage: ")

    @property
    def dimension(self) -> int:
        """ベクトルの次元数"""
        self.load()
        return self._model.get_sentence_embedding_dimension()

    def is_loaded(self) -> bool:
        return self._model is not None
