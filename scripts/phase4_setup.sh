#!/bin/bash
# =============================================================================
# Phase 4: 長期記憶 (RAG) セットアップスクリプト
# ChromaDB + sentence-transformers + 埋め込みモデル
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="${PROJECT_ROOT}/.venv"

echo "=========================================="
echo " Phase 4: 長期記憶 (RAG) セットアップ"
echo "=========================================="

# --- 0. 仮想環境の確認 ---
if [ ! -d "$VENV_DIR" ]; then
    echo "❌ 仮想環境が見つかりません。先に phase2_setup.sh を実行してください。"
    exit 1
fi
source "${VENV_DIR}/bin/activate"

# --- 1. Python パッケージ ---
echo ""
echo "[1/3] Python パッケージのインストール..."
pip install --upgrade pip --quiet
pip install chromadb sentence-transformers --quiet
echo "✅ Python パッケージインストール完了"

pip list --format=columns 2>/dev/null | grep -iE "chromadb|sentence-transformers|torch"

# --- 2. 埋め込みモデルのダウンロード ---
echo ""
echo "[2/3] 埋め込みモデルのダウンロード..."
echo "  モデル: intfloat/multilingual-e5-small (~120MB)"
echo "  (初回はダウンロードに数分かかります)"
python3 -c "
from sentence_transformers import SentenceTransformer
import time
start = time.time()
model = SentenceTransformer('intfloat/multilingual-e5-small', device='cpu')
elapsed = time.time() - start
dim = model.get_sentence_embedding_dimension()
print(f'✅ 埋め込みモデルロード完了 ({elapsed:.1f}秒, {dim}次元)')

# テストエンコード
import numpy as np
vec = model.encode(['query: テスト'], normalize_embeddings=True)
print(f'  テストベクトル: shape={vec.shape}, norm={np.linalg.norm(vec[0]):.4f}')
"

# --- 3. ディレクトリ作成 ---
echo ""
echo "[3/3] ディレクトリの作成..."
mkdir -p "${PROJECT_ROOT}/data/vectordb"
echo "✅ ディレクトリ作成完了"

# --- 完了 ---
echo ""
echo "=========================================="
echo " ✅ Phase 4 セットアップ完了!"
echo "=========================================="
echo ""
echo "RAGは音声対話・テキスト対話・Web UIに自動統合されます。"
echo "会話が長期記憶に自動保存され、関連する過去の文脈が"
echo "LLMのプロンプトに注入されます。"
echo ""
echo "RAGを無効にして起動する場合:"
echo "  python src/audio/main.py --no-rag"
echo ""
