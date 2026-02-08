#!/bin/bash
# =============================================================================
# Phase 2: テキスト対話セットアップスクリプト
# Python仮想環境の作成 + 依存パッケージのインストール
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="${PROJECT_ROOT}/.venv"

echo "=========================================="
echo " Phase 2: テキスト対話セットアップ"
echo "=========================================="

# --- 1. Python仮想環境の作成 ---
echo ""
echo "[1/4] Python仮想環境の作成..."
if [ -d "$VENV_DIR" ]; then
    echo "仮想環境は既に存在します: ${VENV_DIR}"
else
    python3 -m venv "$VENV_DIR"
    echo "✅ 仮想環境を作成しました: ${VENV_DIR}"
fi

# 仮想環境を有効化
source "${VENV_DIR}/bin/activate"
echo "Python: $(python --version)"
echo "pip: $(pip --version)"

# --- 2. pip アップグレード ---
echo ""
echo "[2/4] pip アップグレード..."
pip install --upgrade pip --quiet

# --- 3. 依存パッケージのインストール ---
echo ""
echo "[3/4] 依存パッケージのインストール..."
pip install -r "${PROJECT_ROOT}/requirements.txt" --quiet
echo "✅ パッケージインストール完了"

# インストール済みパッケージ一覧
echo ""
echo "インストール済みパッケージ:"
pip list --format=columns | grep -iE "httpx|anyio|certifi|httpcore|idna|sniffio"

# --- 4. ディレクトリ作成 ---
echo ""
echo "[4/4] データディレクトリの作成..."
mkdir -p "${PROJECT_ROOT}/data/chat_history"
mkdir -p "${PROJECT_ROOT}/config"
echo "✅ ディレクトリ作成完了"

# --- 完了 ---
echo ""
echo "=========================================="
echo " ✅ Phase 2 セットアップ完了!"
echo "=========================================="
echo ""
echo "テキスト対話を開始するには:"
echo "  source ${VENV_DIR}/bin/activate"
echo "  python ${PROJECT_ROOT}/src/chat/main.py"
echo ""
