#!/bin/bash
# =============================================================================
# Phase 6: PCログ収集セットアップスクリプト
# psutil + SQLite (Python標準) によるシステムメトリクス収集
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="${PROJECT_ROOT}/.venv"

echo "=========================================="
echo " Phase 6: PCログ収集 セットアップ"
echo "=========================================="

# --- 0. 仮想環境の確認 ---
if [ ! -d "$VENV_DIR" ]; then
    echo "❌ 仮想環境が見つかりません。先に phase2_setup.sh を実行してください。"
    exit 1
fi
source "${VENV_DIR}/bin/activate"

# --- 1. Python パッケージ ---
echo ""
echo "[1/2] Python パッケージのインストール..."
pip install --upgrade pip --quiet
pip install psutil --quiet
echo "✅ psutil インストール完了"

pip list --format=columns 2>/dev/null | grep -iE "psutil"

# --- 2. データディレクトリ作成 ---
echo ""
echo "[2/2] データディレクトリの作成..."
mkdir -p "${PROJECT_ROOT}/data/metrics"
echo "✅ data/metrics/ ディレクトリ作成完了"

# --- 完了 ---
echo ""
echo "=========================================="
echo " ✅ Phase 6 セットアップ完了!"
echo "=========================================="
echo ""
echo "確認:"
echo "  bash scripts/phase6_verify.sh"
echo ""
echo "使い方:"
echo "  # 音声対話 (モニター有効)"
echo "  python -m src.audio.main"
echo ""
echo "  # 音声対話 (モニター無効)"
echo "  python -m src.audio.main --no-monitor"
echo ""
echo "  # Web UI (モニター自動起動)"
echo "  python -m src.web.server"
