#!/bin/bash
# =============================================================================
# Phase 7: パーソナライズ セットアップスクリプト
# ユーザープロファイル管理 + 会話要約 + プリロード + プロアクティブ発話
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="${PROJECT_ROOT}/.venv"

echo "=========================================="
echo " Phase 7: パーソナライズ セットアップ"
echo "=========================================="

# --- 0. 仮想環境の確認 ---
if [ ! -d "$VENV_DIR" ]; then
    echo "❌ 仮想環境が見つかりません。先に phase2_setup.sh を実行してください。"
    exit 1
fi
source "${VENV_DIR}/bin/activate"

# --- 1. データディレクトリ作成 ---
echo ""
echo "[1/3] データディレクトリの作成..."
mkdir -p "${PROJECT_ROOT}/data/profile/summaries"
echo "✅ data/profile/ ディレクトリ作成完了"
echo "✅ data/profile/summaries/ ディレクトリ作成完了"

# --- 2. デフォルトプロファイル作成 ---
echo ""
echo "[2/3] デフォルトプロファイル確認..."
PROFILE_PATH="${PROJECT_ROOT}/data/profile/user_profile.json"
if [ -f "$PROFILE_PATH" ]; then
    echo "✅ プロファイルファイル既存: ${PROFILE_PATH}"
else
    cat > "$PROFILE_PATH" << 'EOF'
{
  "name": "",
  "nickname": "",
  "preferences": {},
  "habits": {},
  "schedule": [],
  "notes": [],
  "extracted_facts": [],
  "updated_at": ""
}
EOF
    echo "✅ デフォルトプロファイル作成: ${PROFILE_PATH}"
fi

# --- 3. Pythonモジュール確認 ---
echo ""
echo "[3/3] Pythonモジュール確認..."

python3 -c "
import sys; sys.path.insert(0, '${PROJECT_ROOT}')
from src.persona.profile import UserProfile
from src.persona.summarizer import ConversationSummarizer
from src.persona.preloader import SessionPreloader
from src.persona.proactive import ProactiveEngine
print('OK')
" && echo "✅ 全モジュールインポート成功" || echo "❌ モジュールインポート失敗"

# --- 完了 ---
echo ""
echo "=========================================="
echo " ✅ Phase 7 セットアップ完了!"
echo "=========================================="
echo ""
echo "確認:"
echo "  bash scripts/phase7_verify.sh"
echo ""
echo "使い方:"
echo "  # 音声対話 (パーソナライズ有効)"
echo "  python -m src.audio.main"
echo ""
echo "  # 音声対話 (パーソナライズ無効)"
echo "  python -m src.audio.main --no-persona"
echo ""
echo "  # Web UI (パーソナライズ自動起動)"
echo "  python -m src.web.server"
echo ""
echo "  # プロファイル編集"
echo "  data/profile/user_profile.json を直接編集"
echo "  または Web API: POST /api/persona/profile"
