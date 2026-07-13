#!/bin/bash
# ===================================================
# Phase 10 セットアップスクリプト
# ウェイクワード検知 — OpenWakeWord インストール
# ===================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"

echo "================================================="
echo " Phase 10: ウェイクワード検知 — セットアップ"
echo "================================================="
echo ""

# --- 0. 仮想環境の確認 ---
if [ ! -d "$VENV_DIR" ]; then
    echo "❌ 仮想環境が見つかりません。先に phase2_setup.sh を実行してください。"
    exit 1
fi
source "${VENV_DIR}/bin/activate"
echo "✅ 仮想環境: ${VENV_DIR}"

# --- 1. openwakeword インストール ---
echo ""
echo "[1/2] openwakeword パッケージインストール..."
pip install openwakeword 2>&1 | tail -5
echo "✅ openwakeword インストール完了"

# --- 2. モジュール確認 ---
echo ""
echo "[2/2] モジュール確認..."
python3 -c "
import sys
sys.path.insert(0, '${PROJECT_ROOT}')
from src.audio.wakeword import WakeWordDetector
detector = WakeWordDetector(model_names=['hey_jarvis'])
print('  WakeWordDetector インスタンス化OK')
if detector.load():
    print(f'  ロード済みモデル: {detector.loaded_models}')
    print(f'  閾値: {detector.threshold}')
    print(f'  フレームサイズ: {detector.frame_size} samples (80ms)')
else:
    print('  ⚠️  モデルロードに失敗')
"
echo "✅ モジュール確認完了"

echo ""
echo "================================================="
echo " ✅ Phase 10 セットアップ完了"
echo ""
echo " 使い方:"
echo "   python src/audio/main.py --wakeword"
echo "   python src/audio/main.py --wakeword --wakeword-model hey_jarvis"
echo "================================================="
