#!/bin/bash
# ===================================================
# Phase 10 検証スクリプト
# ウェイクワード検知 — モジュール・機能テスト
# ===================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"

PASS=0
FAIL=0

check() {
    local name="$1"
    local cmd="$2"
    echo -n "  ${name}... "
    if eval "${cmd}" > /dev/null 2>&1; then
        echo "✅ OK"
        PASS=$((PASS + 1))
    else
        echo "❌ FAIL"
        FAIL=$((FAIL + 1))
    fi
}

echo "================================================="
echo " Phase 10: ウェイクワード検知 — 検証"
echo "================================================="
echo ""

# 仮想環境
if [ ! -d "$VENV_DIR" ]; then
    echo "❌ 仮想環境が見つかりません。"
    exit 1
fi
source "${VENV_DIR}/bin/activate"

# --- ファイル存在確認 ---
echo "[ファイル存在確認]"
check "wakeword.py" "test -f ${PROJECT_ROOT}/src/audio/wakeword.py"
check "phase10_setup.sh" "test -f ${PROJECT_ROOT}/scripts/phase10_setup.sh"
check "phase10_verify.sh" "test -f ${PROJECT_ROOT}/scripts/phase10_verify.sh"

# --- パッケージインポート ---
echo ""
echo "[パッケージインポート]"
check "openwakeword" "python3 -c \"import openwakeword\""

# --- モジュールインポート ---
echo ""
echo "[モジュールインポート]"
check "WakeWordDetector" "python3 -c \"import sys; sys.path.insert(0, '${PROJECT_ROOT}'); from src.audio.wakeword import WakeWordDetector\""

# --- インスタンス化テスト ---
echo ""
echo "[インスタンス化テスト]"
check "WakeWordDetector(hey_jarvis)" "python3 -c \"
import sys; sys.path.insert(0, '${PROJECT_ROOT}')
from src.audio.wakeword import WakeWordDetector
d = WakeWordDetector(model_names=['hey_jarvis'], threshold=0.5)
assert d.threshold == 0.5
assert d.frame_size == 1280
assert d.sample_rate == 16000
assert not d.is_loaded
\""

check "WakeWordDetector(default)" "python3 -c \"
import sys; sys.path.insert(0, '${PROJECT_ROOT}')
from src.audio.wakeword import WakeWordDetector
d = WakeWordDetector()
assert d.model_names is None
assert d.threshold == 0.5
\""

# --- モデルロードテスト ---
echo ""
echo "[モデルロードテスト]"
check "load()" "python3 -c \"
import sys; sys.path.insert(0, '${PROJECT_ROOT}')
from src.audio.wakeword import WakeWordDetector
d = WakeWordDetector(model_names=['hey_jarvis'])
assert d.load() == True
assert d.is_loaded
assert len(d.loaded_models) > 0
\""

# --- process_frame テスト (無音→検知なし) ---
echo ""
echo "[process_frame テスト]"
check "process_frame(silence)" "python3 -c \"
import sys; sys.path.insert(0, '${PROJECT_ROOT}')
import numpy as np
from src.audio.wakeword import WakeWordDetector
d = WakeWordDetector(model_names=['hey_jarvis'])
d.load()
# 無音フレーム (80ms @ 16kHz)
silence = np.zeros(1280, dtype=np.float32)
result = d.process_frame(silence)
assert result is None, 'Expected None for silence'
\""

check "get_info()" "python3 -c \"
import sys; sys.path.insert(0, '${PROJECT_ROOT}')
from src.audio.wakeword import WakeWordDetector
d = WakeWordDetector(model_names=['hey_jarvis'])
d.load()
info = d.get_info()
assert info['loaded'] == True
assert 'hey_jarvis' in str(info['models'])
assert info['threshold'] == 0.5
\""

# --- reset テスト ---
echo ""
echo "[reset テスト]"
check "reset()" "python3 -c \"
import sys; sys.path.insert(0, '${PROJECT_ROOT}')
from src.audio.wakeword import WakeWordDetector
d = WakeWordDetector(model_names=['hey_jarvis'])
d.load()
d.reset()  # should not raise
\""

# --- cleanup テスト ---
check "cleanup()" "python3 -c \"
import sys; sys.path.insert(0, '${PROJECT_ROOT}')
from src.audio.wakeword import WakeWordDetector
d = WakeWordDetector(model_names=['hey_jarvis'])
d.load()
d.cleanup()
assert not d.is_loaded
\""

# --- CLI ヘルプ確認 ---
echo ""
echo "[CLI ヘルプ確認]"
check "--wakeword in help" "python3 ${PROJECT_ROOT}/src/audio/main.py --help 2>&1 | grep -q 'wakeword'"

# --- 結果 ---
echo ""
echo "================================================="
TOTAL=$((PASS + FAIL))
echo " 結果: ${PASS}/${TOTAL} テスト成功"
if [ "$FAIL" -gt 0 ]; then
    echo " ⚠️  ${FAIL} テスト失敗"
else
    echo " ✅ 全テスト成功！"
fi
echo "================================================="
