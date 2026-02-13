#!/bin/bash
# ===================================================
# Phase 9 検証スクリプト
# GPU換装 — モジュールのデバイス設定確認
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
        ((PASS++))
    else
        echo "❌ FAIL"
        ((FAIL++))
    fi
}

echo "================================================="
echo " Phase 9: GPU換装 — 検証"
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
check "gpu_config.py" "test -f ${PROJECT_ROOT}/src/service/gpu_config.py"
check "phase9_setup.sh" "test -f ${PROJECT_ROOT}/scripts/phase9_setup.sh"

# --- モジュールインポート ---
echo ""
echo "[モジュールインポート]"
check "gpu_config" "python3 -c \"import sys; sys.path.insert(0, '${PROJECT_ROOT}'); from src.service.gpu_config import detect_gpu, get_device_config, DeviceConfig\""
check "power (GPU presets)" "python3 -c \"import sys; sys.path.insert(0, '${PROJECT_ROOT}'); from src.service.power import GpuPowerManager, GPU_POWER_PRESETS\""

# --- GPU検出 ---
echo ""
echo "[GPU検出テスト]"
check "detect_gpu()" "python3 -c \"
import sys; sys.path.insert(0, '${PROJECT_ROOT}')
from src.service.gpu_config import detect_gpu
info = detect_gpu()
assert isinstance(info.available, bool)
assert isinstance(info.vram_mb, int)
\""

check "get_device_config()" "python3 -c \"
import sys; sys.path.insert(0, '${PROJECT_ROOT}')
from src.service.gpu_config import get_device_config
config = get_device_config()
assert config.profile in ('p40', 'gtx1060', 'cpu')
assert config.stt_device in ('cpu', 'cuda')
assert config.embedding_device in ('cpu', 'cuda')
assert 'CPUExecutionProvider' in config.onnx_providers
\""

# --- STT device=auto ---
echo ""
echo "[STT device=auto]"
check "WhisperSTT(auto)" "python3 -c \"
import sys; sys.path.insert(0, '${PROJECT_ROOT}')
from src.audio.stt import WhisperSTT
stt = WhisperSTT()  # device=auto
assert stt.device in ('cpu', 'cuda')
assert stt.compute_type in ('int8', 'float16')
assert stt.model_size in ('small', 'medium', 'large')
\""

check "WhisperSTT(explicit)" "python3 -c \"
import sys; sys.path.insert(0, '${PROJECT_ROOT}')
from src.audio.stt import WhisperSTT
stt = WhisperSTT(model_size='small', device='cpu', compute_type='int8')
assert stt.device == 'cpu'
assert stt.compute_type == 'int8'
assert stt.model_size == 'small'
\""

# --- Embedding device=auto ---
echo ""
echo "[Embedding device=auto]"
check "EmbeddingModel(auto)" "python3 -c \"
import sys; sys.path.insert(0, '${PROJECT_ROOT}')
from src.memory.embedding import EmbeddingModel
emb = EmbeddingModel()  # device=auto
assert emb.device in ('cpu', 'cuda')
\""

check "EmbeddingModel(explicit)" "python3 -c \"
import sys; sys.path.insert(0, '${PROJECT_ROOT}')
from src.memory.embedding import EmbeddingModel
emb = EmbeddingModel(device='cpu')
assert emb.device == 'cpu'
\""

# --- VectorStore device=auto ---
echo ""
echo "[VectorStore device=auto]"
check "VectorStore(auto)" "python3 -c \"
import sys; sys.path.insert(0, '${PROJECT_ROOT}')
from src.memory.vectorstore import VectorStore
vs = VectorStore(embedding_device='auto')
assert vs.embedding_device == 'auto'
\""

# --- Vision ONNX providers ---
echo ""
echo "[Vision ONNX providers]"
check "_detect_onnx_providers()" "python3 -c \"
import sys; sys.path.insert(0, '${PROJECT_ROOT}')
from src.vision.detector import _detect_onnx_providers
providers = _detect_onnx_providers()
assert 'CPUExecutionProvider' in providers
assert isinstance(providers, list)
\""

# --- Power presets ---
echo ""
echo "[Power GPU presets]"
check "GpuPowerManager(auto)" "python3 -c \"
import sys; sys.path.insert(0, '${PROJECT_ROOT}')
from src.service.power import GpuPowerManager
mgr = GpuPowerManager()
assert mgr.idle_watts > 0
assert mgr.active_watts > mgr.idle_watts
\""

check "GPU_POWER_PRESETS" "python3 -c \"
import sys; sys.path.insert(0, '${PROJECT_ROOT}')
from src.service.power import GPU_POWER_PRESETS
assert 'P40' in GPU_POWER_PRESETS
assert 'GTX 1060' in GPU_POWER_PRESETS
assert GPU_POWER_PRESETS['P40'] == (100, 250)
assert GPU_POWER_PRESETS['GTX 1060'] == (80, 120)
\""

# --- gpu_config CLI ---
echo ""
echo "[gpu_config CLI]"
check "gpu_config main()" "python3 -c \"
import sys; sys.path.insert(0, '${PROJECT_ROOT}')
from src.service.gpu_config import main
main()
\""

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
