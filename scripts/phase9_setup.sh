#!/bin/bash
# ===================================================
# Phase 9 セットアップスクリプト
# GPU換装準備 — ソフトウェア側の設定確認・更新
# ===================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"

echo "================================================="
echo " Phase 9: GPU換装 — セットアップ"
echo "================================================="
echo ""

# --- 0. 仮想環境の確認 ---
if [ ! -d "$VENV_DIR" ]; then
    echo "❌ 仮想環境が見つかりません。先に phase2_setup.sh を実行してください。"
    exit 1
fi
source "${VENV_DIR}/bin/activate"
echo "✅ 仮想環境: ${VENV_DIR}"

# --- 1. GPU 検出 ---
echo ""
echo "[1/4] GPU 検出..."
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader 2>/dev/null || echo "unknown")
    GPU_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null || echo "0")
    DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo "unknown")
    echo "  GPU: ${GPU_NAME}"
    echo "  VRAM: ${GPU_VRAM} MB"
    echo "  Driver: ${DRIVER}"
else
    echo "⚠️  nvidia-smi が見つかりません。GPU なし環境です。"
    GPU_NAME="none"
    GPU_VRAM="0"
fi

# --- 2. gpu_config モジュールテスト ---
echo ""
echo "[2/4] GPU 設定モジュール確認..."
python3 -c "
import sys
sys.path.insert(0, '${PROJECT_ROOT}')
from src.service.gpu_config import get_device_config
config = get_device_config()
print(f'  Profile: {config.profile}')
print(f'  STT: device={config.stt_device}, compute={config.stt_compute_type}, model={config.stt_model_size}')
print(f'  Embedding: device={config.embedding_device}')
print(f'  ONNX providers: {config.onnx_providers}')
print(f'  推奨LLMモデル: {config.recommended_model}')
"
echo "✅ gpu_config OK"

# --- 3. onnxruntime 確認 ---
echo ""
echo "[3/4] ONNX Runtime 確認..."
python3 -c "
import onnxruntime as ort
providers = ort.get_available_providers()
print(f'  利用可能プロバイダー: {providers}')
if 'CUDAExecutionProvider' in providers:
    print('  ✅ CUDAExecutionProvider 利用可能')
else:
    print('  ℹ️  CPUExecutionProvider のみ (GPU ONNX が必要な場合: pip install onnxruntime-gpu)')
"

# --- 4. P40 換装チェックリスト ---
echo ""
echo "[4/4] P40 換装チェックリスト..."
if echo "${GPU_VRAM}" | grep -qE '^[2-9][0-9]{4}'; then
    echo "  ✅ 大容量VRAM GPU 検出 — P40モードで動作します"
    echo ""
    echo "  📋 推奨アクション:"
    echo "    1. LLMモデルをアップグレード:"
    echo "       ollama pull qwen2.5:14b-instruct-q4_K_M"
    echo "    2. config/chat_config.json の model を変更:"
    echo '       "model": "qwen2.5:14b-instruct-q4_K_M"'
    echo "    3. STT medium モデルをダウンロード (初回自動):"
    echo "       python3 -c \"from faster_whisper import WhisperModel; WhisperModel('medium', device='cuda')\""
else
    echo "  ℹ️  現在 GTX 1060 相当の環境です。"
    echo "  P40 換装後に再度このスクリプトを実行してください。"
    echo ""
    echo "  📋 P40 換装手順:"
    echo "    1. P40 を物理的に取り付け"
    echo "    2. 電源 500W→650W への換装を推奨 (P40 TDP: 250W)"
    echo "    3. BIOS で iGPU を映像出力に設定 (P40 は映像出力なし)"
    echo "    4. Ubuntu 起動後 nvidia-smi で認識確認"
    echo "    5. bash scripts/phase9_setup.sh を再実行"
    echo "    6. config/chat_config.json の model を 14b に変更"
fi

echo ""
echo "================================================="
echo " ✅ Phase 9 セットアップ完了"
echo "================================================="
