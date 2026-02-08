#!/bin/bash
# =============================================================================
# Phase 1: 環境検証スクリプト
# すべてのコンポーネントが正しくセットアップされたか確認する
# =============================================================================

echo "=========================================="
echo " Phase 1: 環境検証"
echo "=========================================="

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

# --- OS ---
echo ""
echo "[OS]"
check "Ubuntu 24.04" "grep -q '24.04' /etc/os-release"

# --- NVIDIA ---
echo ""
echo "[NVIDIA]"
check "nvidia-smi" "command -v nvidia-smi"
check "GPU検出" "nvidia-smi --query-gpu=name --format=csv,noheader | grep -qi 'gtx\|tesla\|geforce'"
check "CUDA (nvcc)" "command -v nvcc"

# --- Ollama ---
echo ""
echo "[Ollama]"
check "ollama コマンド" "command -v ollama"
check "ollama サービス起動" "systemctl is-active --quiet ollama"
check "ollama API応答" "curl -s http://localhost:11434/api/tags > /dev/null"
check "7Bモデル存在" "ollama list | grep -qi '7b\|8b'"

# --- 基本ツール ---
echo ""
echo "[基本ツール]"
check "git" "command -v git"
check "python3" "command -v python3"
check "pip3" "command -v pip3"
check "curl" "command -v curl"

# --- システム情報 ---
echo ""
echo "=========================================="
echo " システム情報"
echo "=========================================="
echo ""
echo "OS: $(lsb_release -d 2>/dev/null | cut -f2)"
echo "Kernel: $(uname -r)"
echo "CPU: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)"
echo "RAM: $(free -h | awk '/^Mem:/ {print $2}')"
echo ""
if command -v nvidia-smi &> /dev/null; then
    echo "GPU:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/  /'
    echo ""
    echo "CUDA: $(nvcc --version 2>/dev/null | grep 'release' | awk '{print $6}' | sed 's/,//')"
fi
echo ""
if command -v ollama &> /dev/null; then
    echo "Ollama: $(ollama --version 2>&1)"
    echo "モデル一覧:"
    ollama list 2>/dev/null | sed 's/^/  /'
fi

# --- 結果 ---
echo ""
echo "=========================================="
echo " 結果: ✅ ${PASS} PASS / ❌ ${FAIL} FAIL"
echo "=========================================="

if [ ${FAIL} -eq 0 ]; then
    echo ""
    echo " 🎉 Phase 1 セットアップは正常です！"
    echo " 次のステップ: Phase 2 (テキスト対話システム)"
else
    echo ""
    echo " ⚠️  ${FAIL} 件の問題があります。上記のFAIL項目を確認してください。"
fi
