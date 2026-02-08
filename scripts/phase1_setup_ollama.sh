#!/bin/bash
# =============================================================================
# Phase 1: Ollama セットアップ + 7Bモデル導入
# 対象: Ubuntu 24.04 LTS + GTX 1060 (VRAM 6GB)
# 前提: NVIDIA Driver + CUDA が既にインストール済み
# =============================================================================
set -e

echo "=========================================="
echo " Phase 1: Ollama + LLM Setup"
echo " Target: GTX 1060 (6GB VRAM)"
echo "=========================================="

# --- 事前チェック ---
echo ""
echo "[0/4] 事前チェック..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi が見つかりません。先に phase1_setup_nvidia.sh を実行してください。"
    exit 1
fi

echo "GPU状態:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# --- 1. Ollama のインストール ---
echo "[1/4] Ollama のインストール..."
if command -v ollama &> /dev/null; then
    echo "Ollama は既にインストールされています。"
    ollama --version
else
    curl -fsSL https://ollama.com/install.sh | sh
    echo "✅ Ollama インストール完了"
fi

# --- 2. Ollama サービスの起動確認 ---
echo ""
echo "[2/4] Ollama サービスの確認..."
if systemctl is-active --quiet ollama; then
    echo "✅ Ollama サービスは起動済みです"
else
    echo "Ollama サービスを起動します..."
    sudo systemctl enable ollama
    sudo systemctl start ollama
    sleep 2
    echo "✅ Ollama サービスを起動しました"
fi

# --- 3. 7B モデルのダウンロード ---
echo ""
echo "[3/4] LLM モデルのダウンロード..."
echo ""
echo "GTX 1060 (6GB VRAM) に最適なモデル候補:"
echo "  1. qwen2.5:7b-instruct-q4_K_M  — 日本語性能◎、推奨"
echo "  2. gemma2:7b-instruct-q4_K_M   — Google製、日本語○"
echo "  3. llama3.1:8b-instruct-q4_K_M — Meta製、英語◎日本語△"
echo ""

# デフォルトで Qwen2.5 7B をダウンロード（日本語性能が高い）
MODEL="qwen2.5:7b-instruct-q4_K_M"
echo ">>> ${MODEL} をダウンロードします..."
ollama pull "${MODEL}"
echo "✅ ${MODEL} ダウンロード完了"

# --- 4. 動作テスト ---
echo ""
echo "[4/4] 動作テスト..."
echo ""
echo ">>> テストプロンプト送信中..."
RESPONSE=$(ollama run "${MODEL}" "こんにちは！自己紹介を短くお願いします。" 2>&1)
echo ""
echo "--- LLM応答 ---"
echo "${RESPONSE}"
echo "----------------"
echo ""

# GPU使用状況の確認
echo ">>> GPU VRAM 使用状況:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
echo ""

echo "=========================================="
echo " ✅ Phase 1 セットアップ完了！"
echo ""
echo " インストール済み:"
echo "   - NVIDIA Driver + CUDA"
echo "   - Ollama"
echo "   - ${MODEL}"
echo ""
echo " 使い方:"
echo "   ollama run ${MODEL}"
echo ""
echo " API (他のプログラムから利用):"
echo "   curl http://localhost:11434/api/generate \\"
echo "     -d '{\"model\": \"${MODEL}\", \"prompt\": \"こんにちは\"}'"
echo ""
echo " 次のステップ: Phase 2 (テキスト対話システム構築)"
echo "=========================================="
