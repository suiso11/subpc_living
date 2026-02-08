#!/bin/bash
# =============================================================================
# Phase 1: NVIDIA ドライバ + CUDA セットアップスクリプト
# 対象: Ubuntu 24.04 LTS + GTX 1060
# =============================================================================
set -e

echo "=========================================="
echo " Phase 1: NVIDIA Driver + CUDA Setup"
echo " Target: Ubuntu 24.04 + GTX 1060"
echo "=========================================="

# --- 1. システムアップデート ---
echo ""
echo "[1/5] システムアップデート..."
sudo apt update && sudo apt upgrade -y

# --- 2. 必要パッケージのインストール ---
echo ""
echo "[2/5] 必要パッケージのインストール..."
sudo apt install -y \
    build-essential \
    gcc \
    g++ \
    make \
    curl \
    wget \
    git \
    python3 \
    python3-pip \
    python3-venv \
    htop \
    nvtop \
    lm-sensors \
    net-tools \
    software-properties-common

# --- 3. NVIDIA ドライバのインストール ---
echo ""
echo "[3/5] NVIDIA ドライバのインストール..."

# nouveauドライバの無効化確認
if lsmod | grep -q nouveau; then
    echo "nouveauドライバを無効化します..."
    sudo bash -c 'echo "blacklist nouveau" >> /etc/modprobe.d/blacklist-nouveau.conf'
    sudo bash -c 'echo "options nouveau modeset=0" >> /etc/modprobe.d/blacklist-nouveau.conf'
    sudo update-initramfs -u
    echo "⚠️  再起動が必要です。再起動後にこのスクリプトを再実行してください。"
    echo "   sudo reboot"
    exit 0
fi

# ubuntu-drivers で推奨ドライバをインストール
sudo apt install -y ubuntu-drivers-common
echo "検出されたGPU:"
ubuntu-drivers devices
echo ""
echo "推奨ドライバをインストール..."
sudo ubuntu-drivers autoinstall

# --- 4. CUDA Toolkit のインストール ---
echo ""
echo "[4/5] CUDA Toolkit のインストール..."

# CUDA 12.x (Ubuntu 24.04用)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
rm cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit

# 環境変数の設定
echo "" >> ~/.bashrc
echo "# CUDA" >> ~/.bashrc
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# --- 5. インストール確認 ---
echo ""
echo "[5/5] インストール確認..."
echo ""
echo "--- NVIDIA Driver ---"
nvidia-smi || echo "⚠️  nvidia-smi が実行できません。再起動が必要な場合があります。"
echo ""
echo "--- CUDA Version ---"
nvcc --version 2>/dev/null || echo "⚠️  nvcc が見つかりません。再起動後に確認してください。"

echo ""
echo "=========================================="
echo " セットアップ完了！"
echo " ⚠️  再起動を推奨します: sudo reboot"
echo " 再起動後に以下を確認:"
echo "   nvidia-smi"
echo "   nvcc --version"
echo "=========================================="
