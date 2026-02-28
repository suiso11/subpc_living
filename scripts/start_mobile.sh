#!/bin/bash
# ==============================================
#  subpc_living — スマホ向けモバイルアクセス起動
#  Tailscale serve で HTTPS 化 + Web UI 起動
# ==============================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PORT=8000

echo "=============================================="
echo " subpc_living — モバイルアクセス起動"
echo "=============================================="

# --- 1. Tailscale 接続確認 ---
echo "[1/3] Tailscale 接続確認..."
if ! command -v tailscale &> /dev/null; then
    echo "❌ Tailscale がインストールされていません"
    echo "   curl -fsSL https://tailscale.com/install.sh | sh"
    exit 1
fi

TS_IP=$(tailscale ip -4 2>/dev/null || echo "")
if [[ -z "$TS_IP" ]]; then
    echo "❌ Tailscale に接続されていません"
    echo "   sudo tailscale up --ssh"
    exit 1
fi

TS_DNS=$(tailscale status --json 2>/dev/null | python3 -c "
import sys, json
d = json.load(sys.stdin)
dns = d['Self']['DNSName'].rstrip('.')
print(dns)
" 2>/dev/null || echo "")

echo "  Tailscale IP: $TS_IP"
echo "  MagicDNS:     $TS_DNS"

# --- 2. Tailscale Serve 設定 (HTTPS → HTTP プロキシ) ---
echo ""
echo "[2/3] Tailscale Serve 設定 (HTTPS化)..."
echo "  ※ スマホのマイクを使うにはHTTPS接続が必須です"

# 既存の設定をリセット
sudo tailscale serve reset 2>/dev/null || true

# HTTP → HTTPS プロキシを設定 (バックグラウンド)
sudo tailscale serve --bg --https=443 http://localhost:${PORT}
echo "  ✅ HTTPS プロキシ設定完了"
echo ""

# --- 3. Web UI サーバー起動 (systemd サービス経由) ---
echo "[3/3] Web UI サーバー確認..."

# サービスが動いていなければ起動
if ! systemctl --user is-active --quiet subpc-web; then
    echo "  subpc-web サービスを起動中..."
    systemctl --user start subpc-web
    sleep 3
fi

if systemctl --user is-active --quiet subpc-web; then
    echo "  ✅ subpc-web サービス稼働中"
else
    echo "  ❌ subpc-web サービスの起動に失敗"
    echo "  手動確認: systemctl --user status subpc-web"
    exit 1
fi

echo ""
echo "=============================================="
echo " ✅ 準備完了！"
echo ""
echo " 📱 スマホからアクセス:"
echo "   https://${TS_DNS}"
echo ""
echo " 💻 PCからアクセス:"
echo "   http://localhost:${PORT}"
echo ""
echo " 🎤 音声入力: マイクボタンをタップして話しかける"
echo " 🔊 読み上げ: TTS トグルをONにする"
echo ""
echo " 管理コマンド:"
echo "   systemctl --user status subpc-web"
echo "   systemctl --user restart subpc-web"
echo "=============================================="
