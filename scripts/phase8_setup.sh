#!/bin/bash
# =============================================================================
# Phase 8: 常時稼働化 セットアップスクリプト
# systemd サービス登録 + ヘルスチェック + GPU省電力制御
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="${PROJECT_ROOT}/.venv"
SYSTEMD_USER_DIR="${HOME}/.config/systemd/user"

echo "=========================================="
echo " Phase 8: 常時稼働化 セットアップ"
echo "=========================================="

# --- 0. 仮想環境の確認 ---
if [ ! -d "$VENV_DIR" ]; then
    echo "❌ 仮想環境が見つかりません。先に phase2_setup.sh を実行してください。"
    exit 1
fi
source "${VENV_DIR}/bin/activate"

# --- 1. Pythonモジュール確認 ---
echo ""
echo "[1/4] Pythonモジュール確認..."

python3 -c "
import sys; sys.path.insert(0, '${PROJECT_ROOT}')
from src.service.healthcheck import HealthChecker
from src.service.power import GpuPowerManager
print('OK')
" && echo "✅ 全モジュールインポート成功" || { echo "❌ モジュールインポート失敗"; exit 1; }

# --- 2. ヘルスチェック実行 ---
echo ""
echo "[2/4] ヘルスチェック実行..."

python3 -c "
import sys, json; sys.path.insert(0, '${PROJECT_ROOT}')
from src.service.healthcheck import HealthChecker

checker = HealthChecker()
result = checker.check_all()

for name, check in result['checks'].items():
    status = check['status']
    if status == 'ok':
        icon = '✅'
    elif status == 'warning':
        icon = '⚠️'
    elif status == 'skip':
        icon = '⏭️'
    else:
        icon = '❌'
    print(f'  {icon} {name}: {status}')

print(f'  Overall: {result[\"status\"]}')
"

# --- 3. systemd ユニットファイルの準備 ---
echo ""
echo "[3/4] systemd ユニットファイルの準備..."

# ユーザー systemd ディレクトリ作成
mkdir -p "$SYSTEMD_USER_DIR"
echo "✅ ${SYSTEMD_USER_DIR} ディレクトリ確認"

# パスの事前置換: %h を実際のパスに展開したユニットファイルを生成
for SVC_FILE in subpc-web.service subpc-voice.service; do
    SRC="${SCRIPT_DIR}/systemd/${SVC_FILE}"
    DEST="${SYSTEMD_USER_DIR}/${SVC_FILE}"
    if [ -f "$SRC" ]; then
        # シンボリックリンクで配置
        ln -sf "$SRC" "$DEST"
        echo "✅ ${SVC_FILE} → ${DEST} (symlink)"
    else
        echo "❌ ${SRC} が見つかりません"
    fi
done

# daemon-reload
systemctl --user daemon-reload
echo "✅ systemctl --user daemon-reload 完了"

# --- 4. loginctl enable-linger ---
echo ""
echo "[4/4] enable-linger 設定..."

# ログアウト後もユーザーサービスを維持
if loginctl show-user "$USER" 2>/dev/null | grep -q "Linger=yes"; then
    echo "✅ enable-linger は既に有効です"
else
    echo "ログアウト後もサービスを動作させるには enable-linger が必要です。"
    echo "以下のコマンドを実行してください:"
    echo ""
    echo "  sudo loginctl enable-linger $USER"
    echo ""
    echo "⚠️  sudo 権限が必要なためスクリプトでは自動実行しません。"
fi

# --- GPU 省電力 (オプション) ---
echo ""
echo "[オプション] GPU 省電力サービス"
if command -v nvidia-smi &>/dev/null; then
    echo "nvidia-smi が検出されました。"
    echo "GPU 省電力サービスをインストールするには:"
    echo ""
    echo "  sudo cp ${SCRIPT_DIR}/systemd/subpc-gpu-powersave.service /etc/systemd/system/"
    echo "  sudo systemctl daemon-reload"
    echo "  sudo systemctl enable subpc-gpu-powersave"
    echo "  sudo systemctl start subpc-gpu-powersave"
    echo ""
else
    echo "nvidia-smi が見つかりません。GPU 省電力は不要です。"
fi

# --- 完了 ---
echo ""
echo "=========================================="
echo " ✅ Phase 8 セットアップ完了!"
echo "=========================================="
echo ""
echo "確認:"
echo "  bash scripts/phase8_verify.sh"
echo ""
echo "使い方:"
echo "  # 全サービスの状態確認"
echo "  bash scripts/service_ctl.sh status"
echo ""
echo "  # Web UI をサービスとして起動"
echo "  bash scripts/service_ctl.sh start web"
echo ""
echo "  # 全サービス起動"
echo "  bash scripts/service_ctl.sh start all"
echo ""
echo "  # ログ確認"
echo "  bash scripts/service_ctl.sh logs web"
echo ""
echo "  # ヘルスチェック"
echo "  bash scripts/service_ctl.sh health"
echo ""
echo "  # 自動起動有効化 (PC起動時に自動で開始)"
echo "  bash scripts/service_ctl.sh enable web"
echo ""
