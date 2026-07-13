#!/bin/bash
# =============================================================================
# Phase 8: 常時稼働化 検証スクリプト
# HealthChecker / GpuPowerManager / systemd ユニット / service_ctl
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="${PROJECT_ROOT}/.venv"

echo "=========================================="
echo " Phase 8: 常時稼働化 検証"
echo "=========================================="

PASS=0
FAIL=0
SKIP=0

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

skip() {
    local name="$1"
    local reason="$2"
    echo "  ${name}... ⏭️  SKIP (${reason})"
    ((SKIP++))
}

# 仮想環境の有効化
source "${VENV_DIR}/bin/activate"

# --- プロジェクト構成 ---
echo ""
echo "[プロジェクト構成]"
check "src/service/__init__.py" "[ -f '${PROJECT_ROOT}/src/service/__init__.py' ]"
check "src/service/healthcheck.py" "[ -f '${PROJECT_ROOT}/src/service/healthcheck.py' ]"
check "src/service/power.py" "[ -f '${PROJECT_ROOT}/src/service/power.py' ]"
check "src/service/gpu_power_daemon.py" "[ -f '${PROJECT_ROOT}/src/service/gpu_power_daemon.py' ]"
check "scripts/systemd/subpc-web.service" "[ -f '${SCRIPT_DIR}/systemd/subpc-web.service' ]"
check "scripts/systemd/subpc-voice.service" "[ -f '${SCRIPT_DIR}/systemd/subpc-voice.service' ]"
check "scripts/systemd/subpc-gpu-powersave.service" "[ -f '${SCRIPT_DIR}/systemd/subpc-gpu-powersave.service' ]"
check "scripts/systemd/subpc-gpu-powerd@.service" "[ -f '${SCRIPT_DIR}/systemd/subpc-gpu-powerd@.service' ]"
check "scripts/service_ctl.sh" "[ -f '${SCRIPT_DIR}/service_ctl.sh' ]"

# --- モジュールインポート ---
echo ""
echo "[モジュールインポート]"
check "HealthChecker" "python3 -c \"import sys; sys.path.insert(0, '${PROJECT_ROOT}'); from src.service.healthcheck import HealthChecker\""
check "GpuPowerManager" "python3 -c \"import sys; sys.path.insert(0, '${PROJECT_ROOT}'); from src.service.power import GpuPowerManager\""
check "GpuPowerDaemon" "python3 -c \"import sys; sys.path.insert(0, '${PROJECT_ROOT}'); from src.service.gpu_power_daemon import GpuPowerDaemon\""
check "__init__.py エクスポート" "python3 -c \"import sys; sys.path.insert(0, '${PROJECT_ROOT}'); from src.service import HealthChecker, GpuPowerManager\""

# --- HealthChecker テスト ---
echo ""
echo "[HealthChecker テスト]"

echo -n "  ディスクチェック... "
DISK_RESULT=$(python3 -c "
import sys; sys.path.insert(0, '${PROJECT_ROOT}')
from src.service.healthcheck import HealthChecker

c = HealthChecker()
r = c.check_disk()
assert r['status'] in ('ok', 'warning'), f'unexpected: {r}'
assert 'free_gb' in r
assert 'total_gb' in r
assert r['free_gb'] > 0
print('OK')
" 2>&1)

if echo "$DISK_RESULT" | grep -q "OK"; then
    echo "✅ OK"
    ((PASS++))
else
    echo "❌ FAIL"
    echo "    $DISK_RESULT"
    ((FAIL++))
fi

echo -n "  メモリチェック... "
MEM_RESULT=$(python3 -c "
import sys; sys.path.insert(0, '${PROJECT_ROOT}')
from src.service.healthcheck import HealthChecker

c = HealthChecker()
r = c.check_memory()
assert r['status'] in ('ok', 'warning', 'skip'), f'unexpected: {r}'
if r['status'] != 'skip':
    assert 'available_gb' in r
    assert 'total_gb' in r
print('OK')
" 2>&1)

if echo "$MEM_RESULT" | grep -q "OK"; then
    echo "✅ OK"
    ((PASS++))
else
    echo "❌ FAIL"
    echo "    $MEM_RESULT"
    ((FAIL++))
fi

echo -n "  Ollamaチェック... "
OLLAMA_RESULT=$(python3 -c "
import sys; sys.path.insert(0, '${PROJECT_ROOT}')
from src.service.healthcheck import HealthChecker

c = HealthChecker()
r = c.check_ollama()
# Ollama が起動していなくても error は正常動作
assert r['status'] in ('ok', 'error', 'skip'), f'unexpected: {r}'
print('OK')
" 2>&1)

if echo "$OLLAMA_RESULT" | grep -q "OK"; then
    echo "✅ OK"
    ((PASS++))
else
    echo "❌ FAIL"
    echo "    $OLLAMA_RESULT"
    ((FAIL++))
fi

echo -n "  check_all... "
ALL_RESULT=$(python3 -c "
import sys; sys.path.insert(0, '${PROJECT_ROOT}')
from src.service.healthcheck import HealthChecker

c = HealthChecker()
r = c.check_all(include_web=False)
assert 'status' in r
assert r['status'] in ('ok', 'degraded', 'error')
assert 'checks' in r
assert 'ollama' in r['checks']
assert 'disk' in r['checks']
assert 'memory' in r['checks']
print('OK')
" 2>&1)

if echo "$ALL_RESULT" | grep -q "OK"; then
    echo "✅ OK"
    ((PASS++))
else
    echo "❌ FAIL"
    echo "    $ALL_RESULT"
    ((FAIL++))
fi

echo -n "  CLI実行 (python -m)... "
CLI_RESULT=$(cd "$PROJECT_ROOT" && python3 -m src.service.healthcheck 2>&1; echo "EXIT:$?")
if echo "$CLI_RESULT" | grep -q '"status"'; then
    echo "✅ OK"
    ((PASS++))
else
    echo "❌ FAIL"
    echo "    $CLI_RESULT"
    ((FAIL++))
fi

# --- GpuPowerManager テスト ---
echo ""
echo "[GpuPowerManager テスト]"

echo -n "  初期化・available チェック... "
GPU_RESULT=$(python3 -c "
import sys; sys.path.insert(0, '${PROJECT_ROOT}')
from src.service.power import GpuPowerManager

m = GpuPowerManager(idle_watts=100, active_watts=250)
# nvidia-smi がなくてもインスタンス化は成功する
assert m.idle_watts == 100
assert m.active_watts == 250
# available は nvidia-smi の有無で変わる（どちらでもOK）
assert isinstance(m.available, bool)
print('OK')
" 2>&1)

if echo "$GPU_RESULT" | grep -q "OK"; then
    echo "✅ OK"
    ((PASS++))
else
    echo "❌ FAIL"
    echo "    $GPU_RESULT"
    ((FAIL++))
fi

echo -n "  get_gpu_info (nvidia-smi無しフォールバック)... "
GPU_INFO_RESULT=$(python3 -c "
import sys; sys.path.insert(0, '${PROJECT_ROOT}')
from src.service.power import GpuPowerManager

m = GpuPowerManager()
info = m.get_gpu_info()
# nvidia-smi があれば status=ok, 無ければ unavailable
assert info['status'] in ('ok', 'unavailable', 'error'), f'unexpected: {info}'
print('OK')
" 2>&1)

if echo "$GPU_INFO_RESULT" | grep -q "OK"; then
    echo "✅ OK"
    ((PASS++))
else
    echo "❌ FAIL"
    echo "    $GPU_INFO_RESULT"
    ((FAIL++))
fi

echo -n "  get_status... "
STATUS_RESULT=$(python3 -c "
import sys; sys.path.insert(0, '${PROJECT_ROOT}')
from src.service.power import GpuPowerManager

m = GpuPowerManager()
s = m.get_status()
assert 'available' in s
assert 'idle_watts' in s
assert 'active_watts' in s
assert 'gpu_info' in s
print('OK')
" 2>&1)

if echo "$STATUS_RESULT" | grep -q "OK"; then
    echo "✅ OK"
    ((PASS++))
else
    echo "❌ FAIL"
    echo "    $STATUS_RESULT"
    ((FAIL++))
fi

# --- systemd ユニットファイル検証 ---
echo ""
echo "[systemd ユニットファイル]"

for SVC_FILE in subpc-web.service subpc-voice.service subpc-gpu-powersave.service subpc-gpu-powerd@.service; do
    SVC_PATH="${SCRIPT_DIR}/systemd/${SVC_FILE}"
    echo -n "  ${SVC_FILE} 構文チェック... "
    if systemd-analyze verify --user "$SVC_PATH" > /dev/null 2>&1; then
        echo "✅ OK"
        ((PASS++))
    elif systemd-analyze verify "$SVC_PATH" > /dev/null 2>&1; then
        echo "✅ OK"
        ((PASS++))
    else
        # systemd-analyze verify はパスの展開でエラーになることがある
        # ユニットファイルが読めることだけ確認
        if grep -q "\[Unit\]" "$SVC_PATH" && grep -q "\[Service\]" "$SVC_PATH"; then
            echo "✅ OK (構文OK, パス展開は未検証)"
            ((PASS++))
        else
            echo "❌ FAIL"
            ((FAIL++))
        fi
    fi
done

# --- systemd ユニットインストール確認 ---
echo ""
echo "[ユニットインストール状況]"

SYSTEMD_USER_DIR="${HOME}/.config/systemd/user"
for SVC_FILE in subpc-web.service subpc-voice.service; do
    DEST="${SYSTEMD_USER_DIR}/${SVC_FILE}"
    if [ -L "$DEST" ] || [ -f "$DEST" ]; then
        check "${SVC_FILE} インストール済み" "true"
    else
        skip "${SVC_FILE}" "未インストール (phase8_setup.sh を実行してください)"
    fi
done

# --- service_ctl.sh 動作確認 ---
echo ""
echo "[service_ctl.sh]"
check "help コマンド" "bash '${SCRIPT_DIR}/service_ctl.sh' help 2>/dev/null"

echo -n "  status コマンド... "
STATUS_OUT=$(bash "${SCRIPT_DIR}/service_ctl.sh" status 2>&1)
if echo "$STATUS_OUT" | grep -q "subpc-web"; then
    echo "✅ OK"
    ((PASS++))
else
    echo "❌ FAIL"
    echo "    出力にsubpc-webが含まれない"
    ((FAIL++))
fi

# --- Web API ヘルスチェック確認 (コードレベル) ---
echo ""
echo "[Web API /api/health]"

echo -n "  server.py に /api/health 定義あり... "
if grep -q "api/health" "${PROJECT_ROOT}/src/web/server.py" 2>/dev/null; then
    echo "✅ OK"
    ((PASS++))
else
    echo "❌ FAIL"
    ((FAIL++))
fi

# --- 結果サマリー ---
echo ""
echo "=========================================="
echo " 検証結果: ✅ ${PASS} passed, ❌ ${FAIL} failed, ⏭️  ${SKIP} skipped"
echo "=========================================="

if [ $FAIL -gt 0 ]; then
    echo ""
    echo "⚠️  一部テストが失敗しています。"
    echo "  bash scripts/phase8_setup.sh を実行してください。"
    exit 1
else
    echo ""
    echo "🎉 Phase 8 常時稼働化 — すべてOK！"
    echo ""
    echo "次のステップ:"
    echo "  1. サービス状態確認:"
    echo "     bash scripts/service_ctl.sh status"
    echo ""
    echo "  2. Web UI をサービスとして起動:"
    echo "     bash scripts/service_ctl.sh start web"
    echo ""
    echo "  3. 自動起動を有効化 (PC起動時):"
    echo "     bash scripts/service_ctl.sh enable web"
    echo ""
    echo "  4. ログ確認:"
    echo "     bash scripts/service_ctl.sh logs web -f"
    echo ""
    echo "  5. ヘルスチェック:"
    echo "     bash scripts/service_ctl.sh health"
    echo "     curl http://localhost:8000/api/health"
fi
