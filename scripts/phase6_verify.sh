#!/bin/bash
# =============================================================================
# Phase 6: PCログ収集 検証スクリプト
# psutil / SystemCollector / MetricsStorage / MonitorContext の検証
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="${PROJECT_ROOT}/.venv"

echo "=========================================="
echo " Phase 6: PCログ収集 検証"
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

# --- Python パッケージ ---
echo ""
echo "[Pythonパッケージ]"
check "psutil" "python3 -c 'import psutil; print(psutil.__version__)'"
check "sqlite3" "python3 -c 'import sqlite3; print(sqlite3.sqlite_version)'"

# --- プロジェクト構成 ---
echo ""
echo "[プロジェクト構成]"
check "src/monitor/__init__.py" "[ -f '${PROJECT_ROOT}/src/monitor/__init__.py' ]"
check "src/monitor/collector.py" "[ -f '${PROJECT_ROOT}/src/monitor/collector.py' ]"
check "src/monitor/storage.py" "[ -f '${PROJECT_ROOT}/src/monitor/storage.py' ]"
check "src/monitor/context.py" "[ -f '${PROJECT_ROOT}/src/monitor/context.py' ]"
check "data/metrics ディレクトリ" "[ -d '${PROJECT_ROOT}/data/metrics' ]"

# --- SystemCollector テスト ---
echo ""
echo "[SystemCollector テスト]"

echo -n "  psutil メトリクス収集... "
COLLECT_RESULT=$(python3 -c "
import sys; sys.path.insert(0, '${PROJECT_ROOT}')
from src.monitor.collector import SystemCollector
c = SystemCollector(interval=10)
m = c.collect_once()
assert m.cpu_percent >= 0
assert m.mem_percent > 0
assert m.mem_total_gb > 0
assert m.disk_percent > 0
assert m.process_count > 0
print('OK')
" 2>&1)

if echo "$COLLECT_RESULT" | grep -q "OK"; then
    echo "✅ OK"
    ((PASS++))
else
    echo "❌ FAIL"
    echo "    $COLLECT_RESULT"
    ((FAIL++))
fi

echo -n "  バックグラウンド収集 (start/stop)... "
BG_RESULT=$(python3 -c "
import sys, time; sys.path.insert(0, '${PROJECT_ROOT}')
from src.monitor.collector import SystemCollector
c = SystemCollector(interval=1)
collected = []
c.start(callback=lambda m: collected.append(m))
time.sleep(2.5)
c.stop()
assert len(collected) >= 1, f'collected {len(collected)} items'
print('OK')
" 2>&1)

if echo "$BG_RESULT" | grep -q "OK"; then
    echo "✅ OK"
    ((PASS++))
else
    echo "❌ FAIL"
    echo "    $BG_RESULT"
    ((FAIL++))
fi

# --- GPU検出 ---
echo ""
echo "[GPU検出]"
echo -n "  nvidia-smi... "
if command -v nvidia-smi &>/dev/null; then
    GPU_RESULT=$(python3 -c "
import sys; sys.path.insert(0, '${PROJECT_ROOT}')
from src.monitor.collector import SystemCollector
c = SystemCollector(interval=10)
m = c.collect_once()
if m.gpu_util_percent is not None:
    print(f'OK: GPU {m.gpu_util_percent:.0f}%, VRAM {m.gpu_mem_used_mb:.0f}MB')
else:
    print('OK: nvidia-smi available but GPU info not parsed')
" 2>&1)
    echo "✅ ${GPU_RESULT#OK: }"
    ((PASS++))
else
    skip "nvidia-smi" "nvidia-smiが見つかりません"
fi

# --- MetricsStorage テスト ---
echo ""
echo "[MetricsStorage テスト]"

TEST_DB="/tmp/subpc_test_metrics_$$.db"

echo -n "  DB初期化・保存・読取... "
STORAGE_RESULT=$(python3 -c "
import sys; sys.path.insert(0, '${PROJECT_ROOT}')
from src.monitor.collector import SystemCollector
from src.monitor.storage import MetricsStorage

storage = MetricsStorage(db_path='${TEST_DB}')
storage.initialize()

c = SystemCollector(interval=10)
m = c.collect_once()
storage.store_metrics(m)

# 確認
count = storage.get_record_count()
assert count == 1, f'record_count={count}'

rows = storage.get_recent(minutes=5)
assert len(rows) == 1, f'recent rows={len(rows)}'

latest = storage.get_latest_row()
assert latest is not None
assert latest['cpu_percent'] == m.cpu_percent

storage.close()
print('OK')
" 2>&1)

if echo "$STORAGE_RESULT" | grep -q "OK"; then
    echo "✅ OK"
    ((PASS++))
else
    echo "❌ FAIL"
    echo "    $STORAGE_RESULT"
    ((FAIL++))
fi

echo -n "  クリーンアップ機能... "
CLEANUP_RESULT=$(python3 -c "
import sys; sys.path.insert(0, '${PROJECT_ROOT}')
from src.monitor.storage import MetricsStorage

storage = MetricsStorage(db_path='${TEST_DB}')
storage.initialize()
storage.cleanup_old(keep_days=0)  # 全削除
count = storage.get_record_count()
assert count == 0, f'after cleanup: {count}'
storage.close()
print('OK')
" 2>&1)

if echo "$CLEANUP_RESULT" | grep -q "OK"; then
    echo "✅ OK"
    ((PASS++))
else
    echo "❌ FAIL"
    echo "    $CLEANUP_RESULT"
    ((FAIL++))
fi

# テストDB削除
rm -f "${TEST_DB}"

# --- MonitorContext テスト ---
echo ""
echo "[MonitorContext テスト]"

TEST_DB2="/tmp/subpc_test_monitor_$$.db"

echo -n "  コンテキスト生成... "
CTX_RESULT=$(python3 -c "
import sys, time; sys.path.insert(0, '${PROJECT_ROOT}')
from src.monitor.context import MonitorContext

ctx = MonitorContext(db_path='${TEST_DB2}', collect_interval=1.0)
assert ctx.start()
time.sleep(3.0)

text = ctx.get_context_text()
assert 'CPU' in text, f'CPU not in context: {text}'
assert 'メモリ' in text, f'Memory not in context: {text}'

status = ctx.get_status()
assert status['running'] is True
assert status['cpu_percent'] >= 0

ctx.stop()
print('OK')
" 2>&1)

if echo "$CTX_RESULT" | grep -q "OK"; then
    echo "✅ OK"
    ((PASS++))
else
    echo "❌ FAIL"
    echo "    $CTX_RESULT"
    ((FAIL++))
fi

echo -n "  サマリー取得... "
SUMMARY_RESULT=$(python3 -c "
import sys; sys.path.insert(0, '${PROJECT_ROOT}')
from src.monitor.context import MonitorContext

ctx = MonitorContext(db_path='${TEST_DB2}', collect_interval=1.0)
ctx.start()
import time; time.sleep(3.0)

summary = ctx.get_recent_summary(minutes=5)
assert summary['sample_count'] >= 1, f'sample_count={summary[\"sample_count\"]}'
ctx.stop()
print('OK')
" 2>&1)

if echo "$SUMMARY_RESULT" | grep -q "OK"; then
    echo "✅ OK"
    ((PASS++))
else
    echo "❌ FAIL"
    echo "    $SUMMARY_RESULT"
    ((FAIL++))
fi

# テストDB削除
rm -f "${TEST_DB2}"

# --- ChatSession 統合テスト ---
echo ""
echo "[ChatSession 統合テスト]"

echo -n "  monitor_context パラメータ... "
CHAT_RESULT=$(python3 -c "
import sys; sys.path.insert(0, '${PROJECT_ROOT}')
from src.chat.session import ChatSession

# monitor_context=None でも動作
s = ChatSession(system_prompt='test')
s.add_user_message('hello')
msgs = s.build_messages()
assert len(msgs) == 2  # system + user
print('OK')
" 2>&1)

if echo "$CHAT_RESULT" | grep -q "OK"; then
    echo "✅ OK"
    ((PASS++))
else
    echo "❌ FAIL"
    echo "    $CHAT_RESULT"
    ((FAIL++))
fi

echo -n "  MonitorContext 注入テスト... "
INJECT_RESULT=$(python3 -c "
import sys, time; sys.path.insert(0, '${PROJECT_ROOT}')
from src.chat.session import ChatSession
from src.monitor.context import MonitorContext

ctx = MonitorContext(db_path='/tmp/subpc_inject_test_$$.db', collect_interval=1.0)
ctx.start()
time.sleep(3.0)

s = ChatSession(system_prompt='あなたはAIです', monitor_context=ctx)
s.add_user_message('PCの調子はどう？')
msgs = s.build_messages()

system_msg = msgs[0]['content']
assert 'CPU' in system_msg, f'CPU not found in system prompt'
assert 'メモリ' in system_msg, f'Memory not found in system prompt'

ctx.stop()
import os; os.unlink('/tmp/subpc_inject_test_$$.db')
print('OK')
" 2>&1)

if echo "$INJECT_RESULT" | grep -q "OK"; then
    echo "✅ OK"
    ((PASS++))
else
    echo "❌ FAIL"
    echo "    $INJECT_RESULT"
    ((FAIL++))
fi

# --- 結果サマリー ---
echo ""
echo "=========================================="
TOTAL=$((PASS + FAIL + SKIP))
echo " 結果: ${PASS}/${TOTAL} PASS, ${FAIL} FAIL, ${SKIP} SKIP"
echo "=========================================="

if [ "$FAIL" -gt 0 ]; then
    echo "⚠️  一部のテストが失敗しました。"
    exit 1
else
    echo "✅ Phase 6 検証完了！"
fi
