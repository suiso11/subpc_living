#!/bin/bash
# =============================================================================
# Phase 2: テキスト対話 検証スクリプト
# すべてのコンポーネントが正しく動作するか確認する
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="${PROJECT_ROOT}/.venv"

echo "=========================================="
echo " Phase 2: テキスト対話 検証"
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

# 仮想環境の有効化
if [ -d "$VENV_DIR" ]; then
    source "${VENV_DIR}/bin/activate"
fi

# --- Python 環境 ---
echo ""
echo "[Python環境]"
check "Python仮想環境" "[ -d '${VENV_DIR}' ]"
check "Python 3.12+" "python3 --version 2>&1 | grep -q '3.1[2-9]'"
check "httpx パッケージ" "python3 -c 'import httpx'"

# --- Ollama ---
echo ""
echo "[Ollama]"
check "Ollama サービス起動" "systemctl is-active --quiet ollama"
check "Ollama API応答" "curl -s http://localhost:11434/api/tags > /dev/null"
check "7Bモデル存在" "ollama list | grep -qi 'qwen2.5.*7b\|gemma.*7b\|7b\|8b'"

# --- プロジェクト構成 ---
echo ""
echo "[プロジェクト構成]"
check "src/chat/config.py" "[ -f '${PROJECT_ROOT}/src/chat/config.py' ]"
check "src/chat/client.py" "[ -f '${PROJECT_ROOT}/src/chat/client.py' ]"
check "src/chat/session.py" "[ -f '${PROJECT_ROOT}/src/chat/session.py' ]"
check "src/chat/main.py" "[ -f '${PROJECT_ROOT}/src/chat/main.py' ]"
check "config/chat_config.json" "[ -f '${PROJECT_ROOT}/config/chat_config.json' ]"
check "data/chat_history ディレクトリ" "[ -d '${PROJECT_ROOT}/data/chat_history' ]"
check "requirements.txt" "[ -f '${PROJECT_ROOT}/requirements.txt' ]"

# --- API通信テスト ---
echo ""
echo "[API通信テスト]"
check "Ollama チャットAPI" "curl -s -X POST http://localhost:11434/api/chat -d '{\"model\":\"qwen2.5:7b-instruct-q4_K_M\",\"messages\":[{\"role\":\"user\",\"content\":\"こんにちは\"}],\"stream\":false}' | grep -q 'content'"

# --- Pythonモジュール統合テスト ---
echo ""
echo "[統合テスト]"
check "OllamaClient接続" "python3 -c \"
import sys; sys.path.insert(0, '${PROJECT_ROOT}')
from src.chat.client import OllamaClient
c = OllamaClient()
assert c.is_available(), 'not available'
c.close()
\""

check "ChatSession作成" "python3 -c \"
import sys; sys.path.insert(0, '${PROJECT_ROOT}')
from src.chat.session import ChatSession
s = ChatSession(system_prompt='test', history_dir='/tmp/test_chat_hist')
s.add_user_message('hello')
msgs = s.build_messages()
assert len(msgs) == 2
assert msgs[0]['role'] == 'system'
\""

check "ChatConfig ロード" "python3 -c \"
import sys; sys.path.insert(0, '${PROJECT_ROOT}')
from src.chat.config import ChatConfig
c = ChatConfig.load('${PROJECT_ROOT}/config/chat_config.json')
assert c.model == 'qwen2.5:7b-instruct-q4_K_M'
\""

# --- LLM応答テスト ---
echo ""
echo "[LLM応答テスト]"
echo -n "  LLM応答生成 (数秒かかります)... "
RESPONSE=$(python3 -c "
import sys; sys.path.insert(0, '${PROJECT_ROOT}')
from src.chat.client import OllamaClient
c = OllamaClient()
r = c.generate([{'role':'user','content':'1+1は？一言で答えて'}], num_ctx=512)
print(r[:100])
c.close()
" 2>&1)
if [ $? -eq 0 ] && [ -n "$RESPONSE" ]; then
    echo "✅ OK"
    echo "    応答: ${RESPONSE}"
    ((PASS++))
else
    echo "❌ FAIL"
    echo "    ${RESPONSE}"
    ((FAIL++))
fi

# --- 結果サマリー ---
echo ""
echo "=========================================="
echo " 結果: ✅ ${PASS} 成功 / ❌ ${FAIL} 失敗"
echo "=========================================="

if [ $FAIL -eq 0 ]; then
    echo ""
    echo "🎉 Phase 2 検証 すべてパス!"
    echo ""
    echo "テキスト対話を開始するには:"
    echo "  source ${VENV_DIR}/bin/activate"
    echo "  python ${PROJECT_ROOT}/src/chat/main.py"
    exit 0
else
    echo ""
    echo "⚠️  ${FAIL}件の失敗があります。上記を確認してください。"
    exit 1
fi
