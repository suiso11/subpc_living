#!/bin/bash
# =============================================================================
# Phase 4: 長期記憶 (RAG) 検証スクリプト
# ChromaDB / sentence-transformers / VectorStore / RAG の検証
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="${PROJECT_ROOT}/.venv"

echo "=========================================="
echo " Phase 4: 長期記憶 (RAG) 検証"
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
source "${VENV_DIR}/bin/activate"

# --- Python パッケージ ---
echo ""
echo "[Pythonパッケージ]"
check "chromadb" "python3 -c 'import chromadb'"
check "sentence-transformers" "python3 -c 'import sentence_transformers'"

# --- プロジェクト構成 ---
echo ""
echo "[プロジェクト構成]"
check "src/memory/__init__.py" "[ -f '${PROJECT_ROOT}/src/memory/__init__.py' ]"
check "src/memory/vectorstore.py" "[ -f '${PROJECT_ROOT}/src/memory/vectorstore.py' ]"
check "src/memory/embedding.py" "[ -f '${PROJECT_ROOT}/src/memory/embedding.py' ]"
check "src/memory/rag.py" "[ -f '${PROJECT_ROOT}/src/memory/rag.py' ]"
check "data/vectordb ディレクトリ" "[ -d '${PROJECT_ROOT}/data/vectordb' ]"

# --- 埋め込みモデル ---
echo ""
echo "[埋め込みモデル]"
echo -n "  multilingual-e5-small ロード... "
EMB_RESULT=$(python3 -c "
import sys; sys.path.insert(0, '${PROJECT_ROOT}')
from src.memory.embedding import EmbeddingModel
model = EmbeddingModel()
model.load()
dim = model.dimension
vec = model.encode_query('テスト')
print(f'OK: dim={dim}, vec_shape={vec.shape}')
" 2>&1)
if echo "$EMB_RESULT" | grep -q "OK:"; then
    echo "✅ OK"
    echo "    $EMB_RESULT" | grep "OK:"
    ((PASS++))
else
    echo "❌ FAIL"
    echo "    $EMB_RESULT"
    ((FAIL++))
fi

# --- VectorStore ---
echo ""
echo "[VectorStore テスト]"
echo -n "  ChromaDB 初期化 + 保存 + 検索... "
VS_RESULT=$(python3 -c "
import sys, tempfile, os
sys.path.insert(0, '${PROJECT_ROOT}')
from src.memory.vectorstore import VectorStore

# テスト用の一時ディレクトリ
with tempfile.TemporaryDirectory() as tmpdir:
    vs = VectorStore(persist_dir=tmpdir)
    vs.initialize()

    # 会話を保存
    doc_id = vs.store_conversation_turn(
        user_message='好きな食べ物は寿司です',
        assistant_message='寿司が好きなんですね。ネタは何が好きですか？',
        session_id='test_session',
    )
    assert doc_id, 'store failed'
    assert vs.conversation_count == 1

    # 知識を保存
    kid = vs.store_knowledge(
        text='ユーザーは寿司が好き。特にサーモンが好み。',
        category='preference',
    )
    assert kid, 'knowledge store failed'
    assert vs.knowledge_count == 1

    # 検索テスト
    results = vs.search_conversations('寿司')
    assert len(results) > 0, 'search returned no results'
    assert '寿司' in results[0]['document'], 'search result mismatch'

    # 知識検索テスト
    know_results = vs.search_knowledge('好きな食べ物')
    assert len(know_results) > 0, 'knowledge search failed'

    # 統合検索テスト
    all_results = vs.search_all('寿司が好き')
    assert len(all_results) > 0, 'search_all failed'

    print(f'OK: conv={vs.conversation_count}, know={vs.knowledge_count}')
" 2>&1)
if echo "$VS_RESULT" | grep -q "OK:"; then
    echo "✅ OK"
    echo "    $VS_RESULT" | grep "OK:"
    ((PASS++))
else
    echo "❌ FAIL"
    echo "    $VS_RESULT"
    ((FAIL++))
fi

# --- RAG Retriever ---
echo ""
echo "[RAG Retriever テスト]"
echo -n "  RAG コンテキスト構築... "
RAG_RESULT=$(python3 -c "
import sys, tempfile
sys.path.insert(0, '${PROJECT_ROOT}')
from src.memory.vectorstore import VectorStore
from src.memory.rag import RAGRetriever

with tempfile.TemporaryDirectory() as tmpdir:
    vs = VectorStore(persist_dir=tmpdir)
    vs.initialize()
    rag = RAGRetriever(vector_store=vs)

    # データ投入
    rag.store_turn('明日の予定は？', '明日は14時から会議がありますね。', session_id='t1')
    rag.store_turn('最近読んだ本は？', '先週、三体を読み始めたと言っていましたね。', session_id='t1')
    rag.store_knowledge('ユーザーは三体(SF小説)を読んでいる', category='interest')

    # 検索テスト
    context = rag.build_context_prompt('本の話をしよう')
    assert '記憶' in context, f'context missing: {context}'

    # 関係ないクエリでは結果が少ない/ない
    stats = rag.get_stats()
    print(f'OK: conversations={stats[\"conversations\"]}, knowledge={stats[\"knowledge\"]}')
    print(f'  context length: {len(context)} chars')
" 2>&1)
if echo "$RAG_RESULT" | grep -q "OK:"; then
    echo "✅ OK"
    echo "    $RAG_RESULT" | grep "OK:"
    ((PASS++))
else
    echo "❌ FAIL"
    echo "    $RAG_RESULT"
    ((FAIL++))
fi

# --- ChatSession + RAG 統合テスト ---
echo ""
echo "[ChatSession + RAG 統合テスト]"
echo -n "  セッション + RAG コンテキスト注入... "
SESSION_RESULT=$(python3 -c "
import sys, tempfile
sys.path.insert(0, '${PROJECT_ROOT}')
from src.memory.vectorstore import VectorStore
from src.memory.rag import RAGRetriever
from src.chat.session import ChatSession

with tempfile.TemporaryDirectory() as tmpdir:
    vs = VectorStore(persist_dir=tmpdir)
    vs.initialize()
    rag = RAGRetriever(vector_store=vs)

    # 過去の会話をRAGに保存
    rag.store_turn('猫を飼っている', 'かわいいですね！名前は何ですか？', session_id='old')
    rag.store_turn('猫の名前はミケです', 'ミケちゃん、かわいい名前ですね！', session_id='old')

    # 新しいセッションでRAGを使用
    session = ChatSession(
        system_prompt='あなたはAIアシスタントです。',
        rag=rag,
    )
    session.add_user_message('うちのペットの話をしよう')
    messages = session.build_messages()

    # システムプロンプトにRAGコンテキストが注入されているか
    system_msg = messages[0]['content']
    assert '記憶' in system_msg, f'RAG context not injected: {system_msg[:100]}'

    print(f'OK: system_prompt length={len(system_msg)}')
    print(f'  RAG context injected successfully')
" 2>&1)
if echo "$SESSION_RESULT" | grep -q "OK:"; then
    echo "✅ OK"
    echo "    $SESSION_RESULT" | grep "OK:"
    ((PASS++))
else
    echo "❌ FAIL"
    echo "    $SESSION_RESULT"
    ((FAIL++))
fi

# --- 結果サマリー ---
echo ""
echo "=========================================="
echo " 結果: ✅ ${PASS} 成功 / ❌ ${FAIL} 失敗"
echo "=========================================="

if [ $FAIL -eq 0 ]; then
    echo ""
    echo "🎉 Phase 4 検証 すべてパス!"
    echo ""
    echo "長期記憶 (RAG) が有効になりました。"
    echo "以降の会話は自動でベクトルDBに保存され、"
    echo "関連する過去の文脈がLLMに自動注入されます。"
    echo ""
    echo "RAG無効で起動:"
    echo "  python src/audio/main.py --no-rag"
    exit 0
else
    echo ""
    echo "⚠️  ${FAIL}件の失敗があります。上記を確認してください。"
    exit 1
fi
