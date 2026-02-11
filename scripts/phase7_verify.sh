#!/bin/bash
# =============================================================================
# Phase 7: パーソナライズ 検証スクリプト
# UserProfile / ConversationSummarizer / SessionPreloader / ProactiveEngine
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="${PROJECT_ROOT}/.venv"

echo "=========================================="
echo " Phase 7: パーソナライズ 検証"
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
check "src/persona/__init__.py" "[ -f '${PROJECT_ROOT}/src/persona/__init__.py' ]"
check "src/persona/profile.py" "[ -f '${PROJECT_ROOT}/src/persona/profile.py' ]"
check "src/persona/summarizer.py" "[ -f '${PROJECT_ROOT}/src/persona/summarizer.py' ]"
check "src/persona/preloader.py" "[ -f '${PROJECT_ROOT}/src/persona/preloader.py' ]"
check "src/persona/proactive.py" "[ -f '${PROJECT_ROOT}/src/persona/proactive.py' ]"
check "data/profile ディレクトリ" "[ -d '${PROJECT_ROOT}/data/profile' ]"
check "data/profile/summaries ディレクトリ" "[ -d '${PROJECT_ROOT}/data/profile/summaries' ]"

# --- モジュールインポート ---
echo ""
echo "[モジュールインポート]"
check "UserProfile" "python3 -c \"import sys; sys.path.insert(0, '${PROJECT_ROOT}'); from src.persona.profile import UserProfile\""
check "ConversationSummarizer" "python3 -c \"import sys; sys.path.insert(0, '${PROJECT_ROOT}'); from src.persona.summarizer import ConversationSummarizer\""
check "SessionPreloader" "python3 -c \"import sys; sys.path.insert(0, '${PROJECT_ROOT}'); from src.persona.preloader import SessionPreloader\""
check "ProactiveEngine" "python3 -c \"import sys; sys.path.insert(0, '${PROJECT_ROOT}'); from src.persona.proactive import ProactiveEngine\""
check "__init__.py エクスポート" "python3 -c \"import sys; sys.path.insert(0, '${PROJECT_ROOT}'); from src.persona import UserProfile, ConversationSummarizer, SessionPreloader, ProactiveEngine\""

# --- UserProfile テスト ---
echo ""
echo "[UserProfile テスト]"

echo -n "  プロファイル作成・ロード... "
PROFILE_RESULT=$(python3 -c "
import sys, tempfile, os; sys.path.insert(0, '${PROJECT_ROOT}')
from src.persona.profile import UserProfile

# テスト用の一時ファイル
tmp = tempfile.mktemp(suffix='.json')
try:
    p = UserProfile(profile_path=tmp)
    p.load()
    assert p.data is not None
    assert p.name == ''
    assert p.preferences == {}
    print('OK')
finally:
    if os.path.exists(tmp):
        os.remove(tmp)
" 2>&1)

if echo "$PROFILE_RESULT" | grep -q "OK"; then
    echo "✅ OK"
    ((PASS++))
else
    echo "❌ FAIL"
    echo "    $PROFILE_RESULT"
    ((FAIL++))
fi

echo -n "  プロファイル読み書き... "
RW_RESULT=$(python3 -c "
import sys, tempfile, os; sys.path.insert(0, '${PROJECT_ROOT}')
from src.persona.profile import UserProfile

tmp = tempfile.mktemp(suffix='.json')
try:
    p = UserProfile(profile_path=tmp)
    p.load()

    # 名前設定
    p.name = 'テストユーザー'
    assert p.name == 'テストユーザー'

    # 嗜好設定
    p.set_preference('food', 'カレー')
    assert p.preferences['food'] == 'カレー'

    # 習慣設定
    p.set_habit('wake_time', '07:00')
    assert p.habits['wake_time'] == '07:00'

    # メモ追加
    p.add_note('猫を飼っている')
    assert '猫を飼っている' in p.notes

    # 事実追加
    added = p.add_extracted_fact('プログラマー')
    assert added == True
    dup = p.add_extracted_fact('プログラマー')
    assert dup == False  # 重複

    # リロードして永続化確認
    p2 = UserProfile(profile_path=tmp)
    p2.load()
    assert p2.name == 'テストユーザー'
    assert p2.preferences['food'] == 'カレー'
    assert 'プログラマー' in p2.extracted_facts

    print('OK')
finally:
    if os.path.exists(tmp):
        os.remove(tmp)
" 2>&1)

if echo "$RW_RESULT" | grep -q "OK"; then
    echo "✅ OK"
    ((PASS++))
else
    echo "❌ FAIL"
    echo "    $RW_RESULT"
    ((FAIL++))
fi

echo -n "  スケジュール管理... "
SCHEDULE_RESULT=$(python3 -c "
import sys, tempfile, os; sys.path.insert(0, '${PROJECT_ROOT}')
from src.persona.profile import UserProfile
from datetime import date, timedelta

tmp = tempfile.mktemp(suffix='.json')
try:
    p = UserProfile(profile_path=tmp)
    p.load()

    today = date.today().isoformat()
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    past = '2020-01-01'

    p.add_schedule('会議', today, '14:00', 'Zoom')
    p.add_schedule('出張', tomorrow, '09:00')
    p.add_schedule('過去の予定', past, '10:00')

    assert len(p.schedule) == 3
    assert len(p.get_today_schedule()) == 1
    assert p.get_today_schedule()[0]['title'] == '会議'

    upcoming = p.get_upcoming_schedule(days=2)
    assert len(upcoming) >= 1

    removed = p.remove_past_schedule()
    assert removed == 1  # 過去の予定1件削除

    print('OK')
finally:
    if os.path.exists(tmp):
        os.remove(tmp)
" 2>&1)

if echo "$SCHEDULE_RESULT" | grep -q "OK"; then
    echo "✅ OK"
    ((PASS++))
else
    echo "❌ FAIL"
    echo "    $SCHEDULE_RESULT"
    ((FAIL++))
fi

echo -n "  プロファイルテキスト生成... "
TEXT_RESULT=$(python3 -c "
import sys, tempfile, os; sys.path.insert(0, '${PROJECT_ROOT}')
from src.persona.profile import UserProfile

tmp = tempfile.mktemp(suffix='.json')
try:
    p = UserProfile(profile_path=tmp)
    p.load()
    p.name = 'はるか'
    p.set_preference('food', 'カレー')
    p.add_note('猫好き')

    text = p.get_profile_text()
    assert 'はるか' in text
    assert 'カレー' in text
    print('OK')
finally:
    if os.path.exists(tmp):
        os.remove(tmp)
" 2>&1)

if echo "$TEXT_RESULT" | grep -q "OK"; then
    echo "✅ OK"
    ((PASS++))
else
    echo "❌ FAIL"
    echo "    $TEXT_RESULT"
    ((FAIL++))
fi

# --- ConversationSummarizer テスト ---
echo ""
echo "[ConversationSummarizer テスト]"

echo -n "  サマライザ初期化... "
SUM_INIT_RESULT=$(python3 -c "
import sys, tempfile; sys.path.insert(0, '${PROJECT_ROOT}')
from src.persona.summarizer import ConversationSummarizer

tmp_dir = tempfile.mkdtemp()
s = ConversationSummarizer(summaries_dir=tmp_dir)
assert s.summaries_dir.exists()

# 空の場合
recents = s.get_recent_summaries()
assert recents == []
text = s.get_recent_summaries_text()
assert text == ''

print('OK')
" 2>&1)

if echo "$SUM_INIT_RESULT" | grep -q "OK"; then
    echo "✅ OK"
    ((PASS++))
else
    echo "❌ FAIL"
    echo "    $SUM_INIT_RESULT"
    ((FAIL++))
fi

echo -n "  会話フォーマット... "
FMT_RESULT=$(python3 -c "
import sys; sys.path.insert(0, '${PROJECT_ROOT}')
from src.persona.summarizer import ConversationSummarizer

s = ConversationSummarizer()
msgs = [
    {'role': 'system', 'content': 'system prompt'},
    {'role': 'user', 'content': 'こんにちは'},
    {'role': 'assistant', 'content': 'こんにちは！'},
]
text = s._format_conversation(msgs)
assert 'ユーザー: こんにちは' in text
assert 'AI: こんにちは！' in text
assert 'system prompt' not in text  # system除外
print('OK')
" 2>&1)

if echo "$FMT_RESULT" | grep -q "OK"; then
    echo "✅ OK"
    ((PASS++))
else
    echo "❌ FAIL"
    echo "    $FMT_RESULT"
    ((FAIL++))
fi

# --- SessionPreloader テスト ---
echo ""
echo "[SessionPreloader テスト]"

echo -n "  プリロードコンテキスト構築... "
PRELOAD_RESULT=$(python3 -c "
import sys, tempfile, os; sys.path.insert(0, '${PROJECT_ROOT}')
from src.persona.profile import UserProfile
from src.persona.summarizer import ConversationSummarizer
from src.persona.preloader import SessionPreloader

tmp_prof = tempfile.mktemp(suffix='.json')
tmp_sum = tempfile.mkdtemp()
try:
    p = UserProfile(profile_path=tmp_prof)
    p.load()
    p.name = 'テスト'
    p.set_preference('music', 'ジャズ')

    s = ConversationSummarizer(summaries_dir=tmp_sum)

    preloader = SessionPreloader(profile=p, summarizer=s)
    ctx = preloader.build_preload_context()

    assert '現在の状況' in ctx
    assert '時間帯' in ctx
    assert 'テスト' in ctx
    assert 'ジャズ' in ctx

    status = preloader.get_status()
    assert 'current_datetime' in status
    assert 'profile' in status
    print('OK')
finally:
    if os.path.exists(tmp_prof):
        os.remove(tmp_prof)
" 2>&1)

if echo "$PRELOAD_RESULT" | grep -q "OK"; then
    echo "✅ OK"
    ((PASS++))
else
    echo "❌ FAIL"
    echo "    $PRELOAD_RESULT"
    ((FAIL++))
fi

# --- ProactiveEngine テスト ---
echo ""
echo "[ProactiveEngine テスト]"

echo -n "  エンジン起動・停止... "
PROACTIVE_RESULT=$(python3 -c "
import sys, time, tempfile, os; sys.path.insert(0, '${PROJECT_ROOT}')
from src.persona.profile import UserProfile
from src.persona.proactive import ProactiveEngine

tmp = tempfile.mktemp(suffix='.json')
try:
    p = UserProfile(profile_path=tmp)
    p.load()

    engine = ProactiveEngine(profile=p, check_interval=1.0)
    assert not engine.is_running

    triggers = []
    def cb(t, m):
        triggers.append((t, m))

    engine.start(callback=cb)
    assert engine.is_running

    time.sleep(2)
    engine.notify_user_activity()

    status = engine.get_status()
    assert status['running'] == True
    assert status['check_interval'] == 1.0
    assert status['session_duration_min'] >= 0

    engine.stop()
    assert not engine.is_running
    print('OK')
finally:
    if os.path.exists(tmp):
        os.remove(tmp)
" 2>&1)

if echo "$PROACTIVE_RESULT" | grep -q "OK"; then
    echo "✅ OK"
    ((PASS++))
else
    echo "❌ FAIL"
    echo "    $PROACTIVE_RESULT"
    ((FAIL++))
fi

echo -n "  クールダウン制御... "
COOLDOWN_RESULT=$(python3 -c "
import sys, tempfile, os; sys.path.insert(0, '${PROJECT_ROOT}')
from src.persona.proactive import ProactiveEngine
from src.persona.profile import UserProfile

tmp = tempfile.mktemp(suffix='.json')
try:
    p = UserProfile(profile_path=tmp)
    p.load()
    engine = ProactiveEngine(profile=p)

    # 未発火ならfire可能
    assert engine._can_fire('greeting') == True

    # 発火済みにする
    import time
    engine._last_fired['greeting'] = time.time()
    assert engine._can_fire('greeting') == False

    # 古い発火は再fire可能
    engine._last_fired['greeting'] = time.time() - 99999
    assert engine._can_fire('greeting') == True

    print('OK')
finally:
    if os.path.exists(tmp):
        os.remove(tmp)
" 2>&1)

if echo "$COOLDOWN_RESULT" | grep -q "OK"; then
    echo "✅ OK"
    ((PASS++))
else
    echo "❌ FAIL"
    echo "    $COOLDOWN_RESULT"
    ((FAIL++))
fi

# --- ChatSession 統合テスト ---
echo ""
echo "[ChatSession 統合テスト]"

echo -n "  preloader 注入 (build_messages)... "
SESSION_RESULT=$(python3 -c "
import sys, tempfile, os; sys.path.insert(0, '${PROJECT_ROOT}')
from src.chat.session import ChatSession
from src.persona.profile import UserProfile
from src.persona.summarizer import ConversationSummarizer
from src.persona.preloader import SessionPreloader

tmp_prof = tempfile.mktemp(suffix='.json')
tmp_sum = tempfile.mkdtemp()
try:
    p = UserProfile(profile_path=tmp_prof)
    p.load()
    p.name = '統合テスト'

    s = ConversationSummarizer(summaries_dir=tmp_sum)
    preloader = SessionPreloader(profile=p, summarizer=s)

    session = ChatSession(
        system_prompt='あなたはAIです。',
        preloader=preloader,
    )
    session.add_user_message('こんにちは')
    msgs = session.build_messages()

    # systemメッセージにプリロードコンテキストが注入されているか
    system_msg = msgs[0]['content']
    assert '統合テスト' in system_msg
    assert '現在の状況' in system_msg
    print('OK')
finally:
    if os.path.exists(tmp_prof):
        os.remove(tmp_prof)
" 2>&1)

if echo "$SESSION_RESULT" | grep -q "OK"; then
    echo "✅ OK"
    ((PASS++))
else
    echo "❌ FAIL"
    echo "    $SESSION_RESULT"
    ((FAIL++))
fi

# --- Pipeline 統合テスト ---
echo ""
echo "[Pipeline 統合テスト]"

echo -n "  VoicePipeline persona init... "
PIPELINE_RESULT=$(python3 -c "
import sys; sys.path.insert(0, '${PROJECT_ROOT}')
from src.audio.pipeline import VoicePipeline

# persona=True でインスタンス化のみテスト (Ollama不要)
p = VoicePipeline(
    enable_rag=False,
    enable_vision=False,
    enable_monitor=False,
    enable_persona=True,
)
assert p.profile is not None
assert p.summarizer is not None
assert p.preloader is not None
assert p.proactive is not None
assert p.session.preloader is not None
print('OK')
" 2>&1)

if echo "$PIPELINE_RESULT" | grep -q "OK"; then
    echo "✅ OK"
    ((PASS++))
else
    echo "❌ FAIL"
    echo "    $PIPELINE_RESULT"
    ((FAIL++))
fi

echo -n "  VoicePipeline persona disabled... "
DISABLED_RESULT=$(python3 -c "
import sys; sys.path.insert(0, '${PROJECT_ROOT}')
from src.audio.pipeline import VoicePipeline

p = VoicePipeline(
    enable_rag=False,
    enable_vision=False,
    enable_monitor=False,
    enable_persona=False,
)
assert p.profile is None
assert p.preloader is None
assert p.proactive is None
print('OK')
" 2>&1)

if echo "$DISABLED_RESULT" | grep -q "OK"; then
    echo "✅ OK"
    ((PASS++))
else
    echo "❌ FAIL"
    echo "    $DISABLED_RESULT"
    ((FAIL++))
fi

# --- 実在プロファイル確認 ---
echo ""
echo "[実在プロファイル]"
PROFILE_PATH="${PROJECT_ROOT}/data/profile/user_profile.json"
if [ -f "$PROFILE_PATH" ]; then
    check "プロファイルファイル存在" "[ -f '${PROFILE_PATH}' ]"
    echo -n "  プロファイルロード... "
    LOAD_RESULT=$(python3 -c "
import sys, json; sys.path.insert(0, '${PROJECT_ROOT}')
from src.persona.profile import UserProfile
p = UserProfile(profile_path='${PROFILE_PATH}')
p.load()
name = p.name or '(未設定)'
print(f'OK name={name} prefs={len(p.preferences)} facts={len(p.extracted_facts)}')
" 2>&1)
    if echo "$LOAD_RESULT" | grep -q "OK"; then
        echo "✅ $LOAD_RESULT"
        ((PASS++))
    else
        echo "❌ FAIL"
        echo "    $LOAD_RESULT"
        ((FAIL++))
    fi
else
    skip "プロファイルファイル" "未作成 (phase7_setup.sh を実行してください)"
fi

# --- 結果サマリー ---
echo ""
echo "=========================================="
echo " 検証結果: ✅ ${PASS} passed, ❌ ${FAIL} failed, ⏭️  ${SKIP} skipped"
echo "=========================================="

if [ $FAIL -gt 0 ]; then
    echo ""
    echo "⚠️  一部テストが失敗しています。"
    echo "  bash scripts/phase7_setup.sh を実行してください。"
    exit 1
else
    echo ""
    echo "🎉 Phase 7 パーソナライズ — すべてOK！"
    echo ""
    echo "次のステップ:"
    echo "  1. プロファイルを編集:"
    echo "     data/profile/user_profile.json"
    echo ""
    echo "  2. 音声対話で試す:"
    echo "     python -m src.audio.main"
    echo ""
    echo "  3. Web UIで試す:"
    echo "     python -m src.web.server"
    echo "     → /api/persona/profile でプロファイル確認"
    echo "     → /api/persona/context でプリロードコンテキスト確認"
fi
