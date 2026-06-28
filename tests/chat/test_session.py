"""
ChatSession の基本テスト
"""
import json

from src.chat.session import ChatSession


def test_add_messages():
    """メッセージ追加とビルド"""
    session = ChatSession(system_prompt="sys")
    session.add_user_message("hello")
    session.add_assistant_message("hi")

    messages = session.build_messages()
    assert messages[0] == {"role": "system", "content": "sys"}
    assert messages[1] == {"role": "user", "content": "hello"}
    assert messages[2] == {"role": "assistant", "content": "hi"}


def test_history_trimming():
    """履歴ターン数が上限を超えたら古いものから削除されること"""
    session = ChatSession(max_history_turns=2)
    for i in range(5):
        session.add_user_message(f"user{i}")
        session.add_assistant_message(f"assistant{i}")

    # user+assistant で 2 ターン分 = 4 メッセージのみ保持
    assert len(session.messages) == 4
    assert session.messages[0]["content"] == "user3"
    assert session.messages[-1]["content"] == "assistant4"


def test_save_and_load(tmp_path):
    """セッションの保存・読み込み"""
    session = ChatSession(system_prompt="test", history_dir=str(tmp_path))
    session.add_user_message("foo")
    session.add_assistant_message("bar")

    saved = session.save()
    assert saved.exists()

    loaded = ChatSession.load(saved, history_dir=str(tmp_path))
    assert loaded.system_prompt == "test"
    assert len(loaded.messages) == 2
