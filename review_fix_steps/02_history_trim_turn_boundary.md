# History Trim Turn Boundary

## Goal
- 会話履歴の上限超過時に user / assistant のペアを壊さず、常にターン単位で古い履歴を削除する。

## Files
- `src/chat/session.py`

## Steps
1. 履歴トリムを「メッセージ数」ではなく「完了済みターン数」で判断する。
2. 途中ターンの user メッセージを残しつつ、先頭から完了済みの user+assistant ペアだけを削除する。
3. `add_user_message()` / `add_assistant_message()` の呼び出しタイミングでも履歴が壊れないことを確認する。

## Verification
- 小さい `max_history_turns` で user / assistant 交互の履歴を作り、先頭が assistant から始まらないことを確認する。

