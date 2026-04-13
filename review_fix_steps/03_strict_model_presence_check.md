# Strict Model Presence Check

## Goal
- Ollama モデル存在チェックの部分一致をやめ、存在しないタグを起動時に確実に弾く。

## Files
- `src/chat/client.py`

## Steps
1. `list_models()` が返すモデル名との比較を厳密化する。
2. ユーザーがタグ省略名を指定した場合だけ安全なフォールバックを許可するかを整理する。
3. 少なくとも誤った別タグへの部分一致では `True` にならない実装にする。

## Verification
- 静的に比較ロジックを確認し、完全一致と `:latest` の扱いが一貫していることを確認する。
