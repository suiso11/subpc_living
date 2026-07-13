# AGENTS.md

## Project

サブPC常駐のパーソナルAIアシスタント。ローカルLLM (Ollama) + STT (faster-whisper) + TTS を
Web UI・Discord bot・音声パイプラインから使う。

- 仮想環境: `.venv/bin/python`
- テスト: `.venv/bin/python -m unittest discover -s tests -q`
- サービス反映: `systemctl --user restart subpc-discord.service subpc-web.service`
- 実設定 `config/discord.env` は git 管理外。例は `config/discord.env.example`

## No-Claude Orchestration

Claude/Anthropic 系に依存しない構成を既定にする。

- オーケストレーター: 現在の実行エージェントが計画・分解・統合を担当する
- 推論の重い問題: まず直接コードを読む。独立意見が必要なら `codex` または `opencode` を使う
- 機械的な作業: 小さければ直接実装する。大きい boilerplate、テスト追加、整形、単純な一括変更は `opencode run` を使える
- 別視点: `codex exec --sandbox read-only` または `opencode run` に事実だけを渡す
- Claude サブエージェント (`deep-reasoner` など) には依存しない

## Delegation Rules

- プロンプトには目的、関連ファイルパス、制約、期待する出力形式を書く
- 独立検討では、他エージェントの結論を混ぜず、コード・ログ・要件だけを渡す
- 実装を委譲する場合は、指示範囲外のファイルを変更しないよう明示する
- 変更後は `git diff --stat` と関連テストを確認する
- 未コミット変更が多い場合、ユーザー作業を戻さず、今回触った範囲だけ扱う

## Resource Check Before LLM-Heavy Work

Ollama の大きいコンテキストを使う前に確認する。

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.free,memory.total,utilization.gpu --format=csv,noheader,nounits
ollama ps
```

`ollama ps` の `CONTEXT` が目的の値なら、その値ではロード済み。VRAM の残りが薄い場合は、同時処理や別モデル起動を避ける。
