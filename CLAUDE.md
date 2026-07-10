# CLAUDE.md

## プロジェクト概要

サブPC常駐のパーソナルAIアシスタント。ローカルLLM (Ollama) + STT (faster-whisper) + TTS を
Web UI・Discord bot・音声パイプラインから使う。

- 仮想環境: `.venv/bin/python` を使う (システムpythonではなく)
- テスト: `.venv/bin/python -m unittest discover -s tests -q`
- サービス: `systemctl --user restart subpc-discord.service subpc-web.service subpc-voice.service` (コード変更後に反映)
  - subpc-voice も chat 設定 (config/chat_config.json) を起動時に読む。モデル差し替え時に再起動を忘れると、新旧モデルがGPUを取り合ってロードのスラッシングが起きる (P40 24GB に26b級は1つしか載らない)
- 実設定 `config/discord.env` はgit管理外。例は `config/discord.env.example`

## Orchestration workflow

Claude/Anthropic 系の使用量に依存しない構成を既定にする。Claude Code がこのファイルを読んでいる場合でも、
`deep-reasoner` など Claude サブエージェントへの委譲は行わない。

- **オーケストレーター**: 現在の実行エージェントが計画・分解・統合を担当する
- **推論の重いフェーズ**: まずオーケストレーターが直接読む。独立意見が必要なら `codex` または `opencode` を使う
- **機械的な作業**: 小さければ直接実装する。大きい boilerplate、テスト追加、整形、単純な一括変更は `opencode run` に投げてもよい
- **別視点が欲しい問題**: `codex exec --sandbox read-only` または `opencode run` に、事実だけを渡して独立に検討させる

トークン節約ルール:
- ブリッジ系エージェント (`codex-peer` / `opencode-peer` / `fast-worker`) は `model: haiku` 指定済み。中継役なのでこれで十分
- `deep-reasoner` (opus) は既定で無効。ユーザーが名指しで指示したときだけ使う
- 組み込みエージェント (Explore / general-purpose / Plan) を起動する場合は `model: "haiku"` を指定する。
  広範な検索・列挙はまず `Grep`/`Glob` を直接使い、それで足りないときだけ委譲する

運用ルール:
- 委譲するときは、目的・関連ファイルパス・制約・期待する出力形式をプロンプトに明示する
- 独立性を守る: あるエージェントの結論を別のエージェントへの入力に混ぜない
- 些細な一発作業 (1ファイルの小修正、単純な質問) は委譲せず直接やる
- 各エージェントの最終メッセージはユーザーには見えない。重要な内容は最終返答に含める
- Claude がレートリミット中の詳しい退避構成は `docs/orchestration_no_claude.md` を参照する
