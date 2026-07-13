# AGENTS.md

## Project

サブPC常駐のパーソナルAIアシスタント。ローカルLLM (Ollama) + STT (faster-whisper) + TTS を
Web UI・Discord bot・音声パイプラインから使う。

- 仮想環境: `.venv/bin/python`
- テスト: `.venv/bin/python -m unittest discover -s tests -q`
- サービス反映: `systemctl --user restart subpc-discord.service subpc-web.service`
- 実設定 `config/discord.env` は git 管理外。例は `config/discord.env.example`

## Token-Efficient Delegation

オーケストレーター自身のトークン消費を抑えるため、探索・要約・機械的作業は
原則として `opencode run` に委譲し、自らは計画・高リスク判断・統合・最終検証に専念する。

### 原則

- Claude/Anthropic 系には依存しない。委譲の第一選択はクラウドGLMの
  `opencode run --model opencode-go/glm-5.2` とし、`codex` は `opencode` が使えない場合か追加の独立意見が必要な場合に限る
- ローカルOllamaモデル（Qwen等）は常駐アシスタントの実行用とし、コーディング委譲には使わない
- オーケストレーター: 計画・分解・高リスク判断・統合・最終検証を担当
- 委譲対象: リポジトリ探索と要約、boilerplate、テスト追加、文書、機械的一括変更、独立レビュー
- 複数ファイルにまたがる作業や「調査 + 実装 + テスト」を含む作業は、着手前に少なくとも1つの境界明確なサブタスクを `opencode run` に委譲する
- 直接実行でよいもの: 小さな一行修正、秘密情報を含む作業、委譲より直接の方が安価な作業
- 相当量の作業を委譲しなかった場合は、ユーザー向け更新で理由を短く明示する

### 委譲プロンプトの要件

- 目的・関連ファイルパス・制約・期待する出力形式を必ず含める
- 指示範囲外のファイルは変更しないよう明示する
- 独立検討では、他エージェントの結論を混ぜず、コード・ログ・要件だけを渡す
- 必要なコード・ログだけを渡し、リポジトリ全体の再読込や同じ調査の重複を避ける
- 出力は要点、変更ファイル、検証結果に絞らせ、長い思考過程やファイル全文を要求しない

### 委譲後の確認

- `git diff --stat` と関連テストで結果を検証する
- 未コミット変更が多い場合はユーザー作業を戻さず、今回触った範囲だけ扱う

## Resource Check Before LLM-Heavy Work

Ollama の大きいコンテキストを使う前に確認する。

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.free,memory.total,utilization.gpu --format=csv,noheader,nounits
ollama ps
```

`ollama ps` の `CONTEXT` が目的の値なら、その値ではロード済み。VRAM の残りが薄い場合は、同時処理や別モデル起動を避ける。
