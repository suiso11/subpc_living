# AGENTS.md

## Project

サブPC常駐のパーソナルAIアシスタント。ローカルLLM (Ollama) + STT (faster-whisper) + TTS を
Web UI・Discord bot・音声パイプラインから使う。

- 仮想環境: `.venv/bin/python`
- テスト: `.venv/bin/python -m unittest discover -s tests -q`
- サービス反映: `systemctl --user restart subpc-discord.service subpc-web.service`
- 実設定 `config/discord.env` は git 管理外。例は `config/discord.env.example`

## Pi Subagent Orchestration

開発時の親ランタイムは Pi、委譲は `pi-subagents` の named agent を使う。
親は計画・分解・高リスク判断・統合・最終確認に専念し、探索・実装・テスト・レビューを
独立した Pi 子プロセスへ渡す。

起動:

```bash
scripts/pi_codex_orchestrator.sh
```

このスクリプトは `PI_ORCHESTRATOR_MODE=1` を有効にする。親から read/bash/edit/write 等を外し、
`subagent`、`subagent_resume`、`subagent_kill` だけを残す。単純な直接作業をしたい場合は通常の
`pi --approve` を使う。

### 原則

- Claude/Anthropic 系には依存しない。親は既定で `openai-codex/gpt-5.5`、子のモデルは
  `.pi/agents/` で OpenAI Codex 系に固定する
- ローカルOllamaモデル（Qwen等）は常駐アシスタントの実行用とし、コーディング委譲には使わない
- 複数ファイルの作業や「調査 + 実装 + テスト」を含む作業は、Piのnamed agentへ分解する
- 独立タスクは1回の `subagent` 呼び出しの `children` にまとめ、最大4並列にする
- 並列のwriteタスクは変更可能パスが重ならないよう分割する。重なる場合は直列化する
- 秘密情報を含む作業は子へ渡さない。実 `.env` と `config/discord.env` は読ませない
- 子の `extensions: none`、`skills: none`、`spawning: false` を維持し、再帰委譲を防ぐ

### Named agents

| Agent | 権限 | 用途 |
| --- | --- | --- |
| `subpc-scout` | 読み取り専用 | 関連ファイル、呼び出し経路、既存パターンの探索 |
| `subpc-implementer` | 編集・ローカルコマンド | 明示した非重複パス内の実装と関連テスト追加 |
| `subpc-tester` | 読み取り・ローカルコマンド | 独立したテスト実行と失敗診断 |
| `subpc-reviewer` | 読み取り・git参照 | 要件と差分の独立レビュー |

agent定義は `.pi/agents/*.md`。子はbackground・async・lineage-onlyで起動し、結果はPiが親へ
自動配送する。探索後の追加調査や同じ実装の修正は `subagent_resume`、独立レビューは新しい
`subpc-reviewer` セッションを使う。

### 委譲プロンプトの要件

- machine name、3〜8語のtitle、agent名、目的、関連ファイル、制約、期待出力を必ず含める
- writeタスクは変更可能ファイルまたはディレクトリを具体的に列挙する。globで広く渡さない
- 指示範囲外のファイルは変更せず、必要になったら停止して追加スコープを報告させる
- 独立検討では、他エージェントの結論を混ぜず、コード・ログ・要件だけを渡す
- 必要なコード・ログだけを渡し、リポジトリ全体の再読込や同じ調査の重複を避ける
- 出力は要点、変更ファイル、検証結果に絞らせ、長い思考過程やファイル全文を要求しない

### 委譲後の確認

- `git diff --stat` と関連テストで結果を検証する
- 未コミット変更が多い場合はユーザー作業を戻さず、今回触った範囲だけ扱う
- 実装結果を同じエージェント自身の説明だけで承認しない。`subpc-tester` または
  `subpc-reviewer` に独立確認させ、オーケストレーターが最終判断する

## Resource Check Before LLM-Heavy Work

Ollama の大きいコンテキストを使う前に確認する。

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.free,memory.total,utilization.gpu --format=csv,noheader,nounits
ollama ps
```

`ollama ps` の `CONTEXT` が目的の値なら、その値ではロード済み。VRAM の残りが薄い場合は、同時処理や別モデル起動を避ける。
