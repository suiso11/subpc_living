# AGENTS.md

## Project

サブPC常駐のパーソナルAIアシスタント。ローカルLLM (Ollama) + STT (faster-whisper) + TTS を
Web UI・Discord bot・音声パイプラインから使う。

- 仮想環境: `.venv/bin/python`
- テスト: `.venv/bin/python -m unittest discover -s tests -q`
- サービス反映: `systemctl --user restart subpc-discord.service subpc-web.service`
- 実設定 `config/discord.env` は git 管理外。例は `config/discord.env.example`

## Pi Subagent Orchestration

開発時の親ランタイムは Pi。通常の探索・テスト・最終レビューは `pi-subagents` の
named agentへ、実装と機械作業・別モデル視点はOpenCode workerへ委譲する。親は
計画・分解・高リスク判断・統合・最終確認に専念する。

OpenCodeを含む統合起動:

```bash
pi-orch
```

Pi named agentだけで運用する場合:

```bash
scripts/pi_codex_orchestrator.sh
```

プロジェクト側スクリプトは `PI_ORCHESTRATOR_MODE=1` を有効にし、親からread/bash/edit/write等を
外して `subagent`、`subagent_resume`、`subagent_kill` だけを残す。`pi-orch` は加えて
`opencode_task`、`opencode_spawn`、`opencode_workflow` 等を提供する。

### 原則

- Claude/Anthropic系には依存しない。親と最終承認は `openai-codex/gpt-5.6-sol`
- 通常探索・テストは `openai-codex/gpt-5.6-terra`
- 実装の既定バックエンドは OpenCode GLM (`opencode-go/glm-5.2`) を単独で使う。実装用の
  名前付き実装agentは置かない
- 実装は `opencode_change` ツールを既定とし、変更ごとに読み取り専用のKimi K3
  (`opencode-go/kimi-k3`) 自動レビューを経てからSol最終承認へ進む。Kimiは読み取り専用のまま
- OpenCodeの通常タスクも `opencode-go/glm-5.2`。Qwen profileは使わない
- OpenCodeの別視点レビューは `opencode-go/kimi-k3` を読み取り専用で使う
- ローカルOllamaモデル（Qwen等）は常駐アシスタントの実行用とし、コーディング委譲には使わない
- 実装は1タスク1関心事に分割する。1つのwriteタスクが扱う変更可能パスは最大5件まで。
  複数モジュールにまたがる作業や「調査 + 実装 + テスト」を含む作業は2〜4個の
  非重複 `opencode_change` タスクに分解し、GLM workerへ渡す。Piのnamed agentは
  scout/tester/reviewerの読み取り・検証・診断だけに使う
- 独立タスクは1回の `subagent` 呼び出しの `children` にまとめ、最大4並列にする
- 並列のwriteタスクは変更可能パスが重ならないよう分割する。重なる場合は直列化する
- 秘密情報を含む作業は子へ渡さない。実 `.env` と `config/discord.env` は読ませない
- 子の `extensions: none`、`skills: none`、`spawning: false`、`no-context-files: true` を
  維持し、再帰委譲を防ぐ

### Named agents

| Agent | 権限 | 用途 |
| --- | --- | --- |
| `subpc-scout` | 読み取り専用 | 関連ファイル、呼び出し経路、既存パターンの探索 |
| `subpc-tester` | 読み取り・ローカルコマンド | 独立したテスト実行と失敗診断 |
| `subpc-reviewer` | 読み取り・git参照 | 要件と差分の独立レビュー |

agent定義は `.pi/agents/*.md`。子はbackground・async・lineage-onlyで起動し、結果はPiが親へ
自動配送する。探索後の追加調査や同じ実装の修正は `subagent_resume`、独立レビューは新しい
`subpc-reviewer` セッションを使う。実装自体はGLM workerの `opencode_change` タスクへ委譲し、
named agentでは実装しない。

### OpenCode workers

| Profile/Model | 権限 | 用途 |
| --- | --- | --- |
| default / `glm` | read-only または限定write | 探索、機械作業、実装、テスト追加 |
| `kimi_k3` | 読み取り専用 | 広域読解、設計・差分の独立レビュー |

`glm` は `opencode-go/glm-5.2`、`kimi_k3` は `opencode-go/kimi-k3` に解決する。
Kimiへwrite、本番データ操作、秘密情報、サービス操作を渡さない。Kimi/GLMの結論は最終承認にせず、
`subpc-reviewer` または親のGPT-5.6 Solが最終判断する。

`opencode_change` は実装の既定ツール。GLMが変更候補を生成し、完了してKimi K3が読み取り専用で
自動レビューを実施する。PiはKimiの指摘を確認のうえ、最終承認をGPT-5.6 Solへ渡す。
Kimiは変更を書けず、本番データ・秘密情報・サービス操作にも触らない。

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
- すべてのソース変更は、自動または明示的なKimi K3読み取り専用レビューを経てから
  GPT-5.6 Solの最終承認へ進める。Kimi単独で承認しない

## Resource Check Before LLM-Heavy Work

Ollama の大きいコンテキストを使う前に確認する。

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.free,memory.total,utilization.gpu --format=csv,noheader,nounits
ollama ps
```

`ollama ps` の `CONTEXT` が目的の値なら、その値ではロード済み。VRAM の残りが薄い場合は、同時処理や別モデル起動を避ける。
