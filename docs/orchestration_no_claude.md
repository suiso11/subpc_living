# No-Claude Orchestration

Claude/Anthropic 系に依存しない既定構成。

## Roles

- Primary orchestrator: 現在の実行エージェント
- Reasoning: オーケストレーターが直接担当。独立意見が必要なときだけ `codex` または `opencode`
- Mechanical worker: `opencode run`
- Peer review / second opinion: `codex exec --sandbox read-only` または `opencode run`

## Disabled Paths

- `deep-reasoner` は既定で使わない (description にも明記済み)。`model: opus` のため Claude 側レートリミット・トークン消費の影響を受ける。ユーザーが名指しで指示したときだけ使う

## Token-Saving Config (2026-07-05)

- `codex-peer` / `opencode-peer` / `fast-worker` は frontmatter で `model: haiku` を指定済み。
  実作業は外部CLI (codex / opencode) が行い、Claude 側は中継・検証だけなので haiku で十分
- 組み込みエージェント (Explore / general-purpose / Plan) を Claude Code から起動する場合は
  Agent ツールの `model: "haiku"` オーバーライドを使う

## Commands

### Pi + Codex orchestrator

Pi をエージェントランタイム、Codex をオーケストレーター、OpenCode を作業ワーカーとして使う。
プロジェクトローカル拡張 `.pi/extensions/opencode-orchestrator/` が、同期タスク、
バックグラウンドサブエージェント、フェーズ付きworkflowを登録し、目的・対象パス・制約・
期待出力を構造化して `opencode run --format json` へ渡す。

初回だけ通常の `pi` を起動して `/login` を実行し、`ChatGPT Plus/Pro (Codex)` を選ぶ。
以後はリポジトリルートで次を実行する。

```bash
scripts/pi_codex_orchestrator.sh
```

最初の依頼例:

```text
AGENTS.md に従ってこの不具合を修正して。独立した探索・実装・テストは
opencode_spawn で並列委譲し、opencode_wait で回収して。複数フェーズが必要な場合だけ
opencode_workflow を使い、あなたは計画、判断、差分確認、最終検証を担当して。
```

既定値は Pi 側が `openai-codex/gpt-5.5`、OpenCode 側が `opencode-go/glm-5.2`。
必要なら環境変数で上書きできる。

```bash
PI_CODEX_MODEL=openai-codex/gpt-5.5 \
PI_OPENCODE_MODEL=opencode-go/glm-5.2 \
scripts/pi_codex_orchestrator.sh
```

その他の設定:

- `PI_OPENCODE_BIN`: OpenCode CLI のパス。既定は `opencode`
- `PI_OPENCODE_TIMEOUT_MS`: 1ワーカーのタイムアウト。既定は600000ミリ秒、上限30分
- `/opencode-status`: Pi セッション内でワーカーモデルとタイムアウトを表示

### Background workers

- `opencode_spawn`: ワーカーをバックグラウンド起動
- `opencode_wait`: 複数ワーカーの完了待ちと結果回収
- `opencode_check` / `opencode_list`: 状態・直近アクティビティ確認
- `opencode_cancel`: 実行中プロセスを停止
- `opencode_task`: 小さな単発タスクを同期実行する互換用ショートカット

同時実行上限は全体で4。`read_only` は対象パスが重複しても並列実行できる。
`write` も並列実行できるが、`relevant_paths` に指定した具体的なファイルまたはディレクトリが
同一・親子関係になるタスクは競合として拒否する。glob、作業ディレクトリ外のパスも拒否する。
これは宣言スコープによる競合防止でありOSサンドボックスではないため、Codexは完了後に差分を確認する。

### Phased workflows

`opencode_workflow` は2フェーズ以上を必須とし、フェーズを順番に、各フェーズ内のタスクを
最大4並列で実行する。単純な1タスクでは使わず、調査→実装、実装→独立検証など依存関係がある
複雑な作業だけに使う。直前フェーズの結果は上限付きで次フェーズの各ワーカーへ渡す。
同一フェーズ内のwriteスコープ重複は開始前に拒否する。

- `opencode_workflow_wait`: workflow完了待ち
- `opencode_workflow_check` / `opencode_workflow_list`: 状態確認
- `opencode_workflow_cancel`: workflowと配下の実行中ワーカーを停止

バックグラウンド結果は、親Codexが明示的にwaitしなかった場合、Piのfollow-upメッセージとして返る。
実装結果を信用してそのまま完了せず、Codexが関連差分と検証結果を確認する。

読み取り専用の独立意見:

```bash
codex exec --sandbox read-only "$(cat /tmp/codex_prompt.md)"
```

実装を含む機械作業:

```bash
opencode run "$(cat /tmp/opencode_prompt.md)"
```

実装後の最低確認:

```bash
git diff --stat
.venv/bin/python -m unittest discover -s tests -q
```

## Current GPU Check

2026-07-05 時点の確認:

- GPU0: Tesla P40 24576MiB。LLM/Ollama 用。
- GPU1: Quadro P5000 16384MiB。STT/TTS/ONNX 推論用。
- 両GPUとも Compute Capability 6.1 の Pascal 世代。
- 現在の PyTorch 2.10.0+cu128 は `sm_61` 非対応のため、sentence-transformers 等の PyTorch CUDA は使わずCPUへ落とす。
- CTranslate2 4.7.1 の CUDA compute type は `int8`, `float32`, `int8_float32`。`float16` は使わない。
- `ollama ps` は未ロード。`gemma4:26b` + `num_ctx=16384` は以前はP40に乗ったが、空きVRAMは薄い。

推奨:

- 常駐会話: `gemma4:26b` / `num_ctx=16384` は維持可。ただし追加モデル同時起動は避ける。
- プログラミング重視: `qwen3-coder:30b` を使う場合は `num_ctx=8192` から始める。
- 軽快さ重視: `qwen2.5:14b-instruct-q4_K_M` / `num_ctx=8192`。
- Ollama は可能なら systemd override で `CUDA_VISIBLE_DEVICES=0` を設定し、P5000をPython推論用に空ける。

Ollama systemd override:

```bash
sudo install -D -m 0644 \
  scripts/systemd/ollama-gpu-p40.override.conf \
  /etc/systemd/system/ollama.service.d/10-gpu-p40.conf
sudo systemctl daemon-reload
sudo systemctl restart ollama.service
ollama ps
```

`ollama ps` の `PROCESSOR` が `100% GPU` または GPU/CPU 混在になれば正常。`100% CPU` のままなら、GPU discovery がまだ失敗している。
