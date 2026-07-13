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
