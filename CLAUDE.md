# CLAUDE.md

## プロジェクト概要

サブPC常駐のパーソナルAIアシスタント。ローカルLLM (Ollama) + STT (faster-whisper) + TTS を
Web UI・Discord bot・音声パイプラインから使う。

- 仮想環境: サブPCでは `.venv/bin/python` を使う (システムpythonではなく)
- テスト: `.venv/bin/python -m unittest discover -s tests -q`
  - このWindows開発機では `python -m unittest discover -s tests -q`。
    ただし全体実行はWindows環境依存のエラーが元から多数出るため、
    `python -m unittest discover -s tests/<対象> -t . -q` で変更範囲だけ回す
- サービス: `systemctl --user restart subpc-discord.service subpc-web.service subpc-voice.service` (コード変更後に反映)
  - subpc-voice も chat 設定 (config/chat_config.json) を起動時に読む。モデル差し替え時に再起動を忘れると、新旧モデルがGPUを取り合ってロードのスラッシングが起きる (P40 24GB に26b級は1つしか載らない)
- 実設定 `config/discord.env` はgit管理外。例は `config/discord.env.example`

## Orchestration workflow

親オーケストレーターは Claude Code (Opus 5)。計画・分解・ブリーフ作成・統合・最終判断を担当し、
実装と検証は下位モデルへ委譲する。親は自分で大量のコードを書かない。

### 役割分担

| 役 | 実体 | 権限 | 担当 |
| --- | --- | --- | --- |
| 統括 | Claude Code (Opus 5) | 全権 | 計画、分解、ブリーフ作成、統合、最終判断 |
| 実装 | `codex exec -m gpt-5.6-sol -c model_reasoning_effort=high` | 書き込み | 実装・修正のみ |
| 検証 | `verifier` サブエージェント (Sonnet 5) | 読み取り + コマンド実行 | テスト実行、実測、診断 |
| レビュー | `codex exec -m gpt-5.6-luna -c model_reasoning_effort=high` | 読み取り専用 | 受け入れ条件との独立レビュー |

- Sol に自己レビュー・自己検証をさせない。トークンの無駄になり、根拠のない「検証済み」という
  自己申告が混ざる。実装者の申告は結果として数えない
- 検証とレビューを両方Claude側へ寄せない。レビュー役 (Luna) は非Anthropicのまま残し、
  見落としの相関を避ける
- `deep-reasoner` (opus) は既定で無効。ユーザーが名指しで指示したときだけ使う
- 組み込みエージェント (Explore / general-purpose / Plan) を起動する場合は `model: "haiku"` を指定する。
  広範な検索・列挙はまず `Grep`/`Glob` を直接使い、それで足りないときだけ委譲する
- 些細な一発作業 (1ファイルの小修正、単純な質問) は委譲せず直接やる
- opencode (`fast-worker` / `opencode-peer`) は現在の既定フローでは使わない。
  3つ目の視点が必要になったときの予備として残してある

### codex の実行方法 (Windows開発機)

この機ではcodexのサンドボックスが `CreateProcessAsUserW failed` でコマンドを一切起動できない
(`windows.sandbox` が `elevated` / `unelevated` のどちらでも同じ。Store版pwshの起動が原因)。
`python --version` すら実行できないため、`--dangerously-bypass-approvals-and-sandbox` を付けて実行し、
書き込み範囲はサンドボックスではなく**ブリーフ側で担保する**。

```powershell
codex exec --dangerously-bypass-approvals-and-sandbox `
  -m gpt-5.6-sol -c model_reasoning_effort=high `
  -C "<repo>" -o "<scratchpad>\sol_last.md" `
  "指示書 <scratchpad>\brief.md を最初に全文読み、その内容を唯一の仕様として実装を完了させてください。指示書に書かれた変更可能パス以外のファイルは絶対に変更しないこと。git commit はしないこと。"
```

- ブリーフはスクラッチパッドの `.md` に書き、パスだけ渡す (日本語のクォート事故を防ぐ)
- 数分かかるので `run_in_background: true` で起動し、完了通知を待つ。`-o` の中身だけ読む
- 検証とレビューは実装完了後に**並列**で回す
- 独立性を守る: あるエージェントの結論を別のエージェントへの入力に混ぜない

### 委譲ブリーフの必須項目

- 目的と、それが唯一の仕様であること
- **変更してよいパスの列挙** (5件を目安)。それ以外は「1行も変更しない」と明記する
- 前提となる既存APIの写し。憶測でAPIを変えさせない
- 受け入れ条件と、検証に使うコマンド
- 出力形式。要点・変更ファイル・検証結果に絞り、長い思考過程やファイル全文は要求しない
- 読み取り専用役には「ファイルの作成・編集・削除、gitの書き込みを一切行わない」を明記する
- 秘密情報を含む作業は渡さない。実 `.env` と `config/discord.env` は読ませない

### 委譲後の確認

- `git status --short` と `git diff --stat` で許可外の変更が無いか確認する
- 未コミット変更が多い場合はユーザー作業を戻さず、今回触った範囲だけ扱う
- 実装結果をその実装者自身の説明だけで承認しない。検証役の実測と、
  オーケストレーター自身のテスト実行で確認する
- 各エージェントの最終メッセージはユーザーには見えない。重要な内容は最終返答に含める
- Pi/Codex/OpenCode のオーケストレーション環境は
  `https://github.com/suiso11/pi-codex-opencode-orchestrator` で管理する
