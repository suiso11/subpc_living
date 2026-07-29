# Pi Subagent Orchestration

Claude/Anthropic系に依存せず、Piを親ランタイムにする。
`pi-subagents`のnamed agentは探索・テスト・レビューの読み取り専用経路とし、実装は
OpenCode workerのGLM 5.2を既定バックエンド、Kimi K3を読み取り専用の別視点レビューへ使う。
実装は `opencode_change` を既定とし、各変更はKimi K3の読み取り専用自動レビューを経てから
最終承認をGPT-5.6 Solが行う。

## Components

- Pi: `@earendil-works/pi-coding-agent` 0.79以上
- Subagent runtime: `git:github.com/edxeth/pi-subagents`
- Parent model: 既定 `openai-codex/gpt-5.6-sol`
- Project agents: `.pi/agents/subpc-*.md`
- Pi-only launcher: `scripts/pi_codex_orchestrator.sh`
- Combined Pi/OpenCode launcher: `pi-orch`
- OpenCode default: `opencode-go/glm-5.2` (実装の既定バックエンド)
- OpenCode independent review: `opencode-go/kimi-k3` (読み取り専用)
- Implementation default tool: `opencode_change` (GLM生成 → Kimi K3読み取り専用自動レビュー → Sol最終承認)

プロジェクトローカルのパッケージ設定は `.pi/settings.json` に保存する。キャッシュ本体は
`.pi/git/` のignore対象で、リポジトリには設定とキャッシュ用 `.gitignore` だけを残す。

## Install and update

初回導入:

```bash
pi install git:github.com/edxeth/pi-subagents -l --approve
```

導入確認:

```bash
pi list --approve
```

更新:

```bash
pi update git:github.com/edxeth/pi-subagents --approve
```

Pi 0.79未満では動かないため、必要なら先にPi本体を更新する。

## Start

通常はリポジトリルートで次を実行する。

```bash
scripts/pi_codex_orchestrator.sh
```

スクリプトは `PI_ORCHESTRATOR_MODE=1` を設定する。親のread/bash/edit/write等は削除され、
親は `subagent`、`subagent_resume`、`subagent_kill` による分解・委譲・統合だけを行う。
`lineage-only` 子が親セッションへ紐づくため、orchestrator起動時は `--no-session` を付けない。
`-p` / `--print` のone-shot実行時だけ、子の同期結果を親が最終回答へ統合できるよう
coordinator-only turn stopを自動解除する。

親モデルとthinkingは上書きできる。

```bash
PI_CODEX_MODEL=openai-codex/gpt-5.6-sol \
PI_CODEX_THINKING=high \
scripts/pi_codex_orchestrator.sh
```

子モデルはagent定義で固定し、親からのmodel overrideを許可しない。

## Named agents

### `subpc-scout`

- 読み取り専用
- 関連ファイル、呼び出し経路、依存関係、既存パターンの探索
- `openai-codex/gpt-5.6-terra`

### `subpc-tester`

- ソース編集ツールなし
- 関連テスト、全体テスト、失敗診断
- `openai-codex/gpt-5.6-terra`

### `subpc-reviewer`

- ソース編集ツールなし
- 要件と現在の差分を独立レビュー
- `openai-codex/gpt-5.6-sol`

全子agentに以下を共通設定する。

- `mode: background`
- `async: true`
- `auto-exit: true`
- `session-mode: lineage-only`
- `extensions: none`
- `skills: none`
- `spawning: false`
- `no-context-files: true`
- `trust-project: false`
- `parent-close-policy: terminate`

子はプロジェクト拡張をロードせず、再帰的なsubagent起動もできない。agent本文の指示に従い、
作業開始時に `AGENTS.md` をread toolで明示的に読む。

## OpenCode role split

- default / `glm`: `opencode-go/glm-5.2`。実装の既定バックエンド。探索、機械作業、実装、テスト追加
- `kimi_k3`: `opencode-go/kimi-k3`。読み取り専用の広域・独立レビュー
- 実装は `opencode_change` を既定ツールとする。GLMが変更候補を生成し、完了後にKimi K3が
  読み取り専用で自動レビューを実施する。PiはKimiの指摘を確認のうえ、最終承認をGPT-5.6 Solへ渡す
- Kimiは変更を書けず、本番データ・秘密情報・サービス操作にも触らない
- Kimi/GLMの結論は最終承認にせず、`subpc-reviewer`または親のGPT-5.6 Solが最終判断する
- OpenCode出力は最終承認に使わず、`subpc-reviewer`または親のGPT-5.6 Solで確認する

統合構成はリポジトリルートで `pi-orch` を実行する。

## Delegation contract

各子タスクには次を含める。

1. lower-kebab形式のmachine name
2. 3〜8語の短いtitle
3. 使用するagent名
4. 目的
5. 関連ファイルと、writeの場合は変更可能パス
6. 制約と変更禁止範囲
7. 期待する出力と検証条件

### 実装タスクの分解

- 実装は1タスク1関心事に分割する
- 1つのwriteタスクが扱う変更可能パスは最大5件まで。globで広く渡さず具体パスを列挙する
- 複数モジュールにまたがる作業や「調査 + 実装 + テスト」を含む作業は2〜4個の非重複
  `opencode_change` タスクに分解し、GLM workerへ渡す。named agentはscout/tester/reviewerの
  読み取り・検証・診断だけに使い、実装はnamed agentでは行わない
- 並列writeタスクは変更可能パスが重ならないよう分割し、重なる場合は直列化する
- 指示範囲外のファイルは変更せず、必要になったら停止して追加スコープを報告させる
- 実装の既定はOpenCode GLM経由の `opencode_change` とし、各変更はKimi K3読み取り専用自動レビューを経てからSol最終承認へ進む

例:

```text
subpc-scoutへ、src/chat/client.pyとtests/test_chat_client.pyだけを対象に
タイムアウト処理の呼び出し経路を調査させて。変更は禁止。関連シンボル、
既存テスト、最小実装境界を返させる。
```

独立タスクは1回の `subagent` 呼び出しの `children` へまとめる。最大4並列を運用上限とし、
writeタスクの対象パスが同一または親子関係になる場合は並列化しない。

## Sessions and results

子は既定でasync起動される。結果は完了後に親へ自動配送され、新しい親ターンが始まる。
結果待ちの間に親が同じ作業をやり直してはいけない。

- 同じ子が直前に探索した範囲の追加調査・修正: `subagent_resume`
- 独立レビューや別観点の検証: 新しい子をspawn
- 不要または暴走した子: `subagent_kill`

親を閉じると子も停止する。長時間ジョブを親終了後も続ける設定にはしていない。

## Security boundary

Piのtool allowlistと`pi-subagents`はOSサンドボックスではない。以下を運用契約として守る。

- 実 `.env`、`config/discord.env`、秘密情報を子タスクへ含めない
- 子に依存追加、ネットワーク利用、commit/push、サービス再起動をさせない
- writeタスクは具体的な変更可能パスを列挙する
- 未コミット変更がある場合、対象外を戻さない
- 実装結果を同じ実装agentの説明だけで承認しない

強い隔離が必要な作業はコンテナや別worktreeを使う。

## Verification

パッケージとagent rosterの確認:

```bash
pi list --approve
scripts/pi_codex_orchestrator.sh -p \
  "subpc-scoutにAGENTS.mdのテストコマンドだけを確認させて"
```

実装後の最低確認:

```bash
git diff --stat
.venv/bin/python -m unittest discover -s tests -q
```

ソース変更がない設定・文書作業では、Pi起動スモーク、agent roster、`git diff --check`を
優先し、アプリサービスは再起動しない。

## Current GPU Check

Ollamaの大きいコンテキストを使う前はAGENTS.md記載のGPU確認を行う。Pi/Codexの
クラウドモデルとbackground子agentはローカルGPUを使用しない。
