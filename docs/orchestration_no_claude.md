# Pi Subagent Orchestration

Claude/Anthropic系に依存せず、Piを親ランタイム、`pi-subagents`を子プロセス管理に使う。
旧構成の「Pi親 → 独自拡張 → OpenCode worker」は廃止し、探索・実装・テスト・レビューを
Piのnamed agentへ直接委譲する。

## Components

- Pi: `@earendil-works/pi-coding-agent` 0.79以上
- Subagent runtime: `git:github.com/edxeth/pi-subagents`
- Parent model: 既定 `openai-codex/gpt-5.5`
- Project agents: `.pi/agents/subpc-*.md`
- Launcher: `scripts/pi_codex_orchestrator.sh`

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
PI_CODEX_MODEL=openai-codex/gpt-5.5 \
PI_CODEX_THINKING=high \
scripts/pi_codex_orchestrator.sh
```

子モデルはagent定義で固定し、親からのmodel overrideを許可しない。

## Named agents

### `subpc-scout`

- 読み取り専用
- 関連ファイル、呼び出し経路、依存関係、既存パターンの探索
- `openai-codex/gpt-5.4-mini`

### `subpc-implementer`

- read/bash/edit/writeを使用可能
- 委譲タスクで明示した非重複パスだけを変更
- `openai-codex/gpt-5.5`

### `subpc-tester`

- ソース編集ツールなし
- 関連テスト、全体テスト、失敗診断
- `openai-codex/gpt-5.4-mini`

### `subpc-reviewer`

- ソース編集ツールなし
- 要件と現在の差分を独立レビュー
- `openai-codex/gpt-5.5`

全子agentに以下を共通設定する。

- `mode: background`
- `async: true`
- `auto-exit: true`
- `session-mode: lineage-only`
- `extensions: none`
- `skills: none`
- `spawning: false`
- `trust-project: false`
- `parent-close-policy: terminate`

子はプロジェクト拡張をロードせず、再帰的なsubagent起動もできない。agent本文の指示に従い、
作業開始時に `AGENTS.md` をread toolで明示的に読む。

## Delegation contract

各子タスクには次を含める。

1. lower-kebab形式のmachine name
2. 3〜8語の短いtitle
3. 使用するagent名
4. 目的
5. 関連ファイルと、writeの場合は変更可能パス
6. 制約と変更禁止範囲
7. 期待する出力と検証条件

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
