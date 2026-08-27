# Pi / pi-orch Orchestration

Piを親ランタイムとする。通常は `pi-orch` で起動する。親はコーディネーター専任で、
要件定義・分解・スコープ付きブリーフ作成・結果統合・競合解消・最終承認だけを担う。
**親は変更を実装せず、テストや検証コマンドも実行しない。**

- **編集はすべてwrite workerへ委譲する。** 1ファイルの小さな修正、文書・設定のみの変更でも、
  変更可能パスを明示したスコープ付きタスクとして `opencode_task` / `opencode_spawn` へ渡す。
  親が直接ファイルを編集しない
- **テスト・検証コマンドは独立read-only testerへ委譲する。**
  差分・要件レビューは独立read-only reviewerへ委譲する
- 親は統合のための読み取り専用 `git status` / `git diff` メタデータ確認のみ可とし、
  それを検証の代わりに使わない

worker / tester / reviewer のモデルID・プロファイルは pi-orch の設定（環境変数・プロファイル）
に従う。この文書にも agent 定義にも固定のモデル名を書かない。設定値を推測して
仮定せず、不明なら親が設定を確認してから委譲する。

## Components

- Pi: `@earendil-works/pi-coding-agent` 0.84以上
- Subagent runtime: `git:github.com/edxeth/pi-subagents`
- Parent / worker / reviewer のモデルとプロファイル: すべて pi-orch 設定に従う
- Project agents: `.pi/agents/subpc-*.md`（named agentsは探索・テスト・レビューの読み取り専用経路）
- Pi-only launcher: `scripts/pi_subagent_orchestrator.sh`
- Combined Pi/OpenCode launcher: `pi-orch`（リポジトリルートで実行）
- Implementation routes: `opencode_task` / `opencode_spawn`（変更可能パスを明示して委譲）

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
pi update --extension git:github.com/edxeth/pi-subagents --approve
```

Pi 0.79未満では動かないため、必要なら先にPi本体を更新する。

## Start

通常はリポジトリルートで次を実行する。

```bash
pi-orch
```

`pi-orch` は親・worker・`glm` / `kimi_k3` profileの現在設定を環境へ渡す。
モデル名をリポジトリへ固定せず、必要なら `PI_*` 環境変数で実行時の設定を確認する。

Pi named agentsだけを使う委譲専用モードも残す。

```bash
scripts/pi_subagent_orchestrator.sh
```

このスクリプトは `PI_ORCHESTRATOR_MODE=1` を設定し、親をsubagent委譲専用にする。
モデル・thinkingが環境で未指定ならPiの現在既定値を使い、固定fallbackは持たない。

## Named agents

### `subpc-scout`

- 読み取り専用
- 関連ファイル、呼び出し経路、依存関係、既存パターンの探索

### `subpc-tester`

- ソース編集ツールなし
- 関連テスト、全体テスト、失敗診断
- 文書・設定のみの変更が委譲された場合は `git diff --check`、構文チェック、設定ファイルの
  整合性確認など、委譲ブリーフで指定された比例した静的・検証コマンドを実行する

### `subpc-reviewer`

- ソース編集ツールなし
- 要件と現在の差分を独立レビュー

モデルとthinkingはagent定義に固定せず、起動した親Piの設定を継承する。

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

- write worker: 実装の既定バックエンド。探索、機械作業、実装、テスト追加。
  すべての編集（1ファイル修正・文書・設定を含む）はここへ委譲する。
  モデル・プロファイルはpi-orch設定に従う
- tester: 読み取り専用の独立検証子。テスト・検証コマンド（`git diff --check` や
  構文・設定チェックを含む）だけを実行し、変更は書かない
- reviewer: 読み取り専用の広域・独立レビュー。実装workerとは別の子として起動する
- 実装は `opencode_task` / `opencode_spawn` を既定とする。workerが変更を生成し、完了後に
  独立read-onlyレビューを実施する。親は指摘を確認のうえ最終承認を行う
- tester / reviewerは変更を書けず、本番データ・秘密情報・サービス操作にも触らない
- 子の結論は最終承認にせず、親が最終判断する
- OpenCode出力はそのまま承認に使わず、独立レビューまたは親で確認する

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
  writeタスクに分解し、OpenCode workerへ渡す。named agentはscout/tester/reviewerの
  読み取り・検証・診断だけに使い、実装はnamed agentでは行わない
- 並列writeタスクは変更可能パスが重ならないよう分割し、重なる場合は直列化する
- 指示範囲外のファイルは変更せず、必要になったら停止して追加スコープを報告させる
- 実装の既定はOpenCode workerへの `opencode_task` / `opencode_spawn` 委譲とし、各変更は
  独立read-onlyレビューを経てから親の最終承認へ進む

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

- 実 `.env`、`config/discord.env`、秘密情報を子タスクへ含めない。子に読ませない
- 子に依存追加、ネットワーク利用、commit/push、サービス再起動をさせない
- 子の再帰委譲を禁止する (`spawning: false`)
- writeタスクは具体的な変更可能パスを列挙する。範囲外は触れさせず、必要なら停止させて報告させる
- 未コミット変更がある場合、対象外を戻さない
- 実装結果を同じ実装agentの説明だけで承認しない

強い隔離が必要な作業はコンテナや別worktreeを使う。

## Verification

パッケージとagent rosterの確認:

```bash
pi list --approve
scripts/pi_subagent_orchestrator.sh -p \
  "subpc-scoutにAGENTS.mdのテストコマンドだけを確認させて"
```

実装後の確認は親が直接実行せず、委譲で行う:

- 親: `git status --short` / `git diff --stat` の読み取りのみ。統合メタデータであり検証ではない
- テスト・検証コマンド（unittest、`git diff --check`、構文・設定チェック等）:
  独立read-only testerへ委譲
- 差分・要件レビュー: 独立read-only reviewerへ委譲

ソース変更がない設定・文書作業でも編集はwrite workerへ、検証（`git diff --check` や
構文・設定チェック）はtesterへ委譲する。アプリサービスは再起動しない。

## Current GPU Check

Ollamaの大きいコンテキストを使う前はAGENTS.md記載のGPU確認を行う。pi-orch設定の
クラウドモデルとbackground子agentはローカルGPUを使用しない。
