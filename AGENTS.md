# AGENTS.md

## Project

サブPC常駐のパーソナルAIアシスタント。ローカルLLM (Ollama) + STT (faster-whisper) + TTS を
Web UI・Discord bot・音声パイプラインから使う。

- 仮想環境: サブPCでは `.venv/bin/python`
- テスト: `.venv/bin/python -m unittest discover -s tests -q`
  - Windows開発機では `python -m unittest discover -s tests/<対象> -t . -q`。
    全体実行はWindows環境依存のエラーが元から多数出るため、変更範囲だけ回す
- サービス反映: `systemctl --user restart subpc-discord.service subpc-web.service`
- 実設定 `config/discord.env` は git 管理外。例は `config/discord.env.example`

## 委譲workerとして起動された場合 (最優先)

親オーケストレーターからブリーフ (指示書) を渡されて起動されたときは、次を守る。
これは以下のどの記述よりも優先する。

- **ブリーフが唯一の仕様**。設計の再検討や範囲拡大をしない
- **ブリーフが列挙した「変更してよいパス」以外を1行も変更しない**。
  必要になったら停止し、追加スコープを親へ報告する
- **さらに下位へ再委譲しない**。自分で読んで自分で実装する
- 読み取り専用と指定されたら、ファイルの作成・編集・削除とgitの書き込み操作を一切しない
- `git commit` / `git push` / PR作成は、ユーザーが明示的に要求し、かつブリーフがリリース操作を
  明示指定した専用リリースworkerだけが実行する。通常のworkerはブリーフにリリース操作の明示指定が
  ない限り一切しない。コミット判断は親とユーザーが行う
- 自分の成果を自分で「検証済み・レビュー済み」と申告しない
- 出力は要点・変更ファイル・検証結果に絞る。長い思考過程やファイル全文を返さない

## 役割分担 (3層)

| 役 | 実体 | 権限 | 担当 |
| --- | --- | --- | --- |
| コーディネーター | 親Pi | 委譲と読み取り専用の統合メタデータ確認のみ。実装・テスト・検証・サービス操作・変更系コマンドは禁止 | 要件、分解、ブリーフ作成、結果統合、調整、最終承認のみ |
| worker | OpenCode worker | 限定write (リリース操作は明示指定があるときのみ) | すべての実装・修正。1タスク1関心事 |
| リリースworker | OpenCode worker (専用) | 限定write + リリース操作 (commit / push / PR作成) | ユーザー明示要求に基づくパススコープ付きリリース操作のみ |
| tester | 読み取り専用tester | 読み取り専用 | テスト・検証コマンドの独立実行 |
| reviewer | OpenCode reviewer / named reviewer | 読み取り専用 | 差分・要件の独立レビュー |

- **親はコーディネーター専任**。親自身は一切の変更を実装せず、テスト・検証・サービス操作・
  変更系コマンドも実行しない。commit・push・PR作成などのリリース操作コマンドも一切実行しない。
  親が使えるのは委譲と、統合のための読み取り専用メタデータ確認だけ。
  親が持つのは要件、分解、スコープ付きブリーフ、結果統合、競合解消、最終承認だけ
- **編集はすべてwrite workerへ**。1ファイルの小さな修正や文書・設定の編集でも、必ず
  変更可能パスを列挙したスコープ付きwriteタスクとして委譲する。親が直接編集しない
- **commit / push / PRは専用リリースworkerへ**。ユーザーが明示的に要求したときのみ、ブリーフに
  リリース操作を明示指定したパススコープ付きタスクとして委譲する。親がリリース操作を兼ねない
- **テスト・検証コマンドは独立read-only testerへ**。差分・要件レビューは独立read-only
  reviewerへ。親が子の代わりに検証を兼ねない
- 親は統合のための読み取り専用 `git status` / `git diff` メタデータ確認のみ可。それを
  テスト代わりに使わない
- worker / tester / reviewer のモデル・プロファイルは pi-orch の設定に従い、固定値を仮定しない。
  実装は `opencode_task` / `opencode_spawn` 経由で起動する
- reviewer / tester は実装workerとは別の独立した読み取り専用子。write・本番データ・秘密情報・
  サービス操作を渡さない
- 子の結論は最終承認にせず、オペレーターが最終判断する
- Pi named agents (`subpc-scout`, `subpc-tester`, `subpc-reviewer`) は探索・テスト・レビュー用。
  実装は OpenCode worker へ委譲し、named agent では実装しない

## 委譲の原則

- 1つのwriteタスクが扱う変更可能パスは最大5件。複数モジュールにまたがる作業は
  2〜4個の非重複タスクに分割する
- 並列のwriteタスクは変更可能パスが重ならないよう分割する。重なる場合は直列化する
- 独立タスクは1回の起動呼び出しにまとめ、最大4並列にする
- 秘密情報を含む作業は子へ渡さない。実 `.env` と `config/discord.env` は読ませない
- 子の再帰委譲を防ぐ (`spawning: false`, `no-context-files: true`)

## 委譲プロンプトの要件

- machine name、3〜8語のtitle、agent名、目的、関連ファイル、制約、期待出力を必ず含める
- writeタスクは変更可能ファイルまたはディレクトリを具体的に列挙する。globで広く渡さない
- 指示範囲外のファイルは変更せず、必要になったら停止して追加スコープを報告させる
- 独立検討では、他エージェントの結論を混ぜず、コード・ログ・要件だけを渡す
- 出力は要点、変更ファイル、検証結果に絞らせ、長い思考過程やファイル全文を要求しない

## 委譲後の確認 (親はコーディネーターのみ)

- 親は `git status --short` と `git diff --stat` の読み取りで統合状況を把握するだけ。
  これはテスト・検証の代わりにならない
- テスト・検証は独立read-only testerへ委譲し、差分・要件レビューは独立read-only reviewerへ
  委譲した結果だけで判断する
- 未コミット変更が多い場合はユーザー作業を戻さず、今回触った範囲だけ扱う
- 実装結果をworker自身の説明だけで承認しない。独立確認を経てオペレーターが最終判断する

## リリース操作の契約 (commit / push / PR作成)

ユーザーが明示的に要求し、ブリーフがリリース操作を明示指定したパススコープ付きタスクとして、
専用リリースworkerだけが以下を守って実行する。

- ブリーフに変更可能パスとリリース操作 (commit / push / PR作成) を明示する。globで広く渡さない
- 通常の実装workerは、ブリーフがリリース操作を明示指定しない限り commit / push / PR を一切しない
- commit前に `git status` / `git diff --cached` でステージ対象パスを確認し、指示範囲外や
  未確認の変更をコミットしない
- git管理外の秘密情報 (`config/discord.env`、実 `.env`) をステージ・コミットに含めない
- force push・履歴書き換え (rebase --force / reset --hard など) はしない
- ユーザーの未コミット変更をrevert・破棄しない。今回触った範囲だけ扱う
- commitは論理的にスコープを1つに保ち、まとめ過ぎない
- 実行後はブランチ名・リモート名・PR URLを親へ報告する
- GitHub接続・ネットワーク・認証の失敗は報告し、回避・迂回しない

詳細な運用契約は `docs/orchestration.md` を参照する。

## Resource Check Before LLM-Heavy Work

Ollama の大きいコンテキストを使う前に確認する。

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.free,memory.total,utilization.gpu --format=csv,noheader,nounits
ollama ps
```

`ollama ps` の `CONTEXT` が目的の値なら、その値ではロード済み。VRAM の残りが薄い場合は、同時処理や別モデル起動を避ける。
