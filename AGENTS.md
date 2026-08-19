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
- `git commit` はしない。コミット判断は親とユーザーが行う
- 自分の成果を自分で「検証済み・レビュー済み」と申告しない
- 出力は要点・変更ファイル・検証結果に絞る。長い思考過程やファイル全文を返さない

## 役割分担 (3層)

| 役 | 実体 | 権限 | 担当 |
| --- | --- | --- | --- |
| オペレーター | 親 | 全ツール | 計画、分解、ブリーフ作成、統合、最終判断 |
| worker | OpenCode `opencode-go/deepseek-v4-pro` | 限定write | 実装・修正のみ。1タスク1関心事 |
| reviewer | OpenCode `opencode-go/kimi-k3` | 読み取り専用 | 差分・要件の独立レビュー |

- worker の既定は `opencode-go/deepseek-v4-pro`。軽作業は `deepseek-v4-flash`。
  OpenCodeの `opencode_task` / `opencode_spawn` 経由で起動する
- reviewer は常に読み取り専用。write・本番データ・秘密情報・サービス操作を渡さない
- worker / reviewer の結論は最終承認にせず、オペレーターが最終判断する
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

## 委譲後の確認

- `git status --short`、`git diff --stat` と関連テストで結果を検証する
- 未コミット変更が多い場合はユーザー作業を戻さず、今回触った範囲だけ扱う
- 実装結果をworker自身の説明だけで承認しない。reviewerに確認させ、
  オペレーターが最終判断する

## Resource Check Before LLM-Heavy Work

Ollama の大きいコンテキストを使う前に確認する。

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.free,memory.total,utilization.gpu --format=csv,noheader,nounits
ollama ps
```

`ollama ps` の `CONTEXT` が目的の値なら、その値ではロード済み。VRAM の残りが薄い場合は、同時処理や別モデル起動を避ける。
