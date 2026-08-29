# ドキュメント索引

> **状態**: active / canonical
> **位置付け**: ドキュメント群の索引と権限・状態の一覧
> **対象範囲**: docs/ 配下のドキュメント索引・状態マトリクス・時系列
> **作成日**: 2026-08-27
> **更新日**: 2026-08-29
> **日付根拠**: Git commit date

各文書は冒頭に共通のメタデータブロック（6フィールド）を持つ。語彙は
[メタデータ語彙](#メタデータ語彙) を参照。**日付は原則 Git commit date（コミット日）であり、
執筆日とは限らない。**

## 状態マトリクス

### 現在の正典 (Current Canonical)

| 文書 | 状態 | 作成日 | 内容 |
| --- | --- | --- | --- |
| [implementation_status.md](./implementation_status.md) | active / canonical | 2026-08-28 | リポジトリの実装状態の正典（6段階マトリクス）。deployed / verified は該当なし。実装状態の確認はここを正とする |
| [implementation_plan.md](./implementation_plan.md) | active / canonical | 2026-08-28 | 実装タスク順序の正典（P0-1〜P0-4・P1と延期項目）。実装状態は implementation_status.md を正とする |
| [infrastructure_plan.md](./infrastructure_plan.md) | active / canonical | 2026-08-22 | インフラ固有計画と次アクションの正本（I1〜I6）。**インフラ計画の正典**（リポジトリ実装状態は implementation_status.md、タスク順序は implementation_plan.md を参照）。I1はリポジトリ実装済み、サブPC実機確認待ち |
| [orchestration.md](./orchestration.md) | active / canonical | 2026-07-05 | Pi / pi-orch による開発委譲運用の正典 |
| [companion_roadmap.md](./plans/companion_roadmap.md) | active / canonical | 2026-08-16 | ローカル常駐コンパニオンの製品方向と実装ロードマップの正典 |

### 草案 (Drafts)

| 文書 | 状態 | 作成日 | 内容 |
| --- | --- | --- | --- |
| [runbook.md](./runbook.md) | draft / supporting | 2026-08-22 | 障害対応、停止再起動、更新rollback、復元訓練の**デプロイ前ドラフト**（全コマンド★未検証） |
| [docker_setup.md](./docker_setup.md) | draft / supporting | 2026-08-27 | Ubuntu 24.04 (I1) セットアップのドラフト手順（Docker / postgres:16、pre-deployment・未検証） |

### 設計・手順 (Supporting Designs/Procedures)

| 文書 | 状態 | 作成日 | 内容 |
| --- | --- | --- | --- |
| [adaptive_growth.md](./designs/adaptive_growth.md) | active / supporting | 2026-07-14 | 適応成長（Growth Points）の設計 |
| [continuous_conversation_loop.md](./designs/continuous_conversation_loop.md) | active / supporting | 2026-07-14 | 自発会話ループの設計（状態遷移・ケイデンス・quiet hours） |
| [priority_orchestration.md](./designs/priority_orchestration.md) | active / supporting | 2026-07-14 | タスク優先順位付けの設計 |
| [windows_desktop.md](./designs/windows_desktop.md) | active / supporting | 2026-07-14 | Windowsネイティブアプリの設計・ビルド手順 |
| [personal_lora_training.md](./training/personal_lora_training.md) | active / supporting | 2026-07-15 | 個人LoRA学習（SFT / DPO・マージ / GGUF配備）の専門計画 |

### 完了記録 (Completed Records)

> 以下は**リポジトリ実装の完了記録**であり、デプロイされた環境の証拠ではない。

| 文書 | 状態 | 作成日 | 内容 |
| --- | --- | --- | --- |
| [assistant_platform_plan.md](./archive/assistant_platform_plan.md) | completed / supporting | 2026-08-16 | マルチモデル基盤実装の完了記録（Provider / Router / Context / Cloud）。新規ネクストアクションの正本ではない |
| [homelab_operations_plan.md](./archive/homelab_operations_plan.md) | completed / supporting | 2026-08-16 | 運用基盤（Inventory / Health / Backup / Runbook、N1〜N7）の完了記録と運用原則 |

### 決定記録 (Decisions)

| 文書 | 状態 | 作成日 | 内容 |
| --- | --- | --- | --- |
| [test_factory_strategy.md](./decisions/test_factory_strategy.md) | active / canonical | 2026-08-27 | テストデータ・ファクトリ戦略の決定記録（thoughtbot/fishery は今は導入しない） |
| [local_inference_backend.md](./decisions/local_inference_backend.md) | active / canonical | 2026-08-28 | ローカル推論 backend の決定記録（keyless対応ローカルproviderをOllamaと並行導入、Ollama既定維持） |
| [voice_context_parity.md](./decisions/voice_context_parity.md) | active / canonical | 2026-08-29 | 音声対話の会話構成 Parity の決定記録（CLI / Voice CLI / Voice パイプラインは同一の共通構成を共有。Voice 固有の RAG / persona は既定有効（`--no-rag` / `--no-persona` で無効化）、tasks / calendar_context は既定構築・利用可能時のみ文脈（読み取り専用）、カレンダー書込と P0-3 センサーは明示 opt-in、これらはチャネル能力拡張。P0-4） |
| [desktop_quick_chat_hud.md](./decisions/desktop_quick_chat_hud.md) | active / canonical | 2026-08-29 | P1 Desktop Quick-Chat HUD の決定記録（既存 DesktopBridge 再利用・新規エンドポイントなし・count-only Today サマリ・desired-state クリックスルー配線・Ctrl+Alt+Space hotkey 復帰・2D/3D fallback 維持） |
| [task_delivery_consistency.md](./decisions/task_delivery_consistency.md) | active / canonical | 2026-08-29 | タスク配送一貫性の決定記録（リマインド通知とタスク⇔カレンダー同期を at-least-once best-effort 契約に統一。tasks.rev 楽観制御・BEGIN IMMEDIATE claim・revalidate-before-callback・expected_rev 条件付き record・state-driven 同期・マーカー照合/重複整理で非センサー drop/update/snooze 競合を解決。residual: micro-TOCTOU・queue/pull latency・一意 owner 規律・exactly-once / durable outbox なし） |

### 参考 (Non-authoritative References)

| 文書 | 状態 | 作成日 | 内容 |
| --- | --- | --- | --- |
| [references/gitsugest.md](./references/gitsugest.md) | reference / non-authoritative | 2026-02-11 | 外部AIが生成した参考コード・提案の控え。設計として採用済みではない |

## 時系列（作成日順）

| 作成日 | 文書 |
| --- | --- |
| 2026-02-11 | [references/gitsugest.md](./references/gitsugest.md) |
| 2026-07-05 | [orchestration.md](./orchestration.md) |
| 2026-07-14 | [adaptive_growth.md](./designs/adaptive_growth.md) / [continuous_conversation_loop.md](./designs/continuous_conversation_loop.md) / [priority_orchestration.md](./designs/priority_orchestration.md) / [windows_desktop.md](./designs/windows_desktop.md) |
| 2026-07-15 | [personal_lora_training.md](./training/personal_lora_training.md) |
| 2026-08-16 | [assistant_platform_plan.md](./archive/assistant_platform_plan.md) / [companion_roadmap.md](./plans/companion_roadmap.md) / [homelab_operations_plan.md](./archive/homelab_operations_plan.md) |
| 2026-08-22 | [infrastructure_plan.md](./infrastructure_plan.md) / [runbook.md](./runbook.md) |
| 2026-08-27 | [README.md](./README.md) / [docker_setup.md](./docker_setup.md) / [test_factory_strategy.md](./decisions/test_factory_strategy.md) |
| 2026-08-28 | [implementation_status.md](./implementation_status.md) / [implementation_plan.md](./implementation_plan.md) / [decisions/local_inference_backend.md](./decisions/local_inference_backend.md) |
| 2026-08-29 | [decisions/voice_context_parity.md](./decisions/voice_context_parity.md) / [decisions/desktop_quick_chat_hud.md](./decisions/desktop_quick_chat_hud.md) / [decisions/task_delivery_consistency.md](./decisions/task_delivery_consistency.md) |

## インフラの次アクション

- I1実機確認: サブPCでhealth、OS再起動後の自動起動、実`pg_dump`、`--verify-only`、復元訓練
- I2: `PostgresRunLogger` + Alembic + SQLite/PG切替
- I3: Prometheus `/metrics` + Grafana + alerts

進捗を更新するときは、実装状態は`implementation_status.md`、タスク順序は`implementation_plan.md`、
インフラ計画は`infrastructure_plan.md`を更新し、この索引には要約だけ反映する。
完了記録（`completed / supporting`）は現在の次アクションの正本ではない。

## メタデータ語彙

各文書の冒頭ブロックは6フィールドで構成する。

| フィールド | 意味 |
| --- | --- |
| 状態 | 文書の権威・状態。`active / canonical`（現在の正典）、`active / supporting`（現行の設計・手順）、`draft / supporting`（デプロイ前の草案・未検証）、`completed / supporting`（完了記録）、`reference / non-authoritative`（参考・非権威） |
| 位置付け | 文書の役割・権威の位置付け |
| 対象範囲 | 文書が扱うスコープ |
| 作成日 | 初版の日付 |
| 更新日 | 最終更新の日付 |
| 日付根拠 | 日付の出所。原則 **Git commit date**（コミット日）であり、執筆日とは限らない |

例外: `test_factory_strategy.md` は日付根拠を「ドキュメント記録」とし、
作成日はドキュメント記録日（2026-08-27）を使用する。