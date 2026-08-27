# ドキュメント案内

## 現在の主線

1. `infrastructure_plan.md` — **現在の実装計画と次アクションの正本**。I1はリポジトリ実装済み、サブPC実機確認待ち
2. `docker_setup.md` — I1のUbuntu 24.04導入・PostgreSQL起動・バックアップ/復元手順
3. `runbook.md` — 障害対応、更新、rollback、復元訓練

## 製品ロードマップ

- `companion_roadmap.md` — Companion Phase 1〜5完了、Phase 6aと6b静的3D表示まで完了
- `assistant_platform_plan.md` — Provider / Router / Context / Cloud基盤の完了記録。新規ネクストアクションの正本ではない
- `homelab_operations_plan.md` — Inventory / Health / Backup / Runbook（N1〜N7）の完了記録と運用原則

## 個別設計

- `adaptive_growth.md` — 適応・成長
- `continuous_conversation_loop.md` — 継続会話
- `priority_orchestration.md` — タスク優先順位
- `personal_lora_training.md` — 個人LoRA学習
- `windows_desktop.md` — Windowsデスクトップ
- `orchestration.md` — Pi / pi-orch開発オーケストレーション

## インフラの次アクション

- I1実機確認: サブPCでhealth、OS再起動後の自動起動、実`pg_dump`、`--verify-only`、復元訓練
- I2: `PostgresRunLogger` + Alembic + SQLite/PG切替
- I3: Prometheus `/metrics` + Grafana + alerts

進捗を更新するときは、まず`infrastructure_plan.md`を更新し、この索引には要約だけ反映する。
