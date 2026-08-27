# インフラ整備計画 (Infrastructure Plan)

## 位置付け

`subpc_living`は「Webアプリ」ではなく**常駐型ローカルAIシステム**へ向かう。
したがって「落ちない・復旧する・状態を保持する・各サービスを管理する」という
インフラ課題が、そのまま製品品質になる。

本計画は `docs/homelab_operations_plan.md` の運用基盤（Inventory / Health / Backup / Runbook）を
土台に、**サービス分離・状態DB・監視・自動復旧・CI/CD** を段階的に追加する。

### 既存の非目標との整合

`homelab_operations_plan.md` §5 は「PostgreSQLへの一括移行」「全サービスDocker化」を
非目標としていた。本計画でこれらを**意図的に覆す**。ただし覆し方は段階的であり、
「一括移行」「全サービス一斉Docker化」は引き続き避ける。

- PostgreSQL: **一括ではなくストア単位の段階移行**。まず write-heavy で Protocol 分離済みの
  実行ログから着手し、SQLite は移行完了まで並走させる。
- Docker: **全サービス一斉ではなく infra層から導入**。Ollama は GPU pinning と systemd watchdog が
  動いているため、当初はホスト systemd のまま維持する。

### 引き続きやらないこと（判断）

- Kubernetes / Terraform / AWS EKS / Service Mesh: 1〜数台構成では技術を使うための導入になり、
  プロジェクトを良くしない。3台目以降・再構築頻発時に再検討。
- Redis: 現状のキューは in-process + SQLite。Redis導入はサービス分離後に「実際に詰まってから」判断。
- インターネット公開・Port Forward: 不可。外部は Tailscale / VPN のみ。

## 現状

- サブPC (Ubuntu 24.04): systemd user service で web / discord / voice / tts / ollama を常時実行
- SQLite 4種: `data/tasks/tasks.db`, `data/growth/growth.db`,
  `data/metrics/system_metrics.db`, `data/assistant/model_runs.db`
- `AssistantService` が `RunLogger` Protocol 経由で実行ログを記録（`SQLiteRunLogger` が既定）
- `GET /api/health` に modules + providers（provider_id/model/local/available/last_success_at）
- `scripts/backup.sh` / `scripts/restore.sh`（sha256 manifest + 世代管理）
- CI: `.github/workflows/windows-desktop.yml`（Windows runner で EXE ビルド）のみ
- `compose.yaml` + `postgres:16`はI1で導入済み（実機確認待ち）。Dockerfile / Prometheus / Grafana / GitHub Actions test は未導入

## ゴール（完了時の姿）

```text
                SubPC Living (サブPC: Ubuntu)

        ┌──────── Docker Compose ─────────┐
        │  postgres:16   prometheus       │
        │  grafana       node_exporter    │
        │  nvidia_gpu_exporter            │
        │  (後段: api / discord / voice)  │
        └────────────┬───────────────────┘
                     │
        systemd (ホスト) ─ Ollama (P40) / SBV2 TTS
                     │
        GitHub Actions: test → docker build → (承認) deploy
```

- 状態は PostgreSQL に集約され、SQLite はローカル専用の高速・単一用途へ縮小
- 監視: Prometheus + Grafana（ホスト/GPU/アプリの3層）、Discord webhook でアラート
- デプロイ: main merge → test → image build → 手動承認 → サブPC pull で更新、旧タグへ即 rollback

## 段階 (Phases)

各フェーズは単独で完了・検証・rollback 可能。変更パスは最小に保つ。

### I1: Docker 基盤と PostgreSQL（リポジトリ実装完了・実機確認待ち）

- サブPCに Docker Engine + Compose plugin を導入（`docs` に手順、Ansible は使わない）
- `compose.yaml` + `.env.example`: `postgres:16`（healthcheck付き、`POSTGRES_*` は .env から）
- volume は `infra/pgdata` へ。LAN/lo のみ bind、外部公開しない
- `scripts/backup.sh` へ `pg_dump` を追加、`restore.sh` へ `pg_restore` を追加

受け入れ条件: 再起動後も postgres が自動起動し、`pg_dump` が定期バックアップへ載る

リポジトリ側の完了内容:

- `compose.yaml`: `postgres:16`、healthcheck、`restart: unless-stopped`、`infra/pgdata`永続化、loopback bind既定
- `.env.example`: 秘密値を含まない設定テンプレート（実`.env`はgitignore）
- `docs/docker_setup.md`: Ubuntu 24.04導入・起動・確認・更新・rollback手順
- `scripts/backup.sh` / `scripts/restore.sh`: custom形式`pg_dump`と明示opt-inの破壊的`pg_restore`
- fake dockerを使う分離テスト。実DB・実コンテナはテストから操作しない

残る完了条件はサブPC実機でのDocker導入、再起動後の自動復旧、実dump/restore訓練。

### I2: 実行ログを PostgreSQL へ（最初のストア移行）

- `src/assistant/run_logger.py` に `PostgresRunLogger` を追加（`SQLiteRunLogger` と同 Protocol）
- SQLAlchemy 2.0 + Alembic を導入（初回 migration で `model_runs` / `route_decisions` を作成）
- factory は `ASSISTANT_RUN_LOG_DSN` 環境変数があれば PG、無ければ SQLite（既定維持＝無設定で挙動不変）
- 受け入れ条件: DSN 指定時は PG に記録、未指定時は従来どおり SQLite。両方のテストが通る
- rollback: DSN を外すだけで SQLite へ戻る

### I3: 監視 (Prometheus / Grafana / アプリ計装)

- compose へ `prometheus` / `grafana` / `node_exporter`（両ノード）/ `nvidia_gpu_exporter` を追加
- アプリに `prometheus_client` を導入し `GET /metrics` を公開:
  - リクエストレイテンシ histogram（チャネル別）
  - provider 成功/失敗/fallback カウンタ（`RunLogger` ではなく in-process 集計）
  - tokens/sec（`last_stats` の `eval_count` / `eval_duration` から導出）
- `infra/grafana/dashboards/*.json` をコミット、`infra/prometheus/alerts.yml` で
  「サービスdown」「ディスク90%超」「GPU VRAM 90%超」「provider 失敗率急増」を Discord webhook へ

受け入れ条件: Grafana で3層のメトリクスが見え、アラートが Discord に届く（テスト発火で確認）

### I4: タスクDBの PostgreSQL 移行

- `TaskStore` を Protocol 化し `PostgresTaskStore` を追加（`model_runs` と同様に DSN 切替）
- 会話履歴・RAG・metrics は当面 SQLite/ファイルのまま（用途が単一・書き込み軽量）
- backup/restore/runbook を更新

受け入れ条件: タスクが PG に移行しても既存機能（Tasks authority block 等）が不変

### I5: アプリのコンテナ化とデプロイ自動化

- `Dockerfile`（web/discord 共通イメージ、`requirements.txt` ベース）
- compose の profile で `api` / `discord` / `voice` を段階的に有効化（systemd と並走）
- GitHub Actions:
  - `test.yml`: ubuntu runner で Python 3.12 の全 unittest（Linux 基準環境に一致）
  - `docker.yml`: test 成功後に image build & push（GHCR、SHA tag）
  - `deploy.yml`: environment 承認 → SSH でサブPCへ pull & 再起動、前タグへ rollback 手順付き

受け入れ条件: push → test → build → (承認) → サブPC更新が一周し、失敗時に旧タグへ戻せる

### I6: CI/CD の堅牢化と運用定着

- main ブランチ保護（test 必須）、`git diff --check` / lint を test.yml へ追加
- 月1バックアップ復元訓練と、障害シナリオ（Ollama down / GPU down / DB down）の自動検出確認
- `docs/runbook.md` に infra 障害（postgres down / docker down / prometheus down）を追記

## 監視メトリクスの具体

| 層 | メトリクス | 取得元 |
|---|---|---|
| ホスト | CPU/RAM/disk | node_exporter |
| GPU | VRAM/温度/電力/util | nvidia_gpu_exporter |
| アプリ | レイテンシ・成功/失敗/fallback・tokens/sec | `GET /metrics`（prometheus_client） |
| 推論 | モデルロード状況・availability | `/api/health` providers（既存） |

## セキュリティ境界（不変）

- 実 `.env` / `config/discord.env` / API キーは Git に載せない（`.env.example` のみ追跡）
- Postgres は LAN/lo のみ bind、パスワードは .env から
- デプロイ SSH 鍵は GitHub secrets、sub-pc のデプロイ専用ユーザーに最小権限
- Prometheus/Grafana は Tailscale 経由で参照、インターネット公開しない

## 直近の次アクション

1. **I1実機確認**: サブPCへDockerを導入し、`docs/docker_setup.md`に従ってhealth・再起動・実`pg_dump`・検証・復元訓練を完了する
2. **I2**: `PostgresRunLogger` + Alembic migration + SQLite/PG切替テスト（リポジトリ内で完結）
3. **I3**: `/metrics` エンドポイント + 計装 + Grafana dashboard JSON + alerts

I1のリポジトリ実装は完了したが、受け入れ条件の最終判定にはサブPC実機（またはSSH）が必要。
実機確認と並行してI2の実装は開始できる。

## 関連文書

- `docs/homelab_operations_plan.md`: 既存の運用基盤（Inventory/Health/Backup/Runbook）
- `docs/assistant_platform_plan.md`: Provider/Router/Service 境界
- `docs/runbook.md`: 障害対応手順（infra 追記対象）
