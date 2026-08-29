# Docker Engine + PostgreSQL (I1) セットアップ

> **状態**: draft / supporting（pre-deployment・未検証）
> **位置付け**: Ubuntu 24.04 (I1) セットアップのドラフト手順（未検証）
> **対象範囲**: サブPCでの Docker Engine / Compose plugin / postgres:16 の導入・起動・確認・更新・rollback
> **作成日**: 2026-08-27
> **更新日**: 2026-08-28
> **日付根拠**: Git commit date

> **未検証のドラフト手順**: 本手順はデプロイ前の草案であり、実機での成功は確認されていない。
> 各コマンドの成功出力・稼働状態を「完了」と主張しないこと。実行は要件を満たす実機上で行い、
> 第5節の静的チェックを事前に、実機チェックを実行後に必ず行うこと。

対象: サブPC (Ubuntu 24.04)。Docker 導入はサブPC上で行う（この文書は手順書であり、
リポジトリ側では Docker を実行しない）。Compose で `postgres:16` を常駐させる。

> **絶対に `.env` をコミットしないこと。** `.env` には実パスワードが入る。
> git 管理されるのは `.env.example`（プレースホルダのみ）だけ。`.env` は
> `.gitignore` 済みだが、`git add -f` しない・コピーして共有しない。

## 0. 前提条件（事前確認）

- Ubuntu 24.04 のサブPC実機、`sudo` 権限、ログアウト/再ログインが可能なユーザー
- `scripts/systemd/` の unit と Ollama が動いている環境を想定（本手順は postgres 導入に限定）
- リポジトリの `compose.yaml` / `.env.example` が最新であること（`git pull`）
- 秘密情報: 実 `.env` は git 管理外。`config/*.env` は本ドキュメントの対象外
- **本手順は未検証**。実機で検証できた項目は第5節へ反映し、検証前は「成功した手順」として扱わない

## 1. Docker Engine + Compose plugin のインストール (Ubuntu 24.04)

公式リポジトリ経由での導入（Ansible は使わない）:

```bash
# 前提パッケージ
sudo apt-get update
sudo apt-get install -y ca-certificates curl

# Docker 公式 GPG 鍵とリポジトリ
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
  https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# 本体
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io \
  docker-buildx-plugin docker-compose-plugin

# 自起動の有効化とユーザー権限
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"   # 一度ログアウト/再ログインで反映
docker compose version             # 動作確認
```

## 2. 初回起動（安全な形で）

```bash
cd ~/path/to/subpc_living

# 実 .env を作成（コミットしない）
cp .env.example .env
chmod 600 .env
$EDITOR .env   # POSTGRES_PASSWORD=change-me... を必ず長い乱数へ変更

mkdir -p infra/pgdata   # ホストバインド用データディレクトリ
docker compose up -d
```

- `${POSTGRES_PASSWORD:?}` は未設定・空文字を拒否する。`.env.example` の `change-me` 値そのものは検出できないため、初回起動前に必ず手動で変更する
- 公開ポートは既定で `127.0.0.1:5432` のみ（LAN/インターネットには出ない）。
  変更する場合も `.env` の `POSTGRES_BIND_ADDR` を Tailscale IP など限定的な
  アドレスに限定すること。`0.0.0.0` にはしない
- データは `./infra/pgdata` に永続化され、`restart: unless-stopped` により
  再起動後も自動起動する

## 3. 起動確認（health）

```bash
docker compose ps                       # STATUS が healthy になるまで待つ
docker compose exec postgres sh -c 'pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB"'
ss -ltnp | grep 5432                    # 127.0.0.1 にのみ bind されていることを目視確認
```

`pg_isready` が `accepting connections` を返せば healthcheck も green になる。
`GET /api/health`（web 側）は I2 以降で PG を見るため、現時点では compose の
healthcheck が一次情報。

## 4. 停止・更新・ロールバック

```bash
docker compose stop        # 停止（データは残る）
docker compose start       # 再開
docker compose down        # 停止＋コンテナ削除（infra/pgdata は残る）

# 更新: イメージの新しいパッチを当てる
docker compose pull
docker compose up -d

# ロールバック: 特定バージョンへ戻す
#   compose.yaml の image: postgres:16 を一時的に postgres:16.x にピン留めして
#   docker compose up -d（メジャーバージョンは上げ下げしない。PGのデータ互換性のため）
```

バックアップは `scripts/backup.sh` が起動中の compose `postgres` を検出し、custom形式の
`postgres.dump` をmanifestへ含める。通常運用では `POSTGRES_BACKUP_MODE=required` を指定し、
DB停止やdump失敗をバックアップ成功として扱わない。

```bash
POSTGRES_BACKUP_MODE=required scripts/backup.sh --target-dir /path/to/backup
scripts/restore.sh <backup_dir> --target /tmp/subpc-restore --verify-only
# 破壊的なDB復元。アプリを止め、対象を確認してから明示実行する
scripts/restore.sh <backup_dir> --target /tmp/subpc-restore --restore-postgres
```

`POSTGRES_BACKUP_MODE=auto`（既定）はPostgres未導入環境との後方互換用で、サービス停止時は
dumpをスキップする。`off` はDBを対象外にする。**postgres 実行中に `infra/pgdata` を直接
コピーしない**（整合性が壊れる）。物理コピーする場合は必ずDBを停止する。

## 5. 検証チェックリスト

### 5a. 静的チェック（リポジトリ内で確認可能・未検証の判定材料）

- [ ] `compose.yaml` が `postgres:16`、healthcheck、`restart: unless-stopped`、loopback bind 既定を持つ
- [ ] `.env.example` に `POSTGRES_PASSWORD` がプレースホルダ（`change-me`）で存在する
- [ ] `.env` / `config/*.env` / `infra/pgdata/` が `.gitignore` されており `git status` に出ない
- [ ] `scripts/backup.sh` / `scripts/restore.sh` に `POSTGRES_BACKUP_MODE` と `--restore-postgres` の実装がある

### 5b. 実機チェック（デプロイ後にのみ検証可能。現時点では未検証）

- [ ] `.env` が存在し、パスワードが change-me から変わっている（かつ未コミット）
- [ ] `docker compose ps` が healthy
- [ ] `ss -ltnp` で 5432 が 127.0.0.1 のみに bind
- [ ] OS 再起動後も postgres が自動起動する
- [ ] 実 `pg_dump` / `--verify-only` / 復元訓練が成功する

> 5b は未実施。実機で成功が確認されるまで、本手順を「成功した手順」として扱わない。
