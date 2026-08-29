# 運用Runbook (デプロイ前ドラフト)

> **状態**: draft / supporting
> **位置付け**: デプロイ前の運用Runbook草案。本番サブPCが存在しないため、記載コマンドは
> いずれも**実機検証されていない**。各コマンドに `★未検証` マークを付け、検証は
> 「10. デプロイ検証チェックリスト」で行う。
> **対象範囲**: サブPC常駐AI (`subpc_living`) の障害対応・停止再起動・更新rollback・
> 復元訓練の草案
> **作成日**: 2026-08-22
> **更新日**: 2026-08-28
> **日付根拠**: Git commit date / 作業日

## 0. 本Runbookの前提

- 本リポジトリは**実機未導入のサブPC**を想定した唯一の証跡である。以下は全て
  **追跡されているファイル (scripts/systemd/*.service, scripts/*.sh, src/web/server.py 等)
  から導出した仕様**に基づく草案。
- 記載のサービス名・パス・オプションはリポジトリ上の定義に一致させているが、
  **実機での動作確認が済むまでは「設計想定」として扱う**こと。
- パスは演算子が設定する環境変数 `PROJECT_ROOT` で扱う。各コマンド実行前に次を確認する:

  ```bash
  # ★未検証 (演算子が設定する値。実機導入時に確定)
  export PROJECT_ROOT="$HOME/subpc_living"   # 実機のパスへ置換して使用
  cd "$PROJECT_ROOT"
  ```

- 秘密情報 (`config/*.env`, 実 `.env`) は読まない。バックアップにも含まれない
  (`scripts/backup.sh` の除外定義による)。

### 関連文書

- `docs/archive/homelab_operations_plan.md` — 運用計画全体・バックアップ候補・更新とRollback案
- `docs/plans/companion_roadmap.md` — コンパニオン構想と設計原則
- `docs/archive/assistant_platform_plan.md` — Provider / Router / AssistantService 実装計画
- `scripts/backup.sh` — バックアップスクリプト (実装済み・実機未検証)
- `scripts/restore.sh` — 復元スクリプト (実装済み・実機未検証)
- `scripts/service_ctl.sh` — サービス管理ヘルパー (実装済み・実機未検証)

---

## 1. スコープとステータス

| 領域 | ステータス | 備考 |
|------|-----------|------|
| systemd unit 定義 | リポジトリに実装 | `scripts/systemd/` に定義。実機適用・検証は未実施 |
| バックアップ/復元 | 実装・**未検証** | `scripts/backup.sh` / `scripts/restore.sh` |
| Ollama セットアップ | 実装・**未検証** | `scripts/phase1_setup_ollama.sh` 等 |
| Web/Discord/Voice/TTS | 実装・**未検証** | ユニット定義とソースのみ |
| PostgreSQL (I1) | 実装・**未検証** | `compose.yaml` + backup/restore の postgres 対応 |
| 実機デプロイ | **未実施** | 本ドラフト時点でサブPCなし |

デプロイ前のため「稼働中」「正常応答」「復旧済み」等の実績は**一切報告できない**。
本Runbookに「復旧確認」があるのは将来の確認手順の草案であり、結果は出ていない。

## 2. 前提条件とインベントリ (プレースホルダ)

デプロイ時に演算子が確定させる項目。空欄は**導入時に埋める**:

| 項目 | 値 (プレースホルダ) | 根拠 |
|------|--------------------|------|
| PROJECT_ROOT | `$HOME/subpc_living` 相当 | 演算子設定 |
| サブPC OS | Ubuntu 24.04 相当 | 追跡スクリプトの前提 |
| ホスト名 / メインPC から見た IP | `サブPCのIP` (未確定) | 演算子設定 |
| Tailscale ノード名 | 未確定 | 演算子設定 |
| GPU 構成 | 未確定（演算子が実機導入時に確定し記入） | 追跡unitのコメント・overrideは GPU0=P40系 / GPU1=Quadro P5000系 を**例示**するが、目標・現行構成の選択ではない |
| Ollama モデル名 | 未確定 (導入時 `ollama list` で確認) | 演算子設定 |
| config のチャットモデル | `config/chat_config.json` の `model` 値 | リポジトリ定義 |
| ポート | Web=8000 / Ollama=11434 / SBV2=50121 | `src/web/server.py`, `subpc-sbv2-tts.service` |

> モデル名・IP・ホスト名・GPU構成は本ドラフトで断定しない（GPU構成は追跡unitがP40系 / P5000系を例示するのみ）。検証手順は各セクションに記載。

## 3. リポジトリ定義のサービス構成

`scripts/service_ctl.sh` と `scripts/systemd/*` から導出した構成。

### 3.1 ユーザーサービス (`systemctl --user`)

| ユニット | 内容 | 依存 (After/Wants) |
|----------|------|--------------------|
| `subpc-web.service` | Web UI サーバー (`src/web/server.py`, port 8000) | `ollama.service` |
| `subpc-discord.service` | Discord bot (`src.discord_bot.bot`) | `ollama.service`, `subpc-sbv2-tts.service` |
| `subpc-voice.service` | 音声対話パイプライン (`src/audio/main.py`) | `ollama.service` |
| `subpc-sbv2-tts.service` | Style-Bert-VITS2 TTS サーバー (port 50121) | `network.target` |

依存順序 (unit 定義に基づく):

```text
ollama (system) → subpc-sbv2-tts → subpc-web
                → subpc-voice
subpc-sbv2-tts → subpc-discord
```

### 3.2 システムサービス (`sudo systemctl`)

| ユニット | 内容 | 備考 |
|----------|------|------|
| `ollama.service` | Ollama API (port 11434) | **システムサービス** (`systemctl is-active ollama`)。`--user` を付けない |
| `subpc-gpu-powersave.service` | 起動時 GPU 電力制御 (oneshot) | `nvidia-smi --persistence-mode=1` / power-limit（追跡unitの例示値: GPU0=P40系125W・GPU1=P5000系100W。実機で要調整） |
| `subpc-gpu-powerd@<user>.service` | GPU 動的電力制御デーモン (root) | UNIX socket 経由 |
| `tailscaled.service` / `tailscaled-restart.timer` | Tailscale 3時間おき再起動 | timer はユーザー/システム設定を実機で確認 |

> **注意**: Ollama を `systemctl --user` で操作しない。Ollama はシステムサービス、
> `subpc-*` はユーザーサービスという混在に注意する (`scripts/phase1_setup_ollama.sh`,
> `scripts/service_ctl.sh` より)。

### 3.3 状態確認の基本コマンド

```bash
# ★未検証
bash scripts/service_ctl.sh status     # 全サービスの状態一覧 (推奨)
systemctl --user status subpc-web.service subpc-discord.service subpc-voice.service subpc-sbv2-tts.service
systemctl status ollama                 # システムサービスとして確認 (--user を付けない)
```

## 4. 安全な定期チェック

> すべてのサービスが実機で稼働している**前提の草案**。デプロイ後は以下の頻度・内容を
> 運用として回す。各コマンドは ★未検証。

```bash
# 1. ヘルスチェック (Ollama/ディスク/メモリ)
# ★未検証
"$PROJECT_ROOT/.venv/bin/python" -m src.service.healthcheck

# 2. サービス状態
# ★未検証
bash scripts/service_ctl.sh status

# 3. Web API 疎通 (デフォルトポート 8000)
# ★未検証
curl -s http://localhost:8000/api/health | head -c 200

# 4. Ollama API 疎通
# ★未検証
curl -s http://localhost:11434/api/tags | head -c 200

# 5. GPU 状態
# ★未検証
nvidia-smi --query-gpu=name,memory.used,memory.free --format=csv,noheader
ollama ps

# 6. ディスク
# ★未検証
df -h /
```

- 異常時は「5. サービス別の診断と対処」へ。
- 「ログ」「健康状態」は `scripts/service_ctl.sh logs <service>` でも確認できる (★未検証)。

## 5. サービス別の診断と対処

### 5.1 Ollama が応答しない (システムサービス)

切り分け:

```bash
# ★未検証
systemctl status ollama                  # システムサービスとして確認
curl -s http://localhost:11434/api/tags | head -c 200
journalctl -u ollama -n 50 --no-pager    # システムログ
nvidia-smi --query-gpu=name,memory.used,memory.free --format=csv,noheader
```

対処:

1. 停止/失敗中 → 再起動 (★未検証):
   ```bash
   sudo systemctl restart ollama
   sleep 5
   curl -s http://localhost:11434/api/tags
   ```
2. GPU 認識エラー → ドライバ再初期化。**演算子の確認の上**で実施 (★未検証):
   ```bash
   sudo systemctl restart nvidia-persistenced
   # 必要な場合のみ、以下を演算子確認後に順次実行
   sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia
   sudo modprobe nvidia
   sudo systemctl restart ollama
   ```
3. ポート競合 (11434) → 使用プロセスを特定してから対処 (★未検証):
   ```bash
   ss -ltnp | grep 11434
   ```
4. モデルキャッシュが疑わしい場合 → **直接削除せず退避** (★未検証):
   ```bash
   TS=$(date +%Y%m%d-%H%M%S)
   mv "$HOME/.ollama/models/.cache" "$HOME/.ollama/models/.cache.bak-$TS"
   sudo systemctl restart ollama
   # 動作確認後に退避先を削除するか判断 (演算子確認)
   ```

復旧確認 (★未検証): `curl -s http://localhost:11434/api/tags` でモデル一覧が返ること。

### 5.2 モデルが見つからない / 切り替え

`scripts/switch_chat_model.py` はサブコマンド形式。**位置引数でモデルを渡す旧方式は使わない**。

```bash
# ★未検証
"$PROJECT_ROOT/.venv/bin/python" scripts/switch_chat_model.py show          # 現在のモデル/override/backup有無
"$PROJECT_ROOT/.venv/bin/python" scripts/switch_chat_model.py switch <model> # 切り替え (旧値は .bak に退避)
"$PROJECT_ROOT/.venv/bin/python" scripts/switch_chat_model.py rollback      # 直前へ復元
```

- モデル未導入 → `ollama pull <model_name>` (★未検証)。実機の `ollama list` と
  `config/chat_config.json` の `model` 値を突き合わせる。
- 導入時にモデル名は `ollama list` で確定させ、本Runbookに断定した記述をしない。

### 5.3 Web / Discord が起動しない

切り分け (★未検証):

```bash
systemctl --user status subpc-web.service subpc-discord.service
journalctl --user -u subpc-web.service -n 80 --no-pager
journalctl --user -u subpc-discord.service -n 80 --no-pager
ss -ltnp | grep -E ':8000|:50121'
```

対処:

1. 再起動 (★未検証):
   ```bash
   systemctl --user restart subpc-web.service subpc-discord.service
   ```
2. 依存サービス確認。Ollama が先に起動している必要がある (★未検証):
   ```bash
   systemctl status ollama
   # 停止中なら先に起動
   sudo systemctl start ollama
   sleep 10
   systemctl --user restart subpc-web.service subpc-discord.service
   ```
3. Discord は `subpc-sbv2-tts.service` に依存 (`Wants`)。TTS 停止中なら先に起動 (★未検証):
   ```bash
   systemctl --user status subpc-sbv2-tts.service
   systemctl --user start subpc-sbv2-tts.service
   systemctl --user restart subpc-discord.service
   ```
4. 依存パッケージ問題 → 実機導入時に確認。`requirements.txt` は追跡済みだが
   再インストールは演算子確認の上で実施 (★未検証):
   ```bash
   "$PROJECT_ROOT/.venv/bin/pip" install -r "$PROJECT_ROOT/requirements.txt"
   systemctl --user restart subpc-web.service subpc-discord.service
   ```

復旧確認 (★未検証): `curl -s http://localhost:8000/api/health`

### 5.4 音声パイプライン / TTS が動かない

- `subpc-voice.service` は独立 (Ollama 依存のみ)。TTS サーバー (`subpc-sbv2-tts.service`)
  は `subpc-discord.service` から参照される。
- 切り分け (★未検証):
  ```bash
  systemctl --user status subpc-voice.service subpc-sbv2-tts.service
  journalctl --user -u subpc-sbv2-tts.service -n 50 --no-pager
  curl -s http://127.0.0.1:50121/health   # SBV2 のヘルスエンドポイントは実機で確認
  ```
- 再起動 (★未検証):
  ```bash
  systemctl --user restart subpc-sbv2-tts.service
  systemctl --user restart subpc-voice.service
  ```
- **TTS を停止したままにしない**。Discord が依存するため、起動順は
  `subpc-sbv2-tts` を `subpc-discord` より先に行う。

### 5.5 GPU / VRAM 不足

切り分け (★未検証):

```bash
nvidia-smi
ollama ps
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader
```

対処:

1. モデルをアンロード (★未検証):
   ```bash
   curl http://localhost:11434/api/generate -d '{"model": "<model>", "keep_alive": 0}'
   ```
2. `ollama-gpu-p40.override.conf` に `OLLAMA_MAX_LOADED_MODELS=1`,
   `OLLAMA_LOAD_TIMEOUT=10m` が定義済み（`p40` は追跡overrideの**例示プロファイル名**であり、
   実機GPUの確定ではない）。調整する場合は実機の VRAM 実測と演算子確認を経てから (★未検証):
   ```bash
   sudo systemctl edit ollama   # 追跡 override の内容を実機で適用してから判断
   ```
3. モデル切り替えで対応: `scripts/switch_chat_model.py switch <model>` (★未検証)。
   VRAM 上限の数値 (例: 「16GB/24GB」はP40構成を例示した場合の値) は本ドラフトでは断定しない。実機の
   `nvidia-smi` 実測値で判定する。

復旧確認 (★未検証): `nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader` と
`ollama ps` で余裕があること。

### 5.6 ディスク不足

切り分け (★未検証):

```bash
df -h /
du -sh data/ config/ logs/ "$HOME/.ollama/" 2>/dev/null | sort -rh | head -10
ls -lh data/tasks/tasks.db data/growth/growth.db data/metrics/system_metrics.db 2>/dev/null
```

対処:

1. journal の圧縮 (★未検証):
   ```bash
   journalctl --user --vacuum-size=100M
   journalctl --user --vacuum-time=7d
   ```
2. バックアップの保持数は `backup.sh --keep-daily N` で管理する。**`xargs rm -rf` の
   手動削除をしない**。手動整理が必要なら、まず退避ディレクトリへ移動して演算子確認後
   に削除する (★未検証):
   ```bash
   mv "$PROJECT_ROOT/backups/<対象TS>" "$PROJECT_ROOT/backups/.trash-$(date +%Y%m%d-%H%M%S)"
   ```
3. DB サイズ確認 (★未検証):
   ```bash
   sqlite3 data/tasks/tasks.db "SELECT COUNT(*) FROM tasks;"
   sqlite3 data/growth/growth.db "SELECT COUNT(*) FROM growth_entries;"
   ```
   (sqlite3 が未導入の実機は導入してから。リポジトリのDBスキーマ名はソースで確認)
4. Ollama キャッシュ退避 → 「5.1」の手順 (直接 `rm -rf` しない)。

復旧確認 (★未検証): `df -h /` で空きが確保できたこと。

### 5.7 DB 破損

切り分け (★未検証):

```bash
sqlite3 data/tasks/tasks.db "PRAGMA integrity_check;"
sqlite3 data/growth/growth.db "PRAGMA integrity_check;"
sqlite3 data/metrics/system_metrics.db "PRAGMA integrity_check;"
```

対処 (バックアップ優先):

1. **バックアップからの復元を最優先**。手順は「8. バックアップ・復元訓練」に従う。
   直接稼働中の `data/` へ展開しない。**検証 → サービス停止 → 演算子確認 → 入れ替え**
   の順を守る。
2. 汎用 `.recover` (各DBを一時DBへ抽出して確認後に置換。演算子確認を挟む) (★未検証):
   ```bash
   sqlite3 data/tasks/tasks.db ".recover" | sqlite3 /tmp/tasks_recovered.db
   # integrity_check と件数を確認してから、サービス停止後に置換
   mv data/tasks/tasks.db data/tasks/tasks.db.broken-$(date +%Y%m%d-%H%M%S)
   mv /tmp/tasks_recovered.db data/tasks/tasks.db
   ```
3. PostgreSQL (I1) の復元は `--restore-postgres` (**破壊的**) を演算子確認後にのみ実行
   (★未検証):
   ```bash
   systemctl --user stop subpc-web.service subpc-discord.service subpc-voice.service
   scripts/restore.sh <backup_dir> --target /tmp/subpc-restore --restore-postgres
   docker compose exec postgres sh -c 'pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB"'
   ```
   復元後は忘れず TTS を含む全ユーザーサービスを起動し直す。

復旧確認 (★未検証): `PRAGMA integrity_check` が `ok` になること。

---

## 6. メインPC (Windows開発機) とサブPCの関係

- **Windows 開発機には systemctl はない**。本Runbookの `systemctl` 系コマンドは
  すべてサブPC (Linux) 上での実行である。
- メインPC (Windows) のデスクトップクライアントは Web API
  (`http://<サブPCのIP>:8000` 想定、`src/desktop/api.py` のデフォルトは
  `http://127.0.0.1:8000`) へ接続する。接続先は実機導入時に `SUBPC_DESKTOP_SERVER_URL`
  等で確定させる (★未検証)。
- 疎通確認 (サブPC側) (★未検証):
  ```bash
  systemctl --user status subpc-web.service
  curl -s http://localhost:8000/api/health
  ```
- Tailscale 経路確認はサブPC上で `tailscale status` (★未検証)。ノード名・IPは実機で確認。
- メインPC側に Ollama を置く構成は**追跡されていない**。`local-fast`/`local-strong` の
  フォールバック挙動はソース上の設計であり、実機での動作確認が必須 (★未検証)。

---

## 7. 更新と Rollback

> 原則: **バックアップ先行 → 検証 → 明示停止 → 演算子確認 → 適用**。稼働中の
> checkout へ直接 `git pull` しない。破壊的操作はすべて退避と演算子確認を挟む。

### 7.1 更新前チェック

```bash
# ★未検証
git -C "$PROJECT_ROOT" status --short
git -C "$PROJECT_ROOT" log --oneline -5
df -h /
```

- 未コミットの変更があれば、それは今回の更新対象外であることを確認する。

### 7.2 バックアップ先行

```bash
# ★未検証
bash "$PROJECT_ROOT/scripts/backup.sh" --dry-run            # 計画の確認
bash "$PROJECT_ROOT/scripts/backup.sh"                       # 本実行 (既定: $PROJECT_ROOT/backups)
```

- PostgreSQL を含む場合: `POSTGRES_BACKUP_MODE=required` で停止漏れを検出できる
  (★未検証):
  ```bash
  POSTGRES_BACKUP_MODE=required bash "$PROJECT_ROOT/scripts/backup.sh"
  ```
- 新しいバックアップのタイムスタンプを控える:
  ```bash
  # ★未検証
  ls -dt "$PROJECT_ROOT/backups/"*/ | head -1
  ```

### 7.3 検証 (staging)

```bash
# ★未検証
git -C "$PROJECT_ROOT" fetch origin
git -C "$PROJECT_ROOT" log --oneline HEAD..origin/<branch>
# 差分を確認してから演算子が適用を判断
git -C "$PROJECT_ROOT" diff --stat HEAD origin/<branch>
```

- 可能なら一時クローンでテストを回す (★未検証):
  ```bash
  git clone --depth 1 --branch <target> "$(git -C "$PROJECT_ROOT" remote get-url origin)" /tmp/subpc-staging
  cd /tmp/subpc-staging
  python -m unittest discover -s tests -q
  ```

### 7.4 明示停止 → 適用 → 起動

停止 (依存の逆順。**停止後も下記起動手順で TTS を復帰させる**):

```bash
# ★未検証
systemctl --user stop subpc-discord.service      # sbv2-tts に依存
systemctl --user stop subpc-web.service
systemctl --user stop subpc-voice.service
systemctl --user stop subpc-sbv2-tts.service
# Ollama 更新も行う場合のみ、システムサービスとして停止
# sudo systemctl stop ollama
```

適用 (演算子確認後):

```bash
# ★未検証
git -C "$PROJECT_ROOT" merge --ff-only origin/<branch>   # または pull --ff-only
"$PROJECT_ROOT/.venv/bin/pip" install -r "$PROJECT_ROOT/requirements.txt"
```

起動 (依存順。**TTS を停止したままにしない**):

```bash
# ★未検証
sudo systemctl start ollama            # 停止していた場合のみ (システムサービス)
sleep 10
systemctl --user start subpc-sbv2-tts.service
systemctl --user start subpc-web.service
systemctl --user start subpc-voice.service
systemctl --user start subpc-discord.service
```

### 7.5 Rollback

1. **バックアップからの復元**が最優先。手順は「8」参照。稼働中へ直接展開しない。
2. Git ベースの巻き戻しは演算子確認の上で (★未検証):
   ```bash
   git -C "$PROJECT_ROOT" log --oneline -5
   git -C "$PROJECT_ROOT" revert <bad_commit>       # または reset は演算子判断
   systemctl --user restart subpc-sbv2-tts.service subpc-web.service subpc-voice.service subpc-discord.service
   ```

---

## 8. バックアップ・復元訓練

> 目的: 「作成できる」だけでなく「復元できる」ことを確認する。
> `backup.sh` / `restore.sh` のオプションは追跡スクリプトの実装に一致させる。

- `backup.sh`: `[--dry-run] [--target-dir DIR] [--keep-daily N]` / env `POSTGRES_BACKUP_MODE`
- `restore.sh`: `<backup_timestamp_dir> --target <restore_root> [--verify-only] [--restore-postgres]`

### 8.1 月1回の復元訓練 (草案)

1. バックアップ実行 (★未検証):
   ```bash
   TS=$(date +%Y%m%d-%H%M%S)
   bash "$PROJECT_ROOT/scripts/backup.sh" --target-dir /tmp/restore-test-backup
   ls -dt /tmp/restore-test-backup/*/ | head -1
   ```
2. 検証のみ (★未検証):
   ```bash
   TIMESTAMP=$(ls -dt /tmp/restore-test-backup/*/ | head -1 | xargs basename)
   bash "$PROJECT_ROOT/scripts/restore.sh" "$TIMESTAMP" \
     --target /tmp/restore-test-target --verify-only
   ```
3. 一時ディレクトリへ展開 (★未検証):
   ```bash
   bash "$PROJECT_ROOT/scripts/restore.sh" "$TIMESTAMP" --target /tmp/restore-test-target
   ```
4. 整合性チェック (★未検証):
   ```bash
   ls -la /tmp/restore-test-target/data/tasks/tasks.db
   ls -la /tmp/restore-test-target/data/vectordb/ 2>/dev/null
   sqlite3 /tmp/restore-test-target/data/tasks/tasks.db "PRAGMA integrity_check;"
   sqlite3 /tmp/restore-test-target/data/growth/growth.db "PRAGMA integrity_check;"
   ```
   - `restore.sh` の出力に「IMPORTANT: Real .env files are NOT in backups」と出る。
     `.env` / `config/*.env` は**別途手動復元が必要**な点を確認する (本ドラフト時点では未検証)。
5. 結果記録 (訓練実施後に埋める):

   | 項目 | 結果 | 日付 | 備考 |
   |------|------|------|------|
   | backup.sh 実行 | 未実施 | | |
   | restore.sh --verify-only | 未実施 | | |
   | restore.sh 実展開 | 未実施 | | |
   | tasks.db PRAGMA | 未実施 | | |
   | growth.db PRAGMA | 未実施 | | |
   | vectordb ファイル数 | 未実施 | | |
6. クリーンアップ (★未検証):
   ```bash
   rm -rf /tmp/restore-test-backup /tmp/restore-test-target
   ```

### 8.2 実機へのデータ復元 (事故対応)

直接の稼働中 checkout へ展開しない。**検証 → 明示停止 → 演算子確認 → 入れ替え**:

1. `--verify-only` で sha256 検証 (★未検証):
   ```bash
   bash "$PROJECT_ROOT/scripts/restore.sh" <TS> --target /tmp/subpc-restore --verify-only
   ```
2. 一時ディレクトリへ展開して内容確認 (★未検証):
   ```bash
   bash "$PROJECT_ROOT/scripts/restore.sh" <TS> --target /tmp/subpc-restore
   sqlite3 /tmp/subpc-restore/data/tasks/tasks.db "PRAGMA integrity_check;"
   ```
3. サービスの明示停止 (★未検証):
   ```bash
   systemctl --user stop subpc-discord.service subpc-web.service subpc-voice.service subpc-sbv2-tts.service
   ```
4. 演算子確認の上で入れ替え (★未検証)。元データは削除せず退避:
   ```bash
   mv "$PROJECT_ROOT/data/tasks" "$PROJECT_ROOT/data/tasks.broken-$(date +%Y%m%d-%H%M%S)"
   cp -a /tmp/subpc-restore/data/tasks "$PROJECT_ROOT/data/tasks"
   ```
5. 起動 (依存順。TTS を忘れない):
   ```bash
   systemctl --user start subpc-sbv2-tts.service subpc-web.service subpc-voice.service subpc-discord.service
   ```

---

## 9. 秘密情報誤出力への対応

### 症状

ログ・チャット出力・Git履歴に APIキーや認証情報が含まれた。

### 対処

1. **即座にキーをローテーション** (演算子が各サービスで実施。本ドラフトは手順の雛形):
   - Discord Bot: https://discord.com/developers/applications → Bot → Reset
   - OpenAI等: 各プロバイダの API キー管理から Revoke → Create
   - Google: Google Cloud Console の Credentials から再発行
2. **Git履歴からの除去** (`git-filter-repo`、履歴書き換えは演算子確認の上で) (★未検証):
   ```bash
   pip install git-filter-repo
   git filter-repo --path config/discord.env --invert-paths
   git filter-repo --path config/.env --invert-paths
   ```
3. **強制プッシュ** (履歴書き換え後。全クローン再取得が必要になる点を演算子と確認) (★未検証):
   ```bash
   git push origin --force --all
   ```
4. **再発防止**:
   - `config/*.env` は `.gitignore` 対象かつ `backup.sh` の除外対象 (追跡定義を確認)。
   - ログ・チャット保存時に秘密情報を混入させない設計を維持する
     (`src/chat/session.py` 等の保存処理を実機で確認)。

復旧確認 (★未検証):

```bash
grep -r "sk-" logs/ 2>/dev/null
grep -r "token" logs/ 2>/dev/null | grep -v "REDACTED"
```

---

## 10. デプロイ検証チェックリスト

デプロイ後に本ドラフトを「active」へ昇格するまで、以下の項目を演算子と共に実行し、
★未検証を消していく。**すべての項目は本ドラフト時点で未実施**。

- [ ] `scripts/phase1_setup_nvidia.sh` / `phase1_setup_ollama.sh` の実行と Ollama が
      システムサービスとして起動することの確認 (`systemctl status ollama`)
- [ ] 追跡 systemd unit の実機適用、`systemctl --user daemon-reload` と
      `systemctl --user status` で全ユーザーサービス active を確認
- [ ] Web: `curl -s http://localhost:8000/api/health` が 200 を返す (port 8000)
- [ ] Ollama: `curl -s http://localhost:11434/api/tags` がモデル一覧を返す。
      導入モデル名を記録し本Runbookへ反映
- [ ] `scripts/switch_chat_model.py show / switch / rollback` の動作確認
- [ ] TTS: `subpc-sbv2-tts.service` と Discord の依存起動順の実機確認
- [ ] GPU: `nvidia-smi` / `ollama ps` で VRAM 使用量を実測し閾値を本Runbookへ反映
- [ ] ディスク: `df -h /` とログ・DB の実サイズ確認
- [ ] バックアップ: `backup.sh --dry-run` → 本実行 → `restore.sh --verify-only` →
      一時展開 → PRAGMA 確認 (8.1 の項目を消化)
- [ ] PostgreSQL (I1) を利用する場合: `POSTGRES_BACKUP_MODE=required` と
      `--restore-postgres` のリハーサル
- [ ] 更新/Rollback: 7.4 / 7.5 の手順を空きタイムで通し演習
- [ ] メインPC (Windows) から `http://<サブPCのIP>:8000/api/health` へ疎通確認
- [ ] Tailscale 経路の確認とノード名・IPの記録
- [ ] 全コマンドの ★未検証 マークを順次除去し、本ドラフトを active へ昇格

---

## 更新履歴

| 日付 | 変更 |
|------|------|
| 2026-08-22 | 初版作成 |
| 2026-08-27 | 旧版の運用メモ更新 (archived) |
| 2026-08-28 | デプロイ前ドラフトとして全面改訂。未検証コマンドへ ★未検証 付与。 |