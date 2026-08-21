# 運用Runbook

## 本Runbookの前提

本Runbookは、サブPC (Ubuntu 24.04) 常駐のパーソナルAIアシスタント (`subpc_living`) の
障害対応手順を網羅する。全サービスは systemd user service で管理する。

### ネットワーク構成

```text
メインPC (Windows) ──LAN── サブPC (Ubuntu)
   │                           │
   │  local-fast               │  local-strong (Ollama)
   │  (Ollama, optional)       │  Web / Discord / Voice / RAG
   │                           │  SQLite DBs / ChromaDB
   └───────────────────────────┘
```

- サブPC: 常時稼働、Ollama / Web / Discord / 音声 / RAG / 監視を実行
- メインPC: デスクトップクライアント、サブPCのWeb APIへ接続
- バックアップ: `scripts/backup.sh` (本Runbookの事前に作成済み)

### 関連文書

- `docs/homelab_operations_plan.md` — 運用計画全体、バックアップ候補一覧、更新とRollback
- `docs/companion_roadmap.md` — コンパニオン構想と設計原則
- `docs/assistant_platform_plan.md` — Provider / Router / AssistantService の実装計画
- `scripts/backup.sh` — バックアップスクリプト
- `scripts/restore.sh` — 復元スクリプト

---

## 1. Ollamaが応答しない

### 症状

Web UIやDiscord botから「応答がありません」「タイムアウト」となる。
チャット画面にレスポンスが表示されない。

### 切り分け

```bash
# Ollamaプロセスの状態確認
systemctl --user status ollama.service

# Ollama APIの疎通確認
curl -s http://localhost:11434/api/tags | head -c 200

# ログ確認（直近50行）
journalctl --user -u ollama.service -n 50 --no-pager

# GPUが認識されているか
nvidia-smi --query-gpu=name,memory.used,memory.free --format=csv,noheader
```

### 対処

1. **サービス停止中** → 再起動:
   ```bash
   systemctl --user restart ollama.service
   sleep 5
   curl -s http://localhost:11434/api/tags
   ```

2. **GPU認識エラー** → NVIDIAドライバ再初期化:
   ```bash
   sudo systemctl restart nvidia-persistenced
   # または
   sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia
   sudo modprobe nvidia
   systemctl --user restart ollama.service
   ```

3. **ポート競合** → 別プロセスが11434を使用している:
   ```bash
   ss -ltnp | grep 11434
   # 該当プロセスを停止してから再起動
   ```

4. **Ollamaアップデート後** → キャッシュクリア:
   ```bash
   rm -rf ~/.ollama/models/.cache
   systemctl --user restart ollama.service
   ```

### 復旧確認

```bash
curl -s http://localhost:11434/api/tags | python3 -m json.tool
# モデル一覧が表示されれば正常
```

---

## 2. モデルが見つからない

### 症状

チャットで「モデルが見つかりません」「model not found」と表示される。
Web UIのモデル選択肢に不足がある。

### 切り分け

```bash
# Ollamaにロード中のモデル一覧
ollama list

# configのモデル名と突き合わせ
cat config/chat_config.json | python3 -m json.tool | grep model

# スイッチスクリプトの存在確認
ls scripts/switch_chat_model.py
```

### 対処

1. **モデル未ダウンロード**:
   ```bash
   ollama pull <model_name>
   ```

2. **configのモデル名が間違っている** → `config/chat_config.json` のモデル名を修正
   または `scripts/switch_chat_model.py` でモデルを切り替え:
   ```bash
   .venv/bin/python scripts/switch_chat_model.py <new_model>
   ```

3. **モデルファイル破損** → 再取得:
   ```bash
   ollama rm <model_name>
   ollama pull <model_name>
   ```

### 復旧確認

```bash
curl -s http://localhost:11434/api/tags | python3 -c "
import json,sys
data=json.load(sys.stdin)
for m in data.get('models',[]):
    print(m['name'], m.get('size',''))
"
```

---

## 3. Web / Discordが起動しない

### 症状

Web UIにアクセスできない。Discord botが応答しない。

### 切り分け

```bash
# サービス状態
systemctl --user status subpc-web.service
systemctl --user status subpc-discord.service

# ログ
journalctl --user -u subpc-web.service -n 80 --no-pager
journalctl --user -u subpc-discord.service -n 80 --no-pager

# ポート競合
ss -ltnp | grep -E ':8080|:3000|:8000'
```

### 対処

1. **サービス停止** → 再起動:
   ```bash
   systemctl --user restart subpc-web.service
   systemctl --user restart subpc-discord.service
   ```

2. **依存サービス** → Ollamaが先に起動している必要がある:
   ```bash
   systemctl --user status ollama.service
   # Ollamaが停止中なら先に起動
   systemctl --user start ollama.service
   sleep 10
   systemctl --user restart subpc-web.service subpc-discord.service
   ```

3. **依存順序（推奨）**:
   ```text
   Ollama (ollama.service) → Web (subpc-web.service) → Discord (subpc-discord.service)
   Voice (subpc-voice.service) は独立起動可
   ```

4. **Python venvの問題** → 依存パッケージ再インストール:
   ```bash
   .venv/bin/pip install -r requirements.txt
   systemctl --user restart subpc-web.service subpc-discord.service
   ```

### 復旧確認

```bash
curl -s http://localhost:8080/api/health 2>/dev/null | head -c 200
# Web UIが表示されれば正常
```

---

## 4. GPU VRAM不足

### 症状

Ollamaが「out of memory」エラーを出す。モデルのロードに失敗する。
他のサービスの動作が不安定になる。

### 切り分け

```bash
# GPU使用状況
nvidia-smi

# Ollamaがロードしているモデル
ollama ps

# VRAM使用量詳細
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader
```

### 対処

1. **不要モデルのアンロード**:
   ```bash
   # 特定モデルをアンロード
   curl http://localhost:11434/api/generate -d '{"model": "<model_name>", "keep_alive": 0}'
   ```

2. **keep_alive調整** → `config/chat_config.json` または環境変数で:
   ```bash
   OLLAMA_KEEP_ALIVE=5m   # デフォルトは5分。短くしてアンロードを早める
   ```

3. **同時ロード数制限**:
   ```bash
   OLLAMA_MAX_LOADED_MODELS=1  # メモリが足りない場合は1つだけ
   ```

4. **P40のVRAM 24GBに収まるか確認** → 超える場合は较小モデルへ切替:
   ```bash
   scripts/switch_chat_model.py <smaller_model>
   ```

### 復旧確認

```bash
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
ollama ps
# 使用量が安全圏内（例: 16GB / 24GB）なら正常
```

---

## 5. ディスク不足

### 症状

「No space left on device」エラー。DB書き込み失敗。ログが止まる。

### 切り分け

```bash
# ディスク使用量
df -h /

# 大きなファイル/ディレクトリ
du -sh data/ config/ logs/ ~/.ollama/ 2>/dev/null | sort -rh | head -10

# SQLite DBサイズ
ls -lh data/tasks/tasks.db data/growth/growth.db data/metrics/system_metrics.db 2>/dev/null
```

### 対処

1. **journalログ圧縮**:
   ```bash
   journalctl --user --vacuum-size=100M
   # 日次ログに圧縮
   journalctl --user --vacuum-time=7d
   ```

2. **古いバックアップ削除**:
   ```bash
   # backup.sh が自動的に保持数を管理するが、手動で:
   ls -dt backups/*/ | tail -n +8 | xargs rm -rf
   ```

3. **DBサイズ確認SQL**:
   ```bash
   sqlite3 data/tasks/tasks.db "SELECT COUNT(*) FROM tasks;"
   sqlite3 data/growth/growth.db "SELECT COUNT(*) FROM growth_entries;"
   # 大きすぎる場合は古いデータのアーカイブ・削除を検討
   ```

4. **Ollamaキャッシュ削除**:
   ```bash
   rm -rf ~/.ollama/models/.cache
   du -sh ~/.ollama/models/
   ```

5. **一時ファイル削除**:
   ```bash
   find /tmp -name '*.wav' -mtime +1 -delete
   find data/ -name '*.tmp' -delete
   ```

### 復旧確認

```bash
df -h /
# 空き容量が 5GB 以上あれば安全圏
```

---

## 6. DB破損

### 症状

SQLite関連で「database disk image is malformed」エラー。データが読めない。

### 切り分け

```bash
# 整合性チェック
sqlite3 data/tasks/tasks.db "PRAGMA integrity_check;"
sqlite3 data/growth/growth.db "PRAGMA integrity_check;"
sqlite3 data/metrics/system_metrics.db "PRAGMA integrity_check;"
```

### 対処

1. **.recoverによる復元** (ParadeDBの拡張がない場合の汎用方法):
   ```bash
   # 破損DBからデータを抽出して新DBへ
   sqlite3 data/tasks/tasks.db ".recover" | sqlite3 data/tasks/tasks_recovered.db
   # 正常なら置き換え
   mv data/tasks/tasks.db data/tasks/tasks.db.broken
   mv data/tasks/tasks_recovered.db data/tasks/tasks.db
   ```

2. **バックアップからの復元** (推奨):
   ```bash
   # 直近のバックアップを確認
   ls -dt backups/*/

   # 検証してから復元
   scripts/restore.sh <最新バックアップのタイムスタンプ> \
     --target /tmp/subpc-restore --verify-only

   # 問題なければ復元実行
   scripts/restore.sh <最新バックアップのタイムスタンプ> \
     --target /tmp/subpc-restore

   # 復元したDBをコピー
   cp /tmp/subpc-restore/data/tasks/tasks.db data/tasks/tasks.db
   ```

3. **WALモードの場合**:
   ```bash
   # WALをコミットしてから復元
   sqlite3 data/tasks/tasks.db "PRAGMA wal_checkpoint(TRUNCATE);"
   ```

### 復旧確認

```bash
sqlite3 data/tasks/tasks.db "PRAGMA integrity_check;"
# "ok" と表示されれば正常
sqlite3 data/tasks/tasks.db "SELECT COUNT(*) FROM tasks;"
```

---

## 7. メインPC停止時の挙動

### 症状

メインPCのデスクトップクライアントがサブPCのWeb APIへ接続できない。
「DESKTOP接続エラー」等の表示。

### 切り分け

```bash
# サブPCのWeb APIが動作中か
systemctl --user status subpc-web.service
curl -s http://localhost:8080/api/health

# Tailscale/VPN経路の確認
tailscale status
```

### 対処

1. **local-fast経路のFallback** → サブPCのlocal-strong(Ollama)が単独で応答する。
   `ProviderRegistry`が自動的にlocal-fastをスキップし、local-strongにフォールバックする。

2. **メインPC復帰手順**:
   - メインPCを起動
   - デスクトップクライアント起動
   - `SUBPC_DESKTOP_SERVER_URL` が正しいことを確認
   - Web APIへの疎通を確認

3. **メインPCのOllama停止時** → Ollamaサービス再起動:
   ```bash
   # メインPC側で
   systemctl --user restart ollama.service
   ```

### 復旧確認

```bash
# メインPCからサブPCのWeb APIへアクセス
curl -s http://<サブPCのIP>:8080/api/health
# 正常応答があれば2ノード間通信OK
```

---

## 8. サブPC停止・再起動手順

### 基本順序 (更新とRollback)

```text
差分確認 → バックアップ → テスト → サブPC更新 → Health確認 → メインPC更新 → End-to-End確認
```

### 停止手順

```bash
# 1. 各サービスを順に停止
systemctl --user stop subpc-discord.service
systemctl --user stop subpc-web.service
systemctl --user stop subpc-voice.service
systemctl --user stop subpc-sbv2-tts.service

# 2. Ollama停止
systemctl --user stop ollama.service

# 3. 更新作業（必要に応じて）
git -C "$HOME/subpc_living" pull

# 4. 依存更新
"$HOME/subpc_living/.venv/bin/pip" install -r "$HOME/subpc_living/requirements.txt"

# 5. サービス再起動（依存順序）
systemctl --user start ollama.service
sleep 10
systemctl --user start subpc-web.service
systemctl --user start subpc-discord.service
systemctl --user start subpc-voice.service
```

### Rollback手順

```bash
# 直前のコミットへ戻す
git -C "$HOME/subpc_living" log --oneline -5
git -C "$HOME/subpc_living" revert HEAD

# またはバックアップからの復元
scripts/restore.sh <backup_timestamp> --target "$HOME/subpc_living"
```

---

## 9. 秘密情報誤出力への対応

### 症状

ログ、チャット出力、Git履歴にAPIキーや認証情報が含まれた。

### 対処

1. **即座にキーをローテーション** (雛形):
   ```bash
   # 各サービスのキーを無効化して新規発行
   # Discord Bot: https://discord.com/developers/applications → Bot → Reset
   # OpenAI: https://platform.openai.com/api-keys → Revoke → Create
   # Google Calendar: https://console.cloud.google.com → Credentials → Regenerate
   ```

2. **Git履歴からの除去** (git filter-repo):
   ```bash
   # インストール
   pip install git-filter-repo

   # ファイル全体を履歴から除去
   git filter-repo --path config/discord.env --invert-paths
   git filter-repo --path config/.env --invert-paths

   # 特定パターンの置換
   git filter-repo --replace-text <(echo 'OLD_API_KEY==>REDACTED<')
   ```

3. **強制プッシュ** (履歴書き換え後):
   ```bash
   # 注意: 全クローンの再cloneが必要
   git push origin --force --all
   ```

4. **ログのMask方針**:
   - ログ出力時に `tasks/safety.py` の検出ロジックを参考にAPIキーを検出
   - `***REDACTED***` に置換
   - `config/*.env` は `.gitignore` に含まれ、バックアップにも含めない
   - チャットセッション保存時も `session.py` の保存パスにAPIキーが混入しないよう注意

### 復旧確認

```bash
# ログに秘密情報が残っていないか確認
grep -r "sk-" logs/ 2>/dev/null
grep -r "token" logs/ 2>/dev/null | grep -v "REDACTED"
# 問題なければ正常
```

---

## 更新とRollback Quick Reference

基本順序:

```text
差分確認
→ バックアップ (scripts/backup.sh)
→ テスト (.venv/bin/python -m unittest discover -s tests -q)
→ サブPC更新 (git pull + pip install)
→ Health確認 (curl /api/health, ollama list, systemctl --user status)
→ メインPC更新
→ End-to-End確認
```

Rollback:

```bash
git revert HEAD
# または
scripts/restore.sh <backup_timestamp> --target "$HOME/subpc_living"
```

---

## 月1復元訓練

バックアップは「作成できる」だけでなく、復元できることを確認する。

### 手順

1. **バックアップ実行**:
   ```bash
   scripts/backup.sh --target-dir /tmp/restore-test-backup
   ```

2. **一時ディレクトリへ復元**:
   ```bash
   TIMESTAMP=$(ls -dt /tmp/restore-test-backup/*/ | head -1 | xargs basename)
   scripts/restore.sh "$TIMESTAMP" --target /tmp/restore-test-target
   ```

3. **整合性チェック**:
   ```bash
   # ファイル存在確認
   ls -la /tmp/restore-test-target/data/tasks/tasks.db
   ls -la /tmp/restore-test-target/data/vectordb/

   # DB整合性
   sqlite3 /tmp/restore-test-target/data/tasks/tasks.db "PRAGMA integrity_check;"
   sqlite3 /tmp/restore-test-target/data/growth/growth.db "PRAGMA integrity_check;"

   # manifest一致確認
   scripts/restore.sh "$TIMESTAMP" --target /tmp/restore-test-target --verify-only
   ```

4. **結果記録**:

   | 項目 | 結果 | 日付 | 備考 |
   |------|------|------|------|
   | backup.sh 実行 | ✅/❌ | | |
   | restore.sh --verify-only | ✅/❌ | | |
   | restore.sh 実復元 | ✅/❌ | | |
   | tasks.db PRAGMA | ✅/❌ | | |
   | growth.db PRAGMA | ✅/❌ | | |
   | vectordb ファイル数 | 件 | | |

5. **クリーンアップ**:
   ```bash
   rm -rf /tmp/restore-test-backup /tmp/restore-test-target
   ```

訓練は月1回以上実施し、結果を記録する。失敗した場合は原因を調査し、
バックアップスクリプトまたは復元手順を改善する。
