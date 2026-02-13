# subpc_living — Usage

## 前提条件

- Ubuntu 24.04 LTS
- Phase 1〜9 のセットアップスクリプトを実行済み
- Ollama がインストール済み・起動中

---

## セットアップ

### Phase 1: 環境構築

```bash
# NVIDIA ドライバ + CUDA
bash scripts/phase1_setup_nvidia.sh

# Ollama インストール
bash scripts/phase1_setup_ollama.sh

# 検証
bash scripts/phase1_verify.sh
```

### Phase 2: テキスト対話

```bash
# Python 仮想環境 + パッケージ
bash scripts/phase2_setup.sh

# 検証
bash scripts/phase2_verify.sh
```

### Phase 3: 音声対話

```bash
# STT/TTS/VAD パッケージ + kokoro-onnx モデル DL
bash scripts/phase3_setup.sh

# 検証 (全テスト実行)
bash scripts/phase3_verify.sh
```

### Silero VAD を有効化する場合 (オプション)

```bash
source .venv/bin/activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Phase 4: 長期記憶 (RAG)

```bash
# ChromaDB + sentence-transformers + 埋め込みモデル DL
bash scripts/phase4_setup.sh

# 検証
bash scripts/phase4_verify.sh
```

### Phase 5: 映像入力

```bash
# OpenCV + 感情推定 ONNX モデル DL
bash scripts/phase5_setup.sh

# 検証
bash scripts/phase5_verify.sh
```

### Phase 6: PCログ収集

```bash
# psutil インストール + データディレクトリ作成
bash scripts/phase6_setup.sh

# 検証
bash scripts/phase6_verify.sh
```

### Phase 7: パーソナライズ

```bash
# プロフィールディレクトリ作成 + デフォルトプロフィール生成
bash scripts/phase7_setup.sh

# 検証
bash scripts/phase7_verify.sh
```
### Phase 8: 常時稼働化

```bash
# systemd ユニットインストール + ヘルスチェック
bash scripts/phase8_setup.sh

# 検証
bash scripts/phase8_verify.sh
```
### Phase 9: GPU換装

```bash
# GPU検出 + 設定確認
bash scripts/phase9_setup.sh

# 検証
bash scripts/phase9_verify.sh
```

> ℹ️ GPU 省電力サービスは sudo 権限が必要です。セットアップスクリプトの指示に従ってください。

### Phase 10: ウェイクワード検知

```bash
# openwakeword インストール + モデル DL
bash scripts/phase10_setup.sh

# 検証
bash scripts/phase10_verify.sh
```


---

## 仮想環境の有効化

すべてのコマンド実行前に有効化が必要:

```bash
source .venv/bin/activate
```

---

## 1. テキスト対話 (Phase 2)

ターミナルでテキストチャット。マイク・スピーカー不要。

```bash
python src/chat/main.py
```

### チャット内コマンド

| コマンド | 説明 |
|---------|------|
| `/help` | コマンド一覧を表示 |
| `/info` | セッション情報 (ターン数等) |
| `/clear` | 会話履歴をクリア |
| `/system` | システムプロンプトを表示 |
| `/save` | 会話を手動保存 |
| `/model` | モデル・パラメータ情報 |
| `/quit` | 終了 (Ctrl+C でも可) |

会話はセッション終了時に `data/chat_history/` に自動保存される。

---

## 2. 音声対話 (Phase 3)

### フル音声対話

マイク → STT → LLM → TTS → スピーカー のフルパイプライン。

```bash
python src/audio/main.py
```

起動時に以下を自動実行:
1. Ollama 接続確認
2. Whisper STT モデルロード (初回は ~500MB DL)
3. kokoro-onnx TTS モデルロード
4. VAD キャリブレーション (Energy VAD の場合は 2秒間のノイズ計測)

話し終わると自動検出 → 認識 → 応答生成 → 読み上げ。Ctrl+C で終了。

### テキスト入力 → 音声再生モード

マイクなしで TTS のテストが可能。テキスト入力 → LLM 応答 → 音声再生。

```bash
python src/audio/main.py --text-mode
```

### CLI オプション一覧

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--stt-model` | `small` | Whisper モデルサイズ: `tiny`, `base`, `small`, `medium` |
| `--tts-voice` | `jf_alpha` | TTS ボイス名 (下表参照) |
| `--text-mode` | ― | テキスト入力モード (マイクなし) |
| `--vad` | `auto` | VAD 方式: `auto`, `silero`, `energy` |
| `--no-streaming-tts` | ― | ストリーミング TTS を無効化 (全文完了後に合成) |
| `--no-rag` | ― | RAG (長期記憶) を無効化 |
| `--no-vision` | ― | Vision (映像入力) を無効化 |
| `--camera-id` | `0` | カメラデバイスID |
| `--no-monitor` | ― | Monitor (PCログ収集) を無効化 |
| `--no-persona` | ― | Persona (パーソナライズ) を無効化 |
| `--wakeword` | ― | ウェイクワードモードを有効化 (呼びかけで起動) |
| `--wakeword-model` | `hey_jarvis` | ウェイクワードモデル名 |
| `--wakeword-threshold` | `0.5` | ウェイクワード検知の閾値 (0.0〜1.0) |

### VAD 方式

| 値 | 説明 |
|----|------|
| `auto` | torch があれば Silero VAD、なければ Energy VAD (デフォルト) |
| `silero` | Silero VAD を強制使用 (torch 必須) |
| `energy` | Energy VAD を強制使用 (RMS エネルギーベース) |

### TTS ボイス一覧

| 名前 | 説明 |
|------|------|
| `jf_alpha` | 日本語 女性 (Alpha) ← デフォルト |
| `jf_gongitsune` | 日本語 女性 (Gongitsune) |
| `jf_nezumi` | 日本語 女性 (Nezumi) |
| `jf_tebukuro` | 日本語 女性 (Tebukuro) |
| `jm_kumo` | 日本語 男性 (Kumo) |

### 使用例

```bash
# デフォルト (Whisper small + jf_alpha + auto VAD + ストリーミングTTS)
python src/audio/main.py

# 軽量モデルで高速応答
python src/audio/main.py --stt-model tiny

# 男性ボイス
python src/audio/main.py --tts-voice jm_kumo

# Energy VAD + ストリーミングTTS無効化
python src/audio/main.py --vad energy --no-streaming-tts

# Silero VAD を指定
python src/audio/main.py --vad silero

# RAG無効で起動
python src/audio/main.py --no-rag

# Vision無効で起動
python src/audio/main.py --no-vision

# カメラデバイスを指定
python src/audio/main.py --camera-id 1

# テキストモードで TTS テスト
python src/audio/main.py --text-mode --tts-voice jf_nezumi

# ウェイクワードモード (「Hey Jarvis」で起動)
python src/audio/main.py --wakeword

# ウェイクワードモード + 閾値調整
python src/audio/main.py --wakeword --wakeword-threshold 0.3
```

---

## 3. Web UI

ブラウザからチャット + TTS。スマホからも LAN 経由でアクセス可能。

```bash
python src/web/server.py
```

### Web UI CLI オプション

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--host` | `0.0.0.0` | バインドアドレス |
| `--port` | `8000` | ポート番号 |
| `--reload` | ― | 開発用ホットリロード |

### アクセス

- PC: http://localhost:8000
- スマホ (LAN): http://<サブPCのIP>:8000

### Web API

| エンドポイント | メソッド | 説明 |
|---------------|---------|------|
| `/api/health` | GET | ヘルスチェック (Ollama/ディスク/メモリ/モジュール状態) |
| `/api/status` | GET | システム状態 (Ollama/TTS/RAG/Vision/Monitor の接続状況) |
| `/api/tts` | POST | テキスト → WAV 音声合成 (`{"text": "..."}`) |
| `/api/tts/voice` | POST | TTS ボイス変更 (`{"voice": "jm_kumo"}`) |
| `/api/vision/status` | GET | 映像入力の状態 (在席/感情/カメラ情報) |
| `/api/vision/snapshot` | GET | 現在のカメラ画像 (JPEG) |
| `/api/vision/context` | GET | 映像コンテキストテキスト (デバッグ用) |
| `/api/monitor/status` | GET | PCモニター状態 (CPU/メモリ/GPU/ディスク等) |
| `/api/monitor/context` | GET | PCモニターコンテキストテキスト (デバッグ用) |
| `/api/monitor/summary?minutes=60` | GET | 直近N分のメトリクスサマリー |
| `/api/persona/status` | GET | パーソナライズ状態 (プロフィール/要約/プリロード) |
| `/api/persona/profile` | GET | ユーザープロフィール取得 |
| `/api/persona/profile` | POST | プロフィール更新 (`{"name": "...", "note": "..."}`) |
| `/api/persona/summaries?count=5` | GET | 直近の会話要約一覧 |
| `/api/persona/context` | GET | プリロードコンテキスト (デバッグ用) |
| `/ws/chat` | WebSocket | ストリーミングチャット (トークン単位) |

---

## 4. 長期記憶 — RAG (Phase 4)

会話が自動でベクトルDB (ChromaDB) に保存され、関連する過去の文脈がLLMのシステムプロンプトに自動注入される。

### 仕組み

1. 会話のたびに user + assistant のペアが ChromaDB に保存される
2. 新しい発言時に、埋め込みモデル (multilingual-e5-small, 384次元) でセマンティック検索
3. 関連する過去の会話・知識がシステムプロンプトに追加される
4. LLM は過去の文脈を参考に応答（不自然に持ち出さないよう指示付き）

### データ保存先

- ベクトルDB: `data/vectordb/`
- 会話履歴 (JSON): `data/chat_history/`

### RAG を無効にする場合

```bash
# 音声対話
python src/audio/main.py --no-rag

# テキスト対話・Web UI は自動有効 (コード上で無効化する場合は ChatSession(rag=None))
```

### 知識の手動追加 (Python)

```python
from src.memory.vectorstore import VectorStore
from src.memory.rag import RAGRetriever

vs = VectorStore(persist_dir="data/vectordb")
vs.initialize()
rag = RAGRetriever(vector_store=vs)

# 知識を追加
rag.store_knowledge("ユーザーは猫のミケを飼っている", category="preference")
rag.store_knowledge("毎週水曜日にジムに行く", category="schedule")
```

### RAG 設定パラメータ

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `max_context_items` | `5` | 検索結果の最大数 |
| `max_context_chars` | `2000` | コンテキストの最大文字数 |
| `relevance_threshold` | `1.5` | 類似度の閾値 (コサイン距離) |

---

## 5. 映像入力 — Vision (Phase 5)

カメラ映像からユーザーの在席状況・表情を解析し、LLMのシステムプロンプトに自動注入する。

### 仕組み

1. バックグラウンドスレッドでカメラフレームを連続取得 (15fps)
2. 2秒間隔で顔検出 (OpenCV Haar Cascade) + 感情推定 (emotion-ferplus ONNX)
3. 在席/離席、表情の状態を追跡
4. 会話時、映像コンテキストがシステムプロンプトに追加される

**LLMに注入されるコンテキスト例:**

```
--- 現在の映像情報 ---
- ユーザーはカメラの前にいます
- ユーザーの表情: 嬉しそう
  (この表情がしばらく続いています)
```

### 感情ラベル

| English | 日本語 |
|---------|--------|
| neutral | 普通 |
| happiness | 嬉しそう |
| surprise | 驚いている |
| sadness | 悲しそう |
| anger | 怒っている |
| disgust | 嫌そう |
| fear | 怖がっている |
| contempt | 冷めている |

### 使用モデル

| モデル | 用途 | サイズ | 実行環境 |
|--------|------|--------|---------|
| OpenCV Haar Cascade | 顔検出 | OpenCV内蔵 | CPU |
| emotion-ferplus-8.onnx | 感情推定 | ~34MB | CPU (onnxruntime) |

### Vision を無効にする場合

```bash
# 音声対話
python src/audio/main.py --no-vision

# カメラデバイスを指定
python src/audio/main.py --camera-id 1
```

カメラが接続されていない場合は自動的にスキップされる（エラーにはならない）。

---

## 6. PCログ収集 — Monitor (Phase 6)

psutil でサブPCのシステムメトリクスを常時収集・SQLiteに蓄積し、LLMのシステムプロンプトに自動注入する。

### 仕組み

1. バックグラウンドスレッドで 30秒間隔でメトリクスを収集 (psutil)
2. SQLite (WALモード) に時系列データとして蓄積
3. 会話時、PCの現在の状態がシステムプロンプトに追加される
4. 異常検知 (CPU過負荷、メモリ逆迫、高温等) は自動で警告注入

**LLMに注入されるコンテキスト例:**

```
--- サブPCの現在の状態 ---
- CPU: 25% (低負荷)
  温度: 52°C
- メモリ: 8.2GB / 15.6GB (53%, 余裕あり)
- GPU: 15% (稼働中)
  VRAM: 1200MB / 6144MB
  温度: 45°C
```

### 収集されるメトリクス

| カテゴリ | 項目 |
|------------|------|
| CPU | 使用率 (全体/コア別)、クロック周波数、ロードアベレージ、温度 |
| メモリ | 合計/使用量/使用率、スワップ |
| ディスク | 合計/使用量/使用率、I/Oレート |
| ネットワーク | 送受信レート |
| GPU | 使用率、VRAM、温度、電力 (nvidia-smi経由) |
| プロセス | 総数、CPUトップ5 |

### Monitor を無効にする場合

```bash
python src/audio/main.py --no-monitor
```

### データ保存先

- DB: `data/metrics/system_metrics.db` (SQLite, WALモード)
- 古いデータは 30日で自動クリーンアップ可能

---

## 7. パーソナライズ — Persona (Phase 7)

ユーザープロフィール管理・会話要約・セッションプリロード・プロアクティブ発話を統合したパーソナライズ機能。

### 仕組み

1. **ユーザープロフィール**: 名前・好み・習慣・スケジュール・メモを JSON で永続化
2. **セッションプリロード**: 会話開始時に日時・プロフィール・スケジュール・直近の会話要約をシステムプロンプトに自動注入
3. **会話要約**: セッション終了時にLLMで会話を要約・ユーザー情報を自動抽出してプロフィールに追記
4. **プロアクティブ発話**: スケジュールリマインド・休憩提案・PC異常通知・時間帯挨拶

**LLMに注入されるコンテキスト例:**

```
--- 現在の状況 ---
- 日時: 2026年02月11日 (水曜日) 21:30
- 時間帯: 夜

--- ユーザープロフィール ---
- ユーザーの名前: はるか
- 好み・嗜好: food: カレー, music: ジャズ
- プログラマー
- 猫を2匹飼っている

--- 今日のスケジュール (02/11 Wednesday) ---
- 14:00 会議 (Zoom)

--- 最近の会話の要約 ---
[2026-02-11] Pythonの非同期処理について議論した。asyncioの基本...
```

### プロアクティブ発話トリガー

| トリガー | 条件 | クールダウン |
|---------|------|-------------|
| `schedule_remind` | 予定の15～5分前 | 30分 |
| `break_suggest` | 2時間以上連続作業 | 1時間 |
| `greeting` | セッション開始時 (朝/深夜) | 12時間 |
| `pc_alert` | CPU/メモリ/温度異常 | 10分 |

### プロフィールの編集

**方法1: JSON直接編集**

`data/profile/user_profile.json` をテキストエディタで編集:

```json
{
  "name": "はるか",
  "nickname": "はるかさん",
  "preferences": {"food": "カレー", "music": "ジャズ"},
  "habits": {"wake_time": "07:00", "sleep_time": "24:00"},
  "schedule": [
    {"date": "2026-02-12", "time": "14:00", "title": "会議", "note": "Zoom"}
  ],
  "notes": ["猫を2匹飼っている", "プログラマー"],
  "extracted_facts": [],
  "updated_at": ""
}
```

**方法2: Web API**

```bash
# プロフィール取得
curl http://localhost:8000/api/persona/profile

# 名前設定
curl -X POST http://localhost:8000/api/persona/profile \
  -H 'Content-Type: application/json' \
  -d '{"name": "はるか"}'

# 好み追加
curl -X POST http://localhost:8000/api/persona/profile \
  -H 'Content-Type: application/json' \
  -d '{"preferences": {"food": "カレー"}}'

# メモ追加
curl -X POST http://localhost:8000/api/persona/profile \
  -H 'Content-Type: application/json' \
  -d '{"note": "猫を2匹飼っている"}'

# スケジュール追加
curl -X POST http://localhost:8000/api/persona/profile \
  -H 'Content-Type: application/json' \
  -d '{"schedule": {"title": "会議", "date": "2026-02-12", "time": "14:00", "note": "Zoom"}}'

# 会話要約一覧
curl http://localhost:8000/api/persona/summaries

# プリロードコンテキスト確認
curl http://localhost:8000/api/persona/context
```

**方法3: Python API**

```python
from src.persona.profile import UserProfile

p = UserProfile("data/profile/user_profile.json")
p.load()

p.name = "はるか"
p.set_preference("food", "カレー")
p.set_habit("wake_time", "07:00")
p.add_note("猫を2匹飼っている")
p.add_schedule("会議", "2026-02-12", "14:00", "Zoom")
```

### データ保存先

- プロフィール: `data/profile/user_profile.json`
- 会話要約: `data/profile/summaries/summary_*.json`

### Persona を無効にする場合

```bash
python src/audio/main.py --no-persona
```

---

## 8. 常時稼働 — Service (Phase 8)

systemd で Web UI・音声対話をサービスとして管理。自動再起動・GPU省電力制御を統合。

### サービス管理 (service_ctl.sh)

```bash
# 全サービスの状態確認
bash scripts/service_ctl.sh status

# Web UI をサービスとして起動
bash scripts/service_ctl.sh start web

# 音声対話をサービスとして起動
bash scripts/service_ctl.sh start voice

# 全サービス起動
bash scripts/service_ctl.sh start all

# サービス停止
bash scripts/service_ctl.sh stop web

# ログ確認 (リアルタイムフォロー)
bash scripts/service_ctl.sh logs web -f

# ヘルスチェック
bash scripts/service_ctl.sh health

# GPU情報
bash scripts/service_ctl.sh gpu
```

### 自動起動 (ブート時)

```bash
# 自動起動を有効化
bash scripts/service_ctl.sh enable web
bash scripts/service_ctl.sh enable voice

# 自動起動を無効化
bash scripts/service_ctl.sh disable web
```

### service_ctl.sh コマンド一覧

| コマンド | 説明 |
|---------|------|
| `status` | 全サービスの状態を表示 |
| `start [web│voice│all]` | サービスを開始 |
| `stop [web│voice│all]` | サービスを停止 |
| `restart [web│voice│all]` | サービスを再起動 |
| `enable [web│voice│all]` | 自動起動を有効化 |
| `disable [web│voice│all]` | 自動起動を無効化 |
| `logs [web│voice] [-f]` | ログを表示 |
| `health` | ヘルスチェック実行 |
| `gpu` | GPU 情報表示 |

### GPU 省電力制御 (オプション)

nvidia-smi で GPU 電力制限を制御。常時稼働時のアイドル消費電力を抑える。

```bash
# システムサービスとしてインストール (root 権限必要)
sudo cp scripts/systemd/subpc-gpu-powersave.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable subpc-gpu-powersave
sudo systemctl start subpc-gpu-powersave
```

| モード | 電力制限 | 用途 |
|--------|-----------|------|
| idle | 100W | アイドル時 (デフォルト) |
| active | 250W | LLM推論時 (P40 TDP) |

### systemd サービス一覧

| サービス名 | 種類 | 説明 |
|-------------|------|------|
| `subpc-web` | ユーザー | Web UI サーバー (Type=notify, Watchdog付き) |
| `subpc-voice` | ユーザー | 音声対話パイプライン |
| `subpc-gpu-powersave` | システム | GPU 省電力制御 (oneshot, 要sudo) |

---

## 9. GPU換装 (Phase 9)

### GPU 自動検出

Phase 9 では GPU を自動検出し、各モジュールの設定を最適化します。

```bash
# 現在の GPU 設定を確認
python3 -c "from src.service.gpu_config import main; main()"
```

| GPU | Profile | STT | Embedding | Vision ONNX | LLM推奨 |
|-----|---------|-----|-----------|-------------|----------|
| P40 (24GB) | `p40` | cuda / float16 / medium | cuda | CUDAExecutionProvider | 14B Q4 |
| GTX 1060 (6GB) | `gtx1060` | cpu / int8 / small | cpu | CPUExecutionProvider | 7B Q4 |
| GPUなし | `cpu` | cpu / int8 / small | cpu | CPUExecutionProvider | 7B Q4 |

### LLM モデルの変更

P40 換装後は大型モデルに切り替え可能です:

```bash
# 14B モデルのダウンロード
ollama pull qwen2.5:14b-instruct-q4_K_M
```

`config/chat_config.json` の `model` を変更:

```json
{
  "model": "qwen2.5:14b-instruct-q4_K_M",
  "num_ctx": 8192
}
```

### P40 換装手順

1. P40 を物理的に取り付け
2. 電源 500W → 650W への換装を推奨 (P40 TDP: 250W)
3. BIOS で iGPU を映像出力に設定 (P40 は映像出力なし)
4. Ubuntu 起動後 `nvidia-smi` で認識確認
5. `bash scripts/phase9_setup.sh` を実行
6. `config/chat_config.json` の model を 14b に変更

---

## 10. ウェイクワード検知 (Phase 10)

特定の呼びかけ（「Hey Jarvis」等）を検知して音声対話モードを自動起動する。常時稼働時は低消費電力でウェイクワード待機。

### 仕組み

1. マイク音声を 80ms フレーム単位で OpenWakeWord モデルに入力
2. スコアが閾値 (デフォルト: 0.5) を超えたらウェイクワード検知
3. 検知後、VAD → STT → LLM → TTS の対話ターンを実行
4. 対話終了後、再びウェイクワード待機に戻る

### 利用可能なウェイクワード

| モデル名 | フレーズ | 言語 |
|-----------|------------|------|
| `hey_jarvis` | "Hey Jarvis" | 英語 |
| `alexa` | "Alexa" | 英語 |
| `hey_mycroft` | "Hey Mycroft" | 英語 |

> ℹ️ 現時点では英語プリトレインモデルのみ対応。日本語ウェイクワードはカスタムトレーニングで対応予定。

### 使用例

```bash
# デフォルト (hey_jarvis, 閾値 0.5)
python src/audio/main.py --wakeword

# 閾値を低くして感度を上げる
python src/audio/main.py --wakeword --wakeword-threshold 0.3

# Alexa モデルを使用
python src/audio/main.py --wakeword --wakeword-model alexa
```

### ウェイクワードなしで起動 (従来通り)

```bash
python src/audio/main.py
```

`--wakeword` を指定しない場合は従来通りの即時VADリスニングモード。

---

## 設定ファイル

### config/chat_config.json

| キー | デフォルト | 説明 |
|------|-----------|------|
| `ollama_base_url` | `http://localhost:11434` | Ollama API の URL |
| `model` | `qwen2.5:7b-instruct-q4_K_M` | 使用する LLM モデル |
| `temperature` | `0.7` | 生成のランダム度 (0.0〜1.0) |
| `top_p` | `0.9` | Nucleus sampling |
| `top_k` | `40` | Top-K sampling |
| `num_ctx` | `4096` | コンテキスト長 (トークン数) |
| `repeat_penalty` | `1.1` | 繰り返しペナルティ |
| `system_prompt` | *(日本語プロンプト)* | AI の振る舞い指示 |
| `max_history_turns` | `20` | 保持する会話ターン上限 |
| `history_dir` | `data/chat_history` | 会話履歴の保存先 |
| `stream` | `true` | ストリーミング出力 |

---

## ディレクトリ構成

```
subpc_living/
├── config/
│   └── chat_config.json       # チャット設定
├── data/
│   ├── chat_history/          # 会話履歴 (JSON)
│   ├── vectordb/              # ChromaDB ベクトルDB (Phase 4)
│   ├── metrics/               # システムメトリクスDB (Phase 6)
│   └── profile/               # ユーザープロフィール + 会話要約 (Phase 7)
│       ├── user_profile.json  # プロフィール
│       └── summaries/         # 会話要約 JSON
├── models/
│   ├── stt/                   # Whisper モデルキャッシュ (自動DL)
│   ├── tts/
│   │   └── kokoro/            # kokoro-onnx モデル
│   └── vision/
│       └── emotion-ferplus-8.onnx  # 感情推定 ONNX モデル
├── scripts/
│   ├── phase1_setup_nvidia.sh
│   ├── phase1_setup_ollama.sh
│   ├── phase1_verify.sh
│   ├── phase2_setup.sh
│   ├── phase2_verify.sh
│   ├── phase3_setup.sh
│   ├── phase3_verify.sh
│   ├── phase4_setup.sh
│   ├── phase4_verify.sh
│   ├── phase5_setup.sh
│   ├── phase5_verify.sh
│   ├── phase6_setup.sh
│   ├── phase6_verify.sh
│   ├── phase7_setup.sh
│   ├── phase7_verify.sh
│   ├── phase8_setup.sh
│   ├── phase8_verify.sh
│   ├── service_ctl.sh            # サービス管理ヘルパー
│   └── systemd/
│       ├── subpc-web.service     # Web UI systemd ユニット
│       ├── subpc-voice.service   # 音声対話 systemd ユニット
│       └── subpc-gpu-powersave.service  # GPU省電力 systemd ユニット
├── src/
│   ├── audio/                 # Phase 3: 音声対話
│   │   ├── main.py            # CLI エントリポイント
│   │   ├── pipeline.py        # VAD→STT→LLM→TTS パイプライン
│   │   ├── stt.py             # faster-whisper STT
│   │   ├── tts.py             # kokoro-onnx TTS
│   │   ├── vad.py             # VAD (Energy + Silero)
│   │   ├── wakeword.py        # ウェイクワード検知 (OpenWakeWord, Phase 10)
│   │   └── audio_io.py        # マイク入力・スピーカー出力
│   ├── chat/                  # Phase 2: テキスト対話
│   │   ├── main.py            # CLI エントリポイント
│   │   ├── client.py          # Ollama API クライアント
│   │   ├── session.py         # 会話セッション管理 + RAG/Vision/Monitor統合
│   │   └── config.py          # 設定管理
│   ├── memory/                # Phase 4: 長期記憶
│   │   ├── embedding.py       # 埋め込みモデル (multilingual-e5-small)
│   │   ├── vectorstore.py     # ChromaDB ベクトルストア
│   │   └── rag.py             # RAG リトリーバー
│   ├── vision/                # Phase 5: 映像入力
│   │   ├── camera.py          # カメラキャプチャ (バックグラウンド)
│   │   ├── detector.py        # 顔検出 + 感情推定
│   │   └── context.py         # 映像コンテキスト管理
│   ├── monitor/               # Phase 6: PCログ収集
│   │   ├── collector.py       # psutilメトリクス収集
│   │   ├── storage.py         # SQLite時系列ストレージ
│   │   └── context.py         # モニターコンテキスト管理
│   ├── persona/               # Phase 7: パーソナライズ
│   │   ├── profile.py         # ユーザープロフィール管理
│   │   ├── summarizer.py      # 会話要約 + 知識抽出
│   │   ├── preloader.py       # セッションプリロード
│   │   └── proactive.py       # プロアクティブ発話エンジン
│   ├── service/               # Phase 8-9: 常時稼働化 + GPU換装
│   │   ├── healthcheck.py     # ヘルスチェック (Ollama/ディスク/メモリ)
│   │   ├── power.py           # GPU省電力制御 (GPU別プリセット)
│   │   └── gpu_config.py      # GPU自動検出・デバイス設定 (Phase 9)
│   └── web/                   # Web UI
│       ├── server.py          # FastAPI サーバー
│       └── static/            # HTML/JS/CSS
├── tools/
│   └── piper/                 # Piper TTS バイナリ (レガシー)
├── requirements.txt
└── readme.md                  # 要件定義書
```

---

## トラブルシューティング

### Ollama に接続できない

```bash
sudo systemctl start ollama
sudo systemctl status ollama
```

### モデルが見つからない

```bash
ollama list                              # インストール済みモデル確認
ollama pull qwen2.5:7b-instruct-q4_K_M  # モデルをDL
```

### マイクが認識されない

```bash
source .venv/bin/activate
python -c "import sounddevice; print(sounddevice.query_devices())"
```

### TTS の GPU 警告 (無害)

```
GPU device discovery failed: ... "/sys/class/drm/card0/device/vendor"
```

TTS は CPU 実行のため無視して問題なし。

### Silero VAD を使いたいが torch がない

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

CPU 版 torch (~200MB)。GPU 版は不要。
