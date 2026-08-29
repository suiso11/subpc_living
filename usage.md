# subpc_living — Usage

## 前提条件

> ⚠️ **実装とデプロイの区別**: 本ドキュメントには2種類のコマンドが混在する。
>
> 1. **リポジトリ内で動作する開発・動作確認用コマンド**（`python src/...`、仮想環境、
>    `curl localhost` 等）— コードから導出した仕様であり、開発機でも単体テストで検証可能。
> 2. **将来の Ubuntu デプロイ用コマンド**（`bash scripts/phase*_setup.sh`、systemd、
>    `sudo tailscale serve` 等）— **実機デプロイは未実施・未検証**。
>
> **デプロイ済みのサブPCは存在しない**。systemd・Ollama・GPU・センサー・Tailscale の
> 稼働は未確認であり、以下の「前提条件」は**将来のデプロイ時の前提**である。

- （デプロイ時前提）Ubuntu 24.04 LTS
- （デプロイ時前提）Phase 1〜9 のセットアップスクリプトを実行済み
- （デプロイ時前提）Ollama がインストール済み・起動中
- 実機での運用（停止・再起動・更新・復旧）はドラフトの **[docs/runbook.md](docs/runbook.md)** を参照（★未検証）

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

## ローカル推論 backend の切り替え (P0-2)

> 実装・テスト状態は **[docs/implementation_status.md](docs/implementation_status.md)**、
> タスク順序は **[docs/implementation_plan.md](docs/implementation_plan.md)** を正とする。
> 下記の **startup コマンド（llama.cpp / LM Studio / vLLM）は外部ツールの例示であり未検証**。
> このリポジトリのテストは FakeTransport / MockTransport 注入のみで、実サーバー・実モデル・
> 実GPU・実ネットワークへの接続（live 検証）は未実施。性能主張はしない。

ローカル推論 backend は **Ollama（既定）** と、OpenAI-compatible なローカルサーバー
（llama.cpp / LM Studio / vLLM 等）の**どちらか**を `config/chat_config.json` で選べる。
無設定時（`local_provider_kind` が空 or `"ollama"`）は従来どおり Ollama のまま。

### 既定: Ollama（変更不要）

```json
{
  "local_provider_kind": "ollama",
  "ollama_base_url": "http://localhost:11434",
  "model": "qwen2.5:7b-instruct-q4_K_M"
}
```

### OpenAI-compatible ローカルサーバーへ切り替える

1. **推論サーバーを起動する**。以下は外部ツールの**例示（未検証）**。いずれも同一マシンの
   loopback で `/v1` の OpenAI互換API（`/chat/completions` と `/models`）を提供する想定:

   ```bash
   # llama.cpp（例示・未検証）
   llama-server -m /path/to/model.gguf --host 127.0.0.1 --port 8080

   # LM Studio（例示・未検証）: アプリ内でモデルをロードし "Local Server" を有効化

   # vLLM（例示・未検証）
   python -m vllm.entrypoints.openai.api_server --model <model-id> --host 127.0.0.1 --port 8080
   ```

   > 上記コマンドはこのプロジェクトの一部ではなく、例示としてのみ記載しています。
   > 実機での導入・起動・接続の検証は実施していません。

2. **設定を切り替える**。[config/chat_config.local-openai.example.json](config/chat_config.local-openai.example.json)
   を雛形にするか、`config/chat_config.json` に次を反映する:

   ```json
   {
     "local_provider_kind": "openai_compatible",
     "local_base_url": "http://localhost:8080/v1",
     "model": "<ローカルサーバーが /v1/models で返すモデルID>"
   }
   ```

   - **loopback 限定の信頼境界**: `local_base_url` は `localhost` または loopback IP のみ許可。
     それ以外（LAN / VPN / 公開ホスト・曖昧なホスト名）は設定検証で `ValueError` になる。
     `local=True` で登録されるため cloud の承認・redaction を迂回するからで、リモートの
     信頼済みノード対応は deferred（[docs/decisions/local_inference_backend.md](docs/decisions/local_inference_backend.md)）。
   - `local_provider_id` を省略・空にすると kind 既定の `local-openai` を使う。
   - `num_ctx` 等の Ollama 固有パラメータは OpenAI-compatible では明示的に無視される。

3. **（任意）API key**。認証を要求するローカルサーバーでは、キー自体は設定に書かず
   **環境変数名だけ**を `local_api_key_env` に指定する:

   ```bash
   export LOCAL_OPENAI_API_KEY=<キー>
   ```

   ```json
   {
     "local_provider_kind": "openai_compatible",
     "local_api_key_env": "LOCAL_OPENAI_API_KEY"
   }
   ```

   キーは実行時に環境変数から解決され、コード・設定・ログへ保存されない。
   `local_api_key_env` が空・未設定なら Authorization ヘッダ自体を送らない（keyless）。

4. **既存の入口のまま起動して確認する**（CLI / Web / Voice / Discord / batch は同じ
   `config/chat_config.json` を読む）:

   ```bash
   python src/chat/main.py        # CLI
   python src/web/server.py       # Web UI
   python src/audio/main.py       # Voice
   # Discord / batch（日記・日次パーソナライズ）も同じ config を継承
   ```

   Discord の default プロファイルは `config/chat_config.json` を継承する。チャンネル別
   プロファイルは同 JSON の `discord_channel_profiles` の `provider_kind` /
   `local_base_url` / `api_key_env` / `model` で個別に上書きできる。

### Ollama へ戻す

`config/chat_config.json` の `local_provider_kind` を `"ollama"`（または未設定）に戻し、
`model` / `ollama_base_url` を元へ戻す。`local_base_url` / `local_api_key_env` は無視される。

### MockTransport と live の区別

- リポジトリのテスト（`tests/llm/test_local_openai_provider.py`、`tests/assistant/test_factory.py` 等）は
  FakeTransport / MockTransport を注入し、実サーバー・実モデル・実GPU・実ネットワークを使わない。
- `is_available` はネットワーク probe をせずライフサイクル状態のみ。到達性は
  `generate` / `generate_stream` で遅延確立される。
- llama.cpp / LM Studio / vLLM の導入・起動・接続の **live 検証は未実施**。
  tokens/sec・レイテンシ・VRAM などの性能主張はしない。

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

> ⚠️ **センサーは既定オフ (opt-in 必須)**: 音声対話はマイクの明示的な同意が無いと
> 起動しない。起動ごとに同意するなら `--microphone`、常時同意するなら
> `SENSOR_MICROPHONE_ENABLED=true` を設定する。カメラ (`--camera` / `SENSOR_CAMERA_ENABLED`)、
> 画面 (`--screen` / `SENSOR_SCREEN_CAPTURE_ENABLED`)、Monitor (`--monitor` /
> `SENSOR_MONITOR_ENABLED`)、活動収集 (`SENSOR_ACTIVITY_ENABLED`) も同様に明示時のみ有効。
> テキストモード (`--text-mode`) はセンサーを使わない。

```bash
python src/audio/main.py --microphone
```

起動時に以下を自動実行:
1. 選択中ローカル推論 backend の接続確認（Ollama は `/api/tags` でモデル存在確認、
   openai_compatible は `/models` を参照。未取得なら生成時に確認）
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
| `--text-mode` | ― | テキスト入力モード (マイクなし・センサー不使用) |
| `--vad` | `auto` | VAD 方式: `auto`, `silero`, `energy` |
| `--no-streaming-tts` | ― | ストリーミング TTS を無効化 (全文完了後に合成) |
| `--no-rag` | ― | RAG (長期記憶) を無効化 |
| `--microphone` | ― | マイク入力を有効化 (音声対話の必須同意。既定: 無効) |
| `--camera` | ― | カメラ (Vision) を有効化 (既定: 無効) |
| `--screen` | ― | Screen (画面認識: スクリーンショット→VLM描写) を有効化 (既定: 無効) |
| `--monitor` | ― | Monitor (PCログ収集) を有効化 (既定: 無効) |
| `--camera-id` | `0` | カメラデバイスID |
| `--no-vision` | ― | (非推奨) Vision を明示的に無効化。`--camera` や `SENSOR_CAMERA_ENABLED=true` より優先 |
| `--no-monitor` | ― | (非推奨) Monitor を明示的に無効化。`--monitor` や `SENSOR_MONITOR_ENABLED=true` より優先 |
| `--no-persona` | ― | Persona (パーソナライズ) を無効化 |
| `--wakeword` | ― | ウェイクワードモードを有効化 (呼びかけで起動。マイクの同意が必要) |
| `--wakeword-model` | `hey_jarvis` | ウェイクワードモデル名 |
| `--wakeword-threshold` | `0.5` | ウェイクワード検知の閾値 (0.0〜1.0) |

> `--microphone` / `--camera` / `--screen` / `--monitor` はその起動一回限りの同意
> (永続化しない)。env の `SENSOR_*_ENABLED=true` は常時同意として同じ経路で有効化
> される。`--wakeword` もマイクが必要なため同じマイク同意ゲートに従う。
> `SENSOR_MICROPHONE_ENABLED` は共有 SensorPolicy の canonical であり、Voice CLI に
> 加えて Discord 通話STT と Desktop の push-to-talk マイクも同じ gate で制御される。

### 音声発話からの予定登録 (独立 opt-in, 既定オフ)

音声対話の「予定入れて」「〜の予定を追加して」等による **Google Calendar への
書き込み**は、マイク同意とは**別の独立 opt-in** `VOICE_CALENDAR_WRITE_ENABLED`
が無い限り行われない (fail closed)。

| env | 既定 | 説明 |
|-----|------|------|
| `VOICE_CALENDAR_WRITE_ENABLED` | `false` | 音声パイプラインのカレンダー**書き込み** (予定直接登録) を許可 |

- 有効化は明示 `true` のみ。false / 未設定 / 不正値は無効 (fail closed)。
- **マイク同意だけではカレンダーへ一切書き込まない**。`--microphone` や
  `SENSOR_MICROPHONE_ENABLED=true` は録音・STT の同意であり、書き込みの同意ではない。
- 無効時の音声ターンは、認識テキストをローカル LLM / セッションの**通常経路へそのまま
  フォールスルー**する。予定登録の意図を含む発話も LLM 応答として扱われ、外部カレンダーへ
  は到達しない。クライアント自体も構築されない。
- 有効時 (`VOICE_CALENDAR_WRITE_ENABLED=true`) は従来の直接登録挙動を維持する。
  書き込み先クライアントの構築は従来どおり `TASKS_CALENDAR_SYNC_ENABLED=true` と
  OAuth 設定 (`GOOGLE_OAUTH_CREDENTIALS`) も必要 (Web / Discord の
  `TASKS_CALENDAR_SYNC_ENABLED` と共通)。
- **保持・副作用の区別**: カレンダーの**読み取り** (予定を `data/calendar/upcoming.json`
  から LLM コンテキストへ注入) と**書き込み** (予定直接登録) は別物。読み取りはこの
  opt-in と無関係に常に動作し、書き込みだけがこの gate で制御される。

```env
VOICE_CALENDAR_WRITE_ENABLED=false
```

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
| `jvnv-F1-jp` | Style-Bert-VITS2 JVNV F1 JP (7スタイル) |
| `tsukuyomi-chan` | Style-Bert-VITS2 つくよみちゃん (Neutralのみ) |

Kokoro以外に Style-Bert-VITS2 を使う場合は、専用venvとTTSサーバーを用意する。

```bash
# JVNV F1 JP (デフォルト)
bash scripts/setup_style_bert_vits2.sh

# つくよみちゃんコーパスモデル
SBV2_REPO=ayousanz/tsukuyomi-chan-style-bert-vits2-model \
SBV2_MODEL_NAME=tsukuyomi-chan \
SBV2_MODEL_FILE=tsukuyomi-chan_e200_s5200.safetensors \
SBV2_FILES=tsukuyomi-chan_e200_s5200.safetensors,config.json,style_vectors.npy \
bash scripts/setup_style_bert_vits2.sh
```

モデルを切り替えたら systemd ユニットの env も合わせる。

```bash
# jvnv の場合
# Environment=SBV2_MODEL_NAME=jvnv-F1-jp
# Environment=SBV2_MODEL_FILE=jvnv-F1-jp_e160_s14000.safetensors

# つくよみちゃんの場合
# Environment=SBV2_MODEL_NAME=tsukuyomi-chan
# Environment=SBV2_MODEL_FILE=tsukuyomi-chan_e200_s5200.safetensors

install -m 0644 scripts/systemd/subpc-sbv2-tts.service ~/.config/systemd/user/subpc-sbv2-tts.service
systemctl --user daemon-reload
bash scripts/service_ctl.sh restart sbv2
curl http://127.0.0.1:50121/health
```

> つくよみちゃんモデルはスタイルが Neutral のみ（感情切り替え不可）。
> 7スタイル（Angry/Sad/Happy等）を使いたい場合は `jvnv-F1-jp` を指定する。
> つくよみちゃんコーパスのライセンスは https://tyc.rei-yumesaki.net/material/corpus/#terms3 に準じる。

### 使用例

```bash
# デフォルト (Whisper small + jf_alpha + auto VAD + ストリーミングTTS + マイク同意)
python src/audio/main.py --microphone

# 軽量モデルで高速応答
python src/audio/main.py --microphone --stt-model tiny

# 男性ボイス
python src/audio/main.py --microphone --tts-voice jm_kumo

# Energy VAD + ストリーミングTTS無効化
python src/audio/main.py --microphone --vad energy --no-streaming-tts

# Silero VAD を指定
python src/audio/main.py --microphone --vad silero

# RAG無効で起動
python src/audio/main.py --microphone --no-rag

# Vision無効で起動 (既定で無効のため通常は不要)
python src/audio/main.py --microphone --no-vision

# カメラデバイスを指定 (カメラは --camera で明示的に有効化)
python src/audio/main.py --microphone --camera --camera-id 1

# テキストモードで TTS テスト (センサー不使用・マイク同意も不要)
python src/audio/main.py --text-mode --tts-voice jf_nezumi

# ウェイクワードモード (「Hey Jarvis」で起動。マイク同意も必要)
python src/audio/main.py --microphone --wakeword

# ウェイクワードモード + 閾値調整
python src/audio/main.py --microphone --wakeword --wakeword-threshold 0.3
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

### Web UI のセンサー opt-in (env gate)

Web UI の機微センサーは共有 SensorPolicy の canonical env で有効化する (既定オフ):

| env | 対象 | 既定 |
|-----|------|------|
| `SENSOR_CAMERA_ENABLED` | Vision (カメラ) | `false` |
| `SENSOR_SCREEN_CAPTURE_ENABLED` | Screen (画面認識 local/remote) | `false` |
| `SENSOR_SCREEN_INGEST_ENABLED` | `/api/screen/ingest` 受信 (token 必須) | `false` |
| `SENSOR_ACTIVITY_ENABLED` | Companion 活動収集 | `false` |
| `SENSOR_MONITOR_ENABLED` | Monitor (PCログ収集) | `false` |
| `SENSOR_MICROPHONE_ENABLED` | Web 音声入力 (`/api/stt` / WS `audio_message` のサーバー側 STT) | `false` |

- 有効化は明示 `true` のみ。false / 未設定 / 不正値は無効 (fail closed)。
- legacy の `WEB_SCREEN_CONTEXT_ENABLED` は `SENSOR_SCREEN_CAPTURE_ENABLED` が
  未設定のときだけ screen_capture として効く。canonical の false は legacy の true を上書き。
- `/api/screen/ingest` は `SENSOR_SCREEN_INGEST_ENABLED=true` **と**共有トークン
  `SCREEN_INGEST_TOKEN` の両方が無ければ 403。token 単独では有効化しない
  (token は認証であり同意ではない)。
- 受信した生 JPEG は保存しない。VLM で 1 回描写した結果だけ
  `data/screen/latest.json` に保持する。
- **Web 音声入力は共有 SensorPolicy のマイク gate を必須とする**: 録音はブラウザの
  `getUserMedia`（HTTPS の権限プロンプト必須）で行い、サーバー自身はマイクを開かない。
  ただしサーバー側 STT の受付（POST `/api/stt` / WS `/ws/chat` の `audio_message`）は
  `SENSOR_MICROPHONE_ENABLED=true` が無ければ 403 / 固定文言 error を返し、
  base64 デコード・STT・一時ファイルへ到達しない。ブラウザ権限だけでポリシーを
  迂回できない。`/api/status` の `stt` は STT engine ロード済み **かつ** policy true の
  ときだけ True になり、Web UI はこの `stt` でマイク入力 UI を gate する
  （env 名・値は frontend へ露出しない）。

### アクセス

- PC: http://localhost:8000
- スマホ (LAN): http://<サブPCのIP>:8000

### スマホから音声対話（Tailscale HTTPS経由）

スマホのブラウザからマイク入力するには **HTTPS が必須**。Tailscale Serve で自動HTTPS化する。

> ⚠️ 本節は**将来の Ubuntu デプロイ時の手順（未検証）**。このプロジェクトでは
> Tailscale ノードは**未検証**であり、MagicDNS 名・IP は実機導入時に確認する。

#### 前提条件（デプロイ時）

- Tailscale セットアップ（詳細は `usage_tailscale_ssh.md`。実機未検証）
- スマホにも Tailscale アプリをインストールし、同じアカウントでログイン
- ffmpeg がインストール済み（`sudo apt install ffmpeg`）

#### 起動方法

**一括起動スクリプト（推奨）:**

```bash
bash scripts/start_mobile.sh
```

このスクリプトは以下を自動で行う:
1. Tailscale 接続確認
2. `tailscale serve` で HTTPS プロキシ設定（443 → 8000）
3. Web UI サーバー起動（STT/TTS 含む）

**手動で起動する場合:**

```bash
# 1. HTTPS プロキシを設定
sudo tailscale serve --bg --https=443 http://localhost:8000

# 2. Web UI サーバー起動
source .venv/bin/activate
python -m src.web.server --host 0.0.0.0 --port 8000
```

#### スマホからのアクセス URL（デプロイ後に確定）

```
https://<サブPCのMagicDNS名>.ts.net
```

> ℹ️ MagicDNS 名は `tailscale status --json | jq -r '.Self.DNSName'` で実機確認する（★未検証）

#### Androidへアプリとして追加

1. AndroidのTailscaleを接続する。
2. Chromeで上記HTTPS URLを開く。
3. Web UI右上の設定を開き、**Androidアプリ → この端末に追加**をタップする。
4. Androidの確認画面で **インストール** を選ぶ。
5. 以後はホーム画面の **BUDDY** アイコンから全画面で起動する。

インストールボタンが表示されない場合は、Chromeの **︙ → ホーム画面に追加** を選ぶ。
更新が反映されない場合は、Chromeでページを一度再読み込みしてからアプリを開き直す。
Tailscale接続中のみサブPCへ到達できるため、外出先でも一般公開せず利用できる。

#### 操作方法

1. **テキスト入力**: メッセージ欄にテキストを入力 → 送信ボタン
2. **音声入力**: 🎤 マイクボタンをタップ → 話す → もう一度タップで停止 → 自動でSTT→LLM→TTS
3. **読み上げ**: 🔊 トグルがONなら応答が自動で音声再生される

#### 音声入力の流れ

```
スマホのマイク → 録音 → base64エンコード
    → WebSocket で送信
    → サーバー側 STT (Whisper medium, CUDA)
    → 認識テキストをLLMに送信
    → ストリーミング応答
    → TTS音声合成 (kokoro-onnx)
    → base64 WAV をスマホに返却 → 再生
```

#### Tailscale Serve の停止

```bash
sudo tailscale serve reset
```

### Web API

| エンドポイント | メソッド | 説明 |
|---------------|---------|------|
| `/api/health` | GET | ヘルスチェック (選択中ローカルbackend到達性/ディスク/メモリ/モジュール状態) |
| `/api/status` | GET | システム状態 (選択中ローカルbackend到達性/TTS/STT/RAG/Vision/Monitor の接続状況)。Web status の payload には `model` / `stt_model` は含まれない（Discord の `/status` とは別） |

> `/api/status` は状態・到達性の公開用であり、選択モデル名や STT モデル名（`model` / `stt_model`）は返さない。モデル設定の確認は設定ファイルまたは Discord の `/status` を参照する（実機・live 状態の表示とは限らない）。
>
> `/api/health`・`/api/status` の backend 判定はハードコードされた "ollama" ではなく
> 起動時に解決した**選択中ローカルProvider** で行う。`provider_kind` と
> `provider_reachability` (`ok`/`error`/`unknown`/`unconfigured`) を返す。
> `unconfigured` は base URL 未解決でネットワークプローブしなかった状態を指す
> （= 接続確認は行っていない）。後方互換として `ollama` / `local_provider` の boolean
> キーは `provider_reachability == "ok"` のときだけ `true` になる。URL・APIキーは応答に含めない。
> `providers` 配列の各 provider entry も到達性・状態の allowlist に限られ、モデル識別子
> (`model`) は含めない。
| `/api/tts` | POST | テキスト → WAV 音声合成 (`{"text": "..."}`) |
| `/api/tts/voice` | POST | TTS ボイス変更 (`{"voice": "jm_kumo"}`) |
| `/api/stt` | POST | 音声 → テキスト変換 (`{"audio": "base64..."}`). WAV形式。`SENSOR_MICROPHONE_ENABLED=true` が無ければ 403 (fail closed) |
| `/api/vision/status` | GET | 映像入力の状態 (allowlist: 稼働/在席/感情検出などの bool のみ。カメラ情報・感情ラベル・解析カウントは含めない) |
| `/api/vision/snapshot` | GET | **廃止** — 生カメラ画像 (JPEG) は未認証公開しない。固定 404 でデータは返さない |
| `/api/vision/context` | GET | **廃止** — デバッグ用映像コンテキストテキストは未認証公開しない。固定 404 でデータは返さない |
| `/api/screen/status` | GET | 画面認識の状態 (allowlist: bool / タイムスタンプ / ソース種別のみ。VLM 描写テキスト・モデル名は含めない) |
| `/api/screen/context` | GET | **廃止** — デバッグ用画面コンテキストテキストは未認証公開しない。固定 404 でデータは返さない |
| `/api/screen/ingest` | POST | メインPCからの画面 push (生 JPEG)。`SENSOR_SCREEN_INGEST_ENABLED` **と** `SCREEN_INGEST_TOKEN` の両方が無ければ 403。生 JPEG は保存せず VLM で 1 回描写 |
| `/api/monitor/status` | GET | PCモニター状態 (allowlist: bool / タイムスタンプ / source=固定 `"monitor"` のみ。CPU/メモリ/GPU/ディスク等のメトリクス集計値・プロセス数・レコード数・DB パス・エラーは含めない) |
| `/api/monitor/context` | GET | **廃止** — デバッグ用PCモニターコンテキストテキストは未認証公開しない。固定 404 でデータは返さない |
| `/api/monitor/summary?minutes=60` | GET | **廃止** — 直近N分のメトリクスサマリーは未認証公開しない。固定 404 でデータは返さない |
| `/api/persona/status` | GET | パーソナライズ状態 (プロフィール/要約/プリロード) |
| `/api/persona/profile` | GET | ユーザープロフィール取得 |
| `/api/persona/profile` | POST | プロフィール更新 (`{"name": "...", "note": "..."}`) |
| `/api/persona/summaries?count=5` | GET | 直近の会話要約一覧 |
| `/api/persona/context` | GET | プリロードコンテキスト (デバッグ用) |
| `/api/idle/status` | GET | アイドル管理状態 (state/idle_seconds/gpu_count) |
| `/ws/chat` | WebSocket | ストリーミングチャット + 音声入力 (トークン単位) |

> **センサー status のプライバシー**: `/api/vision/status` `/api/screen/status`
> `/api/monitor/status` は opt-in に基づく bool / タイムスタンプ / ソース種別
> (Monitor は固定 `"monitor"`) のみを返す。カメラデバイスID・VLM モデル名・画面/
> 映像の描写テキスト・`last_error`・メトリクス集計値・プロセス数・レコード数・
> パスなど、生/派生情報は未認証の Web API では公開しない。VLM 描写・Monitor の
> コンテキストテキストはサーバー内 (ChatSession のシステムプロンプト注入) でのみ
> 使われ、ネットワーク越しには応答しない。`/api/vision/snapshot`
> `/api/vision/context` `/api/screen/context` `/api/monitor/context`
> `/api/monitor/summary` は廃止され、常に固定 404 を返す (データは一切返さない。
> 403 ではなく 404 を使うのは、認証・同意と区別するため)。`/api/status` の
> `vision_status` `monitor_status` も同じ allowlist に従う。remote (ingest) の
> `source` は latest.json の中身に関わらず常に固定 `"remote"`。

### WebSocket メッセージ仕様

#### クライアント → サーバー

| type | フィールド | 説明 |
|------|-----------|------|
| `message` | `text`, `session_id`, `tts` | テキストメッセージ送信 |
| `audio_message` | `data` (base64), `format` (webm/ogg/wav), `session_id`, `tts` | 音声メッセージ送信。`SENSOR_MICROPHONE_ENABLED=true` が無ければ固定 error のみ返し、デコード・STT は実行しない |

#### サーバー → クライアント

| type | フィールド | 説明 |
|------|-----------|------|
| `token` | `content` | ストリーミングトークン (1文字〜数文字) |
| `done` | `full_text` | 応答完了 + 全文テキスト |
| `audio` | `data` (base64 WAV) | TTS音声データ |
| `stt_result` | `text` | STT認識結果テキスト |
| `error` | `message` | エラーメッセージ |

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
python src/audio/main.py --microphone --no-rag

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

> ℹ️ 上記は現行CLIの実装挙動（バックグラウンドでカメラフレームを連続取得）。方針上のカメラ利用条件と
> 実装ギャップについては、後述の「Vision を無効にする場合」を参照。

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
# 音声対話 (Vision は既定で無効。--camera で明示的に有効化するまで動作しない)
python src/audio/main.py --microphone

# カメラデバイスを指定 (--camera で有効化した場合)
python src/audio/main.py --microphone --camera --camera-id 1
```

カメラが接続されていない場合は自動的にスキップされる（エラーにはならない）。

> ⚠️ **オプトイン強制 (P0-3)**: カメラ (Vision) は既定で無効で、明示的な同意
> (`--camera` フラグまたは `SENSOR_CAMERA_ENABLED=true`) があるときだけ有効化される。
> camera を配線済みの入口は Web (`SENSOR_CAMERA_ENABLED`) と Voice CLI
> (`--camera` / `SENSOR_CAMERA_ENABLED`) のみ（Discord / Desktop には未配線）。
> `--no-vision` は明示的な無効上書きとして残る（非推奨）。

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
# 音声対話 (Monitor は既定で無効。--monitor で明示的に有効化するまで動作しない)
python src/audio/main.py --microphone

# Web UI: SENSOR_MONITOR_ENABLED=false (既定) のまま
```

### データ保存先

- DB: `data/metrics/system_metrics.db` (SQLite, WALモード)
- 古いデータは 30日で自動クリーンアップ可能
- **プロセス詳細は既定で保存しない** (default redaction): 収集するのは集計値
  （プロセス件数 `process_count`）のみ。プロセス名・PID・CPUトップ5 などの詳細は
  `SENSOR_PROCESS_DETAILS_ENABLED=true` のときだけ収集・保存される
  （既定オフ / fail closed、共有 SensorPolicy で解決）。

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
python src/audio/main.py --microphone --no-persona
```

---

## 8. 常時稼働 — Service (Phase 8)

> ⚠️ 本節の systemd・Discord 運用は**将来の Ubuntu デプロイ時の手順（未検証）**。
> ユニット定義・スクリプトはリポジトリに実装済みだが、実機での稼働は未確認。
> 運用手順（停止・再起動・rollback・復元）は **[docs/runbook.md](docs/runbook.md)** を参照。

systemd で Web UI・音声対話をサービスとして管理。自動再起動・GPU省電力制御を統合。

Discord を操作コンソールにする場合は、`config/discord.env.example` を参考に
`config/discord.env` を作成し、`DISCORD_BOT_TOKEN` を設定する。

Discord 側の slash command:

| コマンド | 説明 |
|---------|------|
| `/ask` | LLM に質問し、チャンネル単位の会話履歴で応答 |
| `/tts` | テキストを設定済みTTSバックエンドで WAV 化して添付 |
| `/status` | 選択中ローカルbackendの到達性・モデル・ヘルスチェック状態を表示。Ollama 時のみ後方互換の `ollama (legacy)` 行も表示 |
| `/service` | `status` / `logs` / `health` / `gpu` / 許可時のみ `start` `stop` `restart` |
| `/reset` | そのチャンネルの会話履歴をリセット |
| `/diary` | 当日の日記を生成。プレビューまたは日記専用チャンネルへ投稿 |
| `/voice` | 通話チャンネルへ参加し、STTの開始・停止・状態確認を行う |

`DISCORD_AUTO_REPLY_CHANNEL_IDS` を設定したチャンネルでは、slash command なしで
通常投稿されたテキストすべてに LLM が返信する。未設定の場合は
`DISCORD_ALLOWED_CHANNEL_IDS` が自動返信対象として使われる。
この機能には Discord Developer Portal の Bot 設定で
**Message Content Intent** を有効化する必要がある。

### Discord 通話STT

`discord-ext-voice-recv` を使い、実行者が入っているDiscord通話チャンネルの
音声をユーザー別に受信して `faster-whisper` で文字起こしする。
結果は指定したテキストチャンネルへ投稿し、`DISCORD_VOICE_STT_SAVE_TRANSCRIPTS=true`
のときのみ `data/discord_voice/transcripts/YYYY-MM-DD.jsonl` へ保存する
（既定 `false`。`false` ではチャンネル投稿のみでディスク保存しない）。

通話 transcript 由来の **LLM 返信ターン**も、`DISCORD_TRAINING_LOG_ENABLED=true` に
加えて別の明示 opt-in `DISCORD_VOICE_TRAINING_LOG_ENABLED=true` が無い限り学習ログ
(`data/discord_training/conversations.jsonl`) へは保存されない（既定 `false` /
fail closed）。`DISCORD_VOICE_STT_SAVE_TRANSCRIPTS=false` の既定状態では、通話内容は
チャンネル投稿と音声読み上げのみで、STT 原文も学習ログもディスクへは残らない。
TTS 読み上げ・チャンネル投稿はこの opt-in に関係なく動作し、保存だけがスキップされる。

> ⚠️ **二重ゲート（既定オフ）**: 通話STTは安全側デフォルトで無効。開始には
> 次の両方が `true` であること**かつ**明示の `/voice start` が必要
> （どちらかが `false` / 未設定の間、`/voice join|start` は接続前に却下される）:
> 1. `DISCORD_VOICE_STT_ENABLED=true` — 機能自体の opt-in
> 2. `SENSOR_MICROPHONE_ENABLED=true` — 共有 SensorPolicy のマイク gate
>
> bot は自動では通話に入らない。`/voice start` の明示実行でのみ開始する。

```env
DISCORD_VOICE_STT_ENABLED=false
DISCORD_VOICE_TRANSCRIPT_CHANNEL_ID=123456789012345678
DISCORD_VOICE_TIMEZONE=Asia/Tokyo
DISCORD_VOICE_STT_LANGUAGE=ja
DISCORD_VOICE_STT_MODEL=auto
DISCORD_VOICE_STT_DEVICE=auto
DISCORD_VOICE_STT_COMPUTE_TYPE=auto
DISCORD_VOICE_STT_SAVE_TRANSCRIPTS=false
```

主なコマンド:

```text
/voice join
/voice start transcript_channel:#voice-log
/voice stop
/voice leave
/voice status
```

通話の文字起こしは参加者へ明示したうえで使う。botは自動では通話に入らず、
`DISCORD_VOICE_STT_ENABLED=true` **と** `SENSOR_MICROPHONE_ENABLED=true` の両方が
成り立つときだけ `/voice start` で開始する。

- `/voice join` は実行者がいる通話チャンネルへ参加し、既に別チャンネルに接続していれば
  そのチャンネルへ移動する（参加のみ。文字起こしは `/voice start` の明示が必要）。
- `/voice stop` は同意の撤回として、処理中・待機中の音声を **discard**（文字起こし・
  投稿しない）し、リスニング停止・キュー破棄・ワーカーの bounded join まで行う。
  ワーカーが停止に応答せず join タイムアウトした場合は ownership を保持し
  `stop_pending` を真実として公開し、確認死後にのみ解放する。`/voice leave` は
  stop に加えて通話チャンネルから退出する。
- **共通 transcript ゲート**: 通話 transcript の処理は、STT 停止・返信 revoke 後に
  遅れて届いたものを含めて、`on_message` の parsing 直後・全分岐の前に「voice STT が
  存在・listening かつ voice reply ゲートが active」の共通ゲートを適用する。さらに
  Discord メッセージの作成時刻が**現在の STT セッションの開始時刻以降**であることを
  要求するため、再起動・再開前の旧セッションのメッセージは受け付けない。ゲートを
  外れた transcript はタスク/カレンダー直接登録を含む一切の副作用を起こさない。
  通話由来の「タスク:」「予定〜入れて」直接登録ブランチは撤去済みで、受け入れた
  transcript は全てデバウンス → LLM 返信パイプラインのみを通る（テキストチャット側の
  直接登録は従来どおり維持）。
- **返信生成 revoke / 原子履歴コミット**: LLM 返信は `ask_voice_transcript` でセッション
  履歴から切り離して生成する（一時追加 → 必ず除去）。revoke されていない音声返信だけ、生成後に user+assistant をセッションの**インメモリ履歴**へ原子的にコミットするが、RAG と Growth は無効 (`store_memory=false`, `record_growth=false`)。この履歴コミットは学習ログや STT transcript のディスク保存とは別で、各々の明示 opt-in に従う。生成返却後にゲート世代を再チェックし、revoke 済み（gate 非アクティブまたは世代不一致）なら
  LLM・履歴コミット・返信・学習ログ・TTS・リアクションの副作用を一切行わない。
  `/voice stop|leave` は STT 停止より先に reply ゲートを revoke し、進行中返信を
  bounded cancel する。追跡済みの autoread タスクも revoke 時にキャンセルされ、再生中なら playback を停止する。
  TTS 読み上げ・チャンネル投稿はこの opt-in に関係なく動作し、アクティブな返信の
  インメモリ履歴コミットと、学習ログ / transcript のディスク保存はそれぞれ別ゲートで制御される。
- **VAD/STT ログは transcript-safe**: 認識テキストは既定のパイプライン診断・ログへ
  出力されない（STT は所要時間のみ、失敗は例外型名のみ）。認識結果はチャンネル投稿・
  LLM 経路のみへ渡り、ログ・status には残らない。
- Discord側は通常のBot権限に加えて、対象VCへの接続権限と発言権限、
  文字起こし投稿先への送信権限が必要。

#### 診断用デバッグ音声 (opt-in / 期限付き保持)

文字起こしがノイズに見えるとき、Whisper へ渡す前の 16kHz mono 音声を
`DISCORD_VOICE_STT_DEBUG_AUDIO_DIR` で一時保存して実音声を確認できる
（診断専用・**通常は未設定**。保存は best-effort で、失敗しても STT を止めない）。

デバッグ WAV は有界な TTL で自動削除され、無制限に溜まらない:

- `DISCORD_VOICE_STT_DEBUG_AUDIO_TTL_SEC`（秒）— この秒数より古い WAV を
  各書き込みの**前と後**に best-effort 削除する。未設定時は **3600（1時間）** の
  conservative 既定値。
- 削除対象は設定ディレクトリ**直下の通常ファイル `*.wav` のみ**。サブディレクトリ
  への再帰・シンボリックリンクの追従は一切せず、他形式・ディレクトリ・
  シンボリックリンクは触れない。
- TTL が **0 / 負 / 非数値 / 上限超過**のときは fail closed で、デバッグ音声の
  保存自体が無効になる（ディレクトリ未設定と同様に何も書かない）。

```env
DISCORD_VOICE_STT_DEBUG_AUDIO_DIR=data/discord_voice/debug_audio
DISCORD_VOICE_STT_DEBUG_AUDIO_TTL_SEC=3600
```

デバッグ WAV のファイル名・パスはログやステータスへ一切出力しない。

### Discord 通話TTS

通話TTSは Kokoro と Style-Bert-VITS2 を切り替えられる。
Style-Bert-VITS2 は別プロセスの `subpc-sbv2-tts` が `127.0.0.1:50121` で受ける。

```env
DISCORD_VOICE_TTS_BACKEND=style_bert_vits2
DISCORD_VOICE_TTS_VOICE=tsukuyomi-chan
DISCORD_VOICE_TTS_SPEED=0.95
```

`DISCORD_VOICE_TTS_VOICE` は `jvnv-F1-jp` か `tsukuyomi-chan` を指定する。
SBV2サーバ側の `SBV2_MODEL_NAME` / `SBV2_MODEL_FILE` env と一致している必要がある。

Kokoroへ戻す場合は `DISCORD_VOICE_TTS_BACKEND=kokoro` とし、
`DISCORD_VOICE_TTS_VOICE=jf_alpha` など Kokoro の voice 名を指定する。

### Discord 返答ログの学習データ化

Discord bot の自動返信は `data/discord_training/conversations.jsonl` に、
👍/👎 は `feedback.jsonl` に、`修正: ...` 返信は
`training_candidates.jsonl` に保存される。モデル調整に使う場合は、
生ログをそのまま学習に入れず、明示的に修正した候補を優先する。

通話 transcript 由来の返答ターンは、通常の学習ログに加えて
`DISCORD_VOICE_TRAINING_LOG_ENABLED=true` を明示した場合のみ保存される
（既定 `false` / fail closed。テキストチャンネルの自動返信は従来どおり
`DISCORD_TRAINING_LOG_ENABLED=true` だけで保存される）。音声 opt-in が無い
間は通話由来ターンが学習ログへ入らないため、`修正: ...`・👍/👎 の
フィードバックも通話ターンに紐付かない。

STT返答チャンネルなど特定プロファイルだけを DPO 形式に出す例:

```bash
python scripts/export_discord_training.py \
  --format preference \
  --profile voice_short \
  --source discord_voice_transcript \
  --output data/discord_training/exports/voice_short_preference.jsonl
```

SFT形式に出す場合も、デフォルトでは `修正: ...` 済みの候補だけを使う。
👍済みの生返答も混ぜたい場合だけ `--include-positive-feedback` を付ける。

### 日次日記

Discord bot は毎日指定時刻に日記を生成し、通常会話・terminal とは別の
`DISCORD_DIARY_CHANNEL_ID` へ投稿できる。

```env
DIARY_ENABLED=true
DISCORD_DIARY_CHANNEL_ID=123456789012345678
DIARY_POST_TIME=23:50
DIARY_TIMEZONE=Asia/Tokyo
DIARY_PERSONALIZATION_ENABLED=true
DIARY_PERSONALIZATION_MIN_CONFIDENCE=0.72

# Google Calendar MCP を日記の予定ソースに使う場合
DIARY_CALENDAR_ENABLED=true
DIARY_CALENDAR_ID=primary
GOOGLE_OAUTH_CREDENTIALS=/home/haruka/.config/google-calendar-mcp/gcp-oauth.keys.json
```

日記生成に使う材料:

- Google Calendar MCP の当日予定
- `data/discord_training/conversations.jsonl` の当日会話
- `data/discord_voice/transcripts/YYYY-MM-DD.jsonl` の当日通話文字起こし
- `data/profile/summaries/summary_*.json` の直近要約
- `data/metrics/system_metrics.db` の当日PCメトリクス
- `data/profile/user_profile.json` のプロフィールと手動スケジュール

`DIARY_PERSONALIZATION_ENABLED=true` の場合、日記投稿後に同じ日記から
プロフィール更新候補を抽出し、信頼度が
`DIARY_PERSONALIZATION_MIN_CONFIDENCE` 以上のものだけを
`data/profile/user_profile.json` に反映する。全ログを会話コンテキストへ
足し続けるのではなく、安定した嗜好・習慣・メモ・事実へ圧縮する。
各日の抽出結果と適用内容は `data/profile/personalization/YYYY-MM-DD.json`
に監査ログとして保存する。

Google Calendar MCP は OAuth JSON が無い、未認証、取得失敗の状態でも
日記生成全体は止めない。予定は空として扱い、エラーは
`data/diary/YYYY-MM-DD.json` に保存する。
自動投稿の重複防止は `data/diary/posted.json` で管理するため、手動で
Markdownを生成・保存しても、その日が未投稿なら指定時刻に投稿される。

初回の Google Calendar 認証:

```bash
mkdir -p ~/.config/google-calendar-mcp
# Google Cloud Console から Desktop app の OAuth JSON を取得し、以下に置く
# ~/.config/google-calendar-mcp/gcp-oauth.keys.json

GOOGLE_OAUTH_CREDENTIALS="$HOME/.config/google-calendar-mcp/gcp-oauth.keys.json" \
  npx -y @cocal/google-calendar-mcp auth
```

手動で日記を生成して確認:

```bash
# 保存せずプレビュー
.venv/bin/python -m src.diary.main --no-save

# Google Calendar なしで保存
.venv/bin/python -m src.diary.main --no-calendar

# 保存済み日記からプロフィール更新候補を抽出（監査ログのみ）
.venv/bin/python -m src.persona.personalize_daily --dry-run

# 保存済み日記からプロフィールへ反映
.venv/bin/python -m src.persona.personalize_daily

# Discord からは /diary post:false でプレビュー、post:true で日記チャンネルへ投稿
# /personalize dry_run:true で候補確認、dry_run:false で手動反映
```

`/service start|stop|restart` は `DISCORD_ALLOW_SERVICE_CONTROL=true` の場合のみ有効。
外部サーバーに bot を入れる場合は `DISCORD_ALLOWED_USER_IDS` か
`DISCORD_ALLOWED_CHANNEL_IDS` を設定して操作範囲を制限する。

### サービス管理 (service_ctl.sh)

```bash
# 全サービスの状態確認
bash scripts/service_ctl.sh status

# Web UI をサービスとして起動
bash scripts/service_ctl.sh start web

# 音声対話をサービスとして起動
bash scripts/service_ctl.sh start voice

# Discord 操作コンソールを起動
bash scripts/service_ctl.sh start discord

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
bash scripts/service_ctl.sh enable discord

# 自動起動を無効化
bash scripts/service_ctl.sh disable web
```

### service_ctl.sh コマンド一覧

| コマンド | 説明 |
|---------|------|
| `status` | 全サービスの状態を表示 |
| `start [web│voice│sbv2│discord│powerd│all]` | サービスを開始 |
| `stop [web│voice│sbv2│discord│powerd│all]` | サービスを停止 |
| `restart [web│voice│sbv2│discord│powerd│all]` | サービスを再起動 |
| `enable [web│voice│sbv2│discord│powerd│all]` | 自動起動を有効化 |
| `disable [web│voice│sbv2│discord│powerd│all]` | 自動起動を無効化 |
| `logs [web│voice│sbv2│discord│powerd] [-f]` | ログを表示 |
| `health` | ヘルスチェック実行 |
| `gpu` | GPU 情報表示 |

### GPU 省電力制御 (オプション)

nvidia-smi で GPU 電力制限を制御。常時稼働時のアイドル消費電力を抑える。

```bash
# 起動時に idle 電力へ設定する静的サービス (root 権限必要)
sudo cp scripts/systemd/subpc-gpu-powersave.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable subpc-gpu-powersave
sudo systemctl start subpc-gpu-powersave

# IdleManager から active / idle を動的切替する root デーモン (推奨)
sudo cp scripts/systemd/subpc-gpu-powerd@.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable subpc-gpu-powerd@$USER
sudo systemctl start subpc-gpu-powerd@$USER
```

| モード | 電力制限 | 用途 |
|--------|-----------|------|
| idle | 100W | アイドル時 (デフォルト) |
| active | 250W | LLM推論時 (P40 TDP) |

> 電力制限値 (100W / 250W) は P40 クラスの**例示プロファイルに基づく想定値（未検証）**。
> 導入する実際の GPU に応じて変更する。

### アイドル管理 (自動省電力)

`IdleManager` がユーザーの操作を追跡し、GPU 電力・Monitor 収集間隔・Vision 解析を自動で動的制御する。**既定では無効（opt-in）**で、`IDLE_MANAGER_ENABLED=true` を設定したときだけ有効化される。音声パイプライン（`src/audio/pipeline.py`）と Web UI サーバー（`src/web/server.py`）の両方がこの env を読み、`true` の場合のみ起動する。未設定・`false` のときはどちらも IdleManager を起動しない。

> ℹ️ `active` / `idle` の GPU 電力切替を実際に有効にするには、
> root 権限で `subpc-gpu-powerd@$USER` を起動する必要があります。
> 未導入でも Web UI / 音声対話は継続動作し、電力制御のみ無効になります。

#### 状態遷移

| 状態 | 条件 | GPU電力 | 動作 |
|------|------|---------|------|
| `active` | 対話中・操作あり | 250W (フルパワー) | 通常動作 |
| `idle` | 5分間無操作 | 100W (省電力) | GPU電力制限のみ |
| `deep_idle` | 30分間無操作 | 100W (省電力) | Monitor間隔延長 (120秒)、Vision解析一時停止 |

> 電力値 (250W / 100W) は前述の P40 クラス例示プロファイルに基づく想定値（未検証）。実機 GPU の電力設定に応じて変わる。

#### 自動復帰

以下のいずれかが発生すると即座に `active` に復帰:

- **音声パイプライン**: マイク入力検知 (VAD)、ウェイクワード検知
- **Web UI**: WebSocket メッセージ受信 (テキスト/音声)

LLM 推論開始時に GPU をアクティブモードに自動切替し、推論終了後にアクティビティタイマーをリセットする。

#### API

| エンドポイント | メソッド | 説明 |
|---------------|---------|------|
| `/api/idle/status` | GET | アイドル管理状態 (`state`, `idle_seconds`, `gpu_count` 等) |

`/api/health` と `/api/status` にも `idle_manager` フィールドが含まれる。

### systemd サービス一覧

| サービス名 | 種類 | 説明 |
|-------------|------|------|
| `subpc-web` | ユーザー | Web UI サーバー (Type=notify, Watchdog付き) |
| `subpc-voice` | ユーザー | 音声対話パイプライン |
| `subpc-discord` | ユーザー | Discord 操作コンソール |
| `subpc-gpu-powersave` | システム | GPU 省電力制御 (起動時に idle を適用, oneshot, 要sudo) |
| `subpc-gpu-powerd@$USER` | システム | GPU 動的電力制御デーモン (`IdleManager` 用, 要sudo) |

---

## 9. GPU換装 (Phase 9)

> ⚠️ **未検証** — GPU 換装・モデル切替は**将来のデプロイ時の手順**。想定GPU構成の実装は
> リポジトリに存在するが、実機での GPU 検出・稼働は未確認。
>
> 下記のプロファイル名（`p40` / `dual_gpu` / `gtx1060` / `cpu`）と、P40 / P5000 /
> RTX 2070 Super / 750W 電源・推奨モデル（14B / 27B Q4 等）の組み合わせは、
> **リポジトリがサポートする代替プロファイルの例示（未検証）**であり、
> **現在選択中の導入構成ではありません**。実際の GPU は Phase 9 の自動検出で決定される。

### GPU 自動検出

Phase 9 では GPU を自動検出し、各モジュールの設定を最適化します。

```bash
# 現在の GPU 設定を確認
python3 -c "from src.service.gpu_config import main; main()"
```

P40構成の例では、Ollama は P40 のみを見せ、P5000 はPython推論用に空ける（代替プロファイルの例示）:

```bash
sudo install -D -m 0644 \
  scripts/systemd/ollama-gpu-p40.override.conf \
  /etc/systemd/system/ollama.service.d/10-gpu-p40.conf
sudo systemctl daemon-reload
sudo systemctl restart ollama.service
ollama ps
```

| GPU | Profile | STT | Embedding | Vision ONNX | LLM推奨 |
|-----|---------|-----|-----------|-------------|----------|
| P40 (24GB) | `p40` | cuda / int8 / medium | cpu | CUDAExecutionProvider | 14B Q4 |
| P40 + P5000 | `dual_gpu` | cuda:1 / int8 / medium | cpu | CUDAExecutionProvider(device_id=1) | 27B Q4 |
| P40 + RTX 2070S以上 | `dual_gpu` | cuda:1 / float16 / medium | cuda:1 | CUDAExecutionProvider(device_id=1) | 27B Q4 |
| GTX 1060 (6GB) | `gtx1060` | cpu / int8 / small | cpu | CPUExecutionProvider | 7B Q4 |
| GPUなし | `cpu` | cpu / int8 / small | cpu | CPUExecutionProvider | 7B Q4 |

> 上表はコードに実装済みのプロファイル（`p40` / `dual_gpu` / `gtx1060` / `cpu`）ごとに
> GPU・モデルの組み合わせを例示した**代替構成**である。いずれも「現在の対象」ではなく、
> 実機の GPU 検出結果や導入方針に応じて選択する。

### LLM モデルの変更

（例）P40 クラスの GPU 構成では、大型モデルへの切り替えが可能です:

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

### P40 換装手順（デプロイ時の草案・未検証・例示構成）

> 下記は想定される導入構成の**一例（代替プロファイルの例示・未検証）**であり、
> **選択・確定した構成ではない**。750W 電源や P5000 / RTX 2070 Super の追加搭載も
> あくまで例示であり、実機の要件・予算に応じて変わる。

1. P40 を物理的に取り付け
2. 電源 750W 換装（想定・実機未確認）
3. BIOS で iGPU を映像出力に設定 (P40 は映像出力なし)
4. Quadro P5000 または RTX 2070 Super 以上を追加搭載 (推論専用、GPU 1)（想定構成）
5. Ubuntu 起動後 `nvidia-smi` で両GPU認識確認（★未検証）
6. `config/chat_config.json` の model を 14b に変更

---

## 10. ウェイクワード検知 (Phase 10)

特定の呼びかけ（「Hey Jarvis」等）を検知して音声対話モードを自動起動する。デプロイ後は低消費電力でウェイクワード待機する想定（実機未検証）。

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
# デフォルト (hey_jarvis, 閾値 0.5)。マイクの同意も必要
python src/audio/main.py --microphone --wakeword

# 閾値を低くして感度を上げる
python src/audio/main.py --microphone --wakeword --wakeword-threshold 0.3

# Alexa モデルを使用
python src/audio/main.py --microphone --wakeword --wakeword-model alexa
```

### ウェイクワードなしで起動 (従来通り)

```bash
python src/audio/main.py --microphone
```

`--wakeword` を指定しない場合は従来通りの即時VADリスニングモード
（マイクの同意 `--microphone` は引き続き必須）。

---

> ⚠️ **歴史的スナップショット**: 以下の Phase 11〜12 の節は実装時点の記録であり、現在の
> パッケージ・設定と一致しない可能性がある。最新の機能・状態の把握には
> **[readme.md](readme.md)**（要件定義・フェーズ一覧）と **[docs/README.md](docs/README.md)**（ドキュメント索引）を参照。

## 11. エネ化 — 自発発話・画面認識・感情TTS (Phase 11)

AIを「PCに住み着いた相棒」に近づける3機能。いずれもデフォルト無効 (感情タグのみ `chat_config.json` で有効化済み)。

### 11.1 Discord 自発発話 (Proactive → Discord)

音声パイプライン専用だった `ProactiveEngine` (予定リマインド / 休憩提案 / 挨拶 / PC異常) を Discord に配線。トリガー発火時、定型文をLLMでペルソナ口調に言い換えてテキストチャンネルへ投稿し、通話接続中なら autoread でも読み上げる。

プロフィール学習向けの会話開始も有効化できる。会話がない時間が続くと、好み・生活リズム・望む接し方などを一問だけ尋ねる。次の返答は質問文付きで `conversations.jsonl` に保存されるため、「ジャズかな」のような短い返答でも日記パーソナライズ時に文脈を失わない。質問への返答は通常のbot返答と同じく 👍/👎・修正候補の対象になる。

```bash
# config/discord.env
DISCORD_PROACTIVE_ENABLED=true
# 空なら DISCORD_AUTO_REPLY_CHANNEL_IDS の先頭を使う
DISCORD_PROACTIVE_CHANNEL_ID=<投稿先チャンネルID>
DISCORD_PROACTIVE_LLM_REWRITE=true   # ペルソナ口調への言い換え (失敗時は定型文)
DISCORD_PROACTIVE_CHECK_INTERVAL=60  # チェック間隔 (秒)

DISCORD_PROACTIVE_CHAT_ENABLED=true
DISCORD_PROACTIVE_CHAT_INTERVAL_HOURS=6
DISCORD_PROACTIVE_CHAT_IDLE_MINUTES=60
DISCORD_PROACTIVE_CHAT_REPLY_TIMEOUT_HOURS=12
DISCORD_PROACTIVE_CHAT_DAILY_LIMIT=1
DISCORD_PROACTIVE_CHAT_MAX_BACKOFF_HOURS=72
DISCORD_PROACTIVE_CHAT_AUTOREAD=false
DISCORD_PROACTIVE_QUIET_HOURS=1-9
```

自発会話は1日1回を既定上限とし、返信がなければ次回間隔を最大72時間まで
段階的に広げる。送信済み質問とスヌーズは
`data/proactive/conversation_state.json` に保存され、サービス再起動後も維持される。
保留中の質問へ「あとで」と返すと3時間スヌーズ、「今日は静かに」で
翌日までスヌーズする。「自発会話を停止」と「自発会話を再開」は永続する。
ユーザーの返答本文はこの制御状態には保存しない。

`CHAT_INTERVAL_HOURS` は自発質問同士の最短間隔、`CHAT_IDLE_MINUTES` は最後の許可ユーザー発言から質問までの待機時間。`QUIET_HOURS` は日跨ぎに対応し、上例では 1:00 以上 9:00 未満に雑談を開始しない。予定・警告など他の通知はこの雑談用 quiet hours の対象外。

実装: `src/discord_bot/proactive_bridge.py`。許可ユーザーの発言で休憩・自発質問の待機時間がリセットされる。

### 11.2 画面認識 (Screen Context)

スクリーンをキャプチャし、vision対応チャットモデル (gemma4:26b) で「ユーザーが何をしているか」を1〜2文で描写してシステムプロンプトに注入する。X11 前提 (DISPLAY 必須)。解析間隔90秒、描写が10分より古い場合は注入しない。

```bash
# 音声パイプライン (screen_capture の opt-in。マイク同意も必要)
python src/audio/main.py --microphone --screen

# Web UI (env)
SENSOR_SCREEN_CAPTURE_ENABLED=true    # /api/screen/status で状態確認

# Discord bot (config/discord.env)
SENSOR_SCREEN_CAPTURE_ENABLED=true
```

> 画面認識 (screen_capture) は共有 SensorPolicy の canonical 名 `SENSOR_SCREEN_CAPTURE_ENABLED`
> で有効化する (既定オフ)。Discord の legacy `DISCORD_SCREEN_CONTEXT_ENABLED` は非推奨の
> 互換 alias で、canonical が設定されていればその値が確定 (false は legacy の true を上書き)、
> canonical 未設定のときだけ参照される。Web の legacy `WEB_SCREEN_CONTEXT_ENABLED` は
> Discord を有効化しない。

実装: `src/screen/` (capture=mss+Pillow / describer=Ollama VLM / context=バックグラウンドスレッド)。systemd サービスには `Environment=DISPLAY=:0` を追加済み。X接続に失敗した場合は画面情報なしで続行する (クラッシュしない)。
> 画面認識 (ScreenDescriber) は **Ollama `/api/chat` 前提**のため Ollama backend でのみ有効。
> openai_compatible backend では作成せずスキップされる。

#### remoteモード (メインPCの画面を見る)

普段使うPCがサブPCと別の場合は remote モードを使う。メインPC側の軽量エージェントがスクショをpushし、サブPCのWebサーバーがVLMで1回だけ描写して `data/screen/latest.json` に保存。Discord/Web/音声の全プロセスはそのファイルを読むだけ (VLM呼び出しの重複なし)。メインPCからの push が10分途絶えると自動的にコンテキスト注入が止まる。

```
[メインPC] scripts/screen_agent.py ──POST /api/screen/ingest──▶ [サブPC web]
   (mss+Pillow+httpx のみ、90秒毎、             │ policy+token 認証 → 生JPEGは保存せず
    画面が変わらなければ送信スキップ)           │ VLMで1回描写 → latest.json のみ保持
                                                ▼
        Discord bot / Web / 音声パイプライン: RemoteScreenContext が latest.json を読取
```

> **生JPEGは永続化しない**: 受信した raw JPEG は保存せず、VLM で 1 回描写した結果
> (`description` / 取得時刻) だけが `data/screen/latest.json` に残る。レガシーの
> `latest.jpg` は起動・停止・無効状態で best-effort 削除される（**絶対の削除保証では
> ない**。削除失敗時はパス・エラーを外部へ露出せず黙って無視し、新規の保存を行わない
> 運用のみが保証対象）。

サブPC側 (`config/web.env` — subpc-web.service が読む):
```bash
SENSOR_SCREEN_CAPTURE_ENABLED=true
SENSOR_SCREEN_INGEST_ENABLED=true     # ingest 受信の opt-in (token と両方が必須)
SCREEN_CONTEXT_MODE=remote
SCREEN_INGEST_TOKEN=<openssl rand -hex 24 などで生成>
```
Discord側 (`config/discord.env`): `SENSOR_SCREEN_CAPTURE_ENABLED=true` + `SCREEN_CONTEXT_MODE=remote`

> `/api/screen/ingest` は `SENSOR_SCREEN_INGEST_ENABLED=true` **と** `SCREEN_INGEST_TOKEN` の
> 両方が無ければ 403 (安全側デフォルト)。token 単独では有効化しない。token は共有鍵の
> ため、`SCREEN_INGEST_TOKEN` は git 管理外の env にのみ置く。

メインPC (Windows) 側:
```powershell
pip install mss pillow httpx
# スクリプト入手 (サブPCのWeb UIが配信): http://<サブPC>:8000/static/screen_agent.py
python screen_agent.py --enable-screen-capture --url http://<サブPC>:8000 --token <同じトークン> --once   # 動作確認
python screen_agent.py --enable-screen-capture --url http://<サブPC>:8000 --token <同じトークン>          # 常駐
```
画面キャプチャは既定オフで、`--enable-screen-capture` または
`SENSOR_SCREEN_CAPTURE_ENABLED=true` の明示的な source opt-in が必要。これは
`--token`、`--once`、`--url` の指定とは独立しており、それらだけではキャプチャも送信も
開始しない。`--once` は送信回数、`--url` は送信先、token は receiver の認証だけを決める。
配布される `scripts/screen_agent.py` と `/static/screen_agent.py` の2コピーは byte-identical。
診断は固定文言または型名のみで、URL・トークン・画像/本文内容を含めない。
自動起動はタスクスケジューラで「ログオン時」トリガー + `pythonw.exe screen_agent.py ...` (コンソール非表示)。

確認: サブPCで `curl http://localhost:8000/api/screen/status` — `ingest.available` と `ingest.age_seconds` / `ingest.source` が出ていればOK。VLM 描写テキスト (`description`) は未認証公開しないため、`ingest` には含まれない (描写は `data/screen/latest.json` にのみ保持)。

> ⚠️ **status の限界**: `/api/screen/status` や各センサー status が返すのは opt-in の
> boolean と privacy-safe な状態 (出所・取得時刻・保存有無) のみであり、実センサー・
> 実モデル・実機での動作を検証したことを意味しない。live 検証 (実カメラ・実X11画面・
> 実マイク・実メインPC push) は未実施。enabled 表示は「リポジトリ配線上の opt-in」
> であり「センサーが実際に動いている」ことの保証ではない。

### Desktop (活動収集の停止は資源解放)

Desktop アプリ (`src/desktop/bridge.py`) も `create_activity_runtime_from_env` で
活動収集 (activity) を opt-in 起動する (`SENSOR_ACTIVITY_ENABLED=true` / legacy
`COMPANION_ACTIVITY_ENABLED=true`)。オーバーレイ無効化・停止 (stop) は companion を
無効化し、収集タイマーを止めて activity runtime を確実に手放す (資源解放)。UI 上の
非表示・最小化・無効表示だけではセンサー収集は止まらない。

ChatPage の push-to-talk マイクボタン (`bridge.py` の `startRecording`) は共有
SensorPolicy のマイク gate (`SENSOR_MICROPHONE_ENABLED=true`) が無いと録音を開始
しない（無効時は通知のみで recorder を起動しない）。

### 11.3 感情連動TTS (Emotion Tags)

LLMが応答冒頭に `[emo:happy]` 形式のタグを出力し、Style-Bert-VITS2 のスタイルを発話ごとに切り替える。タグは全経路 (履歴 / RAG / トレーニングログ / 画面表示) で除去され、ユーザーには見えない。

- 有効化: `config/chat_config.json` の `"emotion_tag_enabled": true`
- 感情: happy / sad / angry / surprise / fear / disgust / neutral → SBV2スタイル (Happy 等) に1:1マップ
- `jvnv-F1-jp` は7スタイル対応。`tsukuyomi-chan` は常に Neutral。kokoro はスタイル指定を無視
- タグ無し・不正タグは neutral 扱い

実装: `src/chat/emotion.py` (パース / ストリーミングフィルタ / スタイルマップ)。

---

## 12. タスク管理 + エスカレーション催促 (Phase 12)

AIにタスク管理を任せる機能。SQLite (`data/tasks/tasks.db`, WAL) にタスクを保存し、期限に応じて段階的に厳しくなる催促を Discord に自発送信する。未完了タスク (最大8件) はLLMのシステムプロンプトに常時注入され、会話の中で自然に参照・詰められる。

### 登録方法 (5通り)
1. **右クリック**: 任意のメッセージを右クリック (長押し) → アプリ → 「タスクに登録」→ 本文が入力済みのモーダルで期限だけ書いて送信
2. **常設タスクボード**: ピン留めされたボードの【＋追加】ボタン → モーダル入力。ボードでは未完了一覧 (期限順・超過明示・最大15件) と優先順位オーケストレーターの推奨1件を確認でき、【🎯 今やる】で固定、Selectメニューから完了/スヌーズ/削除もできる。ボタンは永続化済みで再起動後も有効。env: `TASKS_BOARD_ENABLED` (default true) / `DISCORD_TASK_BOARD_CHANNEL_ID` (未設定ならリマインド先と同じ)
3. **slash command**: `/task add <title> [due] [priority] [note]` — due は「明日」「7/10」「7/10 15:00」に対応。ほか `/task list` `/task done <id>` `/task snooze <id> <30m|2h|明日>` `/task del <id>`
4. **明示プレフィックス**: 「タスク: レポート提出」とテキストチャットで発言すると確認なしで即登録。
   通話transcript からのタスク/カレンダー直接登録は撤去済みで、通話transcript は
   LLM 返信パイプラインのみを通る。テキストチャット側の直接登録は従来どおり維持される。
5. **自然会話から**: 「明日までにレポート出さないと」と話すと、返信後に非同期でLLMが抽出し「登録する/無視」ボタンで確認 (黙って自動登録はしない)。相対日付はプロンプト内の計算済み換算表で絶対日時化

### エスカレーション催促
期限24h前 (1回) → 3h前 (1回) → 1h前 (30分毎) → 超過 (2時間毎)。催促はペルソナ口調に言い換えられ、[完了][+30分][+2時間] ボタン付きで届く。通話中なら読み上げも。

配送契約は **at-least-once best-effort** であり、exactly-once を保証しない
（詳細: [docs/decisions/task_delivery_consistency.md](docs/decisions/task_delivery_consistency.md)）。
通知状態は `task_notifications` テーブルへ永続化され、`tasks` テーブルの
`rev` をガードに使った `tasks.rev` 楽観制御・`BEGIN IMMEDIATE` による lease
claim・コールバック直前の revalidate・`expected_rev` 条件付き record により、並行の
done / drop / update / snooze を上書きせず再通知を殺さない。ただし**再起動・クラッシュ時に
「重複しない」とは限らない**: 通知送信後に record される前でクラッシュすると再送され得る
（crash callback-before-record redelivery）。residual: revalidate〜コールバック間の
micro-TOCTOU（二重送信は受け入れ・頻度は低い）、lease owner 名は並走エンジン間で一意でなければ
ならない規律、durable outbox なし（外部呼び出しは DB トランザクションにできない）。
実機・実 Discord 常駐での live 検証は未実施（deployed / verified は未主張）。

```bash
# config/discord.env
TASKS_REMINDER_ENABLED=true   # default true
TASKS_QUIET_HOURS=1-8         # この時間帯 (ローカル) は超過以外の催促を抑制
```

送信先は `DISCORD_PROACTIVE_CHANNEL_ID` (未設定なら auto-reply チャンネルの先頭)。

実装: `src/tasks/` (store / reminder) + `src/discord_bot/task_ui.py`。音声パイプラインもタスク一覧を読み取り専用で参照する。

### 優先順位の外注化 (`/focus`)

登録済みタスクを、期限・明示優先度・滞留期間・次の一手の有無から決定的に採点し、
「今やる1件」だけを理由付きで返す。いったん選んだ1件は、新しい緊急タスクが増えても
勝手に切り替えず、完了または明示的な見送りまで固定する。Google Calendar の次の時刻付き
予定も読み、予定の10分前を空けた最大25分の作業枠を提示する。

- `/focus now`: 今やる1件と根拠を表示し、未選択なら固定
- `/focus start`: 開始時刻を記録
- `/focus done`: 完了し、今日の完了数/連続日数を更新して次を自動選定
- `/focus next`: 現候補を既定2時間見送り、次を選定
- `/focus pick <task_id>`: 人間の判断で上書き
- `/focus status`: 委任した決定数、現在の固定、継続記録を表示

優先順位コンテキストはDiscordだけでなく、`ChatSession` を使うWeb・音声にも既存の
タスクコンテキスト経由で注入される。永続状態 `data/tasks/priority_state.json` には本文を
複製せず、タスクID・時刻・見送り回数・日別完了数だけを保存する。

```bash
PRIORITY_ENABLED=true
PRIORITY_SKIP_HOURS=2
PRIORITY_CALENDAR_BUFFER_MIN=10
```

設計と採点規則: `docs/designs/priority_orchestration.md`。

### Google Calendar 連携 (MCP)

`@cocal/google-calendar-mcp` (stdio MCP) 経由の双方向同期。

- **タスク→カレンダー**: 期限付きタスクを登録すると `📋 タイトル` のイベントを自動作成 (時刻あり=期限30分前〜期限、日付のみ=終日)。完了で `✅` に更新、削除でイベント削除。バックグラウンドワーカー処理でDiscord応答をブロックしない。マッピングは tasks テーブルの `calendar_event_id`
- **カレンダー→bot**: `CALENDAR_SYNC_INTERVAL_MIN` (default 20分) ごとに設定済み取得範囲
  （既定 `past_days=14` / `days_ahead=45`、`CALENDAR_PULL_PAST_DAYS` /
  `CALENDAR_SYNC_DAYS_AHEAD` で調整可能）の予定を取得し、`data/calendar/upcoming.json`
  へ保存 (全プロセス共有)。LLMコンテキストに「予定 (Google Calendar)」ブロックとして注入。
  タスク由来イベント (`subpc-task:` マーカー) は再輸入しない

> **配送一貫性**: タスク⇔カレンダー同期は **state-driven** で、queue イベントのラベル
> (add/update/done) を信頼せず常に現在のタスク状態（open / done / dropped、due 有無）から
> 行動を決める。crash・queue-drop 後の対応付けは pull 側のマーカー照合/重複整理が回復し、
> 重複マーカーは正準1件へ決定的に収束する。契約は at-least-once best-effort
> （[docs/decisions/task_delivery_consistency.md](docs/decisions/task_delivery_consistency.md)）。
> residual: `TaskCalendarSync.enqueue` は `queue.Full` で drop する（durable outbox なし）、
> ワーカーポーリング・pull 間隔による反映遅延（queue/pull latency）、外部カレンダー API は
> DB と原子的にできない（DB 先行コミット・失敗はリトライ/pull 回復）。実機・実 Google Calendar
> での live 検証は未実施（deployed / verified は未主張）。
- **予定リマインド**: プロアクティブ発話が有効 (`DISCORD_PROACTIVE_ENABLED=true`) の場合、当日の予定が UserProfile.schedule に同期され、既存の15分前リマインドが発火する

```bash
# config/discord.env
TASKS_CALENDAR_SYNC_ENABLED=true
# TASKS_CALENDAR_ID=primary          # default: DIARY_CALENDAR_ID → primary
# CALENDAR_SYNC_INTERVAL_MIN=20
```

初回セットアップ: GCPでOAuthクライアント作成 → `~/.config/google-calendar-mcp/gcp-oauth.keys.json` に配置 → Calendar API を有効化 → テストユーザーに自分を追加 → `GOOGLE_OAUTH_CREDENTIALS=$HOME/.config/google-calendar-mcp/gcp-oauth.keys.json npx -y @cocal/google-calendar-mcp auth`

注: `src/integrations/mcp_stdio.py` は改行区切りJSONで通信する (2026-07-04 修正。以前はLSP風Content-Lengthフレーミングでサーバーと噛み合わず全MCP呼び出しがタイムアウトしていた — 日記のカレンダー連携が動かなかった根本原因)。

### 通話STT分断対策 (同時実装)
考えながら話すと一文が断片化して断片ごとに返信が来ていた問題への対策:
- `DISCORD_VOICE_STT_SILENCE_MS` を 700→1000ms に変更 (実config)
- `DISCORD_VOICE_REPLY_DEBOUNCE_MS` (default 3000): 断片を待って結合し、発話が落ち着いてから1回だけ返信。最初の断片から `DISCORD_VOICE_REPLY_DEBOUNCE_MAX_MS` (default 10000) で強制発火。`0` で従来挙動

---

## 13. ログ管理

アプリログ・サービスログ・会話履歴を一元管理する。

### アプリのログ出力
各サービス (subpc-web / subpc-discord / subpc-sbv2-tts) は Python logging に統一済み。stdout (journald) と `logs/<service>.log` (5MB×3世代ローテーション) の両方に出力する。レベルは環境変数 `LOG_LEVEL` (default INFO)。共通実装は `src/service/log_setup.py` (SBV2サーバーは別venvのためスクリプト内に同等実装)。

### 会話履歴
Webチャットは応答完了ごとに `data/chat_history/session_*.json` へ自動保存される (以前はメモリのみで再起動で消えていた)。`HISTORY_MAX_FILES` (default 200) を超えた古いファイルは自動削除。実装: `src/chat/history_admin.py`。

### Web UI ログビューア (`/logs`)
- **サービス**: journalctl のログを閲覧 (unit は `subpc-web` / `subpc-discord` / `subpc-sbv2-tts` / `subpc-gpu-powersave` のホワイトリスト制、最大1000行)
- **アプリログ**: `logs/*.log` の一覧と末尾表示
- **会話ログ**: 履歴の一覧 (プレビュー・ターン数・サイズ)、メッセージ閲覧、削除

API: `GET /api/logs/journal?unit=&lines=` / `GET /api/logs/files` / `GET /api/logs/files/{name}` / `GET,DELETE /api/history/sessions[/{file}]`

---

## 設定ファイル

### config/chat_config.json

| キー | デフォルト | 説明 |
|------|-----------|------|
| `ollama_base_url` | `http://localhost:11434` | Ollama API の URL |
| `model` | `qwen2.5:7b-instruct-q4_K_M` | 使用する LLM モデル |
| `local_provider_kind` | `ollama` | ローカルbackend種別: `ollama`（既定）/ `openai_compatible`（llama.cpp / LM Studio / vLLM 等） |
| `local_base_url` | `""`（空なら `http://localhost:8080/v1`） | openai_compatible の送信先。**loopback 限定**（`localhost` / loopback IP のみ） |
| `local_provider_id` | `""`（空なら kind 別既定） | Registry キー・エラーID。openai_compatible 既定は `local-openai` |
| `local_api_key_env` | `""` | APIキーの**環境変数名のみ**。キー値は保存しない。空なら keyless |
| `temperature` | `0.7` | 生成のランダム度 (0.0〜1.0) |
| `top_p` | `0.9` | Nucleus sampling |
| `top_k` | `40` | Top-K sampling |
| `num_ctx` | `4096` | コンテキスト長 (トークン数)。openai_compatible では無視 |
| `repeat_penalty` | `1.1` | 繰り返しペナルティ |
| `system_prompt` | *(日本語プロンプト)* | AI の振る舞い指示 |
| `max_history_turns` | `20` | 保持する会話ターン上限 |
| `history_dir` | `data/chat_history` | 会話履歴の保存先 |
| `stream` | `true` | ストリーミング出力 |

> ローカルbackend切り替えの手順は「[ローカル推論 backend の切り替え (P0-2)](#ローカル推論-backend-の切り替え-p0-2)」を参照。

---

> ⚠️ **歴史的スナップショット**: 以下のディレクトリ構成は過去のある時点のもので、現在の
> パッケージ構成と一致しない可能性がある。最新のナビゲーション・状態は
> **[readme.md](readme.md)** と **[docs/README.md](docs/README.md)** を参照。

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
│   │   ├── idle.py            # アイドル管理コントローラー (GPU電力/Monitor/Vision自動制御)
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

## 適応成長の確認

Web会話画面上部の「適応成長」に、会話例・検索可能記憶・評価・修正候補・個人化事実の蓄積量が
Growth Pointsとして表示される。これはモデル重みや知能指数ではない。点数とレベルの定義、保存内容は
[`docs/designs/adaptive_growth.md`](docs/designs/adaptive_growth.md) を参照。

```bash
curl -s http://127.0.0.1:8000/api/growth | python -m json.tool
```

---

## トラブルシューティング

> ⚠️ 下記の systemd / Ollama コマンドは**将来の Ubuntu デプロイ時の手順（未検証）**。
> リポジトリ内で動作確認する場合は `python src/...` 系コマンドを開発機で実行する。

### ローカル推論 backend に接続できない (Ollama 固有)

選択中 backend が Ollama のときだけ有効な手順（将来の Ubuntu デプロイ時前提・未検証）。
openai_compatible (llama.cpp / LM Studio / vLLM 等) の場合は、起動コマンドを
「[ローカル推論 backend の切り替え (P0-2)](#ローカル推論-backend-の切り替え-p0-2)」の
例示に従い自分で起動・確認する。

```bash
sudo systemctl start ollama
sudo systemctl status ollama
```

### モデルが見つからない (Ollama 固有)

選択中 backend が Ollama のときだけ有効な手順（デプロイ時前提・未検証）。
openai_compatible では `/models` が返すモデルIDを `config/chat_config.json` の
`model` に合わせる。

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
