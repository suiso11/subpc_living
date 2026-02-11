# subpc_living — Usage

## 前提条件

- Ubuntu 24.04 LTS
- Phase 1〜3 のセットアップスクリプトを実行済み
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

# テキストモードで TTS テスト
python src/audio/main.py --text-mode --tts-voice jf_nezumi
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
| `/api/status` | GET | システム状態 (Ollama/TTS/RAG の接続状況) |
| `/api/tts` | POST | テキスト → WAV 音声合成 (`{"text": "..."}`) |
| `/api/tts/voice` | POST | TTS ボイス変更 (`{"voice": "jm_kumo"}`) |
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
│   └── vectordb/              # ChromaDB ベクトルDB (Phase 4)
├── models/
│   ├── stt/                   # Whisper モデルキャッシュ (自動DL)
│   └── tts/
│       └── kokoro/            # kokoro-onnx モデル
├── scripts/
│   ├── phase1_setup_nvidia.sh
│   ├── phase1_setup_ollama.sh
│   ├── phase1_verify.sh
│   ├── phase2_setup.sh
│   ├── phase2_verify.sh
│   ├── phase3_setup.sh
│   ├── phase3_verify.sh
│   ├── phase4_setup.sh
│   └── phase4_verify.sh
├── src/
│   ├── audio/                 # Phase 3: 音声対話
│   │   ├── main.py            # CLI エントリポイント
│   │   ├── pipeline.py        # VAD→STT→LLM→TTS パイプライン
│   │   ├── stt.py             # faster-whisper STT
│   │   ├── tts.py             # kokoro-onnx TTS
│   │   ├── vad.py             # VAD (Energy + Silero)
│   │   └── audio_io.py        # マイク入力・スピーカー出力
│   ├── chat/                  # Phase 2: テキスト対話
│   │   ├── main.py            # CLI エントリポイント
│   │   ├── client.py          # Ollama API クライアント
│   │   ├── session.py         # 会話セッション管理 + RAG統合
│   │   └── config.py          # 設定管理
│   ├── memory/                # Phase 4: 長期記憶
│   │   ├── embedding.py       # 埋め込みモデル (multilingual-e5-small)
│   │   ├── vectorstore.py     # ChromaDB ベクトルストア
│   │   └── rag.py             # RAG リトリーバー
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
