# Phase 13: Personal LoRA Training

## 1. 目的

subpc_living で使用している Qwen3.6-35B-A3B を、個人用対話モデルとして調整する。

既存の長期記憶、人格プロンプト、日次パーソナライズ、タスク管理は維持しつつ、次の振る舞いをモデルの重みに定着させる。

- 春琴をモチーフにした端正で少し厳しい人格
- ユーザーとの適切な距離感
- 軽い雑談では短く返し、作業相談では次の一手を示す
- 過剰な肯定、説教、質問返し、冗長な説明を抑える
- 体調不良や不安など、厳しさを出すべきでない場面を判別する
- プリロードされた記憶、予定、タスクを自然に会話へ利用する

量子アルゴリズムや新規モデルの事前学習は対象外とする。まずは LoRA による教師あり学習を行い、修正データが十分に集まった後だけ DPO を追加する。

## 2. 現在の構成

### 推論環境

- ベースモデル: `Qwen/Qwen3.6-35B-A3B`
- ローカル推論: Ollama
- 現行タグ: `qwen3.6:35b-a3b-q4_K_M`
- ローカルGPU: Tesla P40をLLM専用として使用
- 補助GPU: P5000などをSTT、Embedding、Visionへ使用
- OS: Ubuntu 24.04 LTS

Ollamaは `scripts/systemd/ollama-gpu-p40.override.conf` により `CUDA_VISIBLE_DEVICES=0` へ固定されている。LoRA学習はローカルGPUで行わず、必要な期間だけH200を借りる。

### 既存の学習データ

`DiscordTrainingLog` が次のファイルを追記形式で保存している。

- `data/discord_training/conversations.jsonl`
  - 通常のユーザー発話とモデル応答
- `data/discord_training/feedback.jsonl`
  - Discordの👍/👎評価
- `data/discord_training/training_candidates.jsonl`
  - ユーザーが明示的に修正した回答
  - `preferred_output` が採用回答
  - `rejected_output` が元のモデル回答

`scripts/export_discord_training.py` は、これらからSFT形式と選好学習形式を出力できる。

## 3. データ作成

### SFTデータ

最初の学習では、明示的な修正回答だけを基本とする。高評価された通常回答は、内容を確認したうえで追加する。

```bash
source .venv/bin/activate

python scripts/export_discord_training.py \
  --format sft \
  --include-positive-feedback \
  --min-score 1 \
  --output data/finetune/sft.jsonl
```

SFTレコードはOpenAI Messages形式を使用する。

```json
{
  "messages": [
    {"role": "user", "content": "今日なにもやる気が出ない"},
    {"role": "assistant", "content": "なら、今日は一つだけ片づけなさい。全部やろうとするから動けんのです。"}
  ]
}
```

### DPOデータ

DPOはSFT版が安定した後に実施する。明示的な修正が存在するターンだけを使う。

```bash
python scripts/export_discord_training.py \
  --format preference \
  --output data/finetune/dpo.jsonl
```

```json
{
  "prompt": "今日は何もする気が起きない",
  "chosen": "なら、一つだけ片づけなさい。全部やろうとするから動けんのです。",
  "rejected": "やる気を出すためにタスクリストを作りましょう。"
}
```

### データ品質の原則

- 生の全会話を無条件に学習しない
- モデル自身の悪い回答をSFTの正解として使わない
- 👍だけでなく、内容を確認した回答を優先する
- 修正回答は短さ、人格、実用性のすべてを満たすように書く
- RAGで扱う個人的事実をモデル重みへ記憶させない
- 秘密情報、トークン、住所、連絡先、第三者の個人情報を除去する
- 感情タグなど、ユーザーに見せない制御情報は学習目的に応じて除去する

### system promptの扱い

現在の `training_candidates.jsonl` は完全な `messages` を保持しているが、エクスポーターはuser/assistantの単発会話へ作り直している。

人格を重みに定着させる `persona_sft` では、system promptを付けないか、ごく短い人格契約だけを付ける。現在の長大なsystem prompt全体を毎サンプルへ複製しない。

記憶やタスクの利用方法を学習する `context_sft` は別データとして扱い、短いsystem contextを含める。

## 4. H200での学習

### 基本方針

H200の141GB VRAMを利用し、4bit QLoRAではなくBF16 LoRAを行う。ベース重みは凍結し、LoRAアダプターだけを学習する。

最初の推奨設定:

```yaml
model_name: Qwen/Qwen3.6-35B-A3B
precision: bf16
max_sequence_length: 2048
micro_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 5.0e-5
epochs: 1
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
gradient_checkpointing: true
assistant_only_loss: true
```

データ量が少ない段階ではepoch数を増やすより、修正データの品質を上げる。学習途中のチェックポイントとサンプル応答を確認し、人格が強すぎる、一般能力が落ちる、返答が単調になる場合は早期に停止する。

### LoRA対象

最初の `persona-conservative` では共有層を中心にする。

- AttentionのQ/K/V/出力射影
- DeltaNetなどの共有射影
- 必要に応じて共有MLP
- MoEルーターは学習しない
- 256個のExpert本体は最初は学習しない
- Vision projectorは対象外

MoE ExpertへLoRAを追加すると、アダプターサイズと推論オーバーヘッドが増える。必要になった場合だけ別実験とし、推論用モデルではベース重みへマージする。

### 比較するアダプター

1. `persona-conservative`
   - rank 16
   - 共有Attentionと共有射影のみ
   - 元モデルの能力維持を優先
2. `persona-strong`
   - rank 32
   - 共有Linear層を広めに対象
   - 人格と回答傾向の定着を優先
3. `persona-dpo`
   - 採用したSFTアダプターから継続
   - chosen/rejectedペアを使用
   - SFTで満足できない選好だけを調整

## 5. H200利用前後の手順

### 借りる前にローカルで行うこと

1. SFT/DPO JSONLを出力する
2. 重複、空文字、秘密情報、不適切な応答を除去する
3. Qwenのchat templateを適用したトークン化を確認する
4. 小型モデルで1バッチのdry runを行う
5. 固定の確認用会話セットを準備する
6. 学習設定、出力先、保存間隔を確定する
7. H200環境で利用するDockerイメージと依存バージョンを固定する

個人データを外部GPUへ転送する場合は、必要な学習JSONLだけを送る。ChromaDB、日記、全会話履歴、認証情報は送らない。学習終了後はリモートストレージから削除する。

### H200上で行うこと

1. 公式BF16モデルを取得する
2. SFT LoRAを学習する
3. チェックポイントごとに固定会話を生成する
4. 最良のSFTアダプターを保存する
5. 必要な場合だけDPOを実行する
6. `adapter_config.json`、`adapter_model.safetensors`、学習設定、ベースモデルrevisionを保存する

### ローカルへ戻した後

1. ベースモデルとLoRAアダプターの組み合わせを確認する
2. ベースへマージしていないアダプターを永久保存する
3. コピー上でLoRAをBF16ベースへマージする
4. llama.cpp形式へ変換する
5. Q4_K_MまたはQ5_K_Mへ量子化する
6. Ollama用Modelfileを作成する
7. 新しいモデル名で登録する
8. `config/chat_config.json` の `model` だけを切り替える

モデル名の例:

- `qwen3.6:35b-a3b-base-q4_K_M`
- `qwen3.6:35b-a3b-shunkin-sft-q4_K_M`
- `qwen3.6:35b-a3b-shunkin-dpo-q4_K_M`

## 6. 確認とロールバック

大規模な自動評価は必須としない。普段使う会話を20〜30件ほど固定し、モデル名を隠して読み比べる。

確認項目:

- 人格がsystem promptなしでも自然に維持されるか
- 厳しさが人格否定や冷たい拒絶になっていないか
- 短い雑談に長文で返さないか
- 作業相談で実用的な次の一手を示すか
- 体調不良や不安の場面で不用意に「阿呆」を使わないか
- 記憶、予定、タスクを不自然に持ち出さないか
- 一般知識、コード、要約能力が明確に低下していないか
- 同じ言い回しを繰り返す人格崩壊が起きていないか

LoRAはベースモデルへ直接上書きしない。ベース、SFT、DPOを別名で保持し、`config/chat_config.json` のモデル名だけで即時に戻せるようにする。

## 7. Phase 13で追加する実装

今後リポジトリへ追加する候補:

- `training/validate_dataset.py`
  - スキーマ、重複、秘密情報、文字数を検査
- `training/train_sft.py`
  - H200用BF16 LoRA
- `training/train_dpo.py`
  - SFTアダプターからのDPO
- `training/merge_adapter.py`
  - アダプターをBF16ベースへマージ
- `training/configs/persona_conservative.yaml`
- `training/configs/persona_strong.yaml`
- `scripts/convert_personal_model_to_gguf.sh`
- `models/ollama/Modelfile.personal.example`
- `tests/test_training_dataset.py`

既存の会話、記憶、Discord、Web、音声処理は変更せず、学習とモデル変換を独立したオフライン工程として追加する。

## 8. 参考資料

- Qwen3.6: https://github.com/QwenLM/Qwen3.6
- Qwen3.6-35B-A3B: https://huggingface.co/Qwen/Qwen3.6-35B-A3B
- PEFT LoRA: https://huggingface.co/docs/peft/main/package_reference/lora
- TRL PEFT integration: https://huggingface.co/docs/trl/peft_integration
- LoRA paper: https://arxiv.org/abs/2106.09685
- DPO paper: https://arxiv.org/abs/2305.18290
