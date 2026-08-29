# Phase 13: Personal LoRA Training

> **状態**: active / supporting
> **位置付け**: 個人化LoRAの専門計画。ベースモデル（例示: Qwen3.6-35B-A3B）・外部学習GPU（例示: H200）は計画上の例示プロファイルであり、選定した目標・現行構成ではない（ローカルパイプラインは外部実行と分離）
> **対象範囲**: Personal LoRA Training（SFT / DPO・ローカルpreflight・マージ / GGUF配備）
> **作成日**: 2026-07-15
> **更新日**: 2026-08-28
> **日付根拠**: Git commit date

> **パスの基準**: 本ドキュメント中の説明・コマンド引数に現れる `training/...` パスは、すべてリポジトリルート基準（`<repo-root>/training/...`）である。

## 1. 目的

subpc_living のローカルLLMを、個人用対話モデルとして調整する。本計画ではベースモデル（例示: Qwen3.6-35B-A3B）・ローカルGPU構成（例示: P40 + P5000）を想定構成の**例**として扱う。これらはプロジェクトの目標・現行構成の選定ではない。

既存の長期記憶、人格プロンプト、日次パーソナライズ、タスク管理は維持しつつ、次の振る舞いをモデルの重みに定着させる。

- 春琴をモチーフにした端正で少し厳しい人格
- ユーザーとの適切な距離感
- 軽い雑談では短く返し、作業相談では次の一手を示す
- 過剰な肯定、説教、質問返し、冗長な説明を抑える
- 体調不良や不安など、厳しさを出すべきでない場面を判別する
- プリロードされた記憶、予定、タスクを自然に会話へ利用する

量子アルゴリズムや新規モデルの事前学習は対象外とする。まずは LoRA による教師あり学習を行い、修正データが十分に集まった後だけ DPO を追加する。

## 2. 想定構成（例示プロファイル）

### 推論環境（例示プロファイル）

> 以下は将来の構成検討に使う**例示プロファイル**であり、プロジェクトの目標・現行構成ではありません。実際のベースモデル・GPU・タグは演算子が決定し、決定後に本節へ反映してください。

- ベースモデル（例示）: `Qwen/Qwen3.6-35B-A3B`
- ローカル推論: Ollama
- Ollamaタグ（例示）: `qwen3.6:35b-a3b-q4_K_M`
- ローカルGPU（例示）: Tesla P40をLLM専用として使用
- 補助GPU（例示）: P5000などをSTT、Embedding、Visionへ使用
- OS: Ubuntu 24.04 LTS

Ollamaは `scripts/systemd/ollama-gpu-p40.override.conf`（`p40` は追跡overrideの例示プロファイル名）により `CUDA_VISIBLE_DEVICES=0` へ固定する想定である。LoRA学習はローカルGPUで行わず、必要な期間だけ外部学習GPU（例示: H200）を借りる。

> **注記**: 上記のベースモデル・Ollama・GPU構成は**計画上の例示プロファイル**であり、プロジェクトの目標・現行構成の記録ではない。このリポジトリは個人化モデル（学習済みアダプター、GGUF、Ollamaへ登録済みのpersonalタグ）の存在や実行実績を主張しない。それらはすべて§7のステータス表で **planned / unverified** である。

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

## 4. 外部学習GPU（例示: H200）での学習

> **状態**: 外部学習GPU（例示: H200）での実学習、成果物転送、マージ、GGUF変換、量子化、Ollama登録、評価、ロールバックはすべて**計画（planned / unverified）**である。このリポジトリで検証済みなのはローカルpreflight（データ検査・tokenize確認・dry-run）と各スクリプトの単体テストのみ。実学習・実モデル配備の実行実績は主張しない（§7のステータス表を参照）。

### 基本方針

例示の外部学習GPU（H200: 141GB VRAM）を利用し、4bit QLoRAではなくBF16 LoRAを行う。ベース重みは凍結し、LoRAアダプターだけを学習する。

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

## 5. 外部学習GPU（例示: H200）利用前後の手順

### 借りる前にローカルで行うこと

1. SFT/DPO JSONLを出力する
2. 重複、空文字、秘密情報、不適切な応答を除去する
3. Qwenのchat templateを適用したトークン化を確認する
4. 小型モデルで1バッチのdry runを行う
5. 固定の確認用会話セットを準備する
6. 学習設定、出力先、保存間隔を確定する
7. H200環境で利用するDockerイメージと依存バージョンを固定する

個人データを外部GPUへ転送する場合は、必要な学習JSONLだけを送る。ChromaDB、日記、全会話履歴、認証情報は送らない。学習終了後はリモートストレージから削除する。

### 外部学習GPU上で行うこと

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

> **状態**: 実モデルによる評価（`training/evaluate.py` のOllama呼び出し）と、`switch_chat_model.py` を使った実環境での切替・ロールバックは**計画（planned / unverified）**である。テストで検証済みなのはスタブ生成とJSON書換・復元のロジックのみ（§7）。

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

## 7. 実装済みパイプライン

### 解消した5つの障害

1. **データ品質と個人情報漏えい**
   - エクスポーターは既定で `metadata` を出力しない。ローカル監査時だけ `--include-metadata` を明示する。
   - `training/validate_dataset.py` がスキーマ、空欄、重複、文字数、秘密情報、メール、電話番号などを検査する。
   - `--clean-output` は問題行と重複行を除外し、metadataも削除した転送用JSONLを作る。検出値そのものはログへ表示しない。
   - SFTは20件、DPOは50件を設定値 `min_dataset_rows` として保持するが、これは設定検証（正の整数の確認）と実行前manifestへの記録のみである。**現在のランチャー（train_sft / train_dpo）はレコード数を数えて最低件数を強制しない**。「件数不足なら停止する」挙動は未実装なので、実データの件数は実行前に手動で確認する。
2. **Qwen3.6と学習ライブラリの不整合**
   - ベースrevisionを `995ad96eacd98c81ed38be0c5b274b04031597b0` に固定した。
   - `requirements-training.txt` は `qwen3_5_moe` を実装したTransformers 5.13.1を含む、検証可能な組み合わせへ固定した。
   - 公式chat templateにはassistantマスクがないため、`training/templates/qwen3_6_assistant.jinja` に `{% generation %}` 範囲を定義した。これにより `assistant_only_loss` が全トークン無効になる事故を防ぐ。
3. **MoE向け安全なSFT/DPO実行系がない**
   - `training/train_sft.py`、`training/train_dpo.py`、`training/runtime.py` を追加した。
   - conservative設定はAttentionとGatedDeltaNet射影だけ、strong設定も共有Expertだけを追加し、256 routed Experts、router、vision、embedding、lm_headを除外する。
   - BF16 LoRA以外、未固定revision、全層学習、量子化学習を設定検証で拒否し、実行前manifestを保存する。
4. **学習後の比較・マージ・配備手順がない**
   - `training/evaluate.py` と25件の固定会話、`training/merge_adapter.py`、`scripts/convert_personal_model_to_gguf.sh`、Ollama Modelfile例を追加した。
   - LoRAをBF16ベースのコピーへマージしてから、フルモデルGGUFへ変換・量子化する。ベースとadapterは上書きしない。
   - `merge_adapter.py` が実行時に検証するのは、adapterパスと `adapter_config.json` の存在・解釈、量子化adapter（QLoRA等）の拒否、ベースモデル名/revisionの一致、出力先の上書き禁止のみである。**`peft_type=LORA` の強制と、expert/router/vision/embed/lm_head/norm などの禁止対象moduleの拒否は実装していない**。禁止対象の除外は学習ランチャー側（train_sft / train_dpo のtarget module解決）でのみ行われる。
5. **モデル切替時に人格promptが評価へ混ざり、ロールバックも危険**
   - `ChatConfig.model_prompt_overrides` でpersonal modelだけ短い人格契約を使い、base modelでは従来の長いpromptを維持する。
   - `scripts/switch_chat_model.py` は `model` 以外のJSON値を維持し、バックアップと本体をアトミックに書き、`rollback` で完全復元する。

### 実装ファイル

- `training/dataset.py`
- `training/validate_dataset.py`
- `training/tokenize_check.py`
- `training/runtime.py`
- `training/train_sft.py`
- `training/train_dpo.py`
- `training/merge_adapter.py`
- `training/evaluate.py`
- `training/eval_prompts.jsonl`
- `training/templates/qwen3_6_assistant.jinja`
- `training/configs/persona_conservative.yaml`
- `training/configs/persona_strong.yaml`
- `training/configs/persona_dpo.yaml`
- `requirements-training.txt`
- `scripts/convert_personal_model_to_gguf.sh`
- `scripts/switch_chat_model.py`
- `models/ollama/Modelfile.personal.example`
- `tests/test_training_*.py`
- `tests/test_chat_model_switch.py`

### 実装・運用ステータス

状態の定義:

- **planned**: 実施予定。未実装、または実環境で未実行
- **implemented**: リポジトリにコード・スクリプトが存在する
- **tested**: リポジトリの単体テストで検証済み（実データ・実モデル・GPUは不使用）
- **integrated**: サービス・実設定へ接続済み
- **deployed**: 実環境へ反映済み
- **verified**: 実データ・実モデルで動作確認済み

| 工程 | スクリプト等 | 状態 | 根拠 |
| --- | --- | --- | --- |
| データ出力 | `scripts/export_discord_training.py` | implemented | リポジトリに実装。テストは変換・スキーマ側のみ（test_training_dataset.py） |
| データ検査・清掃 | `training/validate_dataset.py` | implemented / tested | schema・重複・PII・clean-output をテストで検証 |
| tokenize確認 | `training/tokenize_check.py` | implemented / tested | トークン計測をテストで検証 |
| 設定検証・manifest | `training/runtime.py` | implemented / tested | bf16・revision・rank・target module をテストで検証 |
| SFT / DPO dry-run | `train_sft.py --dry-run` / `train_dpo.py --dry-run` | implemented / tested | dry-runとmanifest書込みをテストで検証 |
| SFT / DPO 実学習 | `train_sft.py` / `train_dpo.py`（live） | planned | liveパスは実装済みだが実行未検証（H200未確保） |
| H200環境構築・データ転送 | `requirements-training.txt` ほか | planned | H200未確保。手順は文書のみ |
| LoRAマージ | `training/merge_adapter.py` | implemented / tested（dry-runのみ） | build_plan / dry_run をテストで検証。実merge未実行 |
| GGUF変換・量子化 | `scripts/convert_personal_model_to_gguf.sh` | implemented / tested（dry-runのみ） | dry-runをテストで検証。実変換未実行 |
| Ollama登録 | `models/ollama/Modelfile.personal.example` | planned | Modelfile例のみ。`ollama create` 未実行 |
| 固定会話評価 | `training/evaluate.py` | implemented / tested（スタブ生成のみ） | Ollama実呼び出しは未実行 |
| モデル切替・ロールバック | `scripts/switch_chat_model.py` | implemented / tested | switch/rollbackのJSON書換・復元をテストで検証。実環境未適用 |
| サービス反映 | `systemctl --user restart ...` | planned | 未実行 |

このリポジトリで **integrated / deployed / verified** に到達している工程はない。学習データ・モデル・GGUF・Ollama登録・H200の存在および実行実績はこの文書では主張しない。

## 8. 実行手順

### 8.1 ローカル: 出力、清掃、tokenize確認

```bash
source .venv/bin/activate

python scripts/export_discord_training.py \
  --format sft --include-positive-feedback --min-score 1 \
  --output data/finetune/sft.jsonl
python scripts/export_discord_training.py \
  --format preference \
  --output data/finetune/dpo.jsonl

python -m training.validate_dataset \
  --input data/finetune/sft.jsonl --format sft \
  --clean-output data/finetune/sft.clean.jsonl --json \
  > data/finetune/sft.validation.json
python -m training.validate_dataset \
  --input data/finetune/dpo.jsonl --format dpo \
  --clean-output data/finetune/dpo.clean.jsonl --json \
  > data/finetune/dpo.validation.json

python -m training.tokenize_check \
  --input data/finetune/sft.clean.jsonl --format sft \
  --tokenizer Qwen/Qwen3.6-35B-A3B \
  --revision 995ad96eacd98c81ed38be0c5b274b04031597b0 \
  --chat-template training/templates/qwen3_6_assistant.jinja \
  --max-tokens 2048
```

過去の検証メモ（2026-07-15時点）にはSFT 73件・DPO 12件が検査を通過し、最大トークン数も2048以内だったと記録がある。ただしデータ本体はgit管理外でこのリポジトリから再現・検証できないため、件数の存在は保証しない。`min_dataset_rows` は強制されない（§7参照）ので、実行前に対象JSONLの件数を手動で確認する。DPOは設定上の最低50件を満たしてから実行する。

### 8.2 外部学習GPU（例示: H200）: 環境確認と学習

> **実行条件**: H200等の高GPU搭載ホスト（例示の外部学習GPU）が必要。**外部・高負荷**の手順で、実学習は数時間かかる。必ず先に `--dry-run` を実行し、manifestに記録されるrevision・対象module・dataset path・`min_dataset_rows` を確認してから実学習に進む。件数強制は未実装なので、JSONLの実レコード数を自分で確認する。実学習は本リポジトリで未実行（planned）。

```bash
python -m venv .venv-training
source .venv-training/bin/activate
pip install -r requirements-training.txt

python -m training.train_sft \
  training/configs/persona_conservative.yaml --dry-run
python -m training.train_sft \
  training/configs/persona_conservative.yaml

# SFT adapterを選び、明示修正ペアが50件以上になった後
python -m training.train_dpo \
  --config training/configs/persona_dpo.yaml --dry-run
python -m training.train_dpo \
  --config training/configs/persona_dpo.yaml
```

H200へ送るのは `*.clean.jsonl`、設定、スクリプトだけとする。実学習前にmanifestのrevision、対象module、dataset pathを確認する。**このリポジトリで完了したのは実行系とローカルpreflightであり、H200上の実学習自体はH200を確保した後に実行する外部工程である。**

### 8.3 ローカル: マージ、GGUF、Ollama

> **実行条件**: 実マージとGGUF変換は**高負荷**。実マージはベースモデル（BF16）のダウンロードと大きなメモリを要し、GGUF変換・量子化はフルモデルを処理する（llama.cppの `convert_hf_to_gguf.py` / `llama-quantize` が必要。数十GBのディスクを要する）。どちらも先に `--dry-run` で計画を確認してから `--dry-run` を外して実行する。実マージ・実変換・Ollama登録は本リポジトリで未実行（planned）。

```bash
python -m training.merge_adapter \
  --config training/configs/persona_conservative.yaml \
  --adapter training/outputs/persona-conservative \
  --output training/outputs/persona-conservative-merged \
  --dry-run

python -m training.merge_adapter \
  --config training/configs/persona_conservative.yaml \
  --adapter training/outputs/persona-conservative \
  --output training/outputs/persona-conservative-merged

bash scripts/convert_personal_model_to_gguf.sh \
  --merged-model training/outputs/persona-conservative-merged \
  --output-dir data/models/personal \
  --basename qwen3.6-shunkin-sft \
  --quant Q4_K_M \
  --llama-dir /path/to/llama.cpp \
  --dry-run
```

`--dry-run` の表示を確認してから外し、`models/ollama/Modelfile.personal.example` をコピーして `__MODEL_GGUF__` を生成済みGGUFの絶対パスへ置換する。

### 8.4 比較、切替、ロールバック

> **実行条件**: `training.evaluate` は `--model` で指定するタグがOllamaに登録済みである必要がある（`ollama list` で事前確認）。未登録のタグでは評価できない。Ollamaへの実呼び出しは本リポジトリで未実行。`switch_chat_model.py` は `config/chat_config.json` を書き換える**設定変更**操作（`.bak` バックアップ作成・原子的書換）で、実行後に `show` で差分を確認する。最後の `systemctl --user restart` は稼働中サービスを再起動する**破壊的操作**なので、反映の意図がある場合のみ実行する。

```bash
python -m training.evaluate \
  --kind baseline --model qwen3.6:35b-a3b-base-q4_K_M \
  --output data/finetune/eval-base.json
python -m training.evaluate \
  --kind sft --model qwen3.6:35b-a3b-shunkin-sft-q4_K_M \
  --output data/finetune/eval-sft.json

python scripts/switch_chat_model.py show
python scripts/switch_chat_model.py switch \
  qwen3.6:35b-a3b-shunkin-sft-q4_K_M
# 問題があれば
python scripts/switch_chat_model.py rollback
```

サービスへ反映する場合だけ、切替後に次を実行する。

```bash
systemctl --user restart subpc-discord.service subpc-web.service
```

## 9. 参考資料

- Qwen3.6: https://github.com/QwenLM/Qwen3.6
- Qwen3.6-35B-A3B: https://huggingface.co/Qwen/Qwen3.6-35B-A3B
- PEFT LoRA: https://huggingface.co/docs/peft/main/package_reference/lora
- TRL PEFT integration: https://huggingface.co/docs/trl/peft_integration
- LoRA paper: https://arxiv.org/abs/2106.09685
- DPO paper: https://arxiv.org/abs/2305.18290
