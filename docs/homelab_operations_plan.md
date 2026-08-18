# AIホームラボ運用計画

## 位置付け

`subpc_living`は、メインPCとサブPCを使う2ノードのAIホームラボとして既に成立している。
ただし現在はアプリケーション機能が先行しており、ノード追加、障害対応、再構築を安全に行うための運用基盤は未完成である。

この文書では、未整備部分と次アクションを管理する。実IP、認証情報、APIキーは記載しない。

関連文書:

- `docs/assistant_platform_plan.md`: Provider、Router、AssistantServiceの実装計画
- `docs/companion_roadmap.md`: 常駐コンパニオン全体構想

## 1. 現在の構成

### メインPC

会話で確認できている情報:

- 普段使いのPC
- NVIDIA RTX 3060 Ti
- RAM 32GB
- 開発時のPi / OpenCode実行環境
- 将来は低レイテンシの軽量モデル、Desktop UI、ローカル操作収集を担当

未確認:

- Ollamaを常時サービスとして動かすか
- OS起動時の自動起動方式
- 固定ホスト名とLAN上の到達方法
- 外部公開を完全に禁止するFirewall設定
- 推論中に普段使いへ与える負荷上限

### サブPC

READMEに記載されている情報:

- Ubuntu 24.04 LTS
- Intel Core i7-9700K
- NVIDIA Tesla P40 24GB
- 第2 GPUはREADMEではRTX 2070 Super 8GB
- RAM 48GB
- Ollama、Web、Discord、音声、RAG、監視などを常時実行
- systemd user serviceでWebとDiscordを管理

要確認:

- 実機の第2 GPUがREADMEどおりか
- 現在ロードしているモデルとContext長
- サービスごとの再起動・依存順序
- データ領域とバックアップ対象の実容量

### アプリケーション

現在あるもの:

- Ollamaチャット
- Web / PWA
- Discord bot
- 音声STT / TTS
- RAGと会話履歴
- タスク、Calendar、日記、プロフィール
- カメラ、画面、PCメトリクスのモジュール
- systemd常駐
- Pi / OpenCode開発オーケストレーション

現在進めているもの:

- `LLMProvider`境界
- 複数ノードを扱うProvider Registry
- Privacyを考慮するModel Router
- 各入口をまとめるAssistantService

## 2. ホームラボとして不足しているもの

### P0: ノードInventory

不足:

- ノードID、役割、OS、GPU、RAMの正本
- LAN内ホスト名
- 提供するProviderとモデル
- Health endpoint
- 同時実行上限と用途

方針:

- 実IPではなくホスト名または環境変数で参照する
- Git管理するのはSchemaとExampleだけ
- 実値と秘密情報は管理外設定に置く

将来の例:

```yaml
nodes:
  main-pc:
    role: interactive
    providers: [local-fast]
    local: true
  sub-pc:
    role: always-on
    providers: [local-strong]
    local: true
```

この形式は設計例であり、実装形式はProvider Registry設計時に確定する。

### P0: ネットワーク境界

不足:

- Ollama / Web APIをどのInterfaceでListenするかの統一方針
- LAN内の許可元
- Firewall Rule
- 認証なしAPIの到達範囲
- メインPCまたはサブPC停止時の挙動

最低条件:

- インターネットへ直接公開しない
- LAN内でも不要な端末からの接続を許可しない
- APIキーや秘密値をURLへ含めない
- 外部アクセスが必要になった場合は、直接Port ForwardせずVPN等を別途検討する

### P0: Health CheckとFallback

不足:

- ノード単位のHealth
- Provider単位のHealth
- モデル存在確認
- TimeoutとFallbackの共通ルール
- 障害理由の記録

次の実装で`ProviderRegistry`と`ModelRouter`へ組み込む。

### P0: バックアップと復元

バックアップ候補:

- タスクDB
- 会話履歴
- RAGデータ
- プロフィール
- 日記
- Calendar同期状態
- モデル設定
- systemd unitとExample設定

除外候補:

- 再取得可能なモデル本体
- 一時音声、カメラフレーム、画面キャプチャ
- Cache
- ログの無期限保存

不足:

- バックアップ頻度
- 保存先
- 世代数
- 暗号化
- 復元テスト

バックアップは「作成できる」だけでなく、別ディレクトリへ復元して起動確認できることを完了条件にする。

### P0: 秘密情報管理

現在の原則:

- 実`.env`と`config/discord.env`はGit管理しない
- Exampleだけを追跡する
- APIキーをJSON、Markdown、ログへ書かない

不足:

- 必須環境変数の一覧
- 起動時の不足検出
- Key Rotation手順
- ログのMask規則

### P1: 監視とログ

不足:

- 2台のサービス状態を一画面で確認する方法
- Provider別レイテンシ、失敗率、Fallback回数
- ディスク残量とDBサイズ
- GPU VRAM、温度、電力
- ログ保持期間

当面は既存のSQLiteと標準ログを使う。Prometheus、Grafana等は、既存方式で不足が確認されてから導入する。

### P1: 再現可能なセットアップ

不足:

- 新しいサブPCへ導入する順序
- Python venvと依存Version
- systemd user serviceの配置
- Windows側自動起動
- Ollama model取得手順
- Smoke Test

最初はREADMEとShell / PowerShell Scriptで十分。Ansible等は3台目を追加するか、再構築作業が繰り返し発生してから判断する。

#### 決定: ノードごとに何を入れるか

同一リポジトリを使うが、**インストールする依存と動かすプロセスはノードごとに違う**。
「2台に同じものを入れて両方動かす」構成にはしない。

| | サブPC (Ubuntu) | メインPC (Windows) |
| --- | --- | --- |
| 依存 | `requirements.txt` | `requirements-desktop.txt` |
| 動かすもの | Ollama、Web、Discord、音声、RAG (systemd user service) | `src/desktop` のネイティブクライアント |
| 接続 | — | `SUBPC_DESKTOP_SERVER_URL` でサブPCのWeb APIへ |

メインPCはCIが `subpc-desktop.spec` からビルドする実行ファイルを置くだけでよく、
リポジトリのチェックアウトは開発時にしか必要ない。「更新とRollback」の
「メインPC更新」は、バックエンドではなくデスクトップクライアントとOllamaの更新を指す。

#### 決定: メインPCをProviderノードにする場合

メインPCが担う低レイテンシの軽量モデル (`local-fast`) には、アプリケーションコードを置かない。
メインPCでOllamaをLANに向けてlistenさせ、サブPC側の`ProviderRegistry`へIDで登録する。

```python
registry.register("local-fast",   OllamaProvider(base_url="http://main-pc:11434", ...), local=True)
registry.register("local-strong", OllamaProvider(base_url="http://localhost:11434", ...), local=True)
```

これによりノード追加はRegistry登録とInventory更新だけになり、
アプリケーションの二重配置と更新順序の複雑化を避けられる。

#### 決定: `local` フラグの意味

`ProviderEntry.local` は **「信頼するLAN内にある」** の意味で使う。
「同一ホストで動いている」の意味ではない。したがってLAN越しのメインPCのOllamaは
`local=True` で登録する。この定義は次を意味する。

- `local=False` はクラウドなど外部サービスだけに使う。構築済みmessages経路では実行しない
- LANの境界そのものは`local`フラグではなく、Firewallとlisten Interfaceで守る (「P0: ネットワーク境界」)
- Phase K (Cloudと承認) を始める前に、この定義を`docs/assistant_platform_plan.md`側の
  Privacy設計と突き合わせる

### P1: 更新とRollback

不足:

- Pull前の確認
- DB Migration順序
- サービス停止が必要な変更の分類
- 更新失敗時のRollback
- 複数ノードの更新順序

基本順序:

```text
差分確認
→ バックアップ
→ テスト
→ サブPC更新
→ Health確認
→ メインPC更新
→ End-to-End確認
```

## 3. ネクストアクション

### N1: Provider Adapter（完了）

- `OllamaProvider`で既存`OllamaClient`を包む
- Generation Optionsとエラーを共通化
- 実Ollama不要のテストを追加

完了条件:

- 既存PayloadとStream挙動が変わらない
- TimeoutとRequest Errorが識別できる

### N2: Provider RegistryとStatic Router（完了）

- `local-fast`と`local-strong`をIDで扱う
- 初期設定は単一Providerでもよい
- `local_only`を強制する
- Fallback循環を禁止する

完了条件:

- 2ノードを追加してもUIコードを変更しない
- 選択理由を返せる

### N3: Inventory Example

Provider Registryの設定形式が確定してから追加する。

- ノードSchema
- Git管理可能なExample
- 実設定の管理外Path
- 設定検証コマンド

実IP、秘密情報、実APIキーはCommitしない。

### N4: AssistantServiceとCLI移行（次アクション）

- Fake ProviderでServiceを検証
- CLIを最初のEnd-to-End Adapterにする
- Web、Discord、音声はまだ変更しない

### N5: Health API

最低限返す値:

- node ID
- service status
- Provider ID
- model
- local / remote
- availability
- 最終成功時刻

秘密情報、内部Prompt、個人Contextは返さない。

### N6: バックアップ設計

- 対象Path一覧
- 世代管理
- 復元コマンド
- 月1回以上の復元確認

頻度や保存先は、現在のデータ量を測定してから決める。

### N7: 運用Runbook

最低限必要な手順:

- Ollamaが応答しない
- モデルが見つからない
- Web / Discordが起動しない
- GPU VRAM不足
- ディスク不足
- DB破損
- メインPC停止
- サブPC停止
- 秘密情報を誤って出力した

## 4. 成熟度の目安

### 現在: 2ノードAIホームラボ

- 複数PCとGPUがある
- LAN内でローカルAIサービスを利用できる
- サブPCで常駐サービスを動かしている

### 次段階: マルチノード推論基盤

- RegistryとRouterがある
- HealthとFallbackがある
- 各入口がAssistantServiceを使う
- 経路と失敗を記録できる

### 運用可能: 再現可能なホームラボ

- Inventoryが正本化されている
- BackupとRestoreを確認済み
- 新規ノードを手順どおり追加できる
- 更新とRollbackのRunbookがある
- ネットワークと秘密情報の境界が明文化されている

## 5. やらないこと

現時点では次を必須にしない。

- Kubernetes
- Kafka
- Redis Cluster
- PostgreSQLへの一括移行
- 全サービスDocker化
- インターネット公開
- 高可用性Cluster

2台構成では、systemd、Windows自動起動、Python、SQLite、LAN内APIで十分である。
