# ローカル常駐コンパニオン構想

## この文書の位置付け

ブラウザで行った3つの設計対話を、`subpc_living`で実行可能な方針へ統合したもの。
対話ログの転載ではなく、今後の設計判断と実装順序の基準とする。

## 1. プロダクトの中心

`subpc_living`を単なるローカルLLMチャットではなく、次の存在へ育てる。

> 生活環境をローカルで知覚し、言葉・表情・動作を使い分けて支援する常駐型コンパニオン

価値の中心はモデルの大きさではない。

- PC、予定、タスクなどの状況を必要最小限だけ理解する
- 必要なときだけ介入し、集中中は黙る
- 文章だけでなく表情、視線、小物、短いHUDで伝える
- 取得中の情報、保存先、利用目的を常に確認できる
- 個人情報を含む生データを原則ローカルから出さない

## 2. 設計原則

1. **ローカルファースト**
   カメラ、音声、画面、操作履歴、予定、タスク、プロフィール、長期記憶は原則ローカル固定とする。
2. **生データより意味イベント**
   画像や操作ログをLLMへ流し続けず、ローカルで`focused`、`idle`、`away`などの短いイベントへ変換する。
3. **認識・状態・判断・表現を分離**
   認識モデルに発話可否を直接決めさせない。介入頻度、禁止時間、重複抑止は決定的なPolicyで管理する。
4. **境界を先に作り、全面改修しない**
   Pythonモジュラーモノリス、SQLite、Ollama、既存のWeb・Discord・音声・Desktop・テストを維持しながら段階移行する。
5. **明示的な権限と停止手段**
   「ローカルだから何を取得してもよい」としない。センサーはオプトイン、利用中表示、即時停止、監査可能を基本にする。
6. **キャラクターは状態UI**
   アバターを装飾にせず、現在状態、通知、承認、エラーを非言語で伝えるインターフェースとする。

## 3. 目標アーキテクチャ

```mermaid
flowchart TD
    A[Web / Discord / Voice / Desktop] --> B[AssistantService]
    B --> C[Model Router]
    B --> D[Context Policy]
    B --> E[Tools / Approval Workflows]
    C --> F[Local LLM Providers]
    C --> G[Optional Cloud Provider]
    D --> H[Tasks / Calendar / Memory]

    I[Camera / Mic / PC Activity] --> J[Local Perception]
    J --> K[Semantic Events]
    K --> L[State Aggregator]
    L --> M[Proactive Policy]
    M --> N[Companion State]
    N --> O[3D Avatar / Voice / Notification / HUD]

    D -. local context .-> L
    B -. conversation state .-> N
```

### 中核の責務

| 境界 | 責務 |
|---|---|
| `LLMProvider` | Ollamaなど個別の推論APIを共通契約で包む |
| `AssistantService` | 各入口から共通Requestを受け、応答処理を集約する |
| `ModelRouter` | プライバシー、用途、明示選択に基づきモデルを選ぶ |
| `ContextPolicy` | Contextの出所・機密度・送信可否を機械的に制限する |
| `EventCollector` / `Perception` | 生のセンサー入力をローカルで意味イベントへ変換する |
| `StateAggregator` | 複数イベントを安定したユーザー・相棒状態へ集約する |
| `ProactivePolicy` | 話す、表示だけする、黙る、の判断を行う |
| `Expression` | 表情、モーション、音声、HUD、通知へ変換する |

## 4. 共通データ契約の方向

将来的に次の値を明示的なオブジェクトとして扱う。

- `AssistantRequest`: 本文、会話ID、入口、用途、プライバシー、クラウド許可
- `AssistantResponse`: 本文、Provider、モデル、経路理由、レイテンシ
- `ContextBlock`: 出所、本文、機密度、許可プロファイル、優先度
- `RouteDecision`: 選択モデル、理由、信頼度、許可Context、Fallback
- `CompanionState`: 在席、活動モード、集中時間、割込可否、表示状態
- `PerceptionEvent`: 種別、時刻、要約状態、信頼度、生データ保持有無

Contextをすべて構築してから送信先を決めてはいけない。先に送信候補を決め、その送信先へ許可されたContextだけを収集する。

## 5. センサーとプライバシー

### 権限レベル

| Level | 情報 | 初期方針 |
|---:|---|---|
| 0 | 時刻、PCメトリクス、粗い在席状態 | ローカル利用可、個別停止可能 |
| 1 | 使用アプリの分類、活動時間 | オプトイン |
| 2 | ウィンドウタイトル | 個別許可 |
| 3 | 画面キャプチャ、カメラフレーム | 明示的な利用中のみ |
| 4 | OCR、画面内容、姿勢・表情推定 | 用途ごとの承認 |
| 5 | ファイル変更、外部送信 | 実行ごとの承認 |

### 守ること

- カメラ、画面認識、詳細な活動収集は初期状態で無効
- 利用中は、対象・目的・保存有無・停止操作を見える場所へ表示
- フレームは原則メモリ上で処理し、保存しない
- 顔識別ではなく在席や姿勢など粗い状態を優先
- 「今見ないで」で関連センサーを即時停止
- 生データ、個人Context、秘密情報をクラウドへ送らない
- READMEにある常時ON表現は、実装フェーズでこの方針に合わせて更新する

## 6. モデル利用方針

当面はローカルOllamaを既定とする。複数PC・複数モデル化はProviderとRouterの後に行う。

初期Routerは自動判定より手動選択を優先する。

- fast: 低レイテンシのローカルモデル
- strong: 通常・個人Contextを扱うローカルモデル
- local: 外部送信禁止
- auto: ルールベース選択
- cloud: 将来の明示承認付き経路

クラウドを導入する場合も、ローカルで匿名化した独立問題だけを送る。画面、カメラ、タスク一覧、予定、プロフィール、生の会話履歴を同伴させない。

## 7. コンパニオン体験

### 通常時

- 画面端にVRM 1.0キャラクターを透過表示
- 待機、作業中、考え中、会話中、離席、予定接近、エラーを有限状態で表現
- パネル外はクリック透過可能
- フルスクリーン、画面共有、集中モードでは縮小または非表示

### 必要なとき

- キャラクターの横へ短い半透過HUDを1枚だけ展開
- 会話、今日、タスク、PC状態、センサーを同時に常設しない
- 選択肢は2〜3個に絞り、詳細は既存Web UIへ渡す
- センサー参照時は、出所、取得時刻、保存有無を表示

### 見た目

- 全面ガラスUIにはしない
- 暗い半透明面、細い輪郭、最小限のぼかし、一本のアクセント色
- 大量の角丸カード、紫グラデーション、英語の小見出し、計器風ダッシュボードを避ける
- 学園アイドルマスター等からは「キャラクターが主役で、情報の優先順位が明快」という構造だけを参考にし、素材や画面を複製しない
- 既存Hallmarkテーマの視覚表現は継承しないが、アクセシビリティ、余白、レスポンシブ、reduced-motionの規律は残す

## 8. 実装ロードマップ

進捗は対応する節の完了条件に基づき、完了 / 一部完了 / 未着手で示す。

### Phase 1: LLM境界（完了）

- `LLMProvider`共通契約
- `FakeProvider`とContract Test
- 既存`OllamaClient`の挙動を変えず、後続Adapterの土台を作る

### Phase 2: AssistantService（完了）

- `AssistantRequest` / `AssistantResponse`
- CLIを最初のAdapterとして接続
- Web、Discord、音声を一つずつ移行

### Phase 3: ContextとRouter（完了）

- `ContextBlock`契約とmetadataベース`ContextPolicy`（完了）
- History Context Provider / Builder移行とChatSessionへの適用（完了）
- Preload Context Provider移行（完了・SessionPreloader由来のprofile/schedule/summaryを一括で包む）
- RAG Context Provider移行（完了・prompt injectionのみ）
- Web search Context Provider移行（完了・Context Provider化のみ、検索ロジックは未変更）
- Monitor Context Provider移行（完了）
- Vision / Screen Context Provider移行（完了・secret / local_only）
- Calendar Context Provider移行（完了）
- Tasks Context Provider移行（完了・最終権威ブロックとして末尾）
- ローカルモデルRegistry（完了）
- 手動ルーティングと経路ログ（完了）

### Phase 4: 知覚イベントと状態（完了）

- 分類済みapp_categoryとidle_secondsから意味イベント生成（完了・`ActivityEventCollector`）
- `focused` / `idle` / `away`の3状態（完了）
- `CompanionState`と決定的`StateAggregator`（完了）
- 生データを保存していないことのテスト（完了）
- プラットフォーム別ActivitySource adapter（完了・Windows / Linux(X11)）
- Web runtime wiringと読み取り専用状態API（完了・`/api/companion/state`、オプトイン）
- UI消費とWeb以外（Discord / Voice / Desktop）へのruntime wiring（完了・各入口とも `COMPANION_ACTIVITY_ENABLED=true` オプトイン時のみ起動し、privacy-safe な `companion_state_payload` のみ公開）

次アクション: Phase 5 の接続層（Policy と既存 `ProactiveEngine`・各入口の統合）。カメラ・画面・常時生データ収集は実装しない。

### Phase 5: Proactive Policy（完了）

- 予定接近、長時間作業、離席復帰のルール
- 黙る条件、再通知間隔、拒否フィードバック
- 提案と実行を分け、変更操作は承認必須

#### 完了: 決定的 Policy エンジン基盤

- `PolicyDecision` / `PolicyContext` 契約と `DeterministicProactivePolicy`（`src/companion/policy.py`）。
  CompanionState + 現在時刻 + 予定接近情報を入力とし、LLM に判断させず閾値ベースで
  介入可否・内容・種別を決定する
- ルール: focused 中は黙る（interruptible な予定接近を除く）、長時間作業で `break_suggest`、
  予定接近で `schedule_remind`、離席復帰で `away_return`、cooldown 中は silent
- 提案と実行の分離: 変更操作は行わず `requires_approval` は現状常に False（将来の拡張用）
- privacy: `message_hint` は個人情報・生データ・エラー本文を含まない定数ヒントのみ
- テスト 14件追加 (focused silent, break_suggest, schedule_remind, away_return, cooldown,
  interruptible, 純粋関数性)。companion 全体 56テスト OK

#### 完了: Policy の接続層

- `DeterministicProactivePolicy` と既存 `ProactiveEngine` (`src/persona/proactive.py`) の統合: `ProactiveEngine` は `companion_policy` / `companion_getter` / `calendar_source` / `state_path` を受け付け、既定で `DeterministicProactivePolicy()` を使う
- CalendarSource 接続: `src/companion/calendar.py` が `UserProfile.get_upcoming_schedule` から `next_event_at` への橋渡しを実装し、`schedule_remind` に反映
- 拒否フィードバックの永続化: `record_rejection` が `cooldown_key` へ反映され、再通知間隔を調整（state_path へ永続化）
- Discord wiring: `src/discord_bot/proactive_bridge.py` が `companion_getter` を `ProactiveEngine` へ渡し、活動状態でゲート
- 音声 wiring: `src/audio/pipeline.py` の `VoicePipeline` が `companion_getter=self._companion_state` を `ProactiveEngine` へ渡し、`ActivityRuntime.state` でゲート（None・例外時は従来挙動）
- Desktop wiring: desktop は `ProactiveEngine` を持たず、companion state を表示のみに利用するため Policy ゲートの対象なし。入力である companion state は既に `companionState` Property で露出済み

### Phase 6: 3DデスクトップShell（未着手）

- 仮VRMによる透明ウィンドウ
- 待機、視線、表情、リップシンク
- クイック会話HUDとセンサー確認パネル
- ユーザー所有VRMの読み込みを優先し、第三者モデルを製品へ無断同梱しない

### Phase 7: 任意の高度機能（未着手）

- 明示承認付きクラウド推論
- LangGraphによる承認・中断・再開が必要な処理だけをWorkflow化
- コーディングJob Adapter
- カメラによる在席検知は安全UI完成後に追加

## 9. 現在の実装進行単位

Phase 1〜2とRegistry/Router、CLI・Web・Discord・Voice移行、実行ログの実装とruntime wiringは完了し、Phase 3の`ContextBlock` / `ContextPolicy`基盤、History・Preload・RAG・Web search・Monitor・Vision・Screen・Calendar・Tasksの各Context Provider移行も完了した。Phase J全体は完了し、Phase 4の基盤実装（`PerceptionEvent` / `CompanionState` / `StateAggregator` / `ActivityEventCollector`）とプラットフォーム別ActivitySource adapter、Web runtime wiring・`/api/companion/state`、およびDiscord / 音声 / Desktop / Web UI へのruntime wiringが完了した。Phase 5の決定的Policyエンジン基盤（`DeterministicProactivePolicy`）と接続層（`ProactiveEngine`・CalendarSource・拒否cooldown永続化・Discord / 音声 / Desktop 各入口のPolicy参照wiring）も完了した。次アクションは Phase 6（3DデスクトップShell）へ進むが、本計画の境界はここまでとし、3D UI は `docs/companion_roadmap.md` の Phase 6 で別途進める。

### 完了: 実行ログ

- `src/assistant/run_logger.py`（`RunLogger` / `SQLiteRunLogger`）
- `tests/assistant/test_run_logger.py`
- runtime wiring済み（Service側へ組み込み済み）
- SQLiteへ保存するのはchannel、profile、provider、model、local、latency、success、error等の経路決定と実行結果のみ
- 会話本文、個人Context、APIキーなどの秘密情報、画面・カメラ・タスク・予定の本文は保存しない

受け入れ条件（達成済み）:

- ログ失敗で会話を失敗させない
- first-write-winsで同一request IDの重複を抑える
- 経路と統計だけで再現テストできる

### 完了: ContextBlock / ContextPolicy基盤

- `ContextBlock`契約とmetadataベースの`ContextPolicy`（sensitivity / local_only / 送信先 / priority）を実装し、テスト済み
- Tasksは最終権威ブロックとして最後に移行する方針を維持

### 完了: History Context Provider / Builder移行

- `HistoryContextProvider`が現在の履歴を不変な`ContextMessage`列へコピーし、`ContextBlock`（source=history / sensitivity=personal / local_only=True）として返す
- `ContextBuilder`がPolicy通過済みブロックを既存互換のrole/content dict列へ描画し、`ChatSession.build_messages()`へ組み込み済み
- 次はPreloadをContext Provider化し、続いてRAG、Web search、予定、画面などを順次分離する（Tasksは最後）。（以降の分離はすべて完了済み）

### 完了: Preload Context Provider移行（profile/schedule/summaryを一括で包む）

- `PreloadContextProvider`が`build_preload_context()`の結果をsource=preload / sensitivity=personal / local_only=Trueのstr ContextBlockとして返す
- この結果はSessionPreloaderがprofile・schedule・summary・時刻を一つのstrへまとめたPreloadであり、独立したProfile Providerの成果ではない
- 収集失敗は本文をログせず型名だけwarningし、会話を止めず継続する
- `ContextBuilder.build_system_content()`がstr blockだけをbase systemへ連結し、構造化blockは`StructuredBlockNotAllowedError`で明示拒否する
- `ChatSession.build_messages()`はPreloadを`ContextPolicy`経由（local_only / local target）で描画し、既存のsystem_prompt直後・RAG前の位置を維持
- `PreloadContextProvider`と`StructuredBlockNotAllowedError`は`src.context`から公開し、root公開APIテスト済み
- 次はMonitor Context Provider移行（Tasksは最後）（以降の分離はすべて完了済み）

### 完了: RAG Context Provider移行（prompt injectionのみ）

- `RAGContextProvider.collect(retriever, query)`が`build_context_prompt(query)`の結果をsource=rag / sensitivity=personal / local_only=Trueのstr ContextBlockとして返す
- `RAGSource` Protocolで型契約だけを定義し、`RAGRetriever`本体は変更しない。store_turn / store_knowledge / retrieveの実装自体は変更・呼び出しせず、RAGはプロンプトへの注入のみを行う
- 空・空白のみ・非strの結果、およびretrieverの例外時はNoneを返す。例外は型名だけlogging.warningに残し、query・本文・例外本文はログしない。収集失敗で会話は止まらず継続する
- `ChatSession.build_messages()`はRAGを`ContextBuilder` / `ContextPolicy`経由（local_only / local target）で描画し、Preload直後・Web search前の既存位置を維持する
- `store_turn`による会話記録は従来どおり呼び出される
- `RAGContextProvider`と`RAGSource`は`src.context`から公開し、root公開APIテスト済み

### 完了: Web search Context Provider移行（Context Provider化のみ・検索ロジックは未変更）

- `WebSearchContextProvider.collect(web_search, query)`が`WebSearchContext.build_context_prompt(query)`の結果をsource=web_search / sensitivity=personal / local_only=Trueのstr ContextBlockとして返す（queryを含み得るためlocal_only）
- `WebSearchSource` Protocolで型契約だけを定義し、`WebSearchContext`本体は変更しない。search / should_search / cache / 判定ロジック自体は変更・呼び出しせず、Context Provider化のみ行う
- 空・空白のみ・非strの結果、および例外時はNoneを返す。例外は型名だけlogging.warningに残し、query・本文・例外本文はログしない。収集失敗で会話は止まらず継続する
- `ChatSession.build_messages()`はWeb searchを`ContextBuilder` / `ContextPolicy`経由（local_only / local target）で描画し、RAG直後・Vision前の既存位置を維持する
- `WebSearchContextProvider`と`WebSearchSource`は`src.context`から公開し、root公開APIテスト済み

### 完了: Monitor Context Provider移行

- `MonitorContextProvider.collect(monitor)`が`MonitorContext`の結果をsource=monitor / sensitivity=personal / local_only=Trueのstr ContextBlockとして返す
- `MonitorSource` Protocolで型契約だけを定義し、`MonitorContext`本体は変更しない。収集失敗は型名だけwarningし、query・本文・例外本文はログしない。会話は止まらず継続する
- `ChatSession.build_messages()`はMonitorを`ContextBuilder` / `ContextPolicy`経由（local_only / local target）で描画し、Vision直後・Screen前の既存位置を維持する
- `MonitorContextProvider`と`MonitorSource`は`src.context`から公開し、root公開APIテスト済み

### 完了: Vision / Screen Context Provider移行

- `VisionContextProvider.collect(vision)`がsource=vision / sensitivity=secret / local_only=True、`ScreenContextProvider.collect(screen)`がsource=screen / sensitivity=secret / local_only=Trueのstr ContextBlockとして返す
- どちらもsecret / local_onlyのため非Local経路へ渡らない
- `ChatSession.build_messages()`はVisionをWeb search直後・Monitor前、ScreenをMonitor直後・Calendar前の既存位置に`ContextBuilder` / `ContextPolicy`経由で描画する
- 各収集失敗は型名だけwarningし、本文をログせず会話を継続する
- 各Providerは`src.context`から公開し、root公開APIテスト済み

### 完了: Calendar Context Provider移行

- `CalendarContextProvider.collect(calendar)`がsource=calendar / sensitivity=personal / local_only=Trueのstr ContextBlockとして返す
- `CalendarSource` Protocolで型契約だけを定義し、`CalendarContext`本体は変更しない（ファイル読取のみ）
- `ChatSession.build_messages()`はCalendarをScreen直後・Emotion前の既存位置に`ContextBuilder` / `ContextPolicy`経由で描画する
- 収集失敗は型名だけwarningし、本文をログせず会話を継続する
- `CalendarContextProvider`と`CalendarSource`は`src.context`から公開し、root公開APIテスト済み

### 完了: Tasks Context Provider移行

- `TasksContextProvider.collect(task_store)`がsource=TASKS_SOURCE / sensitivity=personal / local_only=Trueのstr ContextBlockとして返す
- Tasksは最終権威ブロックとしてsystem本文の末尾に配置され、0件でも必ず注入される
- `ChatSession.build_messages()`はTasksを`ContextBuilder` / `ContextPolicy`経由で描画し、Historyのrole messagesより後に最終文字列blockとして置く
- 収集失敗は型名だけwarningし、本文をログせず会話を継続する
- `TasksContextProvider`は`src.context`から公開し、root公開APIテスト済み

### 完了: Phase 4 知覚イベントと状態の基盤

- `PerceptionEvent` / `CompanionState`契約（`src/companion/contracts.py`）。raw payload・本文・metadataを持たず、状態遷移に必要な最小フィールドのみ
- 決定的`StateAggregator`（`src/companion/state.py`）。min_confidence未満とout-of-orderは状態を変えず、`raw_data_retained=True`は`PrivacyViolationError`で拒否。時刻はイベント注入値のみで決定
- `ActivityEventCollector` / `ActivitySample`（`src/perception/activity.py`）。入力は分類済みapp_category（work / communication / media / system / unknown）とidle_secondsのみで、rawなアプリ名・window title・text・path・pid・raw inputは扱わない
- 生データ非保持（`raw_data_retained=False`固定）とprivacy違反拒否のテスト済み。`src.perception` / `src.companion`から公開し、root公開APIテスト済み
- プラットフォーム別ActivitySource adapter（`WindowsActivitySource` / `LinuxActivitySource`）は完了。Linux/X11では`xprintidle`（idle取得）と`xdotool`（アクティブウィンドウPID取得）が必要で、施設が無ければ`ActivitySourceUnavailableError`で明確に失敗する
- Web runtime wiringは完了: `COMPANION_ACTIVITY_ENABLED=true` のオプトイン時だけActivityRuntimeを起動し、`GET /api/companion/state`（読み取り専用・privacy-safe）でActivityRuntimeStatusの集計カウンタと`CompanionState`フィールドのみ公開。起動失敗は例外型名だけログしてcompanion機能のみ無効化し、Web起動は続行する。プロセス名・PID・アプリ分類・window title・エラー本文・生サンプル/イベントは公開しない
- `StateAggregator`からDiscord / 音声 / Desktopへのruntime wiringとUI消費は完了:
  - Discord (`src/discord_bot/bot.py`): 起動・停止で ActivityRuntime を生成・破棄
  - 音声 (`src/audio/main.py` + `src/audio/pipeline.py`): パイプライン起動に runtime を組込み
  - Desktop (`src/desktop/bridge.py` + `api.py` + `qml/Main.qml`): bridge が runtime 管理、
    API は読取専用 payload、QML は控えめな最小表示
  - Web UI (`src/web/static/{index.html,app.js,style.css}`): `/api/companion/state` を
    ポーリング取得し既存デザイントークンに合わせた最小表示
  - いずれも `COMPANION_ACTIVITY_ENABLED=true` オプトイン時のみ起動し、
    privacy-safe な `companion_state_payload` のみ公開。プロセス名・PID・アプリ分類・
    window title・エラー本文・生サンプルは出さない

次の実装単位: Phase 6（3DデスクトップShell）。Cloud経由の実送信Provider swap と LangGraph Workflow は `docs/assistant_platform_plan.md` Phase K/L の通り別判断。

## 10. 現時点の非目標

- 全面LangChain / LangGraph化
- マイクロサービス、Redis、Kafka、Kubernetesの導入
- SQLiteや既存データストアの一括移行
- いきなり全入口を`AssistantService`へ切り替えること
- 感情認識やカメラ常時利用を先行させること
- 高品質な専用3Dモデル制作をソフトウェア境界より先に行うこと
- 第三者のVRChatモデルを製品へ再配布すること

## 11. 未決事項

- `AssistantRequest`の最終フィールドと同期・非同期境界
- 複数ローカルモデルの命名・Registry形式
- デスクトップShellを既存QML、Tauri、Electronのどれで進めるか
- VRM RendererとPythonバックエンド間のIPC方式
- イベント保存期間と監査ログの粒度
- 製品用オリジナルキャラクターと衣装規格

## 12. 元になった設計対話

- [アーキテクチャ、モデルルーティング、段階改修](https://chatgpt.com/share/6a818ce2-f970-83e8-b59e-25408a0a469e)
- [UI、相棒中心の体験、ローカル知覚](https://chatgpt.com/share/6a8185b1-dc3c-83ee-be4c-7175c0eaa612)
- [製品化、3Dキャラクター、透過HUD](https://chatgpt.com/share/6a818d11-f610-83e8-ad95-5e202ba3f5a5)
