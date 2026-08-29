# 実装ステータス (Implementation Status)

> **状態**: active / canonical
> **位置付け**: リポジトリの実装状態の正典（cross-project status ledger）。実装状態の確認は本ファイルを正とする
> **対象範囲**: 全プロジェクト領域（Assistant / Provider / Router / Context、Cloud、Web、Discord、Voice/STT/TTS、Tasks/優先順位、Growth/Memory、Desktop/3D、Perception/センサー、infra/systemd、Docker/PostgreSQL、backup/restore、監視/CI/CD、個人LoRA）
> **作成日**: 2026-08-28
> **更新日**: 2026-08-29
> **日付根拠**: Git commit date

## 位置付け

- 本ファイルはリポジトリに追跡されたソース・テスト・スクリプト・設定例・unit定義から読み取れる
  **実装状態の一覧**である。**計画の正典**は [infrastructure_plan.md](./infrastructure_plan.md)、
  製品方向の正典は [plans/companion_roadmap.md](./plans/companion_roadmap.md)。
- **証跡の範囲**: GitHub に追跡された資産のみ。**デプロイ済みのサブPCは存在せず**、
  Ubuntu / systemd / Ollama / GPU / センサー / Docker / PostgreSQL の実機稼働はいずれも未確認。
- **deployed / verified は未主張**: 本ファイルを含め、リポジトリのどの文書・コードも
  deployed（実環境反映）と verified（実機・実データ・実サービスでの動作確認）を主張しない。

## 状態の語彙（6段階）

| 段階 | 定義 |
| --- | --- |
| planned | 実施予定。リポジトリに未実装、または実環境で未実行 |
| implemented | リポジトリにソース・スクリプト・設定例・unit定義として存在する |
| tested | リポジトリの単体・分離テストで検証済み（実モデル・実GPU・実サービス・実ネットワーク・実データは不使用） |
| integrated | 通常の製品経路（Web / Discord / Voice / CLI / Desktop）または実経路へ配線済み |
| deployed | 実環境（サブPC実機等）へ反映済み。**現状はどこにも該当なし** |
| verified | 実データ・実モデル・実サービス・実機で動作確認済み。**現状はどこにも該当なし** |

- `tested` は単体・分離テストの意味であり、統合（`integrated`）・デプロイ・実機稼働の証拠にはならない。
- 凡例: ○=該当・成立、△=一部・未完成、-=該当なし・未実施。

## 状態マトリクス

| 領域 | implemented | tested | integrated | deployed | verified | 根拠（主要） |
| --- | --- | --- | --- | --- | --- | --- |
| 中核 (Assistant / Provider / Router / Context) | ○ | ○ | ○（CLI / Web / Discord / Voice / batch が `build_local_service` / `build_local_provider` 経由で配線。Web は `SUBPC_NODE_INVENTORY` 時 `build_node_service`。backend は Ollama 既定 / `openai_compatible` 交換可・loopback限定） | - | - | [src/assistant/service.py](../src/assistant/service.py) / [src/llm/registry.py](../src/llm/registry.py) / [src/context/builder.py](../src/context/builder.py) / [src/chat/session.py](../src/chat/session.py) / [src/llm/providers/local_openai.py](../src/llm/providers/local_openai.py) / [config/chat_config.local-openai.example.json](../config/chat_config.local-openai.example.json) / [tests/assistant/test_service.py](../tests/assistant/test_service.py) / [tests/llm/test_provider.py](../tests/llm/test_provider.py) / [tests/llm/test_local_openai_provider.py](../tests/llm/test_local_openai_provider.py) / [tests/context/test_history_builder.py](../tests/context/test_history_builder.py) |
| Cloud 経路 (Phase K) | ○（基盤のみ） | ○（unit。実送信は MockTransport 注入のみ） | △（通常の Web / Discord / Voice / CLI 経路へ未統合） | - | - | [src/llm/cloud_config.py](../src/llm/cloud_config.py) / [src/llm/providers/cloud_http.py](../src/llm/providers/cloud_http.py) / [src/llm/approval.py](../src/llm/approval.py) / [src/assistant/cloud_service.py](../src/assistant/cloud_service.py) / [tests/llm/test_cloud_provider.py](../tests/llm/test_cloud_provider.py) |
| Web | ○ | ○ | ○（AssistantService + Stream Queue Adapter） | - | - | [src/web/server.py](../src/web/server.py) / [tests/test_web_health_providers.py](../tests/test_web_health_providers.py) ほか `tests/test_web_*.py` |
| Discord | ○ | ○ | ○（bot / proactive / voice_stt / voice_tts / task / training 配線） | - | - | [src/discord_bot/bot.py](../src/discord_bot/bot.py) / [tests/test_discord_*.py](../tests/test_discord_assistant_route.py) |
| Voice / STT / TTS | ○ | ○ | ○（pipeline が `generate_stream` + TTS へ接続。P0-4: CLI / Voice CLI（audio-text）/ Voice パイプラインが共通文脈構成（`ChatSession.build_blocks` → `AssistantService.respond_stream` → `ContextBuilder.build_messages`）を共有し、Voice 拡張（rag / persona は既定有効・`--no-rag` / `--no-persona` で無効化、task_store / calendar_context は既定構築・データ/設定利用可能時のみ文脈、カレンダー書込は `VOICE_CALENDAR_WRITE_ENABLED` 明示 `true` のみ、P0-3 センサーは既定オフ opt-in）はチャネル能力拡張として注入） | - | - | [src/audio/pipeline.py](../src/audio/pipeline.py) / [src/audio/main.py](../src/audio/main.py) / [tests/audio/test_pipeline_llm.py](../tests/audio/test_pipeline_llm.py) / [tests/audio/test_context_parity.py](../tests/audio/test_context_parity.py) / [tests/chat/test_session_build_blocks.py](../tests/chat/test_session_build_blocks.py) / [tests/chat/test_cli_loop.py](../tests/chat/test_cli_loop.py) / [tests/test_audio_tts.py](../tests/test_audio_tts.py) / [tests/test_tts_factory.py](../tests/test_tts_factory.py) / [decisions/voice_context_parity.md](./decisions/voice_context_parity.md) |
| Tasks / 優先順位 | ○ | ○ | ○（`TasksContextProvider` が最終権威ブロックとして注入。リマインド配送は at-least-once best-effort 契約: rev 楽観制御・`BEGIN IMMEDIATE` claim・revalidate-before-callback・`expected_rev` 条件付き record・state-driven カレンダー同期・マーカー照合/重複整理。オフライン unit テスト120件） | - | - | [src/tasks/store.py](../src/tasks/store.py) / [src/tasks/prioritizer.py](../src/tasks/prioritizer.py) / [src/tasks/reminder.py](../src/tasks/reminder.py) / [src/tasks/calendar_sync.py](../src/tasks/calendar_sync.py) / [tests/test_tasks_store.py](../tests/test_tasks_store.py) / [tests/test_tasks_reminder.py](../tests/test_tasks_reminder.py) / [tests/test_calendar_sync.py](../tests/test_calendar_sync.py) / [tests/test_prioritizer.py](../tests/test_prioritizer.py) / [decisions/task_delivery_consistency.md](./decisions/task_delivery_consistency.md) |
| Growth / Memory | ○ | ○ | ○（Growth・RAG Context Provider が ChatSession へ配線） | - | - | [src/growth/tracker.py](../src/growth/tracker.py) / [src/memory/rag.py](../src/memory/rag.py) / [tests/test_growth_tracker.py](../tests/test_growth_tracker.py) / [tests/context/test_rag_provider.py](../tests/context/test_rag_provider.py) |
| Desktop / 3D | ○（6a + 6b 静的表示 + P1 Quick-Chat HUD） | ○ | ○（Desktop app へ組込み。companion state は表示のみ。P1 HUD は既存 DesktopBridge のプロパティ・シグナル・送信経路を再利用し、新規 API / エンドポイントなし） | - | - | [src/desktop/shell.py](../src/desktop/shell.py) / [src/desktop/qml/Overlay.qml](../src/desktop/qml/Overlay.qml) / [src/desktop/app.py](../src/desktop/app.py) / [tests/test_desktop_*.py](../tests/test_desktop_shell.py) / [tests/test_desktop_contract.py](../tests/test_desktop_contract.py) / [tests/test_desktop_qml.py](../tests/test_desktop_qml.py) / [decisions/desktop_quick_chat_hud.md](./decisions/desktop_quick_chat_hud.md) |
| Perception / センサー | ○ | ○ | ○（共有 SensorPolicy の opt-in 配線を Web / Discord / Voice CLI / Desktop / Monitor へ適用。camera・screen_capture・screen_ingest・monitor・activity・microphone・process_details。Discord voice STT・Desktop push-to-talk・Web 音声入力（POST `/api/stt` / WS `audio_message`）は共有マイク gate。Vision / Screen は secret / local_only Context Provider。P0-3.3 fail-safe・P0-3.4 privacy-safe はオフライン分離テスト済み。後続の安全仕上げ（Discord 共通 transcript ゲート・返信生成 revoke・未コミットLLM→原子履歴コミット・通話由来直接登録撤去・Voice CLI カレンダー独立 opt-in `VOICE_CALENDAR_WRITE_ENABLED`・ingest 完了 Event/原子 tmp 置換/revoke 後拒否・Remote / Monitor の stop_pending と storage 順序・Monitor 書込の有界リトライ・VAD/STT の transcript-safe ログ）もオフライン分離テスト済み） | - | - | [src/perception/policy.py](../src/perception/policy.py) / [src/perception/bootstrap.py](../src/perception/bootstrap.py) / [src/monitor/context.py](../src/monitor/context.py) / [src/web/server.py](../src/web/server.py) / [src/audio/main.py](../src/audio/main.py) / [src/discord_bot/bot.py](../src/discord_bot/bot.py) / [src/discord_bot/voice_stt.py](../src/discord_bot/voice_stt.py) / [src/discord_bot/voice_reply_debouncer.py](../src/discord_bot/voice_reply_debouncer.py) / [src/desktop/bridge.py](../src/desktop/bridge.py) / [tests/perception/test_sensor_policy.py](../tests/perception/test_sensor_policy.py) / [tests/audio/test_sensor_policy.py](../tests/audio/test_sensor_policy.py) / [tests/test_discord_sensor_policy.py](../tests/test_discord_sensor_policy.py) / [tests/test_desktop_companion.py](../tests/test_desktop_companion.py) / [tests/context/test_monitor_provider.py](../tests/context/test_monitor_provider.py) / [tests/test_screen_remote.py](../tests/test_screen_remote.py) / [tests/web/test_companion_state_api.py](../tests/web/test_companion_state_api.py) / [tests/web/test_microphone_policy.py](../tests/web/test_microphone_policy.py) / [tests/web/test_sensor_error_safety.py](../tests/web/test_sensor_error_safety.py) / [tests/test_screen_context_lifecycle.py](../tests/test_screen_context_lifecycle.py) / [tests/test_vision_context_lifecycle.py](../tests/test_vision_context_lifecycle.py) / [tests/test_discord_voice_reply_debouncer.py](../tests/test_discord_voice_reply_debouncer.py) / [tests/test_discord_voice_stt.py](../tests/test_discord_voice_stt.py) / [tests/audio/test_vad.py](../tests/audio/test_vad.py) / [tests/audio/test_pipeline_llm.py](../tests/audio/test_pipeline_llm.py) |
| インフラ / systemd | ○（unit定義・service_ctl.sh・phaseスクリプト） | ○（gpu_config / log_setup / backup-restore の分離テスト） | △（unit定義は各エントリポイントへ対応。実機配線・稼働は未検証） | - | - | [scripts/systemd/subpc-web.service](../scripts/systemd/subpc-web.service) / [scripts/service_ctl.sh](../scripts/service_ctl.sh) / [src/service/gpu_config.py](../src/service/gpu_config.py) / [tests/test_gpu_config.py](../tests/test_gpu_config.py) / [tests/test_log_setup.py](../tests/test_log_setup.py) |
| Docker / PostgreSQL (I1) | ○（compose.yaml / .env.example / docker_setup.md） | ○（fake docker を使う分離テスト。実DB・実コンテナ不使用） | △（backup.sh / restore.sh が `POSTGRES_BACKUP_MODE` 対応。アプリ接続は I2 以降） | - | - | [compose.yaml](../compose.yaml) / [.env.example](../.env.example) / [docker_setup.md](./docker_setup.md) / [tests/test_backup_restore_scripts.py](../tests/test_backup_restore_scripts.py) |
| バックアップ / 復元 | ○ | ○（分離テスト） | △（定期実行は未配線。systemd timer は `tailscaled-restart.timer` のみ） | - | - | [scripts/backup.sh](../scripts/backup.sh) / [scripts/restore.sh](../scripts/restore.sh) / [tests/test_backup_restore_scripts.py](../tests/test_backup_restore_scripts.py) / [runbook.md](./runbook.md) |
| 監視 / CI/CD | △（CI: windows-desktop.yml + LinuxオフラインCI `test.yml` とMarkdownローカルリンクチェッカーは実装済み。windows-desktop.yml は最小権限 `permissions: contents: read`・concurrency group / cancel-in-progress・push / pull_request のパスに `src/perception/**` を追加済み（Desktop マイク / activity ゲートが共有 SensorPolicy に依存）。P0-1.2 で静的チェック `git diff --check HEAD^ HEAD`＋root fallback `git show --check --format= --no-renames HEAD` / `compileall src tests`・chat slice 全体探索・stdlib-only の `tests.perception.test_sensor_policy` / `tests.test_calendar_sync` をゲートへ追加済み（discord.py 必須の `tests.test_task_chat_editor` は tasks slice から除去、discord は追加しない）。Prometheus / Grafana / `/metrics` は未実装＝I3計画） | △（Windows runner のデスクトップCI設定。workflow_dispatch・setup・依存導入・offscreen 環境・`test_desktop_*.py` 全件発見は維持。Linuxオフラインslice・静的チェック・リンクチェッカーはWindows開発機でのローカル実行のみ。**hosted Windows runner での実行は未検証**。GitHub / Linux 上での実行も未検証。監視系テストなし） | △（デスクトップEXEビルドのみ。Linux CI / deploy は I5 計画） | - | - | [.github/workflows/windows-desktop.yml](../.github/workflows/windows-desktop.yml) / [.github/workflows/test.yml](../.github/workflows/test.yml) / [scripts/check_markdown_links.py](../scripts/check_markdown_links.py) / [tests/test_markdown_links.py](../tests/test_markdown_links.py) / [infrastructure_plan.md](./infrastructure_plan.md) |
| 個人 LoRA（Phase 13） | ○（学習実行系・ローカルpreflight・切替スクリプト） | ○（dry-run・スタブ・JSON書換のみ。実学習・実モデル不使用） | △（DiscordTrainingLog 収集は配線。モデル切替は config 書換のみでサービス反映未検証） | - | - | [training/dataset.py](../training/dataset.py) / [scripts/switch_chat_model.py](../scripts/switch_chat_model.py) / [training/personal_lora_training.md](./training/personal_lora_training.md) / [tests/test_training_dataset.py](../tests/test_training_dataset.py) / [tests/test_chat_model_switch.py](../tests/test_chat_model_switch.py) |

### Desktop P1 Quick-Chat HUD（補足）

- P1.1 / P1.2 / P1.3（Quick-Chat HUD）はリポジトリ実装＋オフライン static/unit テストで
  implemented / tested / integrated（Desktop app の `Overlay.qml` expanded パネル・既存
  DesktopBridge 経路の再利用）である。決定記録:
  [decisions/desktop_quick_chat_hud.md](./decisions/desktop_quick_chat_hud.md)。
- **count-only プライバシー**: Today サマリは固定 allowlist のみ（`today_points` / `streak_days` /
  open タスク件数 / 今日以降の予定件数）。タスク本文・予定タイトル・プロセス・センサー詳細・パス・
  モデル・`modelData`・`statusText` / `serverUrl` は HUD に露出しない。状態は固定ラベル＋
  `sensor_provenance`（出所・取得時刻・保存なし）のみ。`/api/` `/ws/` `http(s)://` 文字列は
  `Overlay.qml` に含めない。
- **クリックスルーと hotkey 復帰**: HUD は `bridge.setOverlayClickThrough(bool)` の希望状態のみを
  持ち、`overlayClickThroughRequested` → `OverlayClickThroughController`（`src/desktop/app.py`）→
  `apply_click_through`（`src/desktop/windows.py`、WS_EX_LAYERED | WS_EX_TRANSPARENT）が実適用する。
  復帰は **Ctrl+Alt+Space**（`WindowsHotkeyFilter` / `RegisterHotKey`、MOD_ALT|MOD_CONTROL|
  MOD_NOREPEAT, VK_SPACE）で `restore_interaction()` を先に実行してから本体ウィンドウを toggle。
  停止・終了・失敗時は必ず解除（`_force_overlay_click_through_off` / `disconnect` /
  `apply_default_click_through`）。
- 検証: `tests/test_desktop_contract.py`（`OverlayContractTest` / `DesktopClickThroughContractTest`）、
  `tests/test_desktop_shell.py`（`OverlayClickThroughTestCase` / `TestOverlayClickThroughApply` /
  `TestHotkeyInteractionRestore`）、`tests/test_desktop_qml.py`（QML ロード検査）ほか計151件が
  Windows 開発機で成功（`python -m unittest tests.test_desktop_contract tests.test_desktop_shell
  tests.test_desktop_qml tests.test_desktop_companion tests.test_desktop_api tests.test_desktop_avatar
  tests.test_desktop_overlay_vrm -q`）。実Qt/GL・実Windowsウィンドウ・実クリックスルー・実hotkey・
  実サービスでの live 検証は未実施であり、live / deployed / verified は未主張。

### Web status と Discord 音声メモリ境界（P0-3 補足）

- Web の `/api/health` は `providers` 配列を返すが、各 provider entry は到達性・状態の
  allowlist に限られ、モデル識別子 `model` は含めない。`/api/status` も `provider_kind` /
  `provider_reachability` と各機能状態を返すが、`model` / `stt_model` は payload に含めない。
  これは status/health の仕様であり、実機・live 状態の表示を意味しない。Discord の
  `/status` がモデルを表示することとは別の経路である。
- Discord 通話の返信は、revoke されていない場合に限り user+assistant をセッションの
  **インメモリ履歴だけ**へコミットする。RAG と Growth は無効
  (`store_memory=false` / `record_growth=false`)。学習ログと STT transcript のディスク保存は
  それぞれ独立した明示 opt-in で、履歴コミットの有無から推測しない。
- `/voice stop|leave` の revoke は追跡済み autoread タスクをキャンセルし、再生中の playback
  を停止する。revoke 済み返信はインメモリ履歴にもコミットしない。Discord voice transcript の
  admission は現在 listening 中の STT セッション開始時刻以降のメッセージだけを受け付けるため、
  再起動・再開前の旧セッションのメッセージは返信パイプラインへ入らない。

### Tasks 配送一貫性（at-least-once best-effort 契約）

リマインド通知とタスク⇔カレンダー同期の配送契約は
[decisions/task_delivery_consistency.md](./decisions/task_delivery_consistency.md) で固定している。
非センサーの drop / update / snooze 競合は解決済みで、オフライン unit テスト 120件
（store 69 / reminder 19 / calendar_sync 32）で検証済み: rev 楽観制御・`BEGIN IMMEDIATE` claim・
revalidate-before-callback・`expected_rev` 条件付き record で並行変更を上書きせず、state-driven
同期＋マーカー照合/重複整理で crash・queue-drop 後の対応付けを回復する。残る residual:
micro-TOCTOU（revalidate〜コールバック間・二重送信は受け入れ）、`queue.Full` drop と
pull ポーリング間隔による反映遅延（queue/pull latency）、一意 lease owner 名の規律、
exactly-once / durable outbox なし。実機・実 Discord 常駐・実 Google Calendar・実複数プロセス
同時起動での live 検証は未実施（deployed / verified は未主張）。

### Screen agent source gate（P0-3 補足）

Screen agent の source capture は既定オフで、`--enable-screen-capture` または
`SENSOR_SCREEN_CAPTURE_ENABLED=true` が必要。token・`--once`・URL とは独立する。
`scripts/screen_agent.py` と `src/web/static/screen_agent.py` の配布2コピーは byte-identical。
診断は固定文言/型名のみで URL・画像/本文内容を含めない。receiver の
`SENSOR_SCREEN_INGEST_ENABLED=true` + `SCREEN_INGEST_TOKEN` 二重 gate は維持する。
`tests/test_screen_remote.py` のオフラインテストで source gate、コピー同一性、診断安全性を固定。
Windows 開発機で `python -m unittest tests.test_screen_remote -q`（95件）が成功した。
実センサー・実ネットワーク・実サービスは不使用で、live / deployed / verified は未主張。

## 既知のギャップ（監査・修正記録に基づく）

監査・修正記録 `review_fix_steps/` から確認できる既知ギャップ。

| 記録 | 内容 | 記録上の状態 |
| --- | --- | --- |
| [01_audio_generation_alignment.md](../review_fix_steps/01_audio_generation_alignment.md) | `ChatConfig` / `OllamaClient` / 音声系呼び出しの生成パラメータ不整合の解消 | 修正手順として追跡。検証は静的整合確認（`compileall`・シグネチャ確認） |
| [02_history_trim_turn_boundary.md](../review_fix_steps/02_history_trim_turn_boundary.md) | 履歴トリムのターン境界（user / assistant ペアを壊さない） | 修正手順として追跡。検証は小さい `max_history_turns` での挙動確認 |
| [03_strict_model_presence_check.md](../review_fix_steps/03_strict_model_presence_check.md) | Ollama モデル存在チェックの部分一致の厳密化 | 修正手順として追跡。検証は静的比較ロジック確認 |
| [04_discord_voice_stt_corruption_fix.md](../review_fix_steps/04_discord_voice_stt_corruption_fix.md) | Discord voice STT の `OpusError` クラッシュ・Whisper ハルシネーション修正（PLC・フィルタ追加） | 実装済み・オフラインE2E計測済み（相関0.87）。実機サインオフは未実施 |

マトリクス各行のギャップ（△）:

- **中核 / P0-2（交換可能ローカルbackend）**: `LocalOpenAICompatibleProvider`（[src/llm/providers/local_openai.py](../src/llm/providers/local_openai.py)）・config（[src/chat/config.py](../src/chat/config.py)）・factory（[src/assistant/factory.py](../src/assistant/factory.py)）・NodeInventory（[src/assistant/nodes.py](../src/assistant/nodes.py)）・公開設定例（[config/chat_config.local-openai.example.json](../config/chat_config.local-openai.example.json)）・docs は実装・オフライン分離テスト済み（[tests/llm/test_local_openai_provider.py](../tests/llm/test_local_openai_provider.py) / [tests/assistant/test_factory.py](../tests/assistant/test_factory.py) / [tests/assistant/test_nodes.py](../tests/assistant/test_nodes.py) / [tests/chat/test_config.py](../tests/chat/test_config.py)）。既定は Ollama のまま。残ギャップ: 実サーバー（llama.cpp / LM Studio / vLLM）の導入・起動・live 検証、ScreenDescriber のマルチモーダル移行、性能ベンチマークは deferred（deployed / verified は該当なし）
- **Cloud**: request-preview / IDライフサイクル、ストリーミング利用、実送信・外部モデル接続・デプロイ検証は未完了（[assistant_platform_plan.md](./archive/assistant_platform_plan.md) Phase K）
- **Perception / センサー**: P0-3 の共有 SensorPolicy（全センサー既定オフ・canonical env の明示 `true` のみ有効化・canonical 未設定時のみ legacy 互換・token は consent ではない）は Web（camera / screen_capture / screen_ingest / monitor / activity。さらに Web 音声入力: POST `/api/stt` と WS `audio_message` は共有 SensorPolicy.microphone の fail-closed gate。録音はブラウザ側 `getUserMedia`（HTTPS 権限）でサーバーはマイクキャプチャを持たず受信音声のみ文字起こし。`/api/status` の `stt` は engine ロード済みかつ policy true のときだけ True、UI はこれで gate）、Voice CLI（microphone / camera / screen_capture / monitor / activity）、Discord（activity / screen_capture。Discord-local legacy `DISCORD_SCREEN_CONTEXT_ENABLED` は canonical 未設定時のみ参照で、canonical false が legacy true を上書き。voice STT は共有マイク gate）、Desktop（activity / push-to-talk マイク gate）、Monitor（process_details）へ配線済みでオフライン分離テスト済み。`SENSOR_PROCESS_DETAILS_ENABLED` は `MonitorContext` が共有 SensorPolicy から解決し、既定では集計値（`process_count`）のみ収集・保存され、プロセス名・PID・CPUトップ5 は opt-in 時のみ（default redaction / safe aggregates）。`/api/screen/ingest` は policy + token の両方必須で、raw JPEG は保存せず VLM 描写結果のみ `data/screen/latest.json` に保持（レガシー `latest.jpg` は起動・停止・無効状態で best-effort 削除。**絶対の削除保証ではない**）。remote（ingest）は共有 SensorPolicy.screen_ingest が有効なときだけ読取・公開し（無効時は既存の最新描写も get_state / get_status / get_context_text から隠す fail closed）、`source` は常に固定 `"remote"`。P0-3.3（fail-safe: 例外の型名のみログ・機能のみ無効化・`stop_pending` の stop 所有権・CP932 安全）と P0-3.4（privacy-safe: センサー status の allowlist 最小化・生/派生情報・VLM 描写・メトリクス・パス・`last_error` の非公開・生/デバッグ endpoint の固定 404）はリポジトリ実装＋オフライン分離テスト済み。**後続の安全仕上げ**もオフライン分離テスト済み: (a) Discord 共通 transcript ゲート（on_message の parsing 直後・全分岐の前に voice STT listening かつ voice reply gate active を適用し、ゲート外の transcript はタスク/カレンダー直接登録を含む一切の副作用を起こさない）。(b) 通話由来のタスク・カレンダー直接登録ブランチ撤去（transcript は全てデバウンス→LLM返信パイプラインのみ。テキストチャット側の直接登録は従来どおり維持）。(c) 返信生成 revoke / 原子履歴コミット（`ask_voice_transcript` の一時追加→必ず除去の未コミットLLM生成、世代同期再チェック後の user+assistant のセッションロック下コミット、revoke 済み世代は副作用なし、`/voice stop|leave` は STT 停止より先に `VoiceReplyGate` を revoke）。`tests/test_discord_voice_reply_debouncer.py` / `tests/test_discord_voice_stt.py` で固定。(d) Voice CLI カレンダー書き込みの独立 opt-in（`VOICE_CALENDAR_WRITE_ENABLED=true` のみ有効・既定 false / fail closed。マイク同意だけでは書き込まず、無効時は通常 LLM 経路へフォールスルーしクライアント自体を構築しない）。(e) ingest 完了 Event・原子 tmp 置換・revoke 後拒否（世代ごとの `threading.Event` で完了判定・`latest.json.tmp` への write/fsync → lock 区間での `os.replace`・revoke 後は受付を登録せず固定 503＋`unavailable` で拒否・revoke 済み世代のコミットは起こらない）。(f) Remote / Monitor の stop_pending と storage 順序（stop は先に worker を stop/join、生存中の間は storage を触らず・置き換えず・閉じず ownership 保持・死確認後のみ解放/再 start）。(g) Monitor 書込の有界リトライ（`write_attempts` 既定 3 回・有界バックオフ・exhaustion 時のみ固定カウンタと型名のみの診断・キューなし）。(h) VAD/STT の transcript-safe ログ（認識テキストは既定のパイプライン診断・ログへ出力しない）。配線はリポジトリ内のみ。**live 検証は未実施**: 実センサー・実X11・実マイク・実メインPC push・実サービスでの動作確認はなく、deployed / verified は該当なし。一般の未認証 Web セキュリティ負債（センサー以外のエンドポイントの認証等）は本センサー opt-in 方針とは別の課題として扱う。TaskCalendarSync / reminder の drop 競合はセンサーとは別の非センサー後続バックログとし、この P0-3 の状態（implemented / tested / integrated）には含めない
- **Voice 文脈整合 / P0-4（CLI / Voice CLI / Voice パイプラインの構成 parity）**: [decisions/voice_context_parity.md](./decisions/voice_context_parity.md) の通り、CLI（`src/chat/main.py`）/ Voice CLI・audio-text（`src/audio/main.py`）/ Voice パイプライン（`src/audio/pipeline.py`）は同一の共通構成（`ChatSession.build_blocks` → `AssistantService.respond_stream` → `ContextBuilder.build_messages`、base prompt / history・emotion・ChatConfig 生成パラメータ・stream close・rollback・persistence・共通 provider web_search / growth_tracker）を共有し、request の channel / profile は経路識別子（`channel="voice"` / `profile="voice_fast"` / `privacy="local_only"`）のみ差異。Voice の RAG / persona / tasks / calendar と P0-3 で gated されるセンサー（vision / screen / monitor / activity / microphone）は**チャネル能力拡張**（共通構成の順序・セマンティクスを変えない）であり共通構成の変更ではない。RAG / persona は既定で有効（`--no-rag` / `--no-persona` で無効化）、task_store / calendar_context は既定構築・データ/設定利用可能時のみ文脈を発話（読み取り専用）、カレンダー書込は `VOICE_CALENDAR_WRITE_ENABLED` 明示 `true` のみ、P0-3 センサーのみ既定オフ opt-in。実装・統合・オフライン分離テスト済み（`tests/audio/test_context_parity.py` / `tests/audio/test_pipeline_llm.py` / `tests/chat/test_session_build_blocks.py` / `tests/chat/test_cli_loop.py`）。残ギャップ: 実機 live 検証（実マイク・実TTS・実サービスでの parity 確認）、Voice の RAG / persona / tasks / calendar の既定挙動変更（現行挙動が既定のため、将来の既定変更は性能・プライバシー検討を経た方針判断）、P0-3 センサーの Voice 既定有効化、Web / Discord との構成完全統一は deferred（deployed / verified は該当なし）
- **Desktop / 3D**: P1 Quick-Chat HUD（`src/desktop/qml/Overlay.qml` の expanded パネル・既存
  DesktopBridge 経路再利用・count-only・desired-state クリックスルー＋Ctrl+Alt+Space hotkey 復帰）は
  リポジトリ実装＋オフライン static/unit テスト済み（`tests/test_desktop_contract.py` /
  `tests/test_desktop_shell.py` / `tests/test_desktop_qml.py` ほか計151件成功）。決定記録:
  [decisions/desktop_quick_chat_hud.md](./decisions/desktop_quick_chat_hud.md)。実Qt Quick 3D
  レンダリング・実Windowsウィンドウ・実クリックスルー・実hotkey・GPU挙動・デプロイ検証は未実施。
  Phase 6b残作業（VRM1.0表情・視線・リップシンク）は planned
- **インフラ / systemd**: 実機適用・OS再起動後の自動復旧・GPU電力制御・WatchdogSec の稼働は未検証
- **Docker / PostgreSQL (I1)**: I1受け入れ（実機Docker導入・再起動後自動起動・実`pg_dump`/`pg_restore`訓練）は未完了。`docker_setup.md` は draft / supporting
- **バックアップ / 復元**: 実`pg_dump`/`pg_restore`・月1復元訓練・定期自動化は未実施（[runbook.md](./runbook.md) は draft / supporting）
- **監視 / CI/CD**: `/metrics`・Grafana dashboard・alerts・deploy workflow は未実装（I3 / I5 / I6）。LinuxオフラインCI（`test.yml`）は P0-1.2 として静的チェック（HEADパッチの空白検査 `git diff --check HEAD^ HEAD`＋root fallback `git show --check --format= --no-renames HEAD` / `python -m compileall src tests`）・`tests/chat` 全体探索・stdlib-only の `tests.perception.test_sensor_policy` / `tests.test_calendar_sync` をゲートへ追加済み（discord.py 必須の `tests.test_task_chat_editor` は tasks slice から除去し、discord は追加しない）。`actions/checkout@v4` は `fetch-depth: 2` のまま（通常・PR合成mergeとも first-parent→HEAD の差分＝変更全体を `git diff --check` で検査し、combined diff のみ見る `git show` は使わない。HEAD が root の時のみ fallback）。Windows開発機でのローカル実行のみ。GitHub / Linux 上での workflow 実行確認は未実施（P0-1.3 の外部ブランチ保護は計画のまま）。Markdownリンクチェッカーも実装済みでローカル実行のみ
- **個人 LoRA**: H200実学習・実マージ・GGUF変換・Ollama登録・実評価は未実施。`min_dataset_rows` の件数強制と `merge_adapter.py` の `peft_type=LORA` / 禁止module拒否は未実装

## 関連文書

- [infrastructure_plan.md](./infrastructure_plan.md): 計画・次アクションの正典（I1〜I6）
- [decisions/task_delivery_consistency.md](./decisions/task_delivery_consistency.md): タスク配送一貫性の決定記録（リマインド・カレンダー同期の at-least-once best-effort 契約）
- [plans/companion_roadmap.md](./plans/companion_roadmap.md): 製品方向・ロードマップの正典
- [archive/assistant_platform_plan.md](./archive/assistant_platform_plan.md): 基盤実装の完了記録（リポジトリ実装の記録でありデプロイ証拠ではない）
- [runbook.md](./runbook.md): デプロイ前運用ドラフト（draft / supporting）
- [docker_setup.md](./docker_setup.md): Docker / Postgres セットアップ手順ドラフト（draft / supporting）