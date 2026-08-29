# 実装計画 (Implementation Plan)

> **状態**: active / canonical
> **位置付け**: 実装タスク順序の正典（actionable plan）。**実装状態の正典は implementation_status.md**
> **対象範囲**: P0基盤（Linux CI品質ゲート / 交換可能ローカル推論backend / センサーopt-in強制 / Voice CLI文脈整合）と P1 Desktop quick-chat HUD、および前提未達の延期項目
> **作成日**: 2026-08-28
> **更新日**: 2026-08-29
> **日付根拠**: Git commit date

## 位置付け

- 本計画は**タスク順序の正典**である。各タスクの実際の実装・テスト・統合状態は
  [implementation_status.md](./implementation_status.md)（6段階マトリクス）を正とする。
  **本計画は deployed / verified を主張しない**（現状該当なし）。
- [infrastructure_plan.md](./infrastructure_plan.md)（I1〜I6）はインフラ固有の計画として並存し、
  本計画の延期項目（PostgreSQL / 監視 / deploy）と対応する。重複するP0-1（Linux CI）は
  本計画を品質ゲートとして先に置き、I5 の deploy 連鎖とは別扱いとする。
- [plans/companion_roadmap.md](./plans/companion_roadmap.md) は製品方向・Phase 1〜7 の正典であり、
  本計画はその中の実行単位（特に Phase 6b 残作業と Phase 7）を P0 / P1 の順序で具体化する。
- **実行環境**: 現時点は Windows 開発機でのリポジトリ実装のみ。サブPC実機・実GPU・実サービスへの
  適用・検証は未実施。夜間の連続実行や無人での外部サービス操作を前提としない。

## 進捗の表記と検証ゲート

- 各タスクの**状態**は `planned`（実施予定）/ `blocked`（前提未達）/ `in_progress` / `done` で示す。
  `done` は「リポジトリ実装＋分離テスト済み」であり、deployed / verified ではない。
  実際の6段階状態は implementation_status.md に反映する。
- **検証ゲート**: 変更範囲の unit テスト（Windows開発機では
  `python -m unittest discover -s tests/<対象> -t . -q`）。実モデル・実GPU・実サービス・
  実ネットワーク・実データは使わない（リポジトリの既存慣行）。
- **レビューゲート**: 差分・要件レビューは実装者とは独立に行う。実装者の自己申告だけで
  `done` としない。
- **rollback / 互換**: 各タスクに「戻し方」と「無設定時の挙動（既定）」を明記する。
  既定は既存挙動を維持する。

## マイルストーン概要

| マイルストーン | 内容 | 依存 |
| --- | --- | --- |
| P0-1 | オフラインLinux CI品質ゲート | なし（独立して着手可） |
| P0-2 | 交換可能なローカル推論backend（Ollama既定を維持） | なし（P0-1と並行可） |
| P0-3 | センサーオプトイン強制 | なし（P0-1と並行可） |
| P0-4 | Voice CLI 文脈整合（Context parity） | なし（P0-1と並行可） |
| P1 | Desktop quick-chat HUD（高度3Dより先） | P0-2（backend安定）が目安 |

## P0-1: オフライン Linux CI 品質ゲート

**目的**: Windows 実行環境依存の回帰と、Linux 基準（サブPC相当）との乖離を push 時点で検出する。
**オフライン** = テストは外部サービス・実モデル・実GPU・実ネットワークを使わない（既存の分離テスト前提）。
deploy は含めない（deploy は延期項目 I5）。

| ID | タスク | 依存 | 変更対象（予定ファイル領域） | 受け入れ条件 | 検証/レビューゲート | rollback/互換 | 状態 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P0-1.1 | オフラインslice単位の `test.yml`（Linux CI品質ゲート）とMarkdownローカルリンクチェッカーを追加 | なし | `.github/workflows/test.yml`（新規）、`requirements-ci.txt`、`scripts/check_markdown_links.py`、`tests/test_markdown_links.py` | `test.yml` が ubuntu / Python 3.12 で決定的オフラインslice（LLM / assistant / context / chat / tasks / perception policy / markdown。詳細は `test.yml` 参照）とリンクチェッカーを実行し、失敗で赤 | 本実装時のローカル実行（下記メモ）／独立レビュー | workflow削除・スクリプト削除で元へ。実装資産は変更しない | done |

**P0-1.1 実装/検証メモ（2026-08-28、本実装worker記録）**
- **実装済み**: `test.yml`（permissions read-only、concurrency cancel、timeout 15分）、`requirements-ci.txt`（`httpx>=0.27,<0.29` のみ。sliceのimport調査の結果、非stdlib依存はhttpxのみ）、`scripts/check_markdown_links.py`、`tests/test_markdown_links.py`
- **ローカル検証（Windows開発機）**: `tests/llm`（75）、`tests/assistant`（121）、`tests/context`（191）、`tests.chat.test_config`＋`test_task_decomposer`＋`test_prioritizer`（33）、`tests/test_markdown_links`（30）が全て成功。`scripts/check_markdown_links.py docs readme.md` はexit 0（現状壊れリンクなし）
- **クロスプラットフォーム `%-m`/`%-d` 問題（解消済み）**: 下記「決定」の通り `src/tasks/formatting.py` を datetime 属性組み立てへ修正し、`%-m`/`%-d` 非依存とした。`tests/test_task_formatting.py` をCIのtasks/config sliceへ追加し、Windows開発機でオフライン検証済み。GitHub / Linux 上での実行確認は未実施であり verified とはしない
- **決定（2026-08-28、CI dry-run由来のクロスプラットフォーム修正）**: Linux CI を想定した dry-run（Windows開発機での `%-m` 書式の検証）で、`strftime("%-m/%-d")` が Windows 非対応であることが露見した。`src/tasks/formatting.py` の `format_short_due` を `strftime` 依存から datetime 属性（`f"{month}/{day}"`、時刻は `:02d` ゼロ埋め）による組み立てへ修正し、`%-m`/`%-d` に依存しないクロスプラットフォーム実装とした。`tests/test_task_formatting.py` を新設し、単桁/2桁の月日・ゼロ埋め HH:MM・timezone-aware 入力の保持・date/datetime 粒度の表示差（`format_local_due`）・UTC ISO 保存の瞬間保持（`to_iso`/`from_iso`）を固定した。GitHub CI での workflow 実行確認は依然未実施であり、verified とはしない
- **未実施（本タスクの範囲外）**: GitHub上でのworkflow実行確認は未実施（CIが走ったと主張しない）。ブランチ保護の必須化（P0-1.3）は実施していない
| P0-1.2 | 静的品質チェック（HEADパッチの空白検査 `git diff --check HEAD^ HEAD`＋root fallback `git show --check --format= --no-renames HEAD`、`python -m compileall src tests`）をゲートへ追加し、ゼロ依存カバレッジギャップ（chat slice 全体探索・stdlib-only の `tests.perception.test_sensor_policy` / `tests.test_calendar_sync` を追加、discord.py 必須の `tests.test_task_chat_editor` はtasks sliceから除去）を閉じる | P0-1.1 | `.github/workflows/test.yml`（`pyproject.toml` は変更不要） | 空白エラー・構文エラーで赤になる。追加sliceがオフラインunitジョブで実行される | 静的チェックの失敗注入と追加sliceの実行（ローカル） | 該当step削除で復帰 | done |
| P0-1.3 | main 保護で `test.yml` を必須化（リポジトリ設定） | P0-1.1 | リポジトリ設定（GitHub UI。repo外のため手順を計画に残す） | main merge に Linux テスト通過が必須 | 設定変更の記録確認 | 保護ルール解除で復帰 | planned |

**P0-1.2 実装/検証メモ（2026-08-29、本実装worker記録）**
- **実装済み（`test.yml` のみ変更）**: 静的チェック2ステップをゲート先頭へ追加（checkout直後に
  HEADコミットパッチの空白検査 `git show --check --format= --no-renames HEAD`、setup-python
  後にインストールより先へ `python -m compileall -q src tests`）。
  chat 設定のみの呼び出し（`tests.chat.test_config`）を `python -m unittest discover -s tests/chat
  -t . -q` の全体探索へ置換し、tasks slice へ stdlib-only の `tests.test_calendar_sync` を追加
  （discord.py 必須の `tests.test_task_chat_editor` は依存分離で除外。discord は追加しない）、
  新規 slice `tests.perception.test_sensor_policy` を追加。llm / assistant / context の slice は維持。
  numpy / sounddevice / discord / portaudio や audio-heavy な suite は追加していない。
  permissions（contents: read）/ concurrency（cancel-in-progress）/ timeout（15分）/
  全ブランチpush / pull_request / workflow_dispatch は現状維持
- **空白ゲートの有効化（2026-08-29 追記）**: clean-checkout で no-op だった `git diff --check`
  （working-tree diff は常に空）を HEADコミットパッチの空白検査
  `git show --check --format= --no-renames HEAD` へ置換（ネットワーク・base履歴・fetch-depth 0
  不要）。ローカル検証: 空白エラー注入コミットで exit 2、クリーンコミットで exit 0、
  depth-1 shallow checkout（親履歴なし）でも動作。注意: depth-1 では HEAD が root 扱いとなり
  追跡済みファイル全体の空白も検出するため、過去から残る trailing whitespace が
  `docs/references/gitsugest.md`・`review_fix_steps/` にある
- **浅いチェックアウトの修正（2026-08-29、`fetch-depth: 2` 追加）**:
  `actions/checkout@v4` へ `fetch-depth: 2` を追加し、`git show --check --format= --no-renames
  HEAD` が親を持って HEAD パッチのみを検査するようにした（depth-1 の HEAD=root 扱いによる
  追跡済みファイル全体の空白検出を回避）。base ブランチ依存は持たせず、他ステップ・
  permissions / concurrency / timeout・全ブランチpush / pull_request / workflow_dispatch は
  現状維持。ローカル検証: 親コミットに trailing whitespace を残し HEAD パッチがクリーンな
  depth-2 clone で exit 0、HEAD パッチに空白を注入した depth-2 clone で exit 2、YAML 構造
  パース確認。GitHub / Linux 上での実行確認は未実施（verified は主張しない）
- **ローカル検証（Windows開発機・オフライン）**: YAML構造パース確認（push / pull_request /
  workflow_dispatch・permissions・concurrency・timeout・全steps）。`git show --check --format=
  --no-renames HEAD` exit 0（HEAD自身のパッチは空白クリーン）、
  `python -m compileall -q src tests` exit 0。`tests/chat`（96件）・tasks slice
  （`tests.test_task_decomposer` / `tests.test_prioritizer` / `tests.test_tasks_store` /
  `tests.test_tasks_reminder` / `tests.test_task_formatting` / `tests.test_calendar_sync`、166件）・
  `tests.perception.test_sensor_policy`（29件）が全て成功。既存の `tests/llm`（167件）・
  `tests/assistant`（169件）・`tests/context` も再実行し全て成功。discord を import 不能にした
  スタブ環境で `tests.test_task_chat_editor` は ImportError となることを確認した（除去根拠）
- **未実施（hosted / external）**: GitHub / Linux 上での workflow 実行確認は未実施（CIが走ったと
  主張しない）。P0-1.3 の外部ブランチ保護（GitHub UI設定）は計画のまま
- **依存分離の修正（2026-08-29 追記、httpx-only tasks slice の discord 依存ギャップ）**:
  `tests.test_task_chat_editor` は import 連鎖 `src/tasks/chat_editor.py` →
  `src/discord_bot/task_ui.py` → `import discord` で discord.py 必須のため、httpx-only の
  Linux tasks slice から除去した（discord は追加しない）。代わりに stdlib-only の
  `tests.test_calendar_sync`（`src/tasks/calendar_sync.py` / `src/integrations/google_calendar.py` /
  `src/integrations/mcp_stdio.py` は stdlib のみ）を tasks slice へ追加し、タスク⇔カレンダー
  一貫性（rev 楽観制御・state-driven 同期・マーカー照合/重複整理）の回帰をゲートへ戻した。
  requirements-ci.txt のコメントも実 slice 一覧（llm / assistant / context / chat / tasks /
  perception policy / markdown）へ更新した
- **空白ゲートの権威化（2026-08-29 追記）**: `git show --check --format= --no-renames HEAD` は
  merge コミットでは combined diff のみ検査し、PR 合成 merge が持ち込んだ変更全体の空白を
  取りこぼすため、`git diff --check HEAD^ HEAD` を権威とした（`fetch-depth: 2` のまま。PR 合成
  merge では first-parent（base tip）→HEAD の差分＝merge の変更全体、通常コミットではその
  コミットのパッチ）。HEAD が root（親なし）の時のみ fallback `git show --check --format=
  --no-renames HEAD`。ローカル検証: 通常コミットクリーンで exit 0、空白注入コミットで exit 2、
  root コミットで fallback exit 0、合成 merge（`--no-ff`）で merge に持ち込んだ空白を
  `git diff --check HEAD^ HEAD` が exit 2 で検出（combined の `git show --check` は exit 0 で
  不検出＝権威化の根拠）。YAML 構造パース確認

**関連**: [infrastructure_plan.md](./infrastructure_plan.md) I5 の `test.yml` は deploy 連鎖に含まれる。
本計画の P0-1 は image build / deploy を含まない品質ゲートであり、I5 より先に単独で完了できる。

## P0-2: 交換可能なローカル推論 backend

**目的**: Ollama 以外のローカル推論サーバー（llama.cpp / LM Studio / vLLM）を
OpenAI-compatible 契約で使い、ハードウェア・運用制約に応じた backend を選べるようにする。
決定記録: [decisions/local_inference_backend.md](./decisions/local_inference_backend.md)。

| ID | タスク | 依存 | 変更対象（予定ファイル領域） | 受け入れ条件 | 検証/レビューゲート | rollback/互換 | 状態 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P0-2.1 | key 任意（keyless対応）の `LocalOpenAICompatibleProvider` を追加 | なし | `src/llm/providers/local_openai.py`（新規）、`src/llm/providers/__init__.py` | `/v1/chat/completions` 互換への生成・ストリームが共通契約で動き、key無指定時に Authorization を送らない | `tests/llm/` に FakeTransport 経由の unit テスト。実サーバー不使用 | 新規 module のため既存影響なし | done |
| P0-2.2 | ローカル provider を `local=True` で Registry / Router へ登録し、cloud の承認・redaction セマンティクスを受けないことを保証 | P0-2.1 | `src/llm/registry.py`、`src/llm/routing/static.py`、`tests/llm/` | ローカル provider は CloudRouteBridge / 承認フローを通らず、context は local target のみ | ルーティング決定の unit テスト | 登録を戻せば従来挙動 | done |
| P0-2.3 | factory / config で backend 選択を可能にし、**無設定時は Ollama 既定のまま** | P0-2.2 | `src/assistant/factory.py`、`src/chat/config.py`、`tests/assistant/`、`tests/chat/`、設定例（`.env.example` / `config` の公開例のみ） | 既定パスの挙動が不変（既存テストが全て通る）。設定時のみ新 backend | 既定パス回帰テスト＋新backend unit テスト | 設定を外すだけで Ollama へ戻る | done |
| P0-2.4 | ローカルbackendの入口・例・docs統合（provider側の扱いは実装済み） | P0-2.1 | `src/llm/providers/local_openai.py`、`src/chat/config.py`、`src/assistant/nodes.py`、エントリポイント、`config/chat_config.local-openai.example.json`（公開例）、docs | provider側・エントリポイント・設定例・docs統合は実装・オフライン分離テスト済み: `is_available` はlifecycleのみ（ネットワークprobeなし）、`list_models` / `has_model` は `/v1/models` のベストエフォート探索で失敗時は空（chatのみのサーバーも利用可）、`num_ctx` 等は明示的に無視。`local_base_url` 空時の慣用既定エンドポイントは `http://localhost:8080/v1`。NodeInventory配線は実装・テスト済み（`provider_kind` 既定 `ollama` / `openai_compatible`、`api_key_env` は環境変数名のみ・非保存、openai_compatible はloopback限定・常に `local=True` 登録、未知kind拒否、factory注入互換維持）。公開設定例 `config/chat_config.local-openai.example.json`（loopback `http://localhost:8080/v1`・プレースホルダ model・key env 空・persona/channel データなし）と docs（plan / status / usage / readme）を統合。通常経路 CLI / Web / Discord / Voice / batch（日記・日次パーソナライズ）と NodeInventory をオフライン分離テストで確認。実サーバー（llama.cpp / LM Studio / vLLM）の導入・起動・live 検証、ScreenDescriber のマルチモーダル移行、ベンチマークは deferred | unit テスト | 新規のため既存影響なし | done |

**P0-2.4 実装/検証メモ（2026-08-28、本実装worker記録）**
- **実装済み**: 公開設定例 `config/chat_config.local-openai.example.json`（`local_provider_kind=openai_compatible`、loopback `http://localhost:8080/v1`、`local_provider_id` / `local_api_key_env` は空、プレースホルダ model、persona / user / channel データなし）を追加。`usage.md`（backend切り替え手順・loopback信頼境界・key env名・MockTransport/live区別・外部startupコマンドの例示明記）・`readme.md`（交換可能backend対応の要約）・`implementation_plan.md` / `implementation_status.md` を統合
- **ローカル検証（Windows開発機）**: `ChatConfig.load` で公開例を読み込み `validate_local_provider()` が成功（resolved id `local-openai`、URL `http://localhost:8080/v1`、key None）。`tests/chat/test_config` / `tests/assistant/test_factory` / `tests/llm/test_local_openai_provider` / `tests/assistant/test_nodes` / `tests/assistant/test_nodes_cli` / `tests/audio/test_backend_selection` / `tests/test_markdown_links` と `scripts/check_markdown_links.py docs readme.md` が全て成功
- **未実施（deferred）**: llama.cpp / LM Studio / vLLM の導入・起動・実接続（live 検証）、ScreenDescriber のマルチモーダル移行、性能ベンチマーク。deployed / verified の主張なし

**このマイルストーンでやらないこと**: ScreenDescriber のマルチモーダル移行（決定により Ollama のまま）、
性能主張（ベンチマーク未実施）。

## P0-3: センサーオプトイン強制

**目的**: カメラ・画面・活動収集などは既定で無効とし、明示的なオプトインでのみ動作することを
コードとテストで強制する。既存の `COMPANION_ACTIVITY_ENABLED=true` オプトイン方針を全センサーへ徹底する。

| ID | タスク | 依存 | 変更対象（予定ファイル領域） | 受け入れ条件 | 検証/レビューゲート | rollback/互換 | 状態 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P0-3.1 | 共有 SensorPolicy 解決器（opt-in 強制の基盤）を実装 | なし | `src/perception/policy.py`（新規）、`tests/perception/test_sensor_policy.py`（新規）、`docs/decisions/sensor_opt_in_policy.md`（新規） | 全センサー既定オフ、canonical env の明示 `true` のみ有効化、legacy 互換（canonical 未設定時のみ）、token 単独では有効化しない。payload は boolean と sensor source 名のみ | `tests/perception/test_sensor_policy` の unit テスト（オフライン） | 解決器を戻せば従来挙動（入口配線なしのため既存挙動不変） | done |
| P0-3.2 | ガード未適用の入口へオプトイン強制を実装（既定オフ） | P0-3.1 | 入口監査で判明した入口（`src/perception/`、`src/companion/`、`src/screen/`、`src/web/server.py`、`src/discord_bot/`、`src/audio/`、`src/desktop/`、`src/monitor/`） | どの経路でも既定は無効で、有効化には明示設定が必要 | `tests/perception/`、`tests/companion/`、`tests/test_web_*`、`tests/audio/`、`tests/test_discord_*`、`tests/test_desktop_*`、`tests/test_screen_remote.py`、`tests/context/test_monitor_provider.py` の unit テスト（オフライン） | ガードを既定オンに戻さない（戻す場合は決定記録を要する） | done |
| P0-3.3 | 未設定・利用不可センサーの fail-safe（例外型名のみログ、機能のみ無効化、本体は継続） | P0-3.2 | `src/perception/`、`src/companion/`、各入口配線 | 失敗でアプリ・会話が止まらない（既存方針を維持） | 例外注入 unit テスト | 変更なし | done |
| P0-3.4 | privacy-safe 検証（生データ・本文・プロセス名等が payload に漏れない）をテストで固定 | P0-3.2 | `tests/perception/`、`tests/companion/`、`tests/test_web_*` | 生データ非保存・privacy-safe payload のみ公開が既定で検証される | 監査・差分レビュー | 変更なし | done |

**P0-3.1 実装/検証メモ（2026-08-28、本実装worker記録）**
- **実装済み**: `src/perception/policy.py`（frozen `SensorPolicy` ＋ `resolve_sensor_policy` / `parse_opt_in`）。canonical 名 `SENSOR_*_ENABLED` 7種を共有解決し、既定は全て False（fail closed）。有効化は明示 `true` のみ。canonical 未設定時のみ legacy（`WEB_SCREEN_CONTEXT_ENABLED`→screen_capture、`COMPANION_ACTIVITY_ENABLED`→activity）を参照し、canonical の false は legacy の true を上書き。token（`SCREEN_INGEST_TOKEN`）単独では screen_ingest を有効化しない。公開 API は `is_enabled` / `enabled_sensor_ids` / `as_status_payload`（boolean と sensor source 名のみ）で、env 名・値・token を公開しない
- **決定記録**: `docs/decisions/sensor_opt_in_policy.md`（全センサー既定オフ / CLI の mic・camera・monitor は affirmative flag か canonical env 必須 / token は認証＋opt-in であり consent ではない / process・PID 詳細は別センサー / UI 非表示は停止ではない・停止は資源解放 / remote ingest の raw 画像保持は削除クリーンアップ必須）
- **ローカル検証（Windows開発機）**: `tests/perception/test_sensor_policy` 28件が全て成功。`tests/perception` 全体 143件も成功。実機・実サービス・実モデル・実GPUは不使用（deployed / verified は主張しない）
- **入口配線は本タスクの範囲外**: 入口監査と SensorPolicy 適用は P0-3.2 で行う。`src/perception/__init__.py` への export 配線も P0-3.2 に含める

**P0-3.2 実装/検証メモ（2026-08-28、本実装worker記録）**
- **実装済み（リポジトリ配線）**: 共有 SensorPolicy を各入口へ適用し、既定オフを強制した。
  - Web（`src/web/server.py`）: `SENSOR_CAMERA_ENABLED`（Vision）/ `SENSOR_SCREEN_CAPTURE_ENABLED`（Screen local/remote）/ `SENSOR_MONITOR_ENABLED`（Monitor）/ `SENSOR_ACTIVITY_ENABLED`（companion）で gate。`/api/screen/ingest` は `SENSOR_SCREEN_INGEST_ENABLED=true` と `SCREEN_INGEST_TOKEN` の両方が無ければ 403。受信 raw JPEG は保存せず、VLM 描写結果のみ `data/screen/latest.json` へ。レガシー `latest.jpg` は起動・停止・無効状態で best-effort 削除
  - Voice CLI（`src/audio/main.py`）: `--microphone` / `--camera` / `--screen` / `--monitor` の affirmative flag（起動一回限りの同意）と canonical env を合成。音声対話はマイク同意が無ければパイプライン・STT・音声デバイス・活動収集より前に終了。`--no-vision` / `--no-monitor` は明示的な無効上書き（非推奨）
  - Discord（`src/discord_bot/bot.py`）: activity を `create_activity_runtime_from_env`（SensorPolicy.activity）で gate。screen_capture は `discord_screen_capture_enabled()`（canonical 最優先、canonical 未設定時のみ Discord-local legacy `DISCORD_SCREEN_CONTEXT_ENABLED`、Web legacy は Discord を有効化しない）
  - Desktop（`src/desktop/bridge.py`）: activity を `create_activity_runtime_from_env` で gate。停止（stop）はタイマー停止と runtime の資源解放を伴う（UI 非表示だけでは収集は止まらない）
- **ローカル検証（Windows開発機・オフライン）**: `tests/perception/test_sensor_policy`（28件）、`tests/audio/test_sensor_policy`、`tests/test_discord_sensor_policy`、`tests/audio/test_companion_wiring`、`tests/test_discord_companion_wiring`、`tests/test_desktop_companion`、`tests/test_screen_remote`（ingest gate / latest.jpg 非保持）、`tests/web/test_companion_state_api` を実行し全て成功。実センサー・実機・実サービス・実モデル・実ネットワークは不使用（deployed / verified は主張しない）
- **P0-3.2 の後続追加配線（2026-08-28、本実装worker記録）**: P0-3.2 の後続で以下を
  追加配線し、オフライン分離テストで固定した。(a) `SENSOR_PROCESS_DETAILS_ENABLED` は
  `src/monitor/context.py` の `MonitorContext` が共有 SensorPolicy から解決し、
  Web (`_init_monitor_from_policy`) / Voice pipeline / Discord proactive の Monitor
  構築で適用（既定オフ。集計値 `process_count` のみ常時、プロセス名・PID・CPUトップ5 は
  opt-in 時のみ収集・保存）。`tests/context/test_monitor_provider.py` で固定。
  (b) Discord 通話STT は `DISCORD_VOICE_STT_ENABLED=true` に加え、共有 SensorPolicy の
  マイク gate（`SENSOR_MICROPHONE_ENABLED=true`）を第2ゲートとして要求し、どちらかが
  false なら `/voice join|start` は接続前に却下（`VoiceSTTConfig.from_env` が
  `resolve_sensor_policy().microphone` を読む）。`tests/test_discord_sensor_policy.py` で固定。
  (c) Desktop の push-to-talk マイク（`src/desktop/bridge.py` の `startRecording`）は共有
  SensorPolicy.microphone が false なら録音を開始しない。`tests/test_desktop_companion.py`
  で固定。(d) Voice CLI の `--text-mode` はマイク同意を要求せず、
  `SENSOR_ACTIVITY_ENABLED=true` でも活動収集を構築・開始しない。
  `tests/audio/test_sensor_policy.py`・`tests/audio/test_companion_wiring.py` で固定。
  上記はリポジトリ配線と分離テストの範囲であり、live / deployed / verified は主張しない
- **残ギャップ（明示）**: (1) 実センサー・実X11・実マイク・実メインPC push・実音声デバイス
  の live 検証は未実施。(2) Web 音声入力の録音はブラウザ側 `getUserMedia` で行うため
  サーバー側マイクキャプチャは存在しない（マイク権限は HTTPS のブラウザプロンプト）が、
  サーバー側 STT の受付（POST `/api/stt` / WS `audio_message`）は共有 SensorPolicy の
  microphone gate を必須とし、ブラウザ権限だけでポリシーを迂回しない。これは live /
  deployed / verified ではない

**P0-3.2 Screen agent source gate 補足（2026-08-28、本実装worker記録）**
- Screen agent は source capture を既定オフとし、`--enable-screen-capture` または
  `SENSOR_SCREEN_CAPTURE_ENABLED=true` が必要。これは token、`--once`、URL と独立する。
  `scripts/screen_agent.py` と `src/web/static/screen_agent.py` の配布2コピーは byte-identical。
  診断は固定文言/型名のみで URL・画像/本文内容を含めない。
- receiver 側の `SENSOR_SCREEN_INGEST_ENABLED=true` + `SCREEN_INGEST_TOKEN` 二重 gate は維持。
  `tests/test_screen_remote.py` の offline unit tests で source gate、コピー同一性、診断安全性を固定。
  Windows 開発機で `python -m unittest tests.test_screen_remote -q`（95件）が成功した。
  実センサー・実ネットワーク・実サービスは不使用で、live / deployed / verified は主張しない。

**P0-3.3 / P0-3.4 実装/検証メモ（2026-08-28、本実装worker記録）**
- **実装済み（リポジトリ実装＋オフライン分離テスト）**:
  - P0-3.3（fail-safe）: センサー初期化失敗・status 取得失敗・ingest 描写失敗・STT 失敗時に
    例外の型名（allowlist）のみログ/レスポンスし、例外本文・パス・token・画像・本文テキストは
    露出しない（`_sensor_error_response` / `_safe_get_status` / `_init_*_from_policy` の
    try/except で None 化し Web 起動を継続）。`/api/vision|screen/monitor` の status は
    例外時 500 で `error_type` のみ。start/stop のライフサイクルは join タイムアウト時に
    ownership を保持し `stop_pending` を公開、確認死後にのみ解放・再 start を許可する
    （Screen / Vision / Monitor / RemoteScreen / Discord voice STT）。`/voice stop` は
    処理中・待機中の音声を discard して即時停止する。センサー診断ログは ASCII-only で
    CP932（Windows-31J）厳格エンコードに失敗しない。
  - P0-3.4（privacy-safe）: `/api/vision/status`・`/api/screen/status`・`/api/monitor/status`・
    `/api/status` の各センサー status を allowlist（bool / タイムスタンプ / source 種別）に
    最小化。VLM 描写テキスト・モデル名・メトリクス集計値・プロセス数・レコード数・パス・
    `last_error` は未認証 Web API から除外。`/api/screen/ingest` の VLM 描写は status に
    含めない（`latest.json` のみ）。`/api/vision/snapshot`・`/api/vision/context`・
    `/api/screen/context`・`/api/monitor/context`・`/api/monitor/summary` は廃止され固定 404。
    remote（ingest）の `source` は latest.json の中身に関わらず常に固定 `"remote"`。
  - **検証（Windows開発機・オフライン）**: `tests/web/test_microphone_policy`・
    `tests/web/test_sensor_error_safety`・`tests/test_screen_context_lifecycle`・
    `tests/test_vision_context_lifecycle`・`tests/test_screen_remote`（計125件）を実行し全成功。
    canary な secret・パス・画像内容を例外に混ぜてもログ/JSON/status へ漏れないこと、
    allowlist のみの公開、stop 所有権、CP932 安全を固定。実機・実サービス・実モデル・
    実GPU・実ネットワークは不使用（live / deployed / verified は主張しない）

**P0-3 後続の安全仕上げ（2026-08-28、本実装worker記録）**
- P0-3 の後続として、センサー/音声系の安全仕上げをリポジトリ実装＋オフライン分離テストで
  固定した（全て live / deployed / verified ではなく、リポジトリ配線と分離テストの範囲）。
  - **Discord 共通 transcript ゲート**: `on_message` は parsing 直後・全分岐の前に
    「voice STT が存在・listening かつ voice reply ゲートが active」の共通ゲートを適用し、
    ゲート外の transcript はタスク/カレンダー直接登録を含む一切の副作用を起こさない。
  - **通話由来の直接登録撤去**: 通話 transcript からの「タスク:」「予定〜入れて」直接登録
    ブランチは撤去済み。受け入れた transcript は全てデバウンス → LLM 返信パイプラインのみを通る
    （テキストチャット側の直接登録経路は従来どおり維持）。
  - **返信生成 revoke / 原子履歴コミット**: `handle_voice_reply` は各副作用の前にゲート世代を
    同期再チェックし、revoke 済みなら LLM・履歴コミット・返信・学習・TTS・リアクションの
    副作用を一切行わない。LLM 生成は `ask_voice_transcript` でセッション履歴から切り離し
    （一時追加 → finally で必ず除去）、生成返却後に await を挟まず世代を再チェックしてから
    user+assistant をセッションロック下で原子的にコミットする。`/voice stop|leave` は STT 停止
    より先に `VoiceReplyGate` を revoke し進行中返信を bounded cancel。voice STT 側も
    `_revoke_generation` で保持送信 Future を cancel し、送信直前に世代を再チェックする。
    `tests/test_discord_voice_reply_debouncer.py` / `tests/test_discord_voice_stt.py` で固定。
  - **Voice CLI カレンダー独立 opt-in**: 音声発話からのカレンダー書き込みは
    `VOICE_CALENDAR_WRITE_ENABLED=true` の明示のみ有効（既定 `false` / fail closed）。
    マイク同意だけでは書き込まない。無効時は通常の LLM 経路へフォールスルーし、
    書き込みクライアント自体を構築しない。`tests/audio/test_pipeline_llm.py` で固定。
  - **ingest 完了 Event・原子 tmp 置換・revoke 後拒否**: `/api/screen/ingest` の描写完了は
    世代ごとの `threading.Event` で判定し（Future の done は下位 worker 完了を保証しない）、
    コミットは `latest.json.tmp` への write/fsync → lock 区間での `os.replace` の原子置換。
    revoke 後は受付を一切登録せず固定 503＋`unavailable` で拒否し、revoke 済み世代のコミットは
    起こらない。`tests/test_screen_remote.py`（ScreenIngestLifecycleTest）/
    `tests/web/test_sensor_error_safety.py`（DescribeIngestedGateTest）で固定。
  - **Remote / Monitor の stop_pending と storage 順序**: stop は先に worker を stop/join し、
    join タイムアウトで worker が生存し続ける間は storage を触らず・置き換えず・閉じず
    ownership を保持し `stop_pending` を公開。死確認後のみ解放・再 start 許可。
    `tests/test_screen_remote.py` / `tests/context/test_monitor_provider.py`
    （MonitorContextStopTruthfulTest / MonitorContextStartAdmissionTest）で固定。
  - **Monitor 書込の有界リトライ**: SQLite 書込の一時失敗に対し同じ metrics を
    `write_attempts`（既定 3）回まで `write_retry_delay`（既定 0.25s）の有界バックオフで再試行。
    exhaustion 時のみ固定カウンタと型名のみ・ASCII の診断。キュー・バッファなし。
    `tests/context/test_monitor_provider.py`（MonitorContextRetryPolicyTest）で固定。
  - **VAD/STT transcript-safe ログ**: 認識テキストは既定のパイプライン診断・ログへ出力しない
    （STT は所要時間のみ、失敗は例外型名のみ。Voice pipeline は認識テキストをセッション/LLM
    経路のみへ渡す）。`tests/audio/test_vad.py` / `tests/audio/test_pipeline_llm.py` で固定。
- 上記の詳細と根拠は `docs/decisions/sensor_opt_in_policy.md`（U11）に追記済み。一般の
  TaskCalendarSync / reminder の drop / update / snooze 競合はセンサー opt-in とは別の非センサー
  領域であり、P0-3 の状態（implemented / tested / integrated）には含めない。この配送一貫性は
  別途 [decisions/task_delivery_consistency.md](./decisions/task_delivery_consistency.md) で
  **解決済み**として固定した（`tasks.rev` 楽観制御・`BEGIN IMMEDIATE` claim・
  revalidate-before-callback・`expected_rev` 条件付き record・state-driven 同期・マーカー照合/
  重複整理。オフライン unit テスト 120件: store 69 / reminder 19 / calendar_sync 32）。残る
  residual は micro-TOCTOU・queue/pull latency・一意 lease owner 名の規律・exactly-once /
  durable outbox なしで、deployed / verified / live は未主張。

## P0-4: Voice CLI 文脈整合 (Context parity)

**目的**: 音声経路が CLI 経路と同じ文脈構築（ChatSession / Context Provider / Policy）を経て
同じ品質の応答を得ることを保証し、経路間の文脈差を無くす。
決定記録: [decisions/voice_context_parity.md](./decisions/voice_context_parity.md)。

| ID | タスク | 依存 | 変更対象（予定ファイル領域） | 受け入れ条件 | 検証/レビューゲート | rollback/互換 | 状態 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P0-4.1 | CLI / Voice CLI（audio-text）/ Voice パイプラインの文脈構築パスの差分を調査・記録 | なし | `src/chat/main.py`、`src/chat/session.py`、`src/audio/pipeline.py`、`src/audio/main.py` | 差分を項目別の表で記録し、共通構成と Voice 拡張（チャネル能力拡張）を決定する（実装変更は不要） | 調査結果・決定の独立レビュー | 変更なし | done |
| P0-4.2 | Voice 経路を CLI と同じ共通文脈構築へ統一（`ChatSession.build_blocks` → `AssistantService.respond_stream` → `ContextBuilder` の同一構成を全入口で共有。`generate_stream` 接続は維持） | P0-4.1 | `src/audio/pipeline.py`、`src/audio/main.py`、`src/chat/main.py`、`src/chat/session.py`、`src/assistant/` | 同一入力で CLI / Voice CLI（audio-text）/ Voice パイプラインが同一文脈で応答生成する。emotion 配線・検証付き request factory（`create_request`）・stream close・rollback・カレンダー直接返信のコミットフラグを含む | `tests/audio/test_context_parity.py`、`tests/audio/test_pipeline_llm.py`、`tests/chat/test_session_build_blocks.py`、`tests/chat/test_cli_loop.py` の unit テスト（オフライン） | 変更を戻せば従来挙動（Voice は共通構成のみで動作） | done |
| P0-4.3 | 文脈整合の回帰 guard テストを追加（構成・描画の同一性と guard を固定） | P0-4.2 | `tests/audio/test_context_parity.py`、`tests/audio/test_pipeline_llm.py`、`tests/chat/test_session_build_blocks.py`、`tests/chat/test_cli_loop.py` | equality（CLI / Voice の共通文脈描画が同一）、order（emotion + history 等のブロック順序）、request（ファクトリの明示 channel / profile / privacy）、lifecycle（stream close が必ず1回・主経路例外をマスクしない・エラー時 rollback）が再度崩れたら失敗する | 上記テストへの差分注入・回帰レビュー | テスト追加のみ | done |

**P0-4 実装/検証メモ（2026-08-29、本実装worker記録）**
- **P0-4.1（調査・決定）**: [decisions/voice_context_parity.md](./decisions/voice_context_parity.md)
  で CLI（`src/chat/main.py`）/ Voice CLI・audio-text（`src/audio/main.py`）/ Voice パイプライン
  （`src/audio/pipeline.py`）の会話構成を項目別に比較した。ChatSession build_blocks・
  AssistantService/ContextBuilder・base prompt/history・emotion・ChatConfig 生成パラメータ・
  stream close・rollback・persistence・共通 provider（web_search / growth_tracker）は同一とし、
  request の channel / profile は経路識別子（`channel="voice"` / `profile="voice_fast"` /
  `privacy="local_only"`）のみ差異と確定した
- **P0-4.2（統一・統合）**: 全入口が共通構成（`ChatSession.build_blocks` →
  `AssistantService.respond_stream` → `ContextBuilder.build_messages`（`privacy=request.privacy` /
  `target_local=True`））を共有する。Voice 経路は emotion（`EmotionTagStreamFilter` /
  `emotion_tags=config.emotion_tag_enabled`）、検証付き request factory
  （`src/assistant/requests.create_request` の channel / profile / privacy 明示）、stream close
  （`StreamResult.close` を正常・空応答・生成例外・`respond_stream` 失敗の全経路で finally で
  必ず1回）、生成エラー時の `session.rollback_last_user_message()`、カレンダー直接返信の
  コミットフラグ（`store_memory=False` / `record_growth=False`、`VOICE_CALENDAR_WRITE_ENABLED`
  明示のみ有効）を実装・統合した
- **P0-4.3（回帰 guard）**: `tests/audio/test_context_parity.py`
  （request factory の検証・CLI/Voice の明示メタデータ・共通文脈描画の equality・
  emotion + history の order・audio text の stream close 1回/例外非マスク・カレンダー
  コミットフラグ）、`tests/audio/test_pipeline_llm.py`（rollback・close・idle 復帰の
  lifecycle guard）、`tests/chat/test_session_build_blocks.py`（`build_blocks` ≡
  `build_messages` の parity・provider/emotion 順序・破損/空 provider の fallback）、
  `tests/chat/test_cli_loop.py` で固定
- **ローカル検証（Windows開発機・オフライン）**: `tests/audio/test_context_parity` /
  `tests/chat/test_session_build_blocks` / `tests/chat/test_cli_loop`（53件）と
  `tests/audio/test_pipeline_llm`（16件）が全て成功。実モデル・実マイク・実TTS・実サービス・
  実ネットワーク・実データは不使用（deployed / verified は主張しない）

**Voice のチャネル能力拡張（共通構成の変更ではない）**: rag / persona / tasks / calendar と
P0-3 で gated されるセンサー（vision / screen / monitor / activity / microphone）は
共通構成へ追加されるチャネル能力拡張であり、共通構成の変更ではない。RAG / persona は
既定で有効（`--no-rag` / `--no-persona` で無効化）。task_store / calendar_context は
既定で構築され、データ・設定が利用可能な場合のみ文脈を発話する（読み取り専用）。
カレンダー書込は `VOICE_CALENDAR_WRITE_ENABLED` の明示 `true` のみ有効。
P0-3 センサーのみ既定オフの opt-in を維持する。

**このマイルストーンでやらないこと（deferred 非目標）**: 実機 live 検証（実マイク・実TTS・
実サービスでの parity 確認）、Voice の RAG / persona / tasks / calendar の既定挙動変更
（RAG / persona は既定有効・task_store / calendar_context は既定構築で既に現行挙動。
将来の既定変更は性能・プライバシー検討を経た方針判断として再評価）、P0-3 センサーの
Voice 既定有効化（既定オフの opt-in を維持）、Web / Discord との構成完全統一
（Web / Discord は固有 provider を持つため対象外）。

## P1: Desktop quick-chat HUD（高度3Dより先）

**目的**: 常駐オーバーレイで短い会話・今日・タスク・PC状態の HUD を 1 枚だけ扱えるようにし、
ユーザー価値を先に得る。高度な3D表現（VRM1.0表情・視線・リップシンク）は HUD の後に回す。

| ID | タスク | 依存 | 変更対象（予定ファイル領域） | 受け入れ条件 | 検証/レビューゲート | rollback/互換 | 状態 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P1.1 | quick-chat HUD（半透過1枚、選択肢2〜3個、詳細は既存Web UIへ）をオーバーレイへ実装 | P0-2（backend安定が目安） | `src/desktop/qml/Overlay.qml`、`src/desktop/shell.py`、`src/desktop/bridge.py`、`src/desktop/app.py`、`src/desktop/windows.py` | センサー出所・取得時刻・保存有無の行を維持しつつ、HUDで短い会話が成立する（オプトイン時のみ） | `tests/test_desktop_*.py` の unit テスト＋QML構造レビュー | 既存2D表示へ fallback し続ける（表示のみ失敗時もアプリ継続） | done |
| P1.2 | 既存 bridge チャット経路（`ChatPage.qml` 相当）を HUD から再利用 | P1.1 | `src/desktop/bridge.py`、`src/desktop/qml/` | チャット実装を複製せず既存経路を再利用 | 配線差分レビュー | 変更なし | done |
| P1.3 | HUD のテストと回帰固定 | P1.2 | `tests/test_desktop_*.py` | 状態遷移・出所表示・クリックスルー切替が既定で検証される | unit テスト | テスト追加のみ | done |

**P1 実装/検証メモ（2026-08-29、本実装worker記録）**
- **実装済み（リポジトリ実装＋オフライン static/unit テスト）**: Quick-Chat HUD を
  `src/desktop/qml/Overlay.qml` の expanded パネルとして実装し、既存 DesktopBridge
  （`src/desktop/bridge.py`）のプロパティ・シグナル・送信経路のみを再利用した（新規
  API / エンドポイントは追加しない）。決定記録:
  [decisions/desktop_quick_chat_hud.md](./decisions/desktop_quick_chat_hud.md)。
  - **Today サマリは count-only（allowlist 固定表示のみ）**: `today_points` / `streak_days` /
    open タスク件数 / 今日以降の予定件数の数値のみ。タスク本文・予定タイトル・プロセス・
    センサー詳細・パス・モデル・`modelData`・`statusText` は HUD に露出しない。状態は
    `overlayShell.shell_state` の固定ラベル＋`sensor_provenance`（出所・取得時刻・保存なし）。
  - **オフライン・読み込み中は送信無効化**: `canSend = connected && !loading`。バナーは固定
    文言のみ（読み込み中… / 接続済み / オフライン）で `statusText` / `serverUrl` 非露出。
  - **クリックスルーは desired-state 方式で配線**: HUD は `bridge.setOverlayClickThrough(bool)`
    の希望状態のみを持ち、`overlayClickThroughRequested` シグナル →
    `OverlayClickThroughController`（`src/desktop/app.py`）→ `apply_click_through`
    （`src/desktop/windows.py`、WS_EX_LAYERED | WS_EX_TRANSPARENT）が実適用する。
    復帰は **Ctrl+Alt+Space**（`WindowsHotkeyFilter` / `RegisterHotKey`、
    MOD_ALT|MOD_CONTROL|MOD_NOREPEAT, VK_SPACE, HOTKEY_ID=0xBADD）で `restore_interaction()`
    （クリックスルー解除）を先に実行してから本体ウィンドウを toggle する。停止・終了・失敗時は
    必ず解除（`_force_overlay_click_through_off` / `disconnect` / `apply_default_click_through`）。
  - **既存 2D/3D fallback を維持**（`hasAvatar3D = avatarModel.exists && !modelLoadFailed`、拡張中は
    アバター非表示）。VRM 参照・3D 再設計なし。
- **検証（Windows開発機・オフライン）**: `tests/test_desktop_contract.py`
  （`OverlayContractTest` / `DesktopClickThroughContractTest`）、`tests/test_desktop_shell.py`
  （`OverlayClickThroughTestCase` / `TestOverlayClickThroughApply` / `TestHotkeyInteractionRestore`）、
  `tests/test_desktop_qml.py`（QML ロード検査）、`tests/test_desktop_companion.py`、
  `tests/test_desktop_api.py` / `tests/test_desktop_avatar.py` / `tests/test_desktop_overlay_vrm.py`
  を実行し計151件が成功（`python -m unittest tests.test_desktop_contract tests.test_desktop_shell
  tests.test_desktop_qml tests.test_desktop_companion tests.test_desktop_api tests.test_desktop_avatar
  tests.test_desktop_overlay_vrm -q`）。
- **未実施（live / deployed / verified は主張しない）**: 実Qt / 実GL レンダリング・実 Windows
  ウィンドウ・実 ws/chat 送受信・実クリックスルー・実 hotkey・実サービスでの表示/送信の
  live 検証は未実施。

**高度3D（Phase 6b 残作業）は P1 HUD 完了後に着手**: P1 Quick-Chat HUD は完了済み（上記）。
VRM1.0 humanoid・表情・視線・リップシンクの専用レンダリング制御は以降の Phase 6b 残作業として
planned のまま。

## 延期項目（前提未達）

| 項目 | 前提条件 | 備考 |
| --- | --- | --- |
| Cloud 通常経路統合（Phase K / Phase 7） | P0-2 のローカル backend 安定、承認UX決定、外部モデル契約の実送信検証 | [companion_roadmap.md](./plans/companion_roadmap.md) Phase 7、[archive/assistant_platform_plan.md](./archive/assistant_platform_plan.md) Phase K 参照 |
| LangGraph Workflow | 承認・中断・再開が必要な具体的なユースケースの出現 | 現状は別判断のまま維持 |
| PostgreSQL / 監視 / deploy（I2〜I6） | P0-1 Linux CI がグリーン、P0-2 backend が決定、I1 実機確認済み | [infrastructure_plan.md](./infrastructure_plan.md) I2〜I6 |
| 実 LoRA 学習（Phase 13） | 実GPU環境、データセット整備、評価ハーネス | [training/personal_lora_training.md](./training/personal_lora_training.md) |
| GPU チューニング | 実機GPUの存在、ベンチマーク基準の定義 | **性能主張はハードウェアベンチマークなしでは行わない**（決定記録参照） |

## 直近の次アクション

1. P0-1.1 / P0-1.2: `test.yml`（オフラインslice品質ゲート）実装済み。全ブランチpush反映と
   `tests.test_task_formatting` のtasks slice追加も反映済み。P0-1.2 として静的チェック
   （HEADパッチの空白検査 `git diff --check HEAD^ HEAD`＋root fallback
   `git show --check --format= --no-renames HEAD` / `compileall src tests`）をゲート先頭へ
   追加し、chat 設定のみの呼び出しを `tests/chat` 全体探索へ置換、stdlib-only の
   `tests.perception.test_sensor_policy` と `tests.test_calendar_sync` をオフラインunit
   ジョブへ追加した（discord.py 必須の `tests.test_task_chat_editor` は tasks slice から
   除去。discord は追加しない。Windows開発機の discord 非依存スタブ環境でタスクslice 166件
   が成功、`tests.test_task_chat_editor` は ImportError で失敗することを確認）。残るは
   GitHub 上での実行確認（未実施）、P0-1.3（ブランチ保護）
2. P0-2.4: ローカルbackendの入口・例・docs統合は完了（オフライン分離テスト済み）。公開設定例 `config/chat_config.local-openai.example.json` と docs（plan / status / usage / readme）を統合し、通常経路 CLI / Web / Discord / Voice / batch / NodeInventory をオフライン分離テストで確認。P0-2.1（provider基盤）と P0-2.3（config/factory）も実装・オフライン分離テスト済み。実サーバー（llama.cpp / LM Studio / vLLM）の live 検証・ScreenDescriber マルチモーダル移行・ベンチマークは deferred
3. P0-3.1: 共有 SensorPolicy 解決器の実装・オフライン unit テスト・決定記録（`docs/decisions/sensor_opt_in_policy.md`）は完了。P0-3.2（入口配線）は Web / Voice CLI / Discord / Desktop の activity に加え、Monitor の process_details 配線・Discord 通話STT の共有マイク gate・Desktop push-to-talk マイク gate・Web 音声入力（POST `/api/stt` / WS `audio_message`）の共有マイク gate・`--text-mode` のセンサー不使用を実装・オフライン分離テスト済み。P0-3.3（fail-safe 例外注入の固定）・P0-3.4（privacy-safe 検証の固定）もオフライン分離テスト済み。後続の安全仕上げ（Discord 共通 transcript ゲート・返信生成 revoke・未コミットLLM→原子履歴コミット・通話由来直接登録撤去・Voice CLI カレンダー独立 opt-in・ingest 完了 Event/原子 tmp 置換/revoke 後拒否・Remote/Monitor の stop_pending と storage 順序・Monitor 書込の有界リトライ・VAD/STT の transcript-safe ログ）もオフライン分離テスト済み。残るは live センサー検証（実機・実サービス）のみ。Web 音声の録音はブラウザ側 `getUserMedia`（HTTPS 権限）でサーバーはマイクキャプチャを持たないが、サーバー側 STT 受付は共有 SensorPolicy の microphone gate を必須とする。TaskCalendarSync /
  reminder の配送一貫性（drop / update / snooze 競合）は
  [decisions/task_delivery_consistency.md](./decisions/task_delivery_consistency.md) で解決済み
  （オフライン unit テスト120件）。残る residual は micro-TOCTOU・queue/pull latency・一意 owner
  規律・exactly-once / durable outbox なし
4. P1.1 / P1.2 / P1.3: Desktop Quick-Chat HUD はリポジトリ実装＋オフライン static/unit テスト済み（`src/desktop/qml/Overlay.qml` の expanded パネル・既存 DesktopBridge 経路再利用・count-only Today サマリ・desired-state クリックスルー＋Ctrl+Alt+Space hotkey 復帰）。決定記録 `docs/decisions/desktop_quick_chat_hud.md`。Desktop 関連テスト計151件成功。残るは実Qt/GL・実Windowsウィンドウ・実クリックスルー・実hotkey・実サービスでの live 検証のみ。Phase 6b 残作業（VRM1.0表情・視線・リップシンク）は planned のまま

進捗更新時は、実装状態は [implementation_status.md](./implementation_status.md)、タスク順序は本計画を
更新する。本計画自体は deployed / verified を主張しない。

## 関連文書

- [implementation_status.md](./implementation_status.md): 実装状態の正典（本計画はタスク順序のみ）
- [decisions/task_delivery_consistency.md](./decisions/task_delivery_consistency.md): タスク配送一貫性（リマインド通知・タスク⇔カレンダー同期）の決定記録（at-least-once best-effort / rev・lease・revalidate・expected_rev・state-driven 同期・マーカー照合/重複整理）
- [decisions/local_inference_backend.md](./decisions/local_inference_backend.md): ローカル推論backendの決定記録
- [infrastructure_plan.md](./infrastructure_plan.md): インフラ固有計画（I1〜I6）
- [plans/companion_roadmap.md](./plans/companion_roadmap.md): 製品方向・Phase 1〜7
- [archive/assistant_platform_plan.md](./archive/assistant_platform_plan.md): 基盤実装の完了記録（Phase K）