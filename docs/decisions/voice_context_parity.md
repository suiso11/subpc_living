# 音声対話の会話構成 Parity の決定記録

> **状態**: active / canonical
> **位置付け**: CLI / Voice CLI（音声テキスト） / Voice パイプライン間の会話構成パリティの決定記録
> **対象範囲**: P0-4.1 調査（ChatSession build_blocks・AssistantService/ContextBuilder・base prompt/history・
>   emotion・request channel/profile/privacy・ChatConfig 生成パラメータ・stream close・rollback・
>   persistence・provider set）とパリティ決定、および deferred 非目標
> **作成日**: 2026-08-29
> **更新日**: 2026-08-29
> **日付根拠**: Git commit date

## 背景

会話生成は CLI（`src/chat/main.py`）、Voice CLI（`src/audio/main.py`、音声テキスト双方向・
`--text-mode` 含む）、Voice パイプライン（`src/audio/pipeline.py`）、Web（`src/web/server.py`）、
Discord（`src/discord_bot/bot.py`）の複数入口から同じ構成要素を組み立てる。入口ごとに
構成が分岐すると「同じ入力・同じ provider で異なる応答」という挙動差が生まれ、
テスト・デバッグ・学習ログの信頼性を損なう。P0-4.1 では CLI / audio-text / Voice の
会話構成を項目別に調査し、共通部分は同一にする方針を確定する。

## 経緯（時系列）

| 日付 | 区分 | 内容 |
| --- | --- | --- |
| 2026-08-29 | 調査 | CLI / Voice CLI（audio-text）/ Voice パイプラインの会話構成を項目別に比較（P0-4.1） |
| 2026-08-29 | 決定 | 本記録の決定を採択（同一の共通構成・Voice の拡張はチャネル能力拡張） |

## P0-4.1 調査結果（CLI / audio-text / Voice 比較表）

| 項目 | CLI（`src/chat/main.py`） | Voice CLI / audio-text（`src/audio/main.py`） | Voice パイプライン（`src/audio/pipeline.py`） | 判定 |
| --- | --- | --- | --- | --- |
| ChatSession build_blocks | 共通 `ChatSession.build_blocks`（`src/chat/session.py`） | 同一 | 同一 | 同一実装 |
| AssistantService / ContextBuilder | 共通（`src/assistant/service.py` / `src/context/builder.py`） | 同一 | 同一 | 同一実装 |
| base prompt / history | `config.effective_system_prompt()` + `max_history_turns` + `history_dir` | 同一 | 同一 | 同一 |
| emotion | `emotion_tags=config.emotion_tag_enabled` | 同一 | 同一 | 同一 |
| request channel / profile / privacy | `channel="cli"` / `profile="chat_auto"` / `privacy="local_only"` | `channel="voice"` / `profile="voice_fast"` / `privacy="local_only"` | 音声経路の識別子（`conversation_source="voice"` 等） | 識別子のみ差異 |
| ChatConfig 生成パラメータ | 共通 `ChatConfig`（model / temperature / num_ctx / stream / max_history_turns） | 同一 | 同一 | 同一 |
| stream close | `respond_stream` の iterator と finally での `stream.close()`（正常・空応答・生成例外の全経路で必ず1回・冪等。close 失敗は主経路の例外をマスクしない） | 同一（`_stream_assistant_response` の finally で必ず1回 close。registry も finally で1回 close） | 同一 | 同一 |
| rollback | 例外時に `session.rollback_last_user_message()` | 同一（audio-text の生成例外時にも rollback） | 同一 | 同一 |
| persistence | `session.save()`（`history_dir`）/ `GrowthTracker` | 同一 | 同一 | 同一 |
| provider set（共通） | web_search / growth_tracker | 同一 | web_search / growth_tracker | 同一 |
| provider set（Voice 追加） | — | —（`--text-mode` はセンサー不使用） | rag / persona（既定有効。`--no-rag` / `--no-persona` で無効化）/ task_store / calendar_context（既定構築・データ/設定利用可能時のみ文脈）/ vision / screen / monitor / preloader / P0-3 センサー（既定オフ opt-in） | チャネル能力拡張のみ（共通パリティ入力ではない） |

## 現時点の決定

1. **同一の注入共通 provider に対しては、CLI / Voice CLI（audio-text）/ Voice パイプラインは
   すべて同一の共通構成を維持する**。ChatSession build_blocks・AssistantService/ContextBuilder・
   base prompt/history・emotion・ChatConfig 生成パラメータ・stream close・rollback・persistence・
   provider set（web_search / growth_tracker）は入口間で差をつけない。
2. **Voice 固有の RAG / persona / tasks / calendar と、P0-3 で gated されるセンサー
   （vision / screen / monitor / activity / microphone 等）は「チャネル能力拡張」であり、
   共通構成の変更ではない**。追加はブロックの source として表現され、共通構成の順序・
   セマンティクスを変えない。RAG / persona は既定で有効（`--no-rag` / `--no-persona`
   で無効化）。task_store / calendar_context は既定で構築され、データ・設定が利用可能な
   場合のみ文脈を発話する（読み取り専用）。カレンダー書込は別の
   `VOICE_CALENDAR_WRITE_ENABLED` 明示 `true` のみ有効（既定 fail closed）。
   P0-3 で gated されるセンサーは既定オフの opt-in を維持する。
3. **request の channel / profile は音声経路の識別子としてのみ異なる**。生成パラメータ・
   構成・文脈組み立てには影響しない。privacy は全入口 `local_only`。
4. **構成の同一性はオフライン分離テストで固定する**（下記「検証・確認」）。

## 判断理由

- **1（同一の共通構成）**: 入口ごとの構成分岐は「同じ provider 注入なら同じ応答」を
  保証できなくし、テスト・学習ログ・デバッグの信頼性を損なう。共通実装
  （`ChatSession.build_blocks` → `ContextBuilder.build_messages`）を全入口が共有すれば、
  差分は注入 provider の差だけになる。
- **2（Voice 拡張はチャネル能力拡張）**: rag / persona / vision / screen / monitor /
  preloader / task_store / calendar は音声パイプライン固有の能力であり、共通構成とは
  独立に扱われるチャネル能力拡張である。RAG / persona は既定有効、task_store /
  calendar_context は既定構築・利用可能時のみ文脈を発話する。P0-3 の既定オフ方針
  （`docs/decisions/sensor_opt_in_policy.md`）が適用されるのはセンサーのみであり、
  チャネル能力拡張は共通構成を複雑化しない。
- **3（識別子のみ）**: channel / profile は経路識別（学習ログ・ルーティング用）であり、
  生成内容を変える構成要素ではない。タグの差異を構成差異と誤認しない。
- **4（テストで固定）**: 構成の同一性は文書上の宣言だけでは劣化しやすいため、
  build_blocks と build_messages の一致性を unit テストで固定する。

## 前提（assumptions）

| ID | 前提 | 推奨既定 |
| --- | --- | --- |
| A1 | 全入口は共通の `ChatSession`（`src/chat/session.py`）を共有する | `build_blocks` / `build_messages` は同一実装 |
| A2 | 共通 provider（web_search / growth_tracker）は全入口で同一注入される | 差異を付けない |
| A3 | Voice 固有 provider（rag / persona / task_store / calendar）はチャネル能力拡張。RAG / persona は既定有効（`--no-rag` / `--no-persona` で無効化）、task_store / calendar_context は既定構築・データ/設定利用可能時のみ文脈を発話。P0-3 センサー（vision / screen / monitor / activity / microphone）は既定オフ opt-in | チャネル能力拡張は共通パリティ入力にしない |
| A4 | request の channel / profile は経路識別子であり生成パラメータに影響しない | 音声は `channel="voice"` 等のみ |
| A5 | 構成同一性はオフライン分離テストで維持する | `tests/chat/test_session_build_blocks.py` 等 |

## 解決済み（resolved）

| ID | 事項 | 実装 |
| --- | --- | --- |
| U1 | build_blocks と build_messages の一致性 | `ChatSession.build_messages` は `build_blocks` に委譲し、`ContextBuilder`（`src/context/builder.py`）で messages 化。全 provider・emotion on/off・破損/空 provider の順序・fallback を `tests/chat/test_session_build_blocks.py`（`ChatSessionCompositionParityTest`）で固定 |
| U2 | CLI / Voice CLI の共通構成 | `src/chat/main.py` と `src/audio/main.py` は同一の ChatSession 構成（system_prompt / max_history_turns / history_dir / web_search / growth_tracker / emotion_tags）を共有し、`conversation_source` と request タグのみ異なる |
| U3 | Voice パイプラインの拡張注入 | `src/audio/pipeline.py` は rag / persona（既定有効。`--no-rag` / `--no-persona` で無効化）/ task_store / calendar_context（既定構築・データ/設定利用可能時のみ文脈）/ P0-3 センサー（既定オフ opt-in）を注入するが、いずれもチャネル能力拡張であり共通構成を変更しない。カレンダー書込は `VOICE_CALENDAR_WRITE_ENABLED` 明示 `true` のみ |
| U4 | `--text-mode` の構成 | Voice CLI の `--text-mode` はセンサー（マイク含む）を一切使用せず、共通構成のみで動作。`tests/audio/test_sensor_policy.py` / `tests/audio/test_companion_wiring.py` で固定 |

## 未解決（unresolved）と推奨（deferred 非目標）

| ID | 未解決事項 | 推奨既定（デフォルト明示） |
| --- | --- | --- |
| U5 | 実機 live 検証（実マイク・実 TTS・実サービスでの会話構成パリティ確認） | 未実施。本記録はリポジトリ実装＋オフライン分離テストのみであり、deployed / verified は主張しない |
| U6 | Voice の RAG / persona / tasks / calendar の既定挙動変更 | deferred（方針判断）。RAG / persona は既に既定有効、task_store / calendar_context は既定構築であり、既定注入は現行挙動として実装済み。将来の既定変更（例: RAG / persona を既定無効化、または tasks / calendar 文脈の既定抑制）は性能・プライバシー検討を経てから方針判断として再評価 |
| U7 | P0-3 センサー（vision / screen / monitor / activity）の Voice 既定有効化 | deferred。既定オフの opt-in を維持（`docs/decisions/sensor_opt_in_policy.md` の決定を変更しない） |
| U8 | Web / Discord との構成完全統一 | deferred。Web / Discord は固有 provider（プロファイル・マルチセッション等）を持つため、本記録の対象（CLI / audio-text / Voice）に限定する |

## 検証・確認

- 構成同一性は `tests/chat/test_session_build_blocks.py`
  （`ChatSessionCompositionParityTest`・`ChatSessionAllProviderOrderTest`）で固定され、
  `build_blocks` + `ContextBuilder.build_messages` の出力が `build_messages` と完全一致する。
  補助として `tests/context/test_history_builder.py`（`ContextBuilderTest` /
  `ChatSessionHistoryWiringTest`）、`tests/test_chat_emotion.py`、
  `tests/test_chat_growth.py`、`tests/test_chat_model_switch.py`、
  `tests/audio/test_sensor_policy.py` / `tests/audio/test_companion_wiring.py`
  （`--text-mode` のセンサー不使用）、`tests/audio/test_context_parity.py`
  （audio-text の stream close 1回/例外非マスク・生成例外時の rollback）を参照する。
- 本記録はオフラインでの調査・決定記録であり、実機・実サービス・ネットワークでの
  live 検証は未実施。deployed / verified は false（U5）。
- 実装状態は [implementation_status.md](../implementation_status.md) を正とする。

## 関連文書

- [implementation_plan.md](../implementation_plan.md): P0-4（音声対話と共通構成のパリティ）
- [implementation_status.md](../implementation_status.md): 実装状態の正典
- [sensor_opt_in_policy.md](./sensor_opt_in_policy.md): P0-3（センサー opt-in 強制）
- [local_inference_backend.md](./local_inference_backend.md): P0-2（ローカル推論 backend）
- [src/chat/session.py](../../src/chat/session.py): 共通 ChatSession（build_blocks / build_messages）
- [src/context/builder.py](../../src/context/builder.py): 共通 ContextBuilder
- [src/chat/main.py](../../src/chat/main.py): CLI 入口
- [src/audio/main.py](../../src/audio/main.py): Voice CLI / audio-text 入口
- [src/audio/pipeline.py](../../src/audio/pipeline.py): Voice パイプライン（明示拡張の注入）