# センサー opt-in 方針の決定記録

> **状態**: active / canonical
> **位置付け**: カメラ・画面・活動など全センサーの opt-in 強制方針の決定記録
> **対象範囲**: センサー有効化の既定値・環境変数解決・legacy 互換・token の扱い・
>   CLI 明示フラグ・停止と資源解放・remote ingest の raw 画像保持
> **作成日**: 2026-08-28
> **更新日**: 2026-08-28
> **日付根拠**: Git commit date

## 背景

カメラ・画面・マイク・活動収集などは機微なデータを扱う。既存の
`COMPANION_ACTIVITY_ENABLED=true` オプトイン方針（`src/perception/bootstrap.py`）や
`WEB_SCREEN_CONTEXT_ENABLED=true`（`src/web/server.py`）は入口ごとに個別の
env 名・真偽判定を持ち、共通の安全な既定値と検証がなかった。全入口が同じ
不変 (immutable) の解決器を共有し、既定オフをコードとテストで強制する。

## 経緯（時系列）

| 日付 | 区分 | 内容 |
| --- | --- | --- |
| 2026-08-28 | 議論 | 全センサーへの opt-in 強制（P0-3）の前提として、共有 SensorPolicy 解決器の方針を検討 |
| 2026-08-28 | 決定 | 本記録の決定を採択（既定オフ・canonical env のみ有効化・legacy 互換・token は consent ではない） |
| 2026-08-28 | 実装 | `src/perception/policy.py` と `tests/perception/test_sensor_policy.py` を追加し、オフライン unit テストで固定 |
| 2026-08-28 | 実装 | P0-3.2 の後続配線: process_details（`src/monitor/context.py` 経由）、Discord 通話STT・Desktop push-to-talk の共有マイク gate、Voice CLI `--text-mode` のセンサー不使用をオフライン分離テストで固定 |
| 2026-08-28 | 実装 | P0-3 後続の安全仕上げ: Discord 共通 transcript ゲートと返信生成 revoke / 未コミットLLM→原子履歴コミット、通話由来のタスク・カレンダー直接登録ブランチ撤去、Voice CLI カレンダー書き込みの独立 opt-in（`VOICE_CALENDAR_WRITE_ENABLED=false` 既定）、ingest 完了 Event・原子 tmp 置換・revoke 後拒否、Remote / Monitor の stop_pending と storage 順序、Monitor 書込の有界リトライ、VAD/STT の transcript-safe ログ。オフライン分離テストで固定（live / deployed / verified は主張しない） |

## 現時点の決定

1. **全センサーは既定でオフ**（safe default / fail closed）:
   `camera` / `screen_capture` / `screen_ingest` / `activity` / `monitor` /
   `microphone` / `process_details` は既定 False。有効化するには明示設定が必須。
2. **有効化は canonical 環境変数の明示 `true` のみ**:
   `SENSOR_CAMERA_ENABLED` / `SENSOR_SCREEN_CAPTURE_ENABLED` /
   `SENSOR_SCREEN_INGEST_ENABLED` / `SENSOR_ACTIVITY_ENABLED` /
   `SENSOR_MONITOR_ENABLED` / `SENSOR_MICROPHONE_ENABLED` /
   `SENSOR_PROCESS_DETAILS_ENABLED`。それ以外の値・未設定・不正値は False
   （例外を上げずに必ず False へ倒れる）。
3. **legacy 互換は canonical 未設定時のみ**:
   `WEB_SCREEN_CONTEXT_ENABLED` → screen_capture、
   `COMPANION_ACTIVITY_ENABLED` → activity。canonical 名が存在すればその値が
   確定値で、canonical の false は legacy の true を上書きする。
4. **token は認証＋opt-in の一部であり、consent ではない**:
   共有 token（`SCREEN_INGEST_TOKEN`）の存在だけでは screen_ingest を有効化しない。
   remote ingest の有効化には `SENSOR_SCREEN_INGEST_ENABLED=true` の明示が必要で、
   token は配線時に認証として使う。token の送信・設定は同意の成立を意味しない。
5. **CLI の microphone / camera / monitor は affirmative flag か canonical env が必須**:
   既定の対話だけで起動しない。明示フラグ（例: `--microphone`）または
   canonical env の `true` が無ければ無効のまま。画面エージェントも同様に
   `--enable-screen-capture` または `SENSOR_SCREEN_CAPTURE_ENABLED=true` を要求し、
   source capture は既定オフとする。この gate は token、`--once`、URL の指定とは独立する。
6. **process / PID 詳細は独立センサー**:
   `process_details` は活動分類（activity）とは別。PID・プロセス名・実行パスなどの
   詳細は既定オフとし、既存の「プロセス名は分類後に保存しない」方針を維持する。
7. **UI の非表示は停止ではない**:
   ウィンドウ非表示・最小化・UI 上の無効表示ではセンサー収集は止まらない。
   停止（stop）は明示的操作とし、収集スレッド・ハンドル・デバイスを解放する。
8. **停止は資源を解放する**:
   `stop()` はタイムアウト付きで収集を終了し、カメラ・マイク・画面デバイスや
   OS ハンドルを確実に手放す。再開は再起動（start）を要求する。
9. **remote ingest の raw 画像保持は削除クリーンアップが必須**:
   送信された raw 画像をサーバーが一時保持する場合、保存先・保持時間（TTL）・
   削除手順を明示し、期限切れは自動削除する。無期限保持・手動削除のみは不可。

## 判断理由

- **1・2（既定オフ・明示 true のみ）**: 機微データの収集は、設定漏れ・誤記で
  有効にならないようにしなければならない。明示 `true` 以外をすべて False にする
  fail-closed 方式は、既存の `COMPANION_ACTIVITY_ENABLED=true` 方針と整合する。
  既存入口の緩い真偽判定（`1` / `yes` / `on` を真とする）は引き継がない。
- **3（legacy 互換）**: 既存設定（`WEB_SCREEN_CONTEXT_ENABLED` /
  `COMPANION_ACTIVITY_ENABLED`）を壊さず移行できる。ただし互換は「canonical 未設定時
  のみ」に限定し、canonical の false が必ず勝つようにして新設定への一本化を促す。
- **4（token は consent ではない）**: token は送信者認証の手段であり、ユーザーの
  同意意思を表さない。token の存在を有効化条件にすると「共有キーを設定しただけで
  画面データを送り始める」事故を防げないため、明示の opt-in を別途要求する。
- **5（CLI 明示フラグ）**: 音声対話・CLI 既定経路は「会話しただけでマイクが開く」
  の誤解を招きやすい。affirmative な明示が無い限り起動しない。
- **6（process/PID 別扱い）**: 活動分類（アプリカテゴリと idle 秒）と PID・プロセス名
  などの詳細は感度が異なる。詳細は既定オフの独立センサーとし、既定経路で漏れない。
- **7・8（UI 非表示 ≠ 停止・資源解放）**: 非表示は消費者の見え方の制御であり、
  収集の継続/停止とは別軸。プライバシー期待を誤らせないため、停止は明示操作で
  資源解放まで含めて定義する。
- **9（raw 画像保持の cleanup）**: remote ingest は受信画像を一時保持し得る。
  保持の無限化を防ぎ、自動削除を必須にする。

## 前提（assumptions）

| ID | 前提 | 推奨既定 |
| --- | --- | --- |
| A1 | 全センサーは `SENSOR_*_ENABLED` の canonical 名で解決される | 既定 False、明示 `true` のみ有効化 |
| A2 | legacy 名は移行期間のみ有効 | canonical 未設定時のみ参照。canonical false が legacy true を上書き |
| A3 | remote ingest は共有 token と別に明示 opt-in を要求する | `SENSOR_SCREEN_INGEST_ENABLED=true` |
| A4 | 停止は収集スレッド・デバイス・ハンドルの解放を含む | `stop()` はタイムアウト付きで資源解放まで行う |
| A5 | raw 画像の一時保持は削除クリーンアップが必須 | 保存先・TTL・自動削除を配線時に明示 |

## 解決済み（resolved）

| ID | 事項 | 実装 |
| --- | --- | --- |
| U1 | 解決 API と status payload の内容 | `resolve_sensor_policy()` → frozen `SensorPolicy`。`is_enabled` / `enabled_sensor_ids` / `as_status_payload`（boolean と sensor source 名のみ） |
| U2 | legacy env の扱い | `LEGACY_ENV_ALIASES`（screen_capture / activity）。canonical 未設定時のみ参照 |
| U3 | token 単独での有効化 | 解決器は token を参照せず、単独ではどのセンサーも有効化しない |
| U4 | 不正値の扱い | `parse_opt_in` は明示 `true` 以外すべて False（例外なしで fail closed） |
| U5 | 不変性 | `@dataclass(frozen=True)`。解決後は env 変更に影響されない |
| U6 | 各入口（Web / Discord / Voice / CLI / Desktop）への SensorPolicy 配線 | 配線済みは Web（camera / screen_capture / screen_ingest / monitor / activity。さらに Web 音声入力: POST `/api/stt` と WS `audio_message` は共有 SensorPolicy.microphone の fail-closed gate で、`stt` status は engine ロード済みかつ policy true のときだけ True。録音はブラウザ側 `getUserMedia` でサーバーはマイクキャプチャを持たず、受信音声のみ文字起こしする）、Voice CLI（microphone / camera / screen_capture / monitor / activity。`--no-vision` / `--no-monitor` は明示的な無効上書き）、Discord（activity / screen_capture。Discord-local legacy `DISCORD_SCREEN_CONTEXT_ENABLED` は canonical 未設定時のみ参照で、canonical false が legacy true を上書き。voice STT は共有マイク gate）、Desktop（activity / push-to-talk マイク gate）、Monitor（process_details）。P0-3.3（fail-safe 例外注入の固定）と P0-3.4（privacy-safe 検証の固定）はオフライン分離テスト済み。テスト: `tests/audio/test_sensor_policy.py` / `tests/test_discord_sensor_policy.py` / `tests/test_desktop_companion.py` / `tests/context/test_monitor_provider.py` / `tests/test_screen_remote.py` / `tests/web/test_companion_state_api.py` / `tests/web/test_microphone_policy.py` / `tests/web/test_sensor_error_safety.py` / `tests/test_screen_context_lifecycle.py` / `tests/test_vision_context_lifecycle.py` |
| U7 | remote ingest の raw 画像保持の実装詳細 | サーバーは受信 raw JPEG を一時保持せず保存もしない（TTL・自動削除は不要）。VLM 描写結果のみ `data/screen/latest.json` に保持。レガシー `latest.jpg` は起動・停止・無効状態で best-effort 削除（絶対の削除保証ではない。失敗時は黙って無視） |
| U8 | CLI affirmative flag の形式と既定 | `--microphone` / `--camera` / `--monitor` / `--screen` を採択。既定は起動しない。`--wakeword` もマイク同意ゲートに従う |
| U13 | Screen agent source gate と配布コピー | `--enable-screen-capture` または `SENSOR_SCREEN_CAPTURE_ENABLED=true` が無ければ source capture は起動しない。token・`--once`・URL とは独立。`scripts/screen_agent.py` と `src/web/static/screen_agent.py` は byte-identical。診断は固定文言/型名のみで URL・画像/本文内容を含めない。receiver は引き続き `SENSOR_SCREEN_INGEST_ENABLED=true` と `SCREEN_INGEST_TOKEN` の二重 gate。 |
| U9 | process_details の配線と default redaction | `MonitorContext`（`src/monitor/context.py`）が共有 SensorPolicy から解決。既定では集計値（`process_count`）のみ収集・保存され、プロセス名・PID・CPUトップ5 は `SENSOR_PROCESS_DETAILS_ENABLED=true` のときのみ（safe aggregates）。`tests/context/test_monitor_provider.py` で固定 |
| U10 | Discord 通話STT・Desktop マイク・Web 音声の SensorPolicy 適用範囲 | Discord 通話STT は `DISCORD_VOICE_STT_ENABLED=true` と共有 SensorPolicy.microphone の二重 gate（どちらか false なら `/voice join|start` は接続前に却下。自動起動はしない。`/voice stop` は処理中・待機中の音声を discard して即時停止し、ワーカーを bounded join。join タイムアウト時は ownership を保持し `stop_pending` で真実を公開）。Desktop の push-to-talk は SensorPolicy.microphone が false なら録音を開始しない。Web 音声入力（POST `/api/stt` / WS `audio_message`）も SensorPolicy.microphone で fail-closed gate（false・未解決・不正値は 403 / 固定文言 error で base64 デコード・STT・一時ファイルへ到達しない）。録音自体はブラウザ権限プロンプト（HTTPS 必須）で行うが、サーバー側 STT の受付には SensorPolicy の明示 true が必要（ブラウザ権限だけでポリシーを迂回しない）。UI は `/api/status` の `stt`（engine ロード済みかつ policy true）でマイク入力を gate し、env 名・値は frontend へ露出しない |
| U11 | P0-3 後続の安全仕上げ（共通ゲート・原子コミット・独立opt-in・stop順序・ログ安全） | (a) **Discord 共通 transcript ゲート**: `on_message` は parsing 直後・全分岐の前に「voice STT が存在・listening かつ voice reply ゲートが active」の共通ゲートを適用し、さらにメッセージ作成時刻が現在の STT セッション開始時刻以降であることを要求する（再起動・再開前の旧セッションは受け付けない）。ゲートを外れた transcript はタスク/カレンダー直接登録を含む一切の副作用を起こさない（STT 停止・revoke 後に遅れて届いた transcript が履歴・外部状態へ取り込まれない）。通話由来の「タスク:」「予定〜入れて」直接登録ブランチは撤去済みで、受け入れた transcript は全てデバウンス→LLM返信パイプラインのみを通る（テキストチャット側の直接登録は従来どおり維持）。(b) **返信生成 revoke / 原子履歴コミット**: `handle_voice_reply` は生成開始前・生成返却後・各副作用の前にゲート世代を同期再チェックし、revoke 済み（gate 非アクティブまたは世代不一致）なら LLM・履歴コミット・返信・学習・TTS・リアクションの副作用を一切行わない。LLM 生成は `ask_voice_transcript` でセッション履歴から切り離して実行し（一時追加 → finally で必ず除去。音声返信は revoke されていない場合に限りセッションのインメモリ履歴へコミットされるが、RAG・Growth は無効（`store_memory=false` / `record_growth=false`））、生成返却後は await を挟まず世代を再チェックしてから user+assistant をセッションロック下で原子的にコミットする。`/voice stop|leave` は STT 停止より先に `VoiceReplyGate` を revoke し、進行中返信を bounded cancel。voice STT 側も `_revoke_generation` で保持送信 Future を cancel し、transcript 送信は送信直前に世代を再チェック（revoke 後は投稿・保存されない）。(c) **Voice CLI カレンダー独立 opt-in**: 音声発話からのカレンダー書き込みは `VOICE_CALENDAR_WRITE_ENABLED=true` の明示のみ有効（既定 `false` / fail closed）。マイク同意（`--microphone` / `SENSOR_MICROPHONE_ENABLED`）だけでは書き込まない。無効時は認識テキストを通常の LLM 経路へフォールスルーし、外部カレンダーへの書き込みクライアント自体を構築しない。(d) **ingest 完了 Event・原子 tmp 置換・revoke 後拒否**: `/api/screen/ingest` の描写完了は世代ごとの `threading.Event` で判定し（`run_in_executor` Future の done は下位 worker の完了を保証しないため、worker の finally でのみ set される Event を bounded wait）、コミットは `latest.json.tmp` への write/fsync → lock 区間での `os.replace` の原子置換。revoke（`_ingest_accepting=False`＋世代前進）済み後は受付を一切登録せず固定 503＋`unavailable` で拒否し、revoke 済み世代のコミットは起こらない（fail closed。どちらの順序でも revoke 済み世代の書き込みは抑制）。タイムアウト時は ownership を保持し restart / 新規 ingest と旧 worker が重ならない。(e) **Remote / Monitor の stop_pending と storage 順序**: stop は先に収集・読取 worker を stop/join し、join タイムアウトで worker が生存し続ける間は storage（latest.json 読取 / SQLite）を触らず・置き換えず・閉じず ownership を保持し `stop_pending` を公開。`is_running` は stop 要求後すぐ False（真実公開）。死を確認できたときのみ storage を閉じ・解放し、再 start は死確認後のみ許可（生存 worker を上書きしない）。(f) **Monitor 書込の有界リトライ**: SQLite 書込の一時失敗に対し同じ metrics を `write_attempts`（既定 3）回まで、`write_retry_delay`（既定 0.25s）の有界バックオフで再試行。exhaustion 時のみ固定カウンタ（`dropped_write_count` / `db_error_count`）と型名のみ・ASCII の診断（パス・SQLite メッセージ・プロセス詳細は出さない）。キュー・バッファは持たずメモリは増加しない。(g) **VAD/STT transcript-safe ログ**: 認識テキストは既定のパイプライン診断・ログへ出力しない（STT は所要時間のみ、失敗は例外型名のみ。Voice pipeline は認識テキストをセッション/LLM 経路のみへ渡す）。VAD はモデルロード完了のみ。テスト: `tests/test_discord_voice_reply_debouncer.py` / `tests/test_discord_voice_stt.py` / `tests/test_screen_remote.py` / `tests/web/test_sensor_error_safety.py` / `tests/context/test_monitor_provider.py` / `tests/audio/test_vad.py` / `tests/audio/test_pipeline_llm.py` / `tests/test_discord_assistant_route.py` |

### Discord 音声返信の保持境界（補足）

Discord 通話の音声返信は、生成後に user+assistant をセッションの**インメモリ履歴だけ**へ
コミットする。RAG と Growth への記録は無効であり、学習ログ (`DISCORD_TRAINING_LOG_ENABLED`
かつ `DISCORD_VOICE_TRAINING_LOG_ENABLED`) と STT transcript (`DISCORD_VOICE_STT_SAVE_TRANSCRIPTS`)
のディスク保存は、この履歴コミットとは独立した明示 opt-in である。`/voice stop|leave` の revoke
では追跡済み autoread タスクもキャンセルし、再生中の playback を停止する。revoke 済みの音声返信は
インメモリ履歴にもコミットしない。

## 未解決（unresolved）と推奨

| ID | 未解決事項 | 推奨既定（デフォルト明示） |
| --- | --- | --- |
| U12 | 実機 live 検証（実センサー・実マイク・実X11・実メインPC push・実サービス） | 未実施。deployed / verified は主張しない。P0-3.3（fail-safe 例外注入の固定）と P0-3.4（privacy-safe 検証の固定）および U11 の安全仕上げはリポジトリ実装＋オフライン分離テスト済みであり、実機・実サービスでの動作確認ではない |

## 検証・確認

- 本記録の U1〜U5 は `src/perception/policy.py` と
  `tests/perception/test_sensor_policy.py`（オフライン・Windows 開発機で unit 全成功）
  で固定される。U6〜U11 の配線・安全仕上げは `tests/audio/test_sensor_policy.py` /
  `tests/test_discord_sensor_policy.py` / `tests/test_desktop_companion.py` /
  `tests/context/test_monitor_provider.py` / `tests/test_screen_remote.py` /
  `tests/web/test_companion_state_api.py` / `tests/web/test_microphone_policy.py` /
  `tests/web/test_sensor_error_safety.py` / `tests/test_screen_context_lifecycle.py` /
  `tests/test_vision_context_lifecycle.py` / `tests/test_discord_voice_reply_debouncer.py` /
  `tests/test_discord_voice_stt.py` / `tests/audio/test_vad.py` /
  `tests/audio/test_pipeline_llm.py` / `tests/test_discord_assistant_route.py` の
  オフライン分離テストで固定される。
  P0-3.3（例外の型名のみログ・リソース解放・stop_pending の stop 所有権）と
  P0-3.4（allowlist 最小化・生/派生情報の非公開・CP932 安全）は上記の例外注入
  / privacy-safe テストで固定される。Screen agent の source gate、2コピーの byte equality、
  固定/型名のみの診断（URL・画像/本文内容なし）は `tests/test_screen_remote.py` の
  オフラインテストで固定される。Windows 開発機で `python -m unittest tests.test_screen_remote -q`
  （95件）が成功した。入口配線・実センサー・実サービスはリポジトリ内のみであり、
  deployed / verified とはしない（U12）。
- 実装状態は [implementation_status.md](../implementation_status.md) を正とする。

## 関連文書

- [implementation_plan.md](../implementation_plan.md): P0-3（センサーオプトイン強制）
- [implementation_status.md](../implementation_status.md): 実装状態の正典
- [src/perception/policy.py](../../src/perception/policy.py): 共有 SensorPolicy 解決器
- [src/perception/bootstrap.py](../../src/perception/bootstrap.py): 既存 `COMPANION_ACTIVITY_ENABLED` の活動起動
- [src/web/server.py](../../src/web/server.py): 既存 `WEB_SCREEN_CONTEXT_ENABLED` / `SCREEN_INGEST_TOKEN` の扱い