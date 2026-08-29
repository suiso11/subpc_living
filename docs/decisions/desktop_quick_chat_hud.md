# P1 デスクトップ Quick-Chat HUD の決定記録

> **状態**: active / canonical
> **位置付け**: デスクトップ常駐の Quick-Chat HUD（オーバーレイ拡張パネル）の機能・
>   データ境界・配線の決定記録
> **対象範囲**: P1 Quick-Chat HUD の画面構成（Today サマリ・最近メッセージ・starter・
>   入力欄）、DesktopBridge 既存機能の再利用、privacy-safe な count-only データ境界、
>   クリックスルー desired-state 配線、2D/3D fallback、オフライン時の挙動
> **作成日**: 2026-08-29
> **更新日**: 2026-08-29
> **日付根拠**: Git commit date

## 背景

デスクトップ常駐オーバーレイ（`src/desktop/qml/Overlay.qml`）は常時表示のコンパクト
状態（アバター）しか持たず、会話・タスク・予定の確認には本体ウィンドウを開く必要が
あった。P1 Quick-Chat HUD はオーバーレイを拡張（expanded）パネルにし、開かずに
「今日の状態」と「すばやい会話」を完結させる。実装は DesktopBridge
（`src/desktop/bridge.py`）が既に公開している既存の読み取り専用プロパティと送信経路を
再利用し、重複 API / エンドポイントを新設しない。表示データはプライバシー安全な
count-only に限定し、生データ・センサー詳細・パス・URL・エンドポイント文字列を
HUD に露出しない。

## 経緯（時系列）

| 日付 | 区分 | 内容 |
| --- | --- | --- |
| 2026-08-29 | 決定 | 本記録の決定を採択（既存 DesktopBridge 再利用・count-only・desired-state クリックスルー・2D/3D fallback 維持） |

## 現時点の決定

1. **Quick-Chat HUD は DesktopBridge の既存プロパティ・シグナル・送信経路のみを使い、
   新しいエンドポイント・API を追加しない**。HUD は次を再利用する:
   - `sendMessage(text)`（`/ws/chat` への既存 websocket 送信）
   - `messages`（チャット履歴。表示は直近 4 件・最大 2 行に有界化）
   - `game.starters`（starter プロンプト。先頭 3 件に有界化）
   - `tasks`（開いているタスク数の集計のみ）
   - `growth`（`today_points` / `streak_days` の集計のみ）
   - `calendarEvents`（今日以降の予定数の集計のみ）
   - `overlayShell`（shell_state と provenance）
   - `connected` / `loading`
   `Overlay.qml` 内に `/api/` `/ws/` `http://` `https://` の文字列を持たない。
2. **Today サマリは count-only（allowlist 固定表示のみ）**:
   「今日の成長」（`today_points` / `streak_days`）、「タスク」（open 件数）、
   「予定」（今日以降の件数）。件数は数値のみで、生データ・タスク本文・
   予定タイトル・プロセス・センサー・パス・モデル・snapshot・context を表示しない。
3. **状態表示は固定の PC/活動状態と provenance のみ**:
   shell_state（待機中 / 作業中 / 会話中 / 離席中 / 予定が近づいています / エラー）と
   `sensor_provenance`（出所・取得時刻・保存なし）。詳細は `src/desktop/shell.py` の
   決定論的 mapping のみ。
4. **キー操作は既存動作に揃える**: 拡張時に composer へ
   `forceActiveFocus()`（フォーカス）、`Esc` で collapse。送信は trim→空なら無視→
   `sendMessage`→クリア→フォーカス維持。
5. **オフライン・読み込み中は送信を無効化する**:
   `canSend = connected && !loading`。ステータスバナーは固定文言のみ
   （オフライン / 接続済み / 読み込み中…）で、statusText・serverUrl・env 名・
   値・エンドポイントは露出しない。
6. **クリックスルーは desired-state 方式で bridge → ネイティブ controller に配線する**:
   HUD は `bridge.setOverlayClickThrough(bool)` の「希望状態」だけを持ち、
   `overlayClickThroughRequested` シグナルで `OverlayClickThroughController`
   （`src/desktop/app.py`）へ伝え、`apply_click_through(hwnd, enabled)`
   （`src/desktop/windows.py`、WS_EX_LAYERED | WS_EX_TRANSPARENT）が実適用する。
   HUD は hwnd / winId / HOTKEY_ID / WS_EX などの詳細に触れない。復帰は
   **Ctrl+Alt+Space**（Windows 予約の Win+Space を避けた `RegisterHotKey`）で
   `restore_interaction()` を先に呼んでから本体ウィンドウを toggle する。
   オーバーレイ無効時は有効化要求を拒否し、停止・終了時は必ず解除する。
7. **既存の 2D/3D fallback を維持し、3D の再設計はしない**:
   `hasAvatar3D`（`avatarModel.exists && !modelLoadFailed`）のときのみ
   `OverlayAvatar3D.qml`（既存 QtQuick3D）をロードし、失敗・不在時は既存の
   2D サークルアバターにフォールバックする。VRM 参照・新規 3D コンポーネントは追加しない。
   拡張中はアバターを非表示にする（`!expanded`）。

## 判断理由

- **1（再利用・エンドポイント新設なし）**: 既存 DesktopBridge は
  messages / tasks / game / growth / calendar / overlayShell / sendMessage を既に
  提供しており、HUD 用の新 API はバックエンドとの契約面を増やし、重複実装と
  テスト面を広げるだけ。websocket の `sendMessage` は送信・ストリーム・done 処理
  （`_socket_message`）をそのまま共有でき、TTL などの生成側挙動も揃う。
- **2（count-only）**: HUD は「開かずに概要を見る」用途。タスク本文・予定タイトル・
  成長詳細の表示は本体に委ね、サマリは件数・点数に限定すれば、画面に漏れうる
  データ面を最小化できる。
- **3（固定状態＋provenance）**: 状態遷移は `decide_shell_state` の決定論的な
  優先順位 mapping で、生データや LLM 判断を HUD に持ち込まない。provenance で
  「いつ・どのセンサー由来か」を示し、`saved: false`（保存なし）を明示する。
- **4・5（フォーカス / Esc / オフライン）**: コンパクト入力は
  「開いて・打って・閉じる」の一連が崩れないことが必須。空送信・未接続・読み込み中の
  送信を防ぎ、バナーは固定文言だけにして接続情報・URL・設定値の露出を避ける。
- **6（desired-state）**: QML から OS 窓拡張スタイルを直接触らせない。
  bridge は「希望状態」と表示状態だけを公開し、実際の hwnd 操作は
  `OverlayClickThroughController` が担う。これにより HUD の権限面を小さく保ち、
  テスト（app / windows / shell）と配線（シグナル→ネイティブ）を分離できる。
  Ctrl+Alt+Space は Windows のレイアウト切替（Win+Space）と衝突しない復帰経路。
- **7（2D/3D fallback 維持・再設計なし）**: 既存の 3D ロード失敗 fallback と
  2D 表示を変えず、拡張中はアバターを隠す。3D アバターの再設計は P1 の範囲外で、
  VRM 等の新規依存も増やさない。

## 前提（assumptions）

| ID | 前提 | 推奨既定 |
| --- | --- | --- |
| A1 | DesktopBridge の既存プロパティ（messages / tasks / game / growth / calendarEvents / overlayShell / connected / loading）と `sendMessage` は HUD が利用可能な正の入力源 | HUD はこれら以外にデータを要求しない |
| A2 | サマリは count-only であり、数値集計は HUD 側（`safeCount` / `countOpenTasks` / `countUpcomingCalendar`）で行う | 件数・点数・streak 数のみ表示 |
| A3 | クリックスルーの実適用はネイティブ（`OverlayClickThroughController` → `apply_click_through`）が担う | bridge は desired-state のみ公開 |
| A4 | Ctrl+Alt+Space は復帰＋本体 toggle のグローバル hotkey として登録可能 | 登録失敗時はバナーで通知し動作は止めない |
| A5 | 3D アバターは既存 `OverlayAvatar3D.qml` のみ。再設計・VRM 導入はしない | 不在・失敗時は 2D サークルへフォールバック |

## 解決済み（resolved）

| ID | 事項 | 実装 |
| --- | --- | --- |
| U1 | HUD の入力源（重複 API を避ける配線） | `Overlay.qml` は bridge の `messages`（直近 4 件 / `recentMessages()`）、`game.starters`（先頭 3 件 / `starterPrompts()`）、`tasks`・`growth`・`calendarEvents`（集計のみ）、`overlayShell`・`connected`・`loading` を使用。送信は `sendMessage(text)` のみ（websocket 経由）。`/api/` `/ws/` `http(s)://` 文字列なし |
| U2 | Today サマリの schema | `overlayTodaySummary` ブロック内に固定 3 カード: `overlaySummaryGrowth`（`today_points` / `streak_days` を `safeCount`）、`overlaySummaryTasks`（`countOpenTasks` の open 件数）、`overlaySummaryCalendar`（`countUpcomingCalendar` の今日以降の件数）。数値以外のデータ・`modelData`・`statusText`・`monitor`・`snapshot`・`context`・`process`・`sensor`・`path`・`model` はブロックに持たない |
| U3 | 状態・provenance 表示 | `overlayShell.shell_state` → 固定ラベル（待機中/作業中/会話中/離席中/予定が近づいています/エラー）、`shell.provenance` → `出所: {source_label} · 取得: {HH:MM} · 保存: なし` |
| U4 | フォーカス / Esc / 送信ゲート | `onExpandedChanged` で `composerField.forceActiveFocus()`。`overlayContent` と composer の `Keys.onEscapePressed` で collapse。`send()` は `canSend`（`connected && !loading`）と trim 後の空を gate |
| U5 | オフライン・読み込みバナー | `statusBannerText` は固定 3 値のみ（読み込み中… / 接続済み / オフライン）。`statusText`・`serverUrl` は使用しない |
| U6 | クリックスルー配線 | bridge `setOverlayClickThrough(bool)`（enabled は overlay 有効時のみ・冪等）→ `overlayClickThroughRequested` → `OverlayClickThroughController._on_requested` → `apply_click_through(hwnd, enabled)`。HUD に hwnd / winId / HOTKEY_ID / WS_EX / `apply_click_through` 文字列なし。停止・`stopOverlayFromOverlay`・shutdown 時に `_force_overlay_click_through_off` で解除 |
| U7 | Ctrl+Alt+Space 復帰 | `WindowsHotkeyFilter.register` で `RegisterHotKey`（MOD_ALT|MOD_CONTROL|MOD_NOREPEAT, VK_SPACE, HOTKEY_ID=0xBADD）。`_on_hotkey_activated` は `restore_interaction()` を `_toggle_window` より先に実行 |
| U8 | 2D/3D fallback | `hasAvatar3D = avatarModel.exists && !modelLoadFailed`。Loader は `hasAvatar3D && !expanded` のとき `OverlayAvatar3D.qml` を active、ロード失敗は `modelLoadFailed=true` で 2D へ。2D は `!hasAvatar3D && !expanded`。VRM 参照なし |

## 未解決（unresolved）と推奨（deferred 非目標）

| ID | 未解決事項 | 推奨既定（デフォルト明示） |
| --- | --- | --- |
| U9 | 実機 live 検証（実 Windows ウィンドウ・実 ws/chat・実クリックスルー・実 hotkey・実サービスでの表示/送信確認） | 未実施。本記録はリポジトリ実装＋オフライン static/unit テストのみであり、deployed / verified は主張しない |
| U10 | HUD からのタスク追加・完了・予定作成などの書き込み UI | deferred（範囲外）。HUD は既存 `sendMessage` での会話経由のみ。タスク/カレンダー操作は本体・Discord・Web の既存 UI に委ねる |
| U11 | 3D アバターの再設計（VRM 対応・新モデル形式・アニメーション強化） | deferred。既存 `OverlayAvatar3D.qml`（QtQuick3D + アセット）と 2D fallback を維持する。P1 では 3D の再設計を行わない |
| U12 | Starter プロンプトの個数・出所の設定可能化 | deferred。当面は `game.starters` の先頭 3 件を固定表示（`starterPrompts()` の `slice(0, 3)`） |
| U13 | クリックスルー時の通知・復帰手がかりの追加 | deferred。現在は HUD 内の固定ヒント（「Ctrl+Alt+Space で解除」）のみ。復帰の視覚強調・音声通知は要望に応じて再評価 |

## 検証・確認

- Quick-Chat HUD の静的契約は `tests/test_desktop_contract.py`
  （`OverlayContractTest`・`DesktopClickThroughContractTest`）で固定される:
  objectName 一覧・provenance ラベル・VRM なし・アクションボタン・状態ラベル・
  `sendMessage` 呼び出し 1 箇所・`canSend` 契約・Esc/フォーカス・
  バナー固定文言（statusText / serverUrl なし）・拡張時アバター非表示・
  メッセージ有界（`result.length < 4` / `maximumLineCount: 2`）・
  HUD 内に `/api/` `/ws/` `http(s)://` なし・サマリ allowlist count-only
  （`today_points` / `streak_days` と数値表示のみで `monitor` / `snapshot` /
  `context` / `process` / `sensor` / `path` / `model` / `modelData` なし）・
  クリックスルー契約（`setOverlayClickThrough(` 1 箇所・hwnd/winId/HOTKEY_ID/
  WS_EX なし）。クリックスルー配線・hotkey 順序・失敗/終了時の disconnect は
  `DesktopClickThroughContractTest` で固定。
- 状態 mapping・クリックスルー controller・hotkey は
  `tests/test_desktop_shell.py`（`TestOverlayVisibility` /
  `OverlayClickThroughTestCase` / `TestOverlayClickThroughApply`）と
  `src/desktop/windows.py` / `src/desktop/app.py` のオフライン unit テストで固定される。
  Windows 開発機で `python -m unittest tests.test_desktop_contract tests.test_desktop_shell tests.test_desktop_qml tests.test_desktop_companion -q`
  が成功していることを確認する（QML ロード検査は `tests/test_desktop_qml.py`）。
- 本記録はオフラインでの決定記録であり、実機・実サービス・ネットワークでの
  live 検証は未実施。deployed / verified は false（U9）。
- 実装状態は [implementation_status.md](../implementation_status.md) を正とする。

## 関連文書

- [implementation_plan.md](../implementation_plan.md): P1（Quick-Chat HUD）
- [implementation_status.md](../implementation_status.md): 実装状態の正典
- [sensor_opt_in_policy.md](./sensor_opt_in_policy.md): P0-3（センサー opt-in 強制）
- [voice_context_parity.md](./voice_context_parity.md): 音声対話の会話構成パリティ
- [src/desktop/bridge.py](../../src/desktop/bridge.py): DesktopBridge（再利用するプロパティ・`sendMessage`・クリックスルー desired-state）
- [src/desktop/api.py](../../src/desktop/api.py): DesktopApi（既存 HTTP クライアント。HUD からはエンドポイント追加なし）
- [src/desktop/app.py](../../src/desktop/app.py): `OverlayClickThroughController`・`WindowsHotkeyFilter`（Ctrl+Alt+Space）
- [src/desktop/windows.py](../../src/desktop/windows.py): `apply_click_through`（WS_EX_LAYERED | WS_EX_TRANSPARENT）
- [src/desktop/shell.py](../../src/desktop/shell.py): `decide_shell_state` / `overlay_visibility` / `sensor_provenance`
- [src/desktop/qml/Overlay.qml](../../src/desktop/qml/Overlay.qml): Quick-Chat HUD（expanded パネル）
- [src/desktop/qml/OverlayAvatar3D.qml](../../src/desktop/qml/OverlayAvatar3D.qml): 既存 3D アバター（再設計しない）