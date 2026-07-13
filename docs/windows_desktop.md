# Windowsネイティブアプリ

SUBPC BUDDYの普段使い用クライアントです。64bit Windows 10/11を対象に、HTML/WebViewではなく、PySide6 + Qt Quick/QML（Qt 6.8 LTS系）で描画します。
会話・タスク・記録・実績は既存の `subpc-web` APIと同じデータを利用するため、DiscordやWeb画面との状態が分裂しません。

## 開発実行

Windows PowerShellで次を実行します。

```powershell
py -3.12 -m venv .venv-desktop
.\.venv-desktop\Scripts\python.exe -m pip install -r requirements-desktop.txt
$env:SUBPC_DESKTOP_SERVER_URL = "http://127.0.0.1:8000"
.\.venv-desktop\Scripts\python.exe -m src.desktop
```

バックエンドをサブPCで動かす場合は、接続先をTailscaleアドレスなどへ変更します。

```powershell
.\.venv-desktop\Scripts\python.exe -m src.desktop --server "http://100.x.x.x:8000"
```

接続先は `%APPDATA%\SUBPC BUDDY\desktop.json` に保存されます。

## 操作

- `Ctrl+K`: コマンドパレット
- `Alt+1`〜`Alt+4`: 話す・やること・記録・実績
- `Ctrl+Alt+Space`: どのアプリからでも表示・非表示（Windows）
- 閉じるボタン: タスクトレイへ格納
- マイクボタン長押し: Windows既定入力から録音し、既存STTへ送信
- 二重起動: 新しいプロセスは終了し、既存ウィンドウを前面へ戻す

Windowsの `Win+Space` は入力言語切替に予約されているため、グローバル呼び出しには `Ctrl+Alt+Space` を使います。

## EXEビルド

```powershell
.\scripts\build_windows_desktop.ps1
```

テスト、EXE生成、生成物の起動検査まで成功すると `dist\SUBPC-BUDDY.exe` が生成されます。
初回起動後、設定画面からWindowsログイン時の自動起動を有効にできます。

コード署名はまだ行わないため、別PCへコピーした初回起動時にSmartScreenが確認を出す場合があります。

## バックエンドとの境界

Windowsアプリは重いLLM/STT/TTSモデルを重複ロードしません。HTTP APIとWebSocketで既存サービスへ接続します。
同じWindows PCでバックエンドも動かす場合は `http://127.0.0.1:8000`、別のサブPCならその安全なLAN/Tailscale URLを指定します。
