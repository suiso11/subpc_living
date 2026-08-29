# Tailscale SSH セットアップ・使い方

> ⚠️ **未検証テンプレート**: このプロジェクトでは **Tailscale ノードは未検証**（デプロイ済みの
> サブPCは存在しない）。以下の内容は**汎用テンプレート**であり、ホスト名・IP・OS・バージョンは
> プレースホルダで表す。実機で確定する値は各節の「確認コマンド」で発見し、ここへ反映すること。

マンション共有回線（NAT越え不可）でも外部からSSH接続できるようにするため、Tailscaleを利用する想定。

## 仕組み

- Tailscale = WireGuardベースのP2P VPNメッシュ
- NATの種類を問わず端末同士を直接接続（ポート開放不要）
- 外部にSSHポートを晒さないので安全
- `--ssh` オプションでTailscale SSHを有効化すると、SSH鍵の管理もTailscale側で自動化される

## このマシンの情報（デプロイ後に記入）

以下の項目は**実機導入時に確定・記入する**。本リポジトリからは断定しない。

- ホスト名: `<サブPCのホスト名>`（未確定）
- OS: `<サブPCのOSとバージョン>`（未確定）
- Tailscale バージョン: `<導入時に確認>`（未確定）

---

## 初回セットアップ（未検証テンプレート）

```bash
# インストール
curl -fsSL https://tailscale.com/install.sh | sh

# Tailscale SSH 有効で起動
sudo tailscale up --ssh

# ブラウザで認証URLを開いてログイン
```

## 確認コマンド（発見用・★未検証）

```bash
# 接続状態の確認
tailscale status

# このマシンのTailscale IPを確認（実機で確認して記録）
tailscale ip -4

# ホスト名・MagicDNS名を確認（実機で確認して記録）
hostname
tailscale status --json | jq -r '.Self.DNSName'

# 詳細情報
tailscale status --json | jq '.Self'
```

> **このプロジェクトでは Tailscale ノードは未検証**。上記コマンドの実行結果は
> デプロイ時にここへ反映すること。

---

## 接続側（リモートから繋ぐ端末）のセットアップ

### 1. Tailscaleをインストール

| OS | コマンド/方法 |
|------|-------------|
| Linux | `curl -fsSL https://tailscale.com/install.sh \| sh` |
| macOS | `brew install tailscale` または [App Store](https://apps.apple.com/app/tailscale/id1475387142) |
| Windows | [tailscale.com/download](https://tailscale.com/download) からインストーラ |
| iOS/Android | 各ストアから「Tailscale」をインストール |

### 2. 同じアカウントでログイン

```bash
sudo tailscale up
# ブラウザで認証（同じアカウントでログイン）
```

### 3. SSH接続（★未検証）

```bash
# Tailscale IP で接続（IP は実機で確認）
ssh <ユーザー名>@<サブPCのIP>

# または MagicDNS が有効ならホスト名で接続
ssh <ユーザー名>@<サブPCのホスト名>

# Tailscale SSH (鍵なし) の場合はそのまま繋がる
```

### 4. VS Code Remote SSH で使う場合

`~/.ssh/config` に追加（値は実機で確定・★未検証）：

```
Host subpc
    HostName <サブPCのIP>
    User <ユーザー名>
```

VS Code で `Ctrl+Shift+P` → `Remote-SSH: Connect to Host` → `subpc` を選択。

---

## 運用コマンド

```bash
# Tailscaleの状態確認
tailscale status

# 切断（一時停止）
sudo tailscale down

# 再接続
sudo tailscale up --ssh

# ログアウト（デバイス削除）
sudo tailscale logout

# tailscaled デーモンの状態確認
sudo systemctl status tailscaled

# サービス再起動
sudo systemctl restart tailscaled
```

## ファイル転送（★未検証）

```bash
# Tailscale経由でSCP
scp ./file.txt <ユーザー名>@<サブPCのIP>:/home/<ユーザー名>/

# Tailscale 組み込みのファイル送信（taildrop）
tailscale file cp ./file.txt <サブPCのホスト名>:

# 受信側で受け取り
tailscale file get ./
```

---

## セキュリティ設定（Tailscale管理画面）

[https://login.tailscale.com/admin](https://login.tailscale.com/admin)

- **ACL**: どのデバイスがどのデバイスに接続できるか制御
- **Key Expiry**: 認証キーの有効期限（デフォルト180日、無効化も可能）
- **Tailscale SSH 設定**: SSH接続の許可ルール
- **MagicDNS**: ホスト名でのアクセスを有効/無効

### ACL でSSH接続を特定デバイスに制限する例

管理画面 → Access Controls で以下のようなポリシーを設定：

```json
{
  "ssh": [
    {
      "action": "accept",
      "src":    ["autogroup:members"],
      "dst":    ["autogroup:self"],
      "users":  ["autogroup:nonroot", "root"]
    }
  ]
}
```

---

## トラブルシューティング

| 症状 | 対処 |
|------|------|
| `tailscale up` でタイムアウト | `sudo systemctl restart tailscaled` してリトライ |
| 接続が遅い / DERP経由になる | `tailscale netcheck` でNAT種別確認、DERP relayは正常動作 |
| SSH接続拒否 | `tailscale up --ssh` で起動したか確認 |
| 鍵認証エラー | Tailscale SSH使用時は鍵不要。通常SSH使用時は `~/.ssh/authorized_keys` 確認 |
| デバイスが見えない | 両方のデバイスが同じアカウントで認証済みか確認 |
| キー期限切れ | 管理画面でキーを再認証、または `sudo tailscale up --ssh` で再ログイン |

```bash
# ネットワーク診断
tailscale netcheck

# Pingテスト
tailscale ping <相手のホスト名 or IP>

# ログ確認
sudo journalctl -u tailscaled -f
```

---

## 自動起動（★未検証）

`tailscaled` サービスはインストール時にsystemdで自動有効化される想定：

```bash
# 確認（実機で確認）
sudo systemctl is-enabled tailscaled
```

OS再起動後も自動的に接続される想定。実機での動作確認は未実施。
