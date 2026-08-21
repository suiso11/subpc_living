"""VRMアバターモデルの発見と解決 (Phase 6b groundwork)。

ユーザー所有VRMの読み込みを優先し、第三者モデルを製品へ同梱しない
(roadmap §7 / §8 Phase 6)。VRMファイルはリポジトリに置かず、
``models/vrm/`` (gitignore済み) から実行時に発見する。

解決順序:
1. 明示パス引数
2. ``DESKTOP_VRM_MODEL`` 環境変数
3. ``models/vrm/`` 内の ``*.vrm`` を名前順に走査した先頭
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

VRM_ENV_VAR = "DESKTOP_VRM_MODEL"


def default_vrm_dir(project_root: Path | None = None) -> Path:
    """既定のVRM配置ディレクトリを返す。"""
    root = project_root if project_root is not None else Path(__file__).resolve().parents[2]
    return root / "models" / "vrm"


def discover_vrm_models(directory: Path) -> list[Path]:
    """ディレクトリ内の ``*.vrm`` を名前順で返す。無ければ空リスト。"""
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.glob("*.vrm") if p.is_file())


def is_probable_vrm(path: Path) -> bool:
    """安価なマジック検査: VRM は glTF 基盤なので先頭4バイトが ``glTF``。"""
    try:
        with path.open("rb") as f:
            return f.read(4) == b"glTF"
    except OSError:
        return False


def resolve_avatar_model(
    project_root: Path | None = None,
    *,
    explicit: str | Path | None = None,
    env: Mapping[str, str] | None = None,
) -> Path | None:
    """使用するVRMモデルのパスを解決する。見つからなければ None。

    - ``explicit`` 指定時: 存在するファイルならそのパス、無ければ None
      (呼び出し元へ欠損を委ねる。例外にはしない)
    - 環境変数 ``DESKTOP_VRM_MODEL``: 存在するファイルなら採用
    - 既定ディレクトリ走査: 名前順先頭の ``*.vrm``
    """
    environment = os.environ if env is None else env
    root = project_root if project_root is not None else Path(__file__).resolve().parents[2]

    if explicit is not None:
        candidate = Path(explicit)
        return candidate if candidate.is_file() else None

    env_path = (environment.get(VRM_ENV_VAR) or "").strip()
    if env_path:
        candidate = Path(env_path)
        if candidate.is_file():
            return candidate

    for candidate in discover_vrm_models(default_vrm_dir(root)):
        return candidate
    return None
