#!/usr/bin/env python3
"""Atomically switch only chat_config.json's model field, or roll it back."""
from __future__ import annotations
import argparse, json, os, sys, tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
DEFAULT_CONFIG=ROOT/"config/chat_config.json"


def read_object(path: Path) -> dict:
    data=json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data,dict): raise ValueError("config root must be an object")
    return data


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix=f".{path.name}.",dir=path.parent)
    try:
        with os.fdopen(fd,"wb") as f:
            f.write(payload); f.flush(); os.fsync(f.fileno())
        os.replace(name,path)
    except Exception:
        try: os.unlink(name)
        except FileNotFoundError: pass
        raise


def backup_path(path: Path) -> Path: return path.with_name(path.name+".bak")


def switch(path: Path, model: str) -> int:
    model=model.strip()
    if not model: print("model name is empty",file=sys.stderr); return 2
    original=path.read_bytes(); before=read_object(path)
    if before.get("model")==model: print(f"already selected: {model}"); return 0
    atomic_write(backup_path(path),original)
    after=dict(before); after["model"]=model
    atomic_write(path,(json.dumps(after,ensure_ascii=False,indent=2)+"\n").encode())
    verified=read_object(path)
    expected=dict(before); expected["model"]=model
    if verified!=expected:
        atomic_write(path,original)
        print("verification failed; original restored",file=sys.stderr); return 1
    print(f"switched: {before.get('model')} -> {model}")
    return 0


def rollback(path: Path) -> int:
    backup=backup_path(path)
    if not backup.exists(): print("backup not found",file=sys.stderr); return 1
    payload=backup.read_bytes(); json.loads(payload.decode("utf-8")); atomic_write(path,payload)
    print(f"rolled back: {read_object(path).get('model')}")
    return 0


def show(path: Path) -> int:
    data=read_object(path); model=str(data.get("model","")); overrides=data.get("model_prompt_overrides",{})
    print(f"model: {model}")
    print(f"prompt_override: {'yes' if isinstance(overrides,dict) and model in overrides else 'no'}")
    print(f"backup: {'yes' if backup_path(path).exists() else 'no'}")
    return 0


def main(argv=None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default=str(DEFAULT_CONFIG))
    sub=parser.add_subparsers(dest="command",required=True)
    sub.add_parser("show"); p=sub.add_parser("switch"); p.add_argument("model"); sub.add_parser("rollback")
    args=parser.parse_args(argv); path=Path(args.config).expanduser()
    try:
        if args.command=="show": return show(path)
        if args.command=="switch": return switch(path,args.model)
        return rollback(path)
    except (OSError,ValueError,json.JSONDecodeError) as exc:
        print(f"model switch failed: {exc}",file=sys.stderr); return 1

if __name__=="__main__": raise SystemExit(main())
