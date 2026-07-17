#!/usr/bin/env python3
"""Offline evaluation for the personal models (baseline / SFT / DPO / checkpoints).

The goal of this tool is to produce a comparable, nullable evaluation trace for each
training stage so baseline / SFT / DPO (and any intermediate checkpoint) can be
compared side by side.  Each stage keeps its own distinct tag so results are never
mixed up.

Design notes:
- ``evaluate_against_prompts`` accepts an injectable ``generator`` callable so the
  generation backend can be swapped out.  Tests pass a deterministic stub and never
  invoke any model.
- ``ollama_generator`` uses Ollama's HTTP API so generation options are applied; tests inject a stub.
- baseline / SFT / DPO / checkpoint each get a distinct ``model_tag`` that is embedded
  in every output record and in the report filename.
"""
from __future__ import annotations

import argparse
import json
import urllib.request
import sys
import time
from pathlib import Path
from typing import Any, Protocol

DEFAULT_PROMPTS = Path(__file__).resolve().parent / "eval_prompts.jsonl"

# Distinct tags per stage.  Never reuse a tag across stages.
MODEL_TAGS: dict[str, str] = {
    "baseline": "personal-baseline",
    "sft": "personal-sft",
    "dpo": "personal-dpo",
    "checkpoint": "personal-dpo-checkpoint",
}


class Generator(Protocol):
    def __call__(self, model: str, prompt: str, **opts: Any) -> str: ...


def model_tag(kind: str, checkpoint: str | None = None) -> str:
    """Return the distinct, stage-specific model tag.

    ``checkpoint`` produces a separately named tag so an intermediate DPO checkpoint
    is never confused with the final DPO adapter.
    """
    if kind not in MODEL_TAGS:
        raise ValueError(f"unknown kind: {kind!r}")
    if kind == "checkpoint":
        if not checkpoint:
            raise ValueError("checkpoint kind requires --checkpoint")
        # sanitize the checkpoint id into the tag, keep it distinct from "personal-dpo"
        safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in checkpoint)
        return f"personal-dpo-checkpoint-{safe}"
    return MODEL_TAGS[kind]


def load_eval_prompts(path: Path | str) -> list[dict[str, Any]]:
    """Load the fixed representative Japanese evaluation prompts."""
    p = Path(path)
    rows: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{p}:{lineno}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict) or "prompt" not in row:
                raise ValueError(f"{p}:{lineno}: missing 'prompt' field")
            rows.append(row)
    return rows


def build_ollama_request(
    model: str, prompt: str, *, num_ctx: int = 4096, num_predict: int = 128,
    temperature: float = 0.35,
) -> dict[str, Any]:
    """Build the supported Ollama /api/generate payload."""
    return {
        "model": model, "prompt": prompt, "stream": False,
        "options": {"num_ctx": num_ctx, "num_predict": num_predict, "temperature": temperature},
    }


def evaluate_against_prompts(
    kind: str,
    model: str,
    prompts: list[dict[str, Any]],
    generator: Generator,
    *,
    checkpoint: str | None = None,
    gen_opts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run ``generator`` for each prompt and collect a comparable trace.

    ``generator`` is responsible for actually producing text; this function never
    starts a subprocess.  All generation options are forwarded to ``generator``.
    """
    tag = model_tag(kind, checkpoint)
    gen_opts = dict(gen_opts or {})
    started = time.time()
    rows: list[dict[str, Any]] = []
    for prompt_row in prompts:
        prompt = str(prompt_row.get("prompt", "")).strip()
        if not prompt:
            continue
        t0 = time.time()
        response = str(generator(model, prompt, **gen_opts))
        rows.append({
            "id": prompt_row.get("id"),
            "category": prompt_row.get("category"),
            "prompt": prompt,
            "response": response,
            "elapsed_s": round(time.time() - t0, 4),
        })
    return {
        "kind": kind,
        "model": model,
        "tag": tag,
        "checkpoint": checkpoint,
        "gen_opts": gen_opts,
        "count": len(rows),
        "elapsed_s": round(time.time() - started, 4),
        "rows": rows,
    }


def ollama_generator(model: str, prompt: str, **opts: Any) -> str:
    """Generate through Ollama's HTTP API (CLI does not support generation options)."""
    base_url = str(opts.pop("base_url", "http://localhost:11434")).rstrip("/")
    timeout = float(opts.pop("timeout", 120))
    payload = build_ollama_request(model, prompt, **opts)
    request = urllib.request.Request(
        f"{base_url}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.load(response)
    return str(data.get("response", "")).strip()


def write_report(result: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kind",
        choices=("baseline", "sft", "dpo", "checkpoint"),
        required=True,
        help="Training stage.  Each kind gets a distinct model tag.",
    )
    parser.add_argument("--model", required=True, help="Ollama model name to evaluate.")
    parser.add_argument(
        "--checkpoint",
        help="Checkpoint identifier (required only for kind=checkpoint).",
    )
    parser.add_argument("--prompts", default=str(DEFAULT_PROMPTS))
    parser.add_argument("--output", required=True, help="Output JSON report path.")
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--num-predict", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--ollama-base-url", default="http://localhost:11434")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prompts = load_eval_prompts(args.prompts)
    if not prompts:
        print("no prompts loaded", file=sys.stderr)
        return 2
    if args.kind == "checkpoint" and not args.checkpoint:
        print("--checkpoint is required for kind=checkpoint", file=sys.stderr)
        return 2
    gen_opts = {
        "num_ctx": args.num_ctx,
        "num_predict": args.num_predict,
        "temperature": args.temperature,
        "base_url": args.ollama_base_url,
    }
    result = evaluate_against_prompts(
        args.kind,
        args.model,
        prompts,
        ollama_generator,
        checkpoint=args.checkpoint,
        gen_opts=gen_opts,
    )
    write_report(result, Path(args.output))
    print(f"[{result['tag']}] {result['count']} prompts -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())