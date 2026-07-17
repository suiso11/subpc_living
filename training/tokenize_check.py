#!/usr/bin/env python3
"""Verify Qwen chat-template token lengths without printing record contents."""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

from .dataset import DatasetFormat, iter_jsonl


class Tokenizer(Protocol):
    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class TokenIssue:
    row_index: int
    field: str
    token_len: int
    limit: int
    reason: str = "exceeds max_tokens"
    def as_dict(self) -> dict[str, Any]: return self.__dict__.copy()


@dataclass
class TokenizeStats:
    lengths: list[int] = field(default_factory=list)
    def observe(self, value: int) -> None: self.lengths.append(value)
    @property
    def total(self) -> int: return len(self.lengths)
    @property
    def min_tokens(self) -> int | None: return min(self.lengths) if self.lengths else None
    @property
    def max_tokens(self) -> int | None: return max(self.lengths) if self.lengths else None
    @property
    def avg_tokens(self) -> float: return sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
    @property
    def p99_tokens(self) -> int | None:
        if not self.lengths: return None
        ordered = sorted(self.lengths)
        return ordered[max(0, math.ceil(len(ordered) * 0.99) - 1)]
    def as_dict(self) -> dict[str, Any]:
        return {"total": self.total, "min_tokens": self.min_tokens, "max_tokens": self.max_tokens,
                "avg_tokens": round(self.avg_tokens, 2), "p99_tokens": self.p99_tokens}


@dataclass
class TokenizeReport:
    format: DatasetFormat
    path: str
    max_tokens: int
    stats: TokenizeStats = field(default_factory=TokenizeStats)
    issues: list[TokenIssue] = field(default_factory=list)
    @property
    def ok(self) -> bool: return self.stats.total > 0 and not self.issues
    def as_dict(self) -> dict[str, Any]:
        return {"format": self.format, "path": self.path, "max_tokens": self.max_tokens,
                "ok": self.ok, "stats": self.stats.as_dict(), "issues": [i.as_dict() for i in self.issues]}


def _token_count(value: Any) -> int:
    if isinstance(value, int): return value
    if isinstance(value, dict): value = value.get("input_ids")
    elif hasattr(value, "input_ids"): value = value.input_ids
    if hasattr(value, "tolist"): value = value.tolist()
    if isinstance(value, list) and value and isinstance(value[0], list): value = value[0]
    if isinstance(value, list): return len(value)
    raise TypeError(f"unsupported chat-template output: {type(value).__name__}")


def count_messages(messages: list[dict[str, str]], tokenizer: Tokenizer) -> int:
    value = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, return_tensors=None,
    )
    return _token_count(value)


def count_tokens(text: str, tokenizer: Tokenizer) -> int:
    """Compatibility helper: count one user message through the chat template."""
    return count_messages([{"role": "user", "content": text}], tokenizer)


def scan_tokens(rows: Iterable[dict[str, Any]], fmt: DatasetFormat, tokenizer: Tokenizer,
                *, max_tokens: int = 2048) -> TokenizeReport:
    report = TokenizeReport(format=fmt, path="", max_tokens=max_tokens)
    for row_index, row in enumerate(rows):
        sequences: list[tuple[str, list[dict[str, str]]]] = []
        if fmt == "sft" and isinstance(row.get("messages"), list):
            sequences.append(("messages", row["messages"]))
        elif fmt == "dpo":
            prompt = row.get("prompt")
            if isinstance(prompt, str):
                for field_name in ("chosen", "rejected"):
                    answer = row.get(field_name)
                    if isinstance(answer, str):
                        sequences.append((field_name, [
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": answer},
                        ]))
        for field_name, messages in sequences:
            length = count_messages(messages, tokenizer)
            report.stats.observe(length)
            if length > max_tokens:
                report.issues.append(TokenIssue(row_index, field_name, length, max_tokens))
    return report


def load_tokenizer(model_name: str, revision: str | None = None, chat_template: str | None = None) -> Tokenizer:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, trust_remote_code=True)
    if chat_template:
        tokenizer.chat_template = Path(chat_template).read_text(encoding="utf-8")
    return tokenizer


def supports_assistant_mask(tokenizer: Tokenizer) -> bool:
    probe = [{"role": "user", "content": "確認"}, {"role": "assistant", "content": "応答"}]
    result = tokenizer.apply_chat_template(
        probe, tokenize=True, add_generation_prompt=False, return_dict=True,
        return_assistant_tokens_mask=True, return_tensors=None,
    )
    mask = result.get("assistant_masks") or result.get("assistant_mask")
    return bool(mask and sum(mask) > 0)


def run_tokenize_check(path: Path | str, fmt: DatasetFormat, tokenizer: Tokenizer,
                       *, max_tokens: int = 2048) -> TokenizeReport:
    report = scan_tokens(iter_jsonl(path), fmt, tokenizer, max_tokens=max_tokens)
    report.path = str(path)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--format", choices=("sft", "dpo"), required=True)
    parser.add_argument("--tokenizer", required=True, help="HF model id")
    parser.add_argument("--revision", help="Pinned HF model revision")
    parser.add_argument("--chat-template", help="Jinja template containing generation blocks")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tokenizer = load_tokenizer(args.tokenizer, args.revision, args.chat_template)
    if args.format == "sft" and not supports_assistant_mask(tokenizer):
        print("chat template has no assistant generation mask; assistant_only_loss would train zero tokens", file=sys.stderr)
        return 2
    report = run_tokenize_check(args.input, args.format, tokenizer, max_tokens=args.max_tokens)
    if args.json:
        print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
    else:
        st = report.stats
        print(f"tokenize {report.path} [{report.format}] samples={st.total} ok={report.ok}")
        print(f"min={st.min_tokens} max={st.max_tokens} p99={st.p99_tokens} avg={st.avg_tokens:.2f}")
        for issue in report.issues:
            print(f"row={issue.row_index} field={issue.field} tokens={issue.token_len} limit={issue.limit}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
