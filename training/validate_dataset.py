#!/usr/bin/env python3
"""Privacy-safe dataset preflight for the personal LoRA pipeline.

Combines four independent checks over a JSONL training dataset:

1. schema validation (via :mod:`training.dataset`)
2. duplicate detection
3. per-field character length caps
4. secret / PII scanning

Privacy contract
----------------
The secret/PII scanner reports *which* rule matched and *where* (row index +
field path) but **never** echoes the matched value.  Records that triggered a
finding are referenced solely by their zero-based row index, so reports can be
safely attached to issues or pasted in chat.

The default training-bound export schema contains no ``metadata`` field.  This
preflight is intended to run *before* export and therefore also inspects
``metadata`` values when present, because locally retained audit copies may
carry channel/model provenance that should not leak into training data.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Pattern, Sequence

from .dataset import (
    DatasetFormat,
    DuplicateReport,
    LengthReport,
    SchemaReport,
    detect_duplicates,
    detect_length_issues,
    iter_jsonl,
    validate_schema,
)


class InvalidRecordError(ValueError):
    """Raised when a single record cannot be validated/imported."""


@dataclass(frozen=True)
class PIIRule:
    """A named secret/PII detection pattern.

    ``fields`` restricts which record paths the rule scans.  When empty the
    rule is tested against every string field of the record (content + any
    metadata values).
    """

    name: str
    pattern: Pattern[str]
    description: str = ""
    fields: tuple[str, ...] = ()

    def __repr__(self) -> str:
        return f"PIIRule(name={self.name!r})"


@dataclass(frozen=True)
class PIIIssue:
    """A single secret/PII finding.  Never carries the matched value."""

    row_index: int
    field: str
    rule: str
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "row_index": self.row_index,
            "field": self.field,
            "rule": self.rule,
            "reason": self.reason,
        }


@dataclass
class PIIReport:
    format: DatasetFormat
    ok: bool
    issues: list[PIIIssue] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "ok": self.ok,
            "issues": [i.as_dict() for i in self.issues],
        }


@dataclass
class PreflightReport:
    format: DatasetFormat
    path: str
    total: int
    schema: SchemaReport
    duplicates: DuplicateReport
    length: LengthReport
    pii: PIIReport

    @property
    def ok(self) -> bool:
        return self.schema.ok and self.duplicates.ok and self.length.ok and self.pii.ok

    def as_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "path": self.path,
            "total": self.total,
            "ok": self.ok,
            "schema": self.schema.as_dict(),
            "duplicates": self.duplicates.as_dict(),
            "length": self.length.as_dict(),
            "pii": self.pii.as_dict(),
        }


def _discord_token_re() -> Pattern[str]:
    # Discord bot/user tokens: base64-ish blobs separated by dots.
    return re.compile(r"(?:[\w-]{20,})\.[\w-]{4,}\.[\w-]{20,}")


def _generic_secret_re() -> Pattern[str]:
    # ``key``/``token``/``secret`` assignments in arbitrary quoting.
    return re.compile(
        r"(?i)\b(?:api[_-]?key|secret|access[_-]?token|auth[_-]?token|"
        r"client[_-]?secret|private[_-]?key|password)\b\s*[:=]\s*[\"\']?[A-Za-z0-9_./+=-]{8,}"
    )


def _bearer_re() -> Pattern[str]:
    return re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{16,}")


def _jwt_re() -> Pattern[str]:
    return re.compile(r"\beyJ[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\b")


def _email_re() -> Pattern[str]:
    return re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")


def _jp_phone_re() -> Pattern[str]:
    return re.compile(r"\b0\d{1,3}-\d{2,4}-\d{3,4}\b")


def _credit_card_re() -> Pattern[str]:
    return re.compile(r"\b(?:\d[ -]?){13,16}\b")


def _long_hex_re() -> Pattern[str]:
    # AWS-style access keys / generic 40-char hex hashes used as credentials.
    return re.compile(r"\b[A-Za-z0-9+/]{40,}\b")



def _provider_token_re() -> Pattern[str]:
    return re.compile(r"\b(?:sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|AKIA[A-Z0-9]{16})\b")

def _jp_postal_re() -> Pattern[str]:
    return re.compile(r"(?<!\d)\d{3}-\d{4}(?!\d)")

def default_pii_rules() -> list[PIIRule]:
    """Return the default secret/PII rule set.

    Order matters for reporting clarity: more specific rules first so a
    ``discord_token`` hit is not double-reported as ``generic_secret``.
    """
    return [
        PIIRule("provider_token", _provider_token_re(), "cloud/provider access token"),
        PIIRule("discord_token", _discord_token_re(), "Discord bot/user token"),
        PIIRule("jwt", _jwt_re(), "JSON Web Token"),
        PIIRule("bearer_token", _bearer_re(), "HTTP Bearer token"),
        PIIRule(
            "generic_secret",
            _generic_secret_re(),
            "credential assignment in prose/config",
        ),
        PIIRule(
            "long_hex_blob",
            _long_hex_re(),
            "long opaque blob resembling an API key or hash",
        ),
        PIIRule("credit_card", _credit_card_re(), "possible credit card number"),
        PIIRule("email", _email_re(), "email address"),
        PIIRule("jp_phone", _jp_phone_re(), "Japanese phone number"),
        PIIRule("jp_postal", _jp_postal_re(), "Japanese postal code"),
    ]


def _content_paths(fmt: DatasetFormat) -> tuple[str, ...]:
    if fmt == "sft":
        return ("messages[*].content",)
    return ("prompt", "chosen", "rejected")


def _iter_record_strings(
    row: dict[str, Any], fmt: DatasetFormat
) -> Iterable[tuple[str, str]]:
    """Yield (field_path, value) for every string embedded in ``row``.

    SFT scans each message ``content`` with a path shaped like
    ``messages[3].content``.  DPO scans ``prompt``/``chosen``/``rejected``.
    Metadata, when present, is iterated too so locally retained audit copies
    are verified before export; metadata values are addressed via
    ``metadata.<key>`` paths.
    """
    if fmt == "sft":
        messages = row.get("messages")
        if isinstance(messages, list):
            for i, msg in enumerate(messages):
                if isinstance(msg, dict):
                    content = msg.get("content")
                    if isinstance(content, str):
                        yield f"messages[{i}].content", content
    else:
        for key in ("prompt", "chosen", "rejected"):
            value = row.get(key)
            if isinstance(value, str):
                yield key, value

    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        for key, value in metadata.items():
            if isinstance(value, str):
                yield f"metadata.{key}", value
            elif isinstance(value, (int, float)):
                yield f"metadata.{key}", str(value)


def scan_pii(
    rows: Iterable[dict[str, Any]],
    fmt: DatasetFormat,
    *,
    rules: Sequence[PIIRule] | None = None,
) -> PIIReport:
    """Scan records for secrets/PII.  Never echoes matched values.

    Each record is tested against every rule.  A single (row, field, rule)
    triple is reported at most once even when the rule matches the same field
    repeatedly, to keep reports compact and to avoid leaking information via
    match counts.
    """
    active_rules = list(rules) if rules is not None else default_pii_rules()
    issues: list[PIIIssue] = []
    seen: set[tuple[int, str, str]] = set()
    for index, row in enumerate(rows):
        for field_path, text in _iter_record_strings(row, fmt):
            for rule in active_rules:
                key = (index, field_path, rule.name)
                if key in seen:
                    continue
                if rule.pattern.search(text):
                    seen.add(key)
                    issues.append(
                        PIIIssue(
                            row_index=index,
                            field=field_path,
                            rule=rule.name,
                            reason=rule.description or f"matched {rule.name}",
                        )
                    )
    return PIIReport(format=fmt, ok=not issues, issues=issues)



def _read_preflight_rows(path: Path | str) -> list[dict[str, Any]]:
    """Read every nonblank line; malformed/non-object records become invalid rows."""
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                item = {}
            rows.append(item if isinstance(item, dict) else {})
    return rows

def run_preflight(
    path: Path | str,
    fmt: DatasetFormat,
    *,
    max_field_chars: int = 8192,
    max_messages: int = 32,
    pii_rules: Sequence[PIIRule] | None = None,
) -> PreflightReport:
    """Run the full preflight pipeline against ``path`` (a JSONL file).

    The file is read once into memory so all four checks see identical data.
    For very large datasets this should be invoked on a dedicated slice rather
    than the full history.
    """
    rows = _read_preflight_rows(path)
    schema = validate_schema(rows, fmt)
    duplicates = detect_duplicates(rows, fmt)
    length = detect_length_issues(
        rows, fmt, max_field_chars=max_field_chars, max_messages=max_messages
    )
    pii = scan_pii(rows, fmt, rules=pii_rules)
    return PreflightReport(
        format=fmt,
        path=str(path),
        total=len(rows),
        schema=schema,
        duplicates=duplicates,
        length=length,
        pii=pii,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="JSONL dataset to inspect.")
    parser.add_argument(
        "--format", choices=("sft", "dpo"), required=True
    )
    parser.add_argument("--max-field-chars", type=int, default=8192)
    parser.add_argument("--max-messages", type=int, default=32)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the report as JSON on stdout instead of a human summary.",
    )
    parser.add_argument(
        "--clean-output",
        help="Write only safe, valid, unique rows to this distinct JSONL path; metadata is removed.",
    )
    return parser


def _human_summary(report: PreflightReport) -> str:
    lines = [
        f"preflight {report.path} [{report.format}] total={report.total} ok={report.ok}",
        f"  schema     ok={report.schema.ok} issues={len(report.schema.issues)}",
        f"  duplicates ok={report.duplicates.ok} count={len(report.duplicates.duplicates)}",
        f"  length     ok={report.length.ok} issues={len(report.length.issues)}",
        f"  pii        ok={report.pii.ok} issues={len(report.pii.issues)}",
    ]
    for issue in report.pii.issues:
        # Deliberately never include the matched value.
        lines.append(
            f"  pii row={issue.row_index} field={issue.field} rule={issue.reason}"
        )
    return "\n".join(lines)



def safe_rows(path: Path | str, report: PreflightReport) -> list[dict[str, Any]]:
    """Return rows safe to transfer; unsafe/duplicate rows and metadata are dropped."""
    rows = _read_preflight_rows(path)
    bad = {i.row_index for i in report.schema.issues}
    bad.update(i.row_index for i in report.duplicates.duplicates)
    bad.update(i.row_index for i in report.length.issues)
    bad.update(i.row_index for i in report.pii.issues)
    cleaned: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if index in bad:
            continue
        item = dict(row)
        item.pop("metadata", None)
        cleaned.append(item)
    return cleaned

def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_preflight(
        args.input,
        args.format,  # type: ignore[arg-type]
        max_field_chars=args.max_field_chars,
        max_messages=args.max_messages,
    )
    if args.clean_output:
        source = Path(args.input).resolve()
        destination = Path(args.clean_output).resolve()
        if source == destination:
            print("clean output must differ from input", file=sys.stderr)
            return 2
        from .dataset import write_jsonl
        count = write_jsonl(destination, safe_rows(args.input, report))
        print(f"clean rows written: {count} -> {destination}", file=sys.stderr)
    if args.json:
        print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
    else:
        print(_human_summary(report))
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())