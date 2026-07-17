"""JSONL dataset schema validation, duplicate detection and basic length checks.

This module is import-safe: it only depends on the Python standard library so
repository tests can run without optional training dependencies.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal


DatasetFormat = Literal["sft", "dpo"]
_VALID_ROLES = {"system", "user", "assistant"}


class InvalidRecordError(ValueError):
    """Raised when a single record cannot be validated/imported."""


@dataclass(frozen=True)
class SchemaIssue:
    row_index: int
    field: str
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {"row_index": self.row_index, "field": self.field, "reason": self.reason}


@dataclass
class SchemaReport:
    format: DatasetFormat
    ok: bool
    total: int
    issues: list[SchemaIssue] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "ok": self.ok,
            "total": self.total,
            "issues": [issue.as_dict() for issue in self.issues],
        }


@dataclass(frozen=True)
class DuplicateKey:
    row_index: int
    key: str

    def as_dict(self) -> dict[str, Any]:
        return {"row_index": self.row_index, "key": self.key}


@dataclass
class DuplicateReport:
    format: DatasetFormat
    duplicates: list[DuplicateKey] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.duplicates

    def as_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "ok": self.ok,
            "duplicates": [d.as_dict() for d in self.duplicates],
        }


@dataclass(frozen=True)
class LengthIssue:
    row_index: int
    field: str
    char_len: int
    limit: int
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "row_index": self.row_index,
            "field": self.field,
            "char_len": self.char_len,
            "limit": self.limit,
            "reason": self.reason,
        }


@dataclass
class LengthReport:
    format: DatasetFormat
    ok: bool
    issues: list[LengthIssue] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "ok": self.ok,
            "issues": [issue.as_dict() for issue in self.issues],
        }


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dicts.

    Malformed lines and non-object payloads are skipped, mirroring the
    conservative behavior expected of import tooling.
    """
    p = Path(path)
    if not p.exists():
        return []
    rows: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                rows.append(item)
    return rows


def iter_jsonl(path: Path | str) -> Iterator[dict[str, Any]]:
    """Yield validated rows one at a time for memory-efficient streaming."""
    p = Path(path)
    if not p.exists():
        return
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                yield item


def write_jsonl(path: Path | str, rows: Iterable[dict[str, Any]]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            json.dump(row, f, ensure_ascii=False, separators=(",", ":"))
            f.write("\n")
            count += 1
    return count


def _ai_assert(condition: bool, index: int, field: str, reason: str, issues: list[SchemaIssue]) -> None:
    if not condition:
        issues.append(SchemaIssue(row_index=index, field=field, reason=reason))


def _validate_sft_row(row: dict[str, Any], index: int, issues: list[SchemaIssue]) -> None:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        _ai_assert(False, index, "messages", "must be a non-empty list", issues)
        return
    roles: list[str] = []
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            _ai_assert(False, index, f"messages[{i}]", "must be an object", issues)
            continue
        role = msg.get("role")
        content = msg.get("content")
        if not isinstance(role, str) or role not in _VALID_ROLES:
            _ai_assert(False, index, f"messages[{i}].role", "invalid role", issues)
        else:
            roles.append(role)
        if not isinstance(content, str) or not content.strip():
            _ai_assert(False, index, f"messages[{i}].content", "must be a non-empty string", issues)
    expected = (["system"] if roles and roles[0] == "system" else [])
    remaining = roles[1:] if expected else roles
    expected += ["user" if i % 2 == 0 else "assistant" for i in range(len(remaining))]
    if roles != expected or not roles or roles[-1] != "assistant":
        _ai_assert(False, index, "messages", "must be optional system then alternating user/assistant ending with assistant", issues)
    extra = set(row) - {"messages", "metadata"}
    if extra:
        _ai_assert(False, index, "root", "unexpected fields are not allowed", issues)
    if "metadata" in row and not isinstance(row["metadata"], dict):
        _ai_assert(False, index, "metadata", "must be an object if present", issues)


def _validate_dpo_row(row: dict[str, Any], index: int, issues: list[SchemaIssue]) -> None:
    for field_name in ("prompt", "chosen", "rejected"):
        value = row.get(field_name)
        if not isinstance(value, str) or not value:
            _ai_assert(False, index, field_name, "must be a non-empty string", issues)
    if row.get("chosen") == row.get("rejected") and isinstance(row.get("chosen"), str):
        _ai_assert(False, index, "chosen/rejected", "must differ", issues)
    extra = set(row) - {"prompt", "chosen", "rejected", "metadata"}
    if extra:
        _ai_assert(False, index, "root", "unexpected fields are not allowed", issues)
    if "metadata" in row and not isinstance(row["metadata"], dict):
        _ai_assert(False, index, "metadata", "must be an object if present", issues)


def validate_schema(rows: Iterable[dict[str, Any]], fmt: DatasetFormat) -> SchemaReport:
    """Validate that each row conforms to the requested training format schema.

    Schema validation never inspects ``metadata`` contents beyond its type, so
    it is safe to run on metadata-bearing records produced for local audit.
    """
    issues: list[SchemaIssue] = []
    total = 0
    for index, row in enumerate(rows):
        total += 1
        if not isinstance(row, dict):
            issues.append(SchemaIssue(row_index=index, field="root", reason="row is not an object"))
            continue
        if fmt == "sft":
            _validate_sft_row(row, index, issues)
        elif fmt == "dpo":
            _validate_dpo_row(row, index, issues)
        else:
            raise ValueError(f"unknown dataset format: {fmt!r}")
    if total == 0:
        issues.append(SchemaIssue(row_index=-1, field="root", reason="dataset is empty"))
    return SchemaReport(format=fmt, ok=not issues, total=total, issues=issues)


def _row_key(row: dict[str, Any], fmt: DatasetFormat) -> str | None:
    if fmt == "sft":
        messages = row.get("messages")
        if not isinstance(messages, list):
            return None
        parts = []
        for msg in messages:
            if isinstance(msg, dict):
                parts.append(f"{msg.get('role', '')}:{msg.get('content', '')}")
        return "\n".join(parts)
    if fmt == "dpo":
        prompt = row.get("prompt")
        if not isinstance(prompt, str):
            return None
        return f"{prompt}\x00{row.get('chosen', '')}\x00{row.get('rejected', '')}"
    return None


def detect_duplicates(rows: Iterable[dict[str, Any]], fmt: DatasetFormat) -> DuplicateReport:
    """Detect duplicate training records by normalized content key.

    Runs in a single pass; metadata is deliberately excluded from the key so a
    pair of records with identical content but different audit metadata is
    reported as a duplicate (which is the desired behavior for deduplication
    prior to training-bound export).
    """
    report = DuplicateReport(format=fmt)
    seen: set[str] = set()
    for index, row in enumerate(rows):
        key = _row_key(row, fmt)
        if key is None:
            continue
        if key in seen:
            report.duplicates.append(DuplicateKey(row_index=index, key=_hash_key(key)))
        else:
            seen.add(key)
    return report


def _hash_key(key: str) -> str:
    import hashlib

    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def detect_length_issues(
    rows: Iterable[dict[str, Any]],
    fmt: DatasetFormat,
    *,
    max_field_chars: int = 8192,
    max_messages: int = 32,
) -> LengthReport:
    """Detect records whose fields exceed character/turn-count caps.

    Character caps apply per field (per ``content`` for SFT, per
    ``prompt``/``chosen``/``rejected`` for DPO) to surface data that may
    consume excessive context before tokenizer verification is attempted.
    """
    issues: list[LengthIssue] = []
    for index, row in enumerate(rows):
        if fmt == "sft":
            messages = row.get("messages")
            if isinstance(messages, list):
                if len(messages) > max_messages:
                    issues.append(
                        LengthIssue(
                            row_index=index,
                            field="messages",
                            char_len=len(messages),
                            limit=max_messages,
                            reason="too many turns",
                        )
                    )
                for i, msg in enumerate(messages):
                    if isinstance(msg, dict):
                        content = msg.get("content")
                        if isinstance(content, str) and len(content) > max_field_chars:
                            issues.append(
                                LengthIssue(
                                    row_index=index,
                                    field=f"messages[{i}].content",
                                    char_len=len(content),
                                    limit=max_field_chars,
                                    reason="content exceeds char limit",
                                )
                            )
        elif fmt == "dpo":
            for field_name in ("prompt", "chosen", "rejected"):
                value = row.get(field_name)
                if isinstance(value, str) and len(value) > max_field_chars:
                    issues.append(
                        LengthIssue(
                            row_index=index,
                            field=field_name,
                            char_len=len(value),
                            limit=max_field_chars,
                            reason="field exceeds char limit",
                        )
                    )
        else:
            raise ValueError(f"unknown dataset format: {fmt!r}")
    return LengthReport(format=fmt, ok=not issues, issues=issues)