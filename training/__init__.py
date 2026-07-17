"""Offline personal LoRA training and dataset tooling.

Optional training dependencies are imported only by the executable training paths.
"""
from .dataset import (
    DatasetFormat, DuplicateKey, DuplicateReport, InvalidRecordError, LengthIssue,
    LengthReport, SchemaIssue, SchemaReport, detect_duplicates,
    detect_length_issues, iter_jsonl, read_jsonl, validate_schema, write_jsonl,
)

__all__ = [
    "DatasetFormat", "DuplicateKey", "DuplicateReport", "InvalidRecordError",
    "LengthIssue", "LengthReport", "SchemaIssue", "SchemaReport",
    "detect_duplicates", "detect_length_issues", "iter_jsonl", "read_jsonl",
    "validate_schema", "write_jsonl",
]
