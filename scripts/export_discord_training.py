#!/usr/bin/env python3
"""Export curated Discord training logs for SFT or preference tuning.

Default behavior is intentionally conservative:
- preference export uses only explicit correction candidates;
- SFT export uses only explicit correction candidates unless
  --include-positive-feedback is set.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
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


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            json.dump(row, f, ensure_ascii=False, separators=(",", ":"))
            f.write("\n")
            count += 1
    return count


def feedback_scores(feedback_rows: list[dict[str, Any]]) -> dict[str, int]:
    scores: dict[str, int] = {}
    for row in feedback_rows:
        turn_id = row.get("turn_id")
        if not isinstance(turn_id, str):
            continue
        try:
            value = int(row.get("value", 0))
        except (TypeError, ValueError):
            continue
        scores[turn_id] = scores.get(turn_id, 0) + value
    return scores


def turn_index(turns: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for turn in turns:
        turn_id = turn.get("turn_id")
        if isinstance(turn_id, str):
            result[turn_id] = turn
    return result


def metadata_for(row: dict[str, Any], turn: dict[str, Any] | None = None) -> dict[str, Any]:
    turn = turn or {}
    return {
        "turn_id": row.get("turn_id") or turn.get("turn_id"),
        "created_at": row.get("created_at") or turn.get("created_at"),
        "source": row.get("turn_source") or turn.get("source") or row.get("source"),
        "profile": row.get("profile") or turn.get("profile") or "default",
        "channel_id": row.get("channel_id") or turn.get("channel_id"),
        "model": row.get("model") or turn.get("model"),
        "num_ctx": row.get("num_ctx") or turn.get("num_ctx"),
        "num_predict": row.get("num_predict") or turn.get("num_predict"),
        "temperature": row.get("temperature") or turn.get("temperature"),
    }


def metadata_matches(
    metadata: dict[str, Any],
    *,
    profile: str | None,
    source: str | None,
    channel_id: int | None,
) -> bool:
    if profile is not None and metadata.get("profile") != profile:
        return False
    if source is not None and metadata.get("source") != source:
        return False
    if channel_id is not None and metadata.get("channel_id") != channel_id:
        return False
    return True


def sft_record(prompt: str, response: str, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
        "metadata": metadata,
    }


def preference_record(
    prompt: str,
    chosen: str,
    rejected: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "metadata": metadata,
    }


def export_preference(
    *,
    candidates: list[dict[str, Any]],
    turns_by_id: dict[str, dict[str, Any]],
    profile: str | None,
    source: str | None,
    channel_id: int | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        turn = turns_by_id.get(str(candidate.get("turn_id", "")))
        metadata = metadata_for(candidate, turn)
        if not metadata_matches(metadata, profile=profile, source=source, channel_id=channel_id):
            continue
        prompt = str(candidate.get("input") or (turn or {}).get("user") or "").strip()
        chosen = str(candidate.get("preferred_output") or "").strip()
        rejected = str(candidate.get("rejected_output") or (turn or {}).get("assistant") or "").strip()
        if not prompt or not chosen or not rejected:
            continue
        rows.append(preference_record(prompt, chosen, rejected, metadata))
    return rows


def export_sft(
    *,
    candidates: list[dict[str, Any]],
    turns: list[dict[str, Any]],
    turns_by_id: dict[str, dict[str, Any]],
    scores: dict[str, int],
    profile: str | None,
    source: str | None,
    channel_id: int | None,
    include_positive_feedback: bool,
    min_score: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for candidate in candidates:
        turn = turns_by_id.get(str(candidate.get("turn_id", "")))
        metadata = metadata_for(candidate, turn)
        if not metadata_matches(metadata, profile=profile, source=source, channel_id=channel_id):
            continue
        prompt = str(candidate.get("input") or (turn or {}).get("user") or "").strip()
        response = str(candidate.get("preferred_output") or "").strip()
        key = (prompt, response)
        if prompt and response and key not in seen:
            rows.append(sft_record(prompt, response, metadata))
            seen.add(key)

    if not include_positive_feedback:
        return rows

    for turn in turns:
        turn_id = turn.get("turn_id")
        if not isinstance(turn_id, str) or scores.get(turn_id, 0) < min_score:
            continue
        metadata = metadata_for(turn)
        if not metadata_matches(metadata, profile=profile, source=source, channel_id=channel_id):
            continue
        prompt = str(turn.get("user") or "").strip()
        response = str(turn.get("assistant") or "").strip()
        key = (prompt, response)
        if prompt and response and key not in seen:
            rows.append(sft_record(prompt, response, metadata))
            seen.add(key)

    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", default="data/discord_training")
    parser.add_argument("--output", required=True)
    parser.add_argument("--format", choices=("preference", "sft"), default="preference")
    parser.add_argument("--profile", help="Filter by logged profile name, e.g. voice_short.")
    parser.add_argument("--source", help="Filter by source, e.g. discord_voice_transcript.")
    parser.add_argument("--channel-id", type=int, help="Filter by Discord text channel ID.")
    parser.add_argument(
        "--include-positive-feedback",
        action="store_true",
        help="For SFT only: include turns with net positive reaction feedback.",
    )
    parser.add_argument("--min-score", type=int, default=1)
    parser.add_argument("--max-rows", type=int, default=0, help="0 means no limit.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    training_dir = Path(args.training_dir)
    turns = read_jsonl(training_dir / "conversations.jsonl")
    candidates = read_jsonl(training_dir / "training_candidates.jsonl")
    feedback = read_jsonl(training_dir / "feedback.jsonl")
    turns_by_id = turn_index(turns)

    if args.format == "preference":
        rows = export_preference(
            candidates=candidates,
            turns_by_id=turns_by_id,
            profile=args.profile,
            source=args.source,
            channel_id=args.channel_id,
        )
    else:
        rows = export_sft(
            candidates=candidates,
            turns=turns,
            turns_by_id=turns_by_id,
            scores=feedback_scores(feedback),
            profile=args.profile,
            source=args.source,
            channel_id=args.channel_id,
            include_positive_feedback=args.include_positive_feedback,
            min_score=args.min_score,
        )

    if args.max_rows > 0:
        rows = rows[: args.max_rows]

    count = write_jsonl(Path(args.output), rows)
    print(f"exported {count} rows to {args.output}")


if __name__ == "__main__":
    main()
