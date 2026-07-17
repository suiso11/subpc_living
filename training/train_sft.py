#!/usr/bin/env python3
"""BF16 PEFT LoRA SFT launcher for the personal-model training pipeline.

This is the heavy entry point that turns a validated YAML config (see
``training/configs/*.yaml``) into an actual TRL SFT run on a single H200 BF16
accelerator.  It is intentionally import-safe: nothing in the standard-library
``argparse`` / ``runtime`` path touches ``torch`` / ``transformers`` / ``peft``
/ ``trl`` / ``datasets``, so ``--dry-run`` and repository unit tests run in the
plain assistant environment without any model download or GPU.

Invariants enforced here and in ``training.runtime``:

- BF16 LoRA only.  ``precision`` must be ``"bf16"``; quantized (4bit/8bit) and
  FP16/FP32 full fine-tuning are rejected before any heavy import so the
  pipeline never silently degrades to QLoRA or full-FT.
- The resolved target-module list is always derived by inspecting the loaded
  base model's ``nn.Linear`` submodule names and filtering through
  ``runtime.match_target_modules`` with the configured (default: conservative)
  mode.  Experts / router / vision projector / embed / lm_head / norm are
  always excluded.
- A resolved-settings manifest JSON is written to the output directory BEFORE
  the trainer is constructed, so the exact run can be audited and reproduced
  even if training crashes.
- Dataset and output paths are validated up front so we never start a multi-hour
  run only to abort on a missing file or a clobbered run dir.

The ``--dry-run`` flag exercises the full configuration + path validation +
manifest write path but stops before importing any heavy library, loading any
model, or downloading anything.  It is the default path used by repository
tests.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .runtime import (
    ConfigError,
    TrainingConfig,
    build_manifest,
    load_config,
    match_target_modules,
    validate_runtime_config,
    write_manifest,
)

__all__ = ["build_parser", "main", "dry_run", "discover_linear_modules"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "config",
        help="Path to the training YAML config (e.g. training/configs/persona_conservative.yaml).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate config and paths and write the resolved-settings manifest, "
            "but do not import torch/transformers/peft/trl, load any model, or "
            "start training."
        ),
    )
    parser.add_argument(
        "--revision",
        help="Override the pinned model_revision from the config (explicit only).",
    )
    parser.add_argument(
        "--dataset",
        help="Override dataset_path from the config (must point at a messages JSONL).",
    )
    parser.add_argument(
        "--output-dir",
        help="Override output_dir from the config (manifest + adapter checkpoints).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Override the number of training epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Override the learning rate.",
    )
    parser.add_argument(
        "--manifest-name",
        default="manifest.json",
        help="Filename of the resolved-settings manifest inside --output-dir.",
    )
    return parser


def _apply_overrides(cfg: TrainingConfig, args: argparse.Namespace) -> TrainingConfig:
    """Apply the small, validated set of CLI overrides on top of the config."""
    if args.revision:
        cfg.model_revision = args.revision
    if args.dataset is not None:
        cfg.dataset_path = args.dataset or None
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.learning_rate is not None:
        cfg.learning_rate = args.learning_rate
    return cfg


def _validate_paths(cfg: TrainingConfig) -> None:
    """Validate dataset / output paths before any heavy work.

    Raises ``ConfigError`` (so callers get a uniform error message) if the
    dataset is missing or the output directory would clobber an existing
    non-empty run dir without being explicitly intended.
    """
    if not cfg.dataset_path:
        raise ConfigError("dataset_path is required to launch SFT training")
    dataset = Path(cfg.dataset_path)
    if not dataset.exists():
        raise ConfigError(f"dataset_path does not exist: {dataset}")
    if dataset.is_dir():
        raise ConfigError(f"dataset_path must be a JSONL file, not a directory: {dataset}")
    if dataset.stat().st_size == 0:
        raise ConfigError(f"dataset_path is empty: {dataset}")

    out = Path(cfg.output_dir)
    if out.exists():
        if out.is_file():
            raise ConfigError(f"output_dir is a file, not a directory: {out}")
        # Refuse to silently clobber a previous run that already wrote an adapter.
        for marker in ("adapter_config.json", "adapter_model.safetensors"):
            if (out / marker).exists():
                raise ConfigError(
                    f"output_dir already contains a finished adapter ({marker}); "
                    f"refusing to clobber {out}.  Move it aside or pass a fresh --output-dir."
                )


def _load_jsonl_messages(path: Path | str) -> list[dict[str, Any]]:
    """Read a messages JSONL file and return rows with a ``messages`` list.

    Pure-Python only; ``datasets`` is imported lazily inside :func:`_run`.
    """
    rows: list[dict[str, Any]] = []
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ConfigError(f"invalid JSON on line {index + 1} of {p}: {exc}") from exc
            if not isinstance(row, dict):
                raise ConfigError(f"line {index + 1} of {p} is not a JSON object")
            messages = row.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ConfigError(
                    f"line {index + 1} of {p}: missing or empty 'messages' list"
                )
            rows.append({"messages": messages})
    if not rows:
        raise ConfigError(f"dataset contains no records: {p}")
    return rows


def discover_linear_modules(model: Any) -> list[str]:
    """Return the unique, sorted names of all ``nn.Linear`` submodules of ``model``.

    Used to derive LoRA target modules via :func:`match_target_modules`.  The
    full dotted attribute path (``model.layers.0.self_attn.q_proj``) is kept so
    the conservative matcher can apply keep/exclude substring rules.
    """
    import torch.nn as nn

    seen: set[str] = set()
    names: list[str] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if name and name not in seen:
                seen.add(name)
                names.append(name)
    return sorted(names)


def dry_run(cfg: TrainingConfig, manifest_path: Path) -> Path:
    """Validate everything that can be checked without heavy imports.

    Writes a manifest with an empty ``resolved_target_modules`` list (the real
    list is filled in after model inspection in the live path) and returns the
    manifest path.  Never imports ``torch`` / ``transformers`` / ``peft`` /
    ``trl`` / ``datasets``, never touches the network or GPU.
    """
    _validate_paths(cfg)
    manifest = build_manifest(cfg, resolved_target_modules=[])
    write_manifest(manifest, manifest_path)
    return manifest_path


def _run(cfg: TrainingConfig, manifest_path: Path) -> None:
    """Live SFT path: import heavy libs, inspect model, write manifest, train."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from trl import SFTConfig, SFTTrainer

    # Guardrails: the runtime already rejected non-BF16 / rank<=0, but keep an
    # explicit, understandable failure here so a future refactor cannot slip a
    # quantized or full-FT path past the heavy-import boundary.
    if cfg.precision != "bf16":
        raise ConfigError(
            f"train_sft only supports precision='bf16'; got {cfg.precision!r}"
        )
    if cfg.lora.rank <= 0:
        raise ConfigError("train_sft only supports LoRA (rank>0); full fine-tuning is not supported")

    # No quantization, ever.  Load the base model in BF16 on CUDA.
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; BF16 LoRA SFT requires a CUDA device")

    processor = AutoProcessor.from_pretrained(
        cfg.model_name, revision=cfg.model_revision, trust_remote_code=True,
    )
    tokenizer = getattr(processor, "tokenizer", processor)
    if cfg.chat_template_path:
        template_path = Path(cfg.chat_template_path)
        if not template_path.is_file():
            raise ConfigError(f"chat_template_path does not exist: {template_path}")
        template = template_path.read_text(encoding="utf-8")
        tokenizer.chat_template = template
        processor.chat_template = template
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForImageTextToText.from_pretrained(
        cfg.model_name,
        revision=cfg.model_revision,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # Inspect Linear submodule names and resolve LoRA targets through the
    # conservative matcher so experts / router / vision / embed / lm_head /
    # norm are never targeted.
    candidates = discover_linear_modules(model)
    resolved_targets = match_target_modules(
        candidates,
        mode=cfg.lora.target_mode,
        explicit_targets=cfg.lora.target_modules or None,
    )
    if not resolved_targets:
        raise ConfigError(
            f"no LoRA target modules matched from {len(candidates)} Linear candidates; "
            f"check target_mode/config or use lora.target_mode='explicit'"
        )
    if cfg.lora.target_mode == "explicit":
        # For explicit mode, match_target_modules re-validates exclusions and
        # returns the verbatim list; surface it as the resolved list directly.
        resolved_targets = list(cfg.lora.target_modules)

    # Write the manifest BEFORE constructing the trainer so an audit record
    # exists even if trainer setup crashes.
    manifest = build_manifest(cfg, resolved_target_modules=resolved_targets)
    write_manifest(manifest, manifest_path)

    # Build the chat-format dataset from the messages JSONL.
    rows = _load_jsonl_messages(cfg.dataset_path)  # type: ignore[arg-type]
    dataset = Dataset.from_list(rows)

    lora_config = LoraConfig(
        r=cfg.lora.rank,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=resolved_targets,
        modules_to_save=cfg.lora.modules_to_save or None,
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = SFTConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.micro_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=20,
        save_steps=cfg.save_steps,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=cfg.gradient_checkpointing,
        max_length=cfg.max_sequence_length,
        assistant_only_loss=cfg.assistant_only_loss,
        seed=cfg.seed,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=processor,
    )
    trainer.train()
    trainer.save_model(cfg.output_dir)
    processor.save_pretrained(cfg.output_dir)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"config file not found: {exc}", file=sys.stderr)
        return 2

    cfg = _apply_overrides(cfg, args)

    try:
        validate_runtime_config(cfg, dataset_required=True)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    out_dir = Path(cfg.output_dir)
    manifest_path = out_dir / args.manifest_name

    if args.dry_run:
        try:
            written = dry_run(cfg, manifest_path)
        except ConfigError as exc:
            print(f"dry-run validation failed: {exc}", file=sys.stderr)
            return 2
        except OSError as exc:
            print(f"dry-run manifest write failed: {exc}", file=sys.stderr)
            return 3
        print(f"dry-run ok: manifest -> {written}")
        return 0

    try:
        dry_run(cfg, manifest_path)
    except ConfigError as exc:
        print(f"pre-flight validation failed: {exc}", file=sys.stderr)
        return 2

    try:
        _run(cfg, manifest_path)
    except (ConfigError, RuntimeError) as exc:
        print(f"training aborted: {exc}", file=sys.stderr)
        return 3
    print(f"training complete: adapter -> {cfg.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())