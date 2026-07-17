#!/usr/bin/env python3
"""DPO continuation from an SFT adapter (BF16 LoRA, never quantized, never
full fine-tune).

This CLI is the DPO half of the Phase 13 personal LoRA training pipeline.  It
*MUST* continue from a previously trained SFT LoRA adapter and never quantize
or fully fine-tune the base model.  The runtime contract is enforced through
:mod:`training.runtime`:

- ``stage`` is forced to ``"dpo"`` and ``base_adapter_path`` is required.
- ``precision`` must be ``"bf16"``; 4bit/8bit and FP16/FP32 full-FT are
  rejected by :func:`training.runtime.validate_config` before any heavy import.
- ``lora.rank`` must be positive so the pipeline never degrades to full-FT.

Import safety: this module imports nothing heavy at import time.  All of
``torch`` / ``transformers`` / ``peft`` / ``trl`` are imported lazily inside
:func:`run_training`, so ``--dry-run`` and unit tests run on hosts without the
training stack installed and without any model download.

The resolved settings manifest is written *before* any heavy work, so a crashed
or aborted run can be audited and reproduced from the manifest alone.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from .runtime import (
    ConfigError,
    TrainingConfig,
    build_manifest,
    load_config,
    match_target_modules,
    resolve_for_run,
    validate_config,
    validate_runtime_config,
    write_manifest,
)

__all__ = [
    "build_parser",
    "check_paths",
    "read_adapter_target_modules",
    "run_dry_run",
    "run_training",
    "main",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a DPO training config YAML (stage must be 'dpo').",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and paths, write the manifest, but do not load any "
        "model or run training.  No network access or downloads occur.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Override dataset_path (preference JSONL: prompt/chosen/rejected).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output_dir for the DPO adapter checkpoints and manifest.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Override the pinned base-model revision (commit hash / tag).",
    )
    parser.add_argument(
        "--epochs",
        type=float,
        default=None,
        help="Override the number of DPO epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override the DPO learning rate.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Override the manifest output path "
        "(default: <output_dir>/manifest.json).",
    )
    return parser


def read_adapter_target_modules(adapter_path: Path | str) -> list[str]:
    """Read the LoRA ``target_modules`` from an SFT adapter on disk.

    This reads ``adapter_config.json`` directly so it works in dry-run mode
    without importing ``peft`` or ``torch``.  The returned list is used to
    populate the audit manifest's ``resolved_target_modules`` and to verify
    that the DPO run will continue the *same* modules the SFT adapter trained.
    """
    p = Path(adapter_path) / "adapter_config.json"
    if not p.exists():
        raise ConfigError(
            f"base_adapter_path does not contain adapter_config.json: {p}"
        )
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"invalid adapter_config.json at {p}: {exc}") from exc
    peft = data.get("peft_type") or data.get("peft_type_")
    if peft is not None and str(peft).upper() not in ("LORA", "LORA.CONFIG"):
        # Reject non-LoRA adapters: we never quantize and never full-FT, so a
        # IA3 / prefix-tuning / AdaLoRA adapter is out of scope here.
        raise ConfigError(
            f"base adapter is not a LoRA adapter (peft_type={peft!r}); "
            "DPO continuation only supports LoRA adapters"
        )
    raw = data.get("target_modules") or []
    if not isinstance(raw, list) or not raw:
        raise ConfigError(
            f"adapter_config.json at {p} has no non-empty 'target_modules' list"
        )
    return [str(m) for m in raw]


def check_paths(cfg: TrainingConfig, *, dry_run: bool) -> None:
    """Explicit filesystem checks before any heavy work is attempted.

    These checks never touch the network and never import torch/transformers,
    so they run identically in dry-run and real-run modes.  Failure exits with
    a clear error rather than deep inside the training stack.
    """
    if cfg.stage != "dpo":
        raise ConfigError(f"train_dpo requires stage='dpo', got {cfg.stage!r}")
    if not cfg.base_adapter_path:
        raise ConfigError(
            "train_dpo requires base_adapter_path; DPO must continue from an "
            "SFT LoRA adapter (full-model DPO is not supported)"
        )
    adapter = Path(cfg.base_adapter_path)
    if not adapter.exists() or not adapter.is_dir():
        raise ConfigError(
            f"base_adapter_path not found or not a directory: {adapter}"
        )
    # Validate that it is actually a LoRA adapter on disk (no heavy imports).
    read_adapter_target_modules(adapter)

    if not cfg.dataset_path:
        raise ConfigError("dataset_path is required for DPO training")
    dataset = Path(cfg.dataset_path)
    if not dataset.exists() or not dataset.is_file():
        raise ConfigError(
            f"dataset_path not found or not a file: {dataset}"
        )

    out = Path(cfg.output_dir)
    if out.exists():
        if out.resolve() == adapter.resolve():
            raise ConfigError(
                "output_dir must not equal base_adapter_path; writing DPO "
                "checkpoints into the SFT adapter directory would overwrite it"
            )
        if not (out.is_dir() or out.suffix == ""):
            raise ConfigError(f"output_dir is not a directory: {out}")

    if dry_run:
        # Confirm the preference JSONL is non-empty so a dry-run that passes
        # validation still surfaces an empty-dataset problem without training.
        with dataset.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    return
        raise ConfigError(f"dataset_path is empty (no records): {dataset}")


def _build_overrides(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "dataset_path": args.dataset,
        "output_dir": args.output_dir,
        "revision": args.revision,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
    }


def _manifest_path(cfg: TrainingConfig, override: str | None) -> Path:
    if override:
        return Path(override)
    return Path(cfg.output_dir) / "manifest.json"


def run_dry_run(cfg: TrainingConfig, manifest_override: str | None) -> int:
    """Validate everything possible without importing the training stack.

    Writes the resolved settings manifest (with the adapter's target modules)
    and returns 0 on success.  No downloads, no model loads, no GPU use.
    """
    check_paths(cfg, dry_run=True)
    resolved_targets = read_adapter_target_modules(cfg.base_adapter_path)  # type: ignore[arg-type]
    manifest = build_manifest(cfg, resolved_target_modules=resolved_targets)
    out = _manifest_path(cfg, manifest_override)
    written = write_manifest(manifest, out)
    print(f"[dpo dry-run] config ok, manifest written: {written}")
    print(
        f"[dpo dry-run] base_adapter={cfg.base_adapter_path} "
        f"dataset={cfg.dataset_path} output_dir={cfg.output_dir}"
    )
    print(
        f"[dpo dry-run] resolved_target_modules={len(resolved_targets)} "
        f"(no training performed; no downloads occurred)"
    )
    return 0


def _parse_preference_rows(path: Path | str) -> list[dict[str, Any]]:
    """Parse a preference JSONL file into a list of dicts.

    Each row must have non-empty ``prompt`` / ``chosen`` / ``rejected``
    strings, matching :mod:`training.dataset`'s DPO schema.  Kept inline here
    so this CLI stays a single file with no extra heavy imports on the read
    path.
    """
    rows: list[dict[str, Any]] = []
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ConfigError(f"{p}:{lineno}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ConfigError(f"{p}:{lineno}: row is not an object")
            for field in ("prompt", "chosen", "rejected"):
                value = row.get(field)
                if not isinstance(value, str) or not value:
                    raise ConfigError(
                        f"{p}:{lineno}: field {field!r} must be a non-empty string"
                    )
            rows.append(row)
    if not rows:
        raise ConfigError(f"{p}: preference dataset is empty (no records)")
    return rows


def _resolve_target_modules_from_model(model: Any) -> list[str]:
    """Read the active LoRA target modules from a loaded PEFT model.

    Falls back to the adapter_config.json-derived list via
    :func:`read_adapter_target_modules` if the PEFT config cannot be inspected.
    """
    peft_config = getattr(model, "peft_config", None)
    if peft_config:
        for cfg in peft_config.values():
            tms = getattr(cfg, "target_modules", None)
            if isinstance(tms, (list, tuple)) and tms:
                return [str(m) for m in tms]
            tms = getattr(cfg, "target_modules", None)
            if isinstance(tms, dict) and tms:
                return [str(m) for m in tms.keys()]
    return []


def run_training(cfg: TrainingConfig, manifest_override: str | None) -> int:
    """Run the actual DPO continuation.  Heavy imports happen here only."""
    check_paths(cfg, dry_run=False)

    # Re-validate against the pure-Python runtime one more time, in case an
    # override introduced an inconsistency that the path check does not cover.
    validate_runtime_config(cfg, dataset_required=True)

    rows = _parse_preference_rows(cfg.dataset_path)  # type: ignore[arg-type]

    # Lazy heavy imports: torch / transformers / peft / trl / datasets.
    import torch  # type: ignore[import]
    from datasets import Dataset  # type: ignore[import]
    from peft import PeftModel  # type: ignore[import]
    from transformers import AutoModelForImageTextToText, AutoProcessor  # type: ignore[import]
    from trl import DPOConfig, DPOTrainer  # type: ignore[import]

    if not torch.cuda.is_available():
        raise RuntimeError(
            "DPO continuation requires a CUDA GPU; this host has no CUDA device"
        )
    if cfg.precision != "bf16":
        # Already rejected by validate_config, but reassert right before the
        # model load to guarantee the dtype below is correct.
        raise ConfigError(
            f"precision must be 'bf16' for DPO, got {cfg.precision!r}"
        )

    dtype = torch.bfloat16
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
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base = AutoModelForImageTextToText.from_pretrained(
        cfg.model_name,
        revision=cfg.model_revision,
        dtype=dtype,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    base.config.use_cache = False

    # Continue from the SFT adapter: load it on top of the frozen base and mark
    # it trainable.  We never quantize, never replace the base, and never
    # full-fine-tune.  The adapter's existing LoRA modules are the ones trained.
    model = PeftModel.from_pretrained(
        base, cfg.base_adapter_path, is_trainable=True,  # type: ignore[arg-type]
    )
    if not getattr(model, "peft_config", None):
        raise ConfigError(
            "loaded model has no active PEFT/LoRA config; refusing to run "
            "DPO against a non-LoRA model (full fine-tuning is not supported)"
        )

    resolved_targets = _resolve_target_modules_from_model(model)
    if not resolved_targets:
        # Fall back to the on-disk adapter config so the manifest is never empty.
        resolved_targets = read_adapter_target_modules(cfg.base_adapter_path)  # type: ignore[arg-type]
    # Sanity check: the resolved LoRA modules must not include any of the
    # substrings the runtime explicitly forbids (expert/router/vision/...).
    # match_target_modules with mode="explicit" re-applies those exclusions.
    verified = match_target_modules(resolved_targets, "explicit", explicit_targets=resolved_targets)
    if set(verified) != set(resolved_targets):
        raise ConfigError(
            "adapter target_modules include forbidden submodules "
            f"(expert/router/vision/projector/embed/lm_head/norm): "
            f"{sorted(set(resolved_targets) - set(verified))!r}"
        )

    # Write the manifest BEFORE training begins so a crash is still auditable.
    manifest = build_manifest(cfg, resolved_target_modules=resolved_targets)
    written = write_manifest(manifest, _manifest_path(cfg, manifest_override))
    print(f"[dpo] manifest written before training: {written}")

    try:
        from peft import LoraConfig  # type: ignore[import]
    except Exception:  # pragma: no cover - defensive, peft already imported above
        LoraConfig = None  # type: ignore[assignment]

    # DPO continuation trains the already-loaded SFT adapter.  We do NOT
    # attach a fresh LoRA over a fresh LoRA; passing the existing peft model
    # directly to DPOTrainer continues training its LoRA parameters with the
    # DPO loss.  The ref model is None: TRL derives reference log-probs by
    # disabling the adapter on the same frozen base, which is the standard
    # PEFT-DPO path and avoids loading a second full copy of the model.
    peft_config = None  # no new adapter; continue the existing one.

    dpo_config = DPOConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.micro_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=max(1, cfg.save_steps // 5),
        save_steps=cfg.save_steps,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False}
        if cfg.gradient_checkpointing
        else None,
        seed=cfg.seed,
        report_to="none",
        remove_unused_columns=False,
        max_length=cfg.max_sequence_length,
        max_prompt_length=cfg.max_sequence_length // 2,
        beta=0.1,
        loss_type="sigmoid",
    )

    train_ds = Dataset.from_list(rows)

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=train_ds,
        processing_class=processor,
        peft_config=peft_config,
    )

    train_result = trainer.train()
    trainer.save_model(cfg.output_dir)

    metrics = {str(k): float(v) for k, v in train_result.metrics.items()}
    metrics_path = Path(cfg.output_dir) / "dpo_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(
            {"manifest": asdict(manifest), "train_metrics": metrics}, f,
            ensure_ascii=False, indent=2, sort_keys=True,
        )
        f.write("\n")
    print(f"[dpo] training complete, metrics: {metrics_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    try:
        cfg, _ = resolve_for_run(cfg, _build_overrides(args))
        validate_config(cfg)
        # validate_runtime_config enforces DPO-stage invariants (base_adapter
        # required, SFT must not carry an adapter, dataset required for launch).
        validate_runtime_config(cfg, dataset_required=not args.dry_run)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    try:
        if args.dry_run:
            return run_dry_run(cfg, args.manifest)
        return run_training(cfg, args.manifest)
    except ConfigError as exc:
        print(f"dpo error: {exc}", file=sys.stderr)
        return 2
    except (RuntimeError, OSError) as exc:
        print(f"dpo runtime error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())