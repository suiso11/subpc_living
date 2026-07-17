"""LoRA adapter merge tool.

Merges a trained BF16 LoRA adapter into its base model producing a standalone
merged checkpoint.  Supports a dry-run mode that verifies the adapter config,
base model name/revision and target modules without downloading any weights
or importing the heavy training stack.

Design rules (see docs/personal_lora_training.md and ``training/runtime.py``):

- BF16 LoRA only.  Quantized base models (4bit/8bit, ``quantization`` is not
  ``None``) or quantized adapters are rejected so we never silently produce a
  quantized merge.
- The base model and the adapter are NEVER modified in place.  The merged
  model is written to a fresh ``output_dir`` and we refuse to overwrite an
  existing directory or to point ``output_dir`` at the base/adapter paths.
- Heavy frameworks (``torch``, ``transformers``, ``peft``) are imported lazily
  inside the actual merge entry point, never at module import time, so that
  dry-runs and unit tests run without any download or GPU dependency.
- A JSON manifest describing the merge is always written next to the merged
  model so the exact merge can be audited and reproduced.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from training.runtime import ConfigError, load_config, validate_config

__all__ = [
    "MergePlan",
    "MergeResult",
    "MergeError",
    "build_plan",
    "dry_run",
    "merge",
    "main",
]

_DEFAULT_TORCH_DTYPE = "bfloat16"


class MergeError(RuntimeError):
    """Raised when a merge cannot be performed safely."""


@dataclass
class MergePlan:
    """Validated plan describing a merge operation (no weights loaded)."""

    model_name: str
    model_revision: str
    adapter_path: str
    output_dir: str
    torch_dtype: str = _DEFAULT_TORCH_DTYPE
    device_map: str | None = None
    # Filled in by ``dry_run`` / ``merge`` after inspecting the adapter.
    adapter_config: dict[str, Any] | None = None
    resolved_target_modules: list[str] = field(default_factory=list)
    source_config_path: str | None = None


@dataclass
class MergeResult:
    """Outcome of a completed merge."""

    plan: MergePlan
    output_dir: str
    manifest_path: str
    num_modules_merged: int


def _read_adapter_config(adapter_path: Path | str) -> dict[str, Any]:
    """Read ``adapter_config.json`` from a local adapter directory.

    Does NOT import transformers/peft; reads the JSON directly so the dry-run
    path stays download-free.
    """
    p = Path(adapter_path)
    if not p.exists():
        raise MergeError(f"adapter path not found: {p}")
    if p.is_file():
        raise MergeError(
            f"adapter path must be a directory, got a file: {p}"
        )
    cfg_file = p / "adapter_config.json"
    if not cfg_file.exists():
        raise MergeError(
            f"adapter_config.json not found inside adapter directory: {p}"
        )
    try:
        with cfg_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise MergeError(f"invalid adapter_config.json at {cfg_file}: {exc}") from exc
    if not isinstance(data, dict):
        raise MergeError(
            f"adapter_config.json root must be an object, got {type(data).__name__}"
        )
    return data


def _check_no_quantization(adapter_cfg: dict[str, Any]) -> None:
    """Refuse quantized adapters (QLoRA cannot be merged into BF16)."""
    quant = adapter_cfg.get("quantization_config") or adapter_cfg.get("quant_config")
    bnb = adapter_cfg.get("bnb_4bit_config") or adapter_cfg.get("load_in_4bit")
    if quant is not None or bnb:
        raise MergeError(
            "refusing to merge a quantized adapter (quantization_config / "
            "bnb_4bit_config present).  Only BF16 LoRA adapters may be merged."
        )


def _check_base_identity(
    adapter_cfg: dict[str, Any], model_name: str, model_revision: str,
) -> None:
    """Verify the adapter's recorded base model matches the requested one."""
    base = adapter_cfg.get("base_model_name_or_path")
    if base and base != model_name:
        raise MergeError(
            f"adapter base_model_name_or_path={base!r} does not match "
            f"requested model_name={model_name!r}"
        )
    rev = adapter_cfg.get("base_model_revision") or adapter_cfg.get("revision")
    if model_revision and rev and str(rev) != str(model_revision):
        raise MergeError(
            f"adapter base_model_revision={rev!r} does not match requested "
            f"model_revision={model_revision!r}"
        )
    if model_revision and not rev:
        # Not fatal: many adapter configs omit the base revision.  We keep
        # the requested revision as the source of truth for the manifest but
        # warn via the plan rather than failing.
        pass


def _check_output_dir(
    output_dir: Path | str, adapter_path: Path | str, model_name: str,
) -> Path:
    """Resolve and guard the output directory.

    - Refuse to overwrite an existing non-empty directory.
    - Refuse to point output_dir at the adapter path or the base model name.
    """
    out = Path(output_dir).expanduser().resolve()
    adapter = Path(adapter_path).expanduser().resolve()
    if out == adapter:
        raise MergeError(
            f"output_dir must not equal adapter_path (both resolve to {out})"
        )
    # Guard against pointing output_dir straight at a local base model dir.
    if out.name == os.path.basename(model_name) and out.parent.exists():
        # Only treat as collision if it actually looks like a snapshot dir.
        if (out / "config.json").exists():
            raise MergeError(
                f"output_dir {out} looks like the base model snapshot; "
                "refusing to overwrite the base model"
            )
    if out.exists():
        children = [c for c in out.iterdir() if not c.name.startswith(".")]
        if children:
            raise MergeError(
                f"output_dir {out} already exists and is not empty; "
                "refusing to overwrite (pass a fresh path)"
            )
    return out


def build_plan(
    config_path: Path | str,
    adapter_path: Path | str,
    output_dir: Path | str,
    *,
    torch_dtype: str = _DEFAULT_TORCH_DTYPE,
    device_map: str | None = None,
) -> MergePlan:
    """Build a ``MergePlan`` from a training config + adapter path.

    Reads the training config to recover the base model name/revision and
    the resolved LoRA target modules, then reads the local adapter_config.json
    to verify identity and reject quantized adapters.  Does NOT download
    anything or import the heavy training stack.
    """
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        raise MergeError(f"training config not found: {cfg_path}")
    if torch_dtype != _DEFAULT_TORCH_DTYPE:
        raise MergeError(
            f"only torch_dtype='bfloat16' is supported for BF16 merge, "
            f"got {torch_dtype!r}"
        )
    try:
        cfg = load_config(cfg_path)
    except ConfigError as exc:
        raise MergeError(f"invalid training config: {exc}") from exc
    validate_config(cfg)

    adapter_cfg = _read_adapter_config(adapter_path)
    _check_no_quantization(adapter_cfg)
    _check_base_identity(adapter_cfg, cfg.model_name, cfg.model_revision)

    targets = list(cfg.lora.target_modules or [])
    if cfg.lora.target_mode in ("conservative", "strong") and not targets:
        # The runtime CLI fills these in after inspecting the base model.  For
        # the merge plan we accept whatever the config pins; if empty we fall
        # back to the adapter's recorded target modules.
        targets = list(adapter_cfg.get("target_modules") or [])

    _check_output_dir(output_dir, adapter_path, cfg.model_name)

    return MergePlan(
        model_name=cfg.model_name,
        model_revision=cfg.model_revision,
        adapter_path=str(Path(adapter_path).expanduser().resolve()),
        output_dir=str(Path(output_dir).expanduser().resolve()),
        torch_dtype=torch_dtype,
        device_map=device_map,
        adapter_config=adapter_cfg,
        resolved_target_modules=targets,
        source_config_path=str(cfg_path),
    )


def dry_run(plan: MergePlan) -> dict[str, Any]:
    """Return a JSON-serializable summary of what ``merge`` would do.

    Performs the same checks as ``build_plan`` (which must already have been
    called) plus a final liveness check on the adapter weights file.  No
    downloads, no heavy imports.
    """
    adapter = Path(plan.adapter_path)
    if not adapter.exists():
        raise MergeError(f"adapter path not found: {adapter}")
    safetensors = sorted(adapter.glob("*.safetensors"))
    bins = sorted(adapter.glob("*.bin"))
    weight_files = safetensors or bins
    if not weight_files:
        raise MergeError(
            f"no adapter weight files (*.safetensors / *.bin) found in {adapter}"
        )
    summary = {
        "model_name": plan.model_name,
        "model_revision": plan.model_revision,
        "adapter_path": plan.adapter_path,
        "output_dir": plan.output_dir,
        "torch_dtype": plan.torch_dtype,
        "device_map": plan.device_map,
        "resolved_target_modules": plan.resolved_target_modules,
        "weight_files": [p.name for p in weight_files],
        "source_config_path": plan.source_config_path,
        "dry_run": True,
    }
    return summary


def _heavy_merge_libraries() -> dict[str, Any]:
    """Lazily import the heavy merge stack.  Must never run on the dry-run path."""
    import importlib

    libs: dict[str, Any] = {}
    for name in ("torch", "transformers", "peft"):
        libs[name] = importlib.import_module(name)
    return libs


def _write_manifest(plan: MergePlan, out_dir: Path, num_modules: int) -> Path:
    manifest = {
        "model_name": plan.model_name,
        "model_revision": plan.model_revision,
        "adapter_path": plan.adapter_path,
        "output_dir": str(out_dir),
        "torch_dtype": plan.torch_dtype,
        "device_map": plan.device_map,
        "resolved_target_modules": plan.resolved_target_modules,
        "source_config_path": plan.source_config_path,
        "num_modules_merged": num_modules,
        "adapter_config": plan.adapter_config,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_dir.mkdir(parents=True, exist_ok=False)
    mpath = out_dir / "merge_manifest.json"
    with mpath.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    return mpath


def merge(plan: MergePlan) -> MergeResult:
    """Perform the actual BF16 adapter merge.

    Imports ``torch``/``transformers``/``peft`` lazily, loads the base model
    and adapter, calls ``merge_and_unload`` and saves the merged model + tokenizer
    plus a JSON manifest.  The base model and adapter on disk are never
    modified.
    """
    libs = _heavy_merge_libraries()
    torch = libs["torch"]
    transformers = libs["transformers"]
    peft = libs["peft"]

    out_dir = Path(plan.output_dir)
    # Re-check in case the directory was created between plan and merge.
    if out_dir.exists():
        children = [c for c in out_dir.iterdir() if not c.name.startswith(".")]
        if children:
            raise MergeError(
                f"output_dir {out_dir} already exists and is not empty; "
                "refusing to overwrite"
            )

    dtype = torch.bfloat16
    load_kwargs: dict[str, Any] = {
        "dtype": dtype,
        "revision": plan.model_revision,
    }
    if plan.device_map:
        load_kwargs["device_map"] = plan.device_map

    base = transformers.AutoModelForImageTextToText.from_pretrained(
        plan.model_name, **load_kwargs,
    )
    processor = transformers.AutoProcessor.from_pretrained(
        plan.model_name, revision=plan.model_revision, trust_remote_code=True,
    )

    adapter = peft.PeftModel.from_pretrained(base, plan.adapter_path)

    # Refuse quantized base models (e.g. bitsandbytes).  Quantized base models
    # cannot be merged into BF16 and saved cleanly.
    quant_cfg = getattr(base.config, "quantization_config", None)
    if quant_cfg:
        raise MergeError(
            "base model is quantized (config.quantization_config present); "
            "cannot merge into a quantized base.  Use a BF16 base snapshot."
        )

    merged = adapter.merge_and_unload()
    num_modules = len(plan.resolved_target_modules)
    mpath = _write_manifest(plan, out_dir, num_modules)

    merged.save_pretrained(str(out_dir), safe_serialization=True)
    processor.save_pretrained(str(out_dir))

    return MergeResult(
        plan=plan,
        output_dir=str(out_dir),
        manifest_path=str(mpath),
        num_modules_merged=num_modules,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="merge_adapter",
        description="Merge a BF16 LoRA adapter into its base model.",
    )
    p.add_argument("--config", required=True, help="training YAML config path")
    p.add_argument("--adapter", required=True, help="local adapter directory")
    p.add_argument("--output", required=True, help="fresh output directory")
    p.add_argument(
        "--torch-dtype", default=_DEFAULT_TORCH_DTYPE,
        help=f"torch dtype (only '{_DEFAULT_TORCH_DTYPE}' is supported)",
    )
    p.add_argument("--device-map", default=None, help="transformers device_map")
    p.add_argument(
        "--dry-run", action="store_true",
        help="verify plan without downloading or merging any weights",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    plan = build_plan(
        args.config, args.adapter, args.output,
        torch_dtype=args.torch_dtype, device_map=args.device_map,
    )
    if args.dry_run:
        summary = dry_run(plan)
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    result = merge(plan)
    print(
        f"merged {result.num_modules_merged} modules -> {result.output_dir} "
        f"(manifest: {result.manifest_path})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())