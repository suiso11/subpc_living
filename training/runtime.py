"""Training runtime: config loading, validation, target-module matching and
resolved settings manifest emission for the H200 BF16 LoRA pipeline.

This module is import-safe: it only depends on the Python standard library and
PyYAML, both of which are available in the base assistant environment.  Heavy
training frameworks (``transformers``, ``peft``, ``trl``, ``torch``) MUST be
imported lazily inside the CLI entry points, never at module import time, so
that dry-runs and unit tests run without any model download or GPU dependency.

Design rules enforced here (see docs/training/personal_lora_training.md):

- BF16 LoRA only.  ``precision`` must be ``"bf16"``.  Quantized (4bit/8bit)
  precisions and FP16/FP32 full fine-tuning are rejected explicitly so the
  pipeline never silently degrades to QLoRA or full-FT.
- LoRA rank must be a positive integer (no full fine-tuning).
- Base model ``revision`` is required (no implicit ``"main"`` default).  The
  training config MUST pin an explicit revision so runs are reproducible.
- Conservative target-module matching MUST exclude experts / router / vision
  projector / embed / lm_head / norm, training only shared attention and
  shared projection layers.
- A resolved settings manifest is always written before any heavy work so the
  exact run can be audited and reproduced even if training crashes.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

import yaml

__all__ = [
    "TrainingConfig",
    "LoraSpec",
    "ResolvedSettings",
    "ConfigError",
    "VALID_STAGES",
    "VALID_PRECISIONS",
    "TARGET_MODES",
    "load_config",
    "validate_config",
    "validate_runtime_config",
    "match_target_modules",
    "build_manifest",
    "write_manifest",
]

VALID_STAGES = ("sft", "dpo")
VALID_PRECISIONS = ("bf16",)
TARGET_MODES = ("conservative", "strong", "explicit")

# Substrings that disqualify a module from LoRA targeting in conservative /
# strong modes.  Plural forms ("experts") and singular ("expert") are both
# covered by the substring check below.
_EXCLUDE_SUBSTRINGS = (
    ".experts",
    "router",
    "vision",
    "visual",
    "projector",
    "embed_tokens",
    "lm_head",
    ".norm",
    "input_layernorm",
    "post_attention_layernorm",
)

# Qwen3.6-35B-A3B is implemented as qwen3_5_moe. Conservative LoRA
# targets the text full-attention projections and GatedDeltaNet projections.
# It deliberately excludes shared/routed MLPs. Strong additionally includes
# only the shared expert MLP, never the 256 routed experts or router.
_CONSERVATIVE_SUFFIXES = (
    ".q_proj", ".k_proj", ".v_proj", ".o_proj",
    ".linear_attn.in_proj_qkv", ".linear_attn.in_proj_z",
    ".linear_attn.in_proj_a", ".linear_attn.in_proj_b",
    ".linear_attn.out_proj",
)
_STRONG_SHARED_MARKERS = (".shared_expert.", ".shared_expert_gate")


class ConfigError(ValueError):
    """Raised when a training config fails validation."""


@dataclass
class LoraSpec:
    """LoRA adapter spec.

    ``target_mode`` selects how ``target_modules`` is interpreted:

    - ``conservative``: keep only attention Q/K/V/O + shared projections,
      excluding experts/router/vision/projector/embed/lm_head/norm.
      ``target_modules`` is ignored (derived from candidate module names).
    - ``strong``: as conservative plus shared MLP sub-projections, still
      excluding the routed experts/router/vision.
    - ``explicit``: use the literal ``target_modules`` list verbatim.  The
      caller is responsible for not including excluded substrings.
    """

    rank: int
    alpha: int
    dropout: float
    target_mode: str = "conservative"
    target_modules: list[str] = field(default_factory=list)
    modules_to_save: list[str] = field(default_factory=list)


@dataclass
class TrainingConfig:
    """Validated training config for SFT or DPO BF16 LoRA runs."""

    name: str
    stage: str
    model_name: str
    model_revision: str
    precision: str
    max_sequence_length: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    epochs: int
    lora: LoraSpec
    gradient_checkpointing: bool = True
    assistant_only_loss: bool = True
    chat_template_path: str | None = None
    dataset_path: str | None = None
    min_dataset_rows: int = 1
    base_adapter_path: str | None = None
    output_dir: str = "training/outputs"
    save_steps: int = 500
    seed: int = 42
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    source_path: str | None = None

    def lora_dict(self) -> dict[str, Any]:
        return asdict(self.lora)


@dataclass
class ResolvedSettings:
    """Resolved, frozen view of a training run used for the audit manifest."""

    name: str
    stage: str
    model_name: str
    model_revision: str
    precision: str
    quantization: str | None
    full_fine_tune: bool
    lora: dict[str, Any]
    dataset_path: str | None
    min_dataset_rows: int
    base_adapter_path: str | None
    output_dir: str
    max_sequence_length: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    epochs: int
    gradient_checkpointing: bool
    assistant_only_loss: bool
    chat_template_path: str | None
    save_steps: int
    seed: int
    warmup_ratio: float
    lr_scheduler_type: str
    resolved_target_modules: list[str]
    source_path: str | None
    generated_at: str


def _require(value: Any, field_name: str) -> None:
    if value is None:
        raise ConfigError(f"missing required field: {field_name!r}")


def _load_yaml(path: Path | str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"config file not found: {p}")
    try:
        with p.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {p}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"config root must be a mapping, got {type(data).__name__}")
    return data


def _build_config(data: dict[str, Any], source_path: Path | str | None) -> TrainingConfig:
    lora_data = data.get("lora")
    if not isinstance(lora_data, dict):
        raise ConfigError("missing 'lora' mapping")
    try:
        lora = LoraSpec(
            rank=int(lora_data.get("rank", 0)),
            alpha=int(lora_data.get("alpha", 0)),
            dropout=float(lora_data.get("dropout", 0.0)),
            target_mode=str(lora_data.get("target_mode", "conservative")),
            target_modules=list(lora_data.get("target_modules", []) or []),
            modules_to_save=list(lora_data.get("modules_to_save", []) or []),
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"invalid lora spec: {exc}") from exc

    try:
        cfg = TrainingConfig(
            name=str(data.get("name", "")),
            stage=str(data.get("stage", "")),
            model_name=str(data.get("model_name", "")),
            model_revision=str(data.get("model_revision", "") or ""),
            precision=str(data.get("precision", "")),
            max_sequence_length=int(data.get("max_sequence_length", 0)),
            micro_batch_size=int(data.get("micro_batch_size", 0)),
            gradient_accumulation_steps=int(data.get("gradient_accumulation_steps", 0)),
            learning_rate=float(data.get("learning_rate", 0.0)),
            epochs=int(data.get("epochs", 0)),
            lora=lora,
            gradient_checkpointing=bool(data.get("gradient_checkpointing", True)),
            assistant_only_loss=bool(data.get("assistant_only_loss", True)),
            chat_template_path=data.get("chat_template_path"),
            dataset_path=data.get("dataset_path"),
            min_dataset_rows=int(data.get("min_dataset_rows", 1)),
            base_adapter_path=data.get("base_adapter_path"),
            output_dir=str(data.get("output_dir", "training/outputs")),
            save_steps=int(data.get("save_steps", 500)),
            seed=int(data.get("seed", 42)),
            warmup_ratio=float(data.get("warmup_ratio", 0.03)),
            lr_scheduler_type=str(data.get("lr_scheduler_type", "cosine")),
            source_path=str(source_path) if source_path is not None else None,
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"invalid config value: {exc}") from exc
    return cfg


def load_config(path: Path | str) -> TrainingConfig:
    """Load a YAML config file into a ``TrainingConfig`` and validate it.

    Raises ``ConfigError`` on any structural or semantic problem.  This call
    does NOT import any heavy training framework.
    """
    raw = _load_yaml(path)
    cfg = _build_config(raw, path)
    validate_config(cfg)
    return cfg


def validate_config(cfg: TrainingConfig) -> None:
    """Validate a ``TrainingConfig`` in place.  Pure-Python, no heavy imports."""
    if not cfg.name:
        raise ConfigError("config field 'name' is required and must be non-empty")
    if cfg.stage not in VALID_STAGES:
        raise ConfigError(
            f"stage must be one of {VALID_STAGES!r}, got {cfg.stage!r}"
        )
    if not cfg.model_name:
        raise ConfigError("model_name is required")
    if not cfg.model_revision:
        raise ConfigError(
            "model_revision is required; pin an explicit HF revision "
            "(commit hash or tag) for reproducibility"
        )
    if cfg.precision not in VALID_PRECISIONS:
        raise ConfigError(
            f"precision must be one of {VALID_PRECISIONS!r} (BF16 LoRA only); "
            f"got {cfg.precision!r}.  Quantized (4bit/8bit) and FP16/FP32 full "
            f"fine-tuning are not supported."
        )
    if cfg.lora.rank <= 0:
        raise ConfigError(
            "lora.rank must be a positive integer; rank<=0 implies full "
            "fine-tuning which is not supported"
        )
    if cfg.lora.alpha <= 0:
        raise ConfigError("lora.alpha must be positive")
    if not (0.0 <= cfg.lora.dropout < 1.0):
        raise ConfigError("lora.dropout must be in [0, 1)")
    if cfg.lora.target_mode not in TARGET_MODES:
        raise ConfigError(
            f"lora.target_mode must be one of {TARGET_MODES!r}, got "
            f"{cfg.lora.target_mode!r}"
        )
    if cfg.lora.target_mode == "explicit" and not cfg.lora.target_modules:
        raise ConfigError(
            "lora.target_mode='explicit' requires a non-empty "
            "lora.target_modules list"
        )
    if cfg.lora.target_mode == "explicit":
        bad = [
            m for m in cfg.lora.target_modules if _contains_any(m, _EXCLUDE_SUBSTRINGS)
        ]
        if bad:
            raise ConfigError(
                "explicit target_modules must not include expert/router/vision/"
                f"projector/embed/lm_head/norm submodules: {bad!r}"
            )
    if cfg.min_dataset_rows <= 0:
        raise ConfigError("min_dataset_rows must be positive")
    if cfg.max_sequence_length <= 0:
        raise ConfigError("max_sequence_length must be positive")
    if cfg.micro_batch_size <= 0:
        raise ConfigError("micro_batch_size must be positive")
    if cfg.gradient_accumulation_steps <= 0:
        raise ConfigError("gradient_accumulation_steps must be positive")
    if cfg.learning_rate <= 0.0:
        raise ConfigError("learning_rate must be positive")
    if cfg.epochs <= 0:
        raise ConfigError("epochs must be positive")
    if cfg.stage == "sft" and cfg.assistant_only_loss and not cfg.chat_template_path:
        raise ConfigError("assistant_only_loss requires chat_template_path with generation blocks")
    if cfg.stage == "dpo" and not cfg.base_adapter_path:
        raise ConfigError(
            "stage='dpo' requires base_adapter_path "
            "(continue from an SFT adapter; full-model DPO is not supported)"
        )


def validate_runtime_config(cfg: TrainingConfig, *, dataset_required: bool = True) -> None:
    """Validate fields only meaningful once a run is about to start."""
    validate_config(cfg)
    if dataset_required and not cfg.dataset_path:
        raise ConfigError("dataset_path is required to launch training")
    if cfg.stage == "sft" and cfg.base_adapter_path:
        raise ConfigError(
            "stage='sft' must not set base_adapter_path; "
            "DPO is the only stage that continues from an adapter"
        )


def _contains_any(name: str, substrings: Iterable[str]) -> bool:
    low = name.lower()
    return any(s in low for s in substrings)


def match_target_modules(
    candidate_names: Iterable[str], mode: str, explicit_targets: list[str] | None = None,
) -> list[str]:
    """Select the module names that LoRA should attach to.

    ``candidate_names`` is the list of module-qualified Linear names produced
    by inspecting the loaded base model.  The conservative mode keeps only
    attention projections and shared projections, excluding routed experts,
    the router, the vision projector, embeddings and lm_head/norm.  Strong
    mode adds shared MLP sub-projections while still excluding those.

    The mode determines the keep-list; exclusions always apply first.
    """
    if mode == "explicit":
        if explicit_targets is None:
            raise ConfigError("mode='explicit' requires explicit_targets")
        bad = [m for m in explicit_targets if _contains_any(m, _EXCLUDE_SUBSTRINGS)]
        if bad:
            raise ConfigError(
                "explicit target_modules must not include excluded submodules: "
                f"{bad!r}"
            )
        return list(explicit_targets)
    if mode not in ("conservative", "strong"):
        raise ConfigError(f"unknown target_mode: {mode!r}")

    selected: list[str] = []
    seen: set[str] = set()
    for name in candidate_names:
        if not isinstance(name, str) or not name:
            continue
        low = name.lower()
        if _contains_any(low, _EXCLUDE_SUBSTRINGS):
            continue
        keep = low.endswith(_CONSERVATIVE_SUFFIXES)
        if mode == "strong" and any(marker in low for marker in _STRONG_SHARED_MARKERS):
            keep = low.endswith((".gate_proj", ".up_proj", ".down_proj", ".shared_expert_gate"))
        if not keep or name in seen:
            continue
        seen.add(name)
        selected.append(name)
    return selected


def build_manifest(
    cfg: TrainingConfig,
    *,
    resolved_target_modules: list[str],
) -> ResolvedSettings:
    """Build the resolved settings manifest for a validated config.

    ``resolved_target_modules`` is the concrete list produced after inspecting
    the base model so reviewers can audit which modules actually received
    LoRA adapters.
    """
    return ResolvedSettings(
        name=cfg.name,
        stage=cfg.stage,
        model_name=cfg.model_name,
        model_revision=cfg.model_revision,
        precision=cfg.precision,
        quantization=None,  # BF16 LoRA only: never quantized.
        full_fine_tune=False,  # LoRA rank>0 always.
        lora=cfg.lora_dict(),
        dataset_path=cfg.dataset_path,
        min_dataset_rows=cfg.min_dataset_rows,
        base_adapter_path=cfg.base_adapter_path,
        output_dir=cfg.output_dir,
        max_sequence_length=cfg.max_sequence_length,
        micro_batch_size=cfg.micro_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        epochs=cfg.epochs,
        gradient_checkpointing=cfg.gradient_checkpointing,
        assistant_only_loss=cfg.assistant_only_loss,
        chat_template_path=cfg.chat_template_path,
        save_steps=cfg.save_steps,
        seed=cfg.seed,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        resolved_target_modules=list(resolved_target_modules),
        source_path=cfg.source_path,
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )


def write_manifest(manifest: ResolvedSettings, path: Path | str) -> Path:
    """Write the resolved settings manifest as pretty-printed JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(manifest)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    return p


def _heavy_training_libraries() -> dict[str, Any]:
    """Lazily import the heavy training stack.  Must never be called on the
    dry-run / test path."""
    import importlib

    libs: dict[str, Any] = {}
    for name in ("torch", "transformers", "peft", "trl", "datasets", "accelerate"):
        libs[name] = importlib.import_module(name)
    return libs


def resolve_for_run(
    cfg: TrainingConfig,
    overrides: dict[str, Any] | None = None,
) -> tuple[TrainingConfig, list[str]]:
    """Resolve CLI overrides on top of a validated config.

    Returns the (possibly overridden) config plus an empty resolved-target
    list (filled in by the CLI after inspecting the base model).  Dry-run
    callers pass ``resolved_target_modules=[]`` to ``build_manifest`` or supply
    a candidate-derived list manually.
    """
    overrides = dict(overrides or {})
    cfg = _build_config(
        {**asdict(cfg)},
        cfg.source_path,
    )
    if "dataset_path" in overrides:
        cfg.dataset_path = overrides["dataset_path"] or cfg.dataset_path
    if "output_dir" in overrides and overrides["output_dir"]:
        cfg.output_dir = overrides["output_dir"]
    if "revision" in overrides and overrides["revision"]:
        cfg.model_revision = overrides["revision"]
    if "epochs" in overrides and overrides["epochs"] is not None:
        cfg.epochs = overrides["epochs"]
    if "learning_rate" in overrides and overrides["learning_rate"] is not None:
        cfg.learning_rate = overrides["learning_rate"]
    validate_runtime_config(cfg, dataset_required=False)
    return cfg, []