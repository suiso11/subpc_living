from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from training import merge_adapter, train_dpo, train_sft
from training.runtime import (
    ConfigError, LoraSpec, TrainingConfig, build_manifest, load_config,
    match_target_modules, validate_config, write_manifest,
)

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "training" / "configs"
EXPECTED_REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"


class ConfigTest(unittest.TestCase):
    def test_all_configs_are_pinned_bf16_lora(self):
        expected = {
            "persona_conservative.yaml": ("sft", "conservative", 16),
            "persona_strong.yaml": ("sft", "strong", 32),
            "persona_dpo.yaml": ("dpo", "conservative", 16),
        }
        for name, values in expected.items():
            with self.subTest(name=name):
                cfg = load_config(CONFIGS / name)
                self.assertEqual((cfg.stage, cfg.lora.target_mode, cfg.lora.rank), values)
                self.assertEqual(cfg.precision, "bf16")
                self.assertEqual(cfg.model_revision, EXPECTED_REVISION)
                self.assertNotIn(cfg.model_revision.lower(), {"main", "master", "head"})

    def test_rejects_non_bf16_full_ft_and_missing_revision(self):
        base = TrainingConfig(
            name="x", stage="sft", model_name="m", model_revision="rev",
            precision="bf16", max_sequence_length=32, micro_batch_size=1,
            gradient_accumulation_steps=1, learning_rate=1e-5, epochs=1,
            lora=LoraSpec(rank=8, alpha=16, dropout=0.0),
            chat_template_path="training/templates/qwen3_6_assistant.jinja",
        )
        validate_config(base)
        for field, value in (("precision", "fp16"), ("model_revision", "")):
            setattr(base, field, value)
            with self.assertRaises(ConfigError):
                validate_config(base)
            setattr(base, field, "bf16" if field == "precision" else "rev")
        base.lora.rank = 0
        with self.assertRaises(ConfigError):
            validate_config(base)


class TargetTest(unittest.TestCase):
    NAMES = [
        "model.language_model.layers.0.self_attn.q_proj",
        "model.language_model.layers.0.self_attn.k_proj",
        "model.language_model.layers.0.self_attn.v_proj",
        "model.language_model.layers.0.self_attn.o_proj",
        "model.language_model.layers.1.linear_attn.in_proj_qkv",
        "model.language_model.layers.1.linear_attn.in_proj_z",
        "model.language_model.layers.1.linear_attn.in_proj_a",
        "model.language_model.layers.1.linear_attn.in_proj_b",
        "model.language_model.layers.1.linear_attn.out_proj",
        "model.language_model.layers.1.mlp.shared_expert.gate_proj",
        "model.language_model.layers.1.mlp.shared_expert.up_proj",
        "model.language_model.layers.1.mlp.shared_expert.down_proj",
        "model.language_model.layers.1.mlp.shared_expert_gate",
        "model.language_model.layers.1.mlp.experts.0.gate_proj",
        "model.language_model.layers.1.mlp.router",
        "model.visual.blocks.0.attn.q_proj",
        "model.language_model.embed_tokens",
        "lm_head",
    ]

    def test_conservative_qwen_targets(self):
        selected = match_target_modules(self.NAMES, "conservative")
        self.assertEqual(len(selected), 9)
        self.assertTrue(any("in_proj_qkv" in x for x in selected))
        self.assertFalse(any("shared_expert" in x for x in selected))

    def test_strong_adds_only_shared_expert(self):
        selected = match_target_modules(self.NAMES, "strong")
        self.assertEqual(len(selected), 13)
        self.assertTrue(any("shared_expert.gate_proj" in x for x in selected))
        self.assertFalse(any(".experts." in x or "visual" in x or "router" in x for x in selected))

    def test_explicit_rejects_forbidden(self):
        with self.assertRaises(ConfigError):
            match_target_modules([], "explicit", ["model.visual.q_proj"])


class ManifestTest(unittest.TestCase):
    def test_manifest_is_serializable(self):
        cfg = load_config(CONFIGS / "persona_conservative.yaml")
        manifest = build_manifest(cfg, resolved_target_modules=["x.q_proj"])
        with tempfile.TemporaryDirectory() as td:
            p = write_manifest(manifest, Path(td) / "manifest.json")
            data = json.loads(p.read_text(encoding="utf-8"))
        self.assertEqual(data["precision"], "bf16")
        self.assertFalse(data["full_fine_tune"])
        self.assertIsNone(data["quantization"])


class DryRunTest(unittest.TestCase):
    def _config_copy(self, source: str, td: Path, **updates) -> Path:
        import yaml
        data = yaml.safe_load((CONFIGS / source).read_text(encoding="utf-8"))
        data.update(updates)
        path = td / source
        path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
        return path

    def test_sft_dry_run(self):
        with tempfile.TemporaryDirectory() as raw:
            td = Path(raw); dataset = td / "sft.jsonl"
            dataset.write_text('{"messages":[{"role":"user","content":"x"},{"role":"assistant","content":"y"}]}\n')
            out = td / "out"
            cfg = self._config_copy("persona_conservative.yaml", td, dataset_path=str(dataset), output_dir=str(out), min_dataset_rows=1)
            self.assertEqual(train_sft.main([str(cfg), "--dry-run"]), 0)
            self.assertTrue((out / "manifest.json").exists())

    def test_dpo_dry_run(self):
        with tempfile.TemporaryDirectory() as raw:
            td = Path(raw); dataset = td / "dpo.jsonl"; adapter = td / "adapter"; out = td / "out"
            dataset.write_text('{"prompt":"x","chosen":"y","rejected":"z"}\n')
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text(json.dumps({"peft_type":"LORA","target_modules":["q_proj"]}))
            cfg = self._config_copy("persona_dpo.yaml", td, dataset_path=str(dataset), base_adapter_path=str(adapter), output_dir=str(out), min_dataset_rows=1)
            self.assertEqual(train_dpo.main(["--config", str(cfg), "--dry-run"]), 0)
            self.assertTrue((out / "manifest.json").exists())

    def test_merge_dry_run(self):
        with tempfile.TemporaryDirectory() as raw:
            td = Path(raw); adapter = td / "adapter"; out = td / "merged"
            adapter.mkdir()
            cfg = load_config(CONFIGS / "persona_conservative.yaml")
            (adapter / "adapter_config.json").write_text(json.dumps({
                "peft_type":"LORA", "base_model_name_or_path":cfg.model_name,
                "base_model_revision":cfg.model_revision, "target_modules":["q_proj"]}))
            (adapter / "adapter_model.safetensors").write_bytes(b"fixture")
            plan = merge_adapter.build_plan(CONFIGS / "persona_conservative.yaml", adapter, out)
            summary = merge_adapter.dry_run(plan)
            self.assertTrue(summary["dry_run"])
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
