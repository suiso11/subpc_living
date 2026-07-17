from __future__ import annotations
import json, tempfile, unittest
from pathlib import Path
from scripts.switch_chat_model import main
from src.chat.config import ChatConfig

class ModelPromptTest(unittest.TestCase):
    def test_effective_override_does_not_mutate_base(self):
        cfg=ChatConfig(model="base",system_prompt="LONG",model_prompt_overrides={"personal":"SHORT"})
        self.assertEqual(cfg.effective_system_prompt(),"LONG")
        cfg.model="personal"
        self.assertEqual(cfg.effective_system_prompt(),"SHORT")
        self.assertEqual(cfg.system_prompt,"LONG")

    def test_roundtrip_preserves_nested_config(self):
        cfg=ChatConfig(model="personal",system_prompt="LONG",model_prompt_overrides={"personal":"SHORT"},discord_channel_profiles={"voice":{"system_prompt_suffix":"VOICE"}})
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"config.json"; cfg.save(p); loaded=ChatConfig.load(p)
        self.assertEqual(loaded,cfg)
        self.assertEqual(loaded.effective_system_prompt(),"SHORT")

    def test_discord_profile_suffix_composes_with_override(self):
        from src.discord_bot.bot import DiscordLLMProfile
        cfg=ChatConfig(model="personal",system_prompt="LONG",model_prompt_overrides={"personal":"SHORT"})
        profile=DiscordLLMProfile.from_config(cfg,overrides={"system_prompt_suffix":"SUFFIX"})
        self.assertEqual(profile.system_prompt,"SHORT\nSUFFIX")

class SwitchCliTest(unittest.TestCase):
    def test_switch_changes_only_model_and_rollback_exact(self):
        original={"model":"base","system_prompt":"LONG","unknown":{"keep":True},"model_prompt_overrides":{"personal":"SHORT"}}
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"chat.json"; path.write_text(json.dumps(original,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
            original_bytes=path.read_bytes()
            self.assertEqual(main(["--config",str(path),"switch","personal"]),0)
            switched=json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(switched,{**original,"model":"personal"})
            self.assertEqual((Path(str(path)+".bak")).read_bytes(),original_bytes)
            self.assertEqual(main(["--config",str(path),"rollback"]),0)
            self.assertEqual(path.read_bytes(),original_bytes)

if __name__=="__main__": unittest.main()
