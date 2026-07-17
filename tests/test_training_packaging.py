from __future__ import annotations
import json, os, subprocess, tempfile, unittest
from pathlib import Path
from training.evaluate import build_ollama_request, evaluate_against_prompts, load_eval_prompts, model_tag

ROOT=Path(__file__).resolve().parent.parent
SCRIPT=ROOT/"scripts/convert_personal_model_to_gguf.sh"
MODELFILE=ROOT/"models/ollama/Modelfile.personal.example"
PROMPTS=ROOT/"training/eval_prompts.jsonl"

class EvaluationTest(unittest.TestCase):
    def test_fixed_prompts_and_distinct_tags(self):
        rows=load_eval_prompts(PROMPTS)
        self.assertGreaterEqual(len(rows),8)
        self.assertEqual(len({x["id"] for x in rows}),len(rows))
        self.assertEqual(len({model_tag(x) for x in ("baseline","sft","dpo")}),3)
        self.assertNotEqual(model_tag("checkpoint","step-1"),model_tag("dpo"))

    def test_ollama_http_payload_carries_options(self):
        data=build_ollama_request("m","危険な;文字",num_ctx=2048,num_predict=32,temperature=.2)
        self.assertEqual(data["prompt"],"危険な;文字")
        self.assertFalse(data["stream"])
        self.assertEqual(data["options"]["num_ctx"],2048)

    def test_stub_generation_trace(self):
        result=evaluate_against_prompts("sft","m",[{"id":"1","prompt":"眠い"}],lambda m,p,**o:"返答",gen_opts={"num_ctx":10})
        self.assertEqual(result["count"],1)
        self.assertEqual(result["rows"][0]["response"],"返答")

class PackagingTest(unittest.TestCase):
    def run_script(self,args):
        env=os.environ.copy(); env["LLAMA_DIR"]=""
        return subprocess.run(["bash",str(SCRIPT),*args],cwd=ROOT,env=env,text=True,capture_output=True)

    def test_dry_run_full_merged_model_has_no_side_effect(self):
        with tempfile.TemporaryDirectory() as raw:
            td=Path(raw); out=td/"out"
            r=self.run_script(["--dry-run","--merged-model",str(td/"merged model"),"--output-dir",str(out),"--basename","personal-sft","--llama-dir",str(td/"llama cpp")])
            self.assertEqual(r.returncode,0,r.stderr)
            self.assertFalse(out.exists())
            self.assertIn("convert_hf_to_gguf.py",r.stdout)
            self.assertIn("personal-sft-Q4_K_M.gguf",r.stdout)
            self.assertNotIn("convert_lora_to_gguf",r.stdout)

    def test_missing_merged_model_fails_real_run(self):
        with tempfile.TemporaryDirectory() as raw:
            td=Path(raw)
            r=self.run_script(["--merged-model",str(td/"missing"),"--output-dir",str(td/"out"),"--basename","personal-dpo","--llama-dir",str(td)])
            self.assertNotEqual(r.returncode,0)

    def test_rejects_overwrite_and_bad_quant(self):
        r=self.run_script(["--dry-run","--merged-model","same","--output-dir","same","--basename","x"])
        self.assertNotEqual(r.returncode,0)
        r=self.run_script(["--dry-run","--merged-model","a","--output-dir","b","--basename","x","--quant","Q2_K"])
        self.assertNotEqual(r.returncode,0)

    def test_modelfile_is_full_model_template(self):
        text=MODELFILE.read_text(encoding="utf-8")
        self.assertIn("FROM __MODEL_GGUF__",text)
        self.assertNotIn("ADAPTER",text)
        for name in ("base-q4_K_M","shunkin-sft-q4_K_M","shunkin-dpo-q4_K_M"):
            self.assertIn(name,text)

if __name__=="__main__": unittest.main()
