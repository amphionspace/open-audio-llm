"""Eval configs for one checkpoint on the context-correction eval set v2, think off and on.

Copied from runs/zh-ctx-clue-target-20261117/scripts/make_eval_configs.py (oal-clue-target, the start arm's own eval),
which was copied from 09-ctx-v2-eval of runs/zh-spelling-think-grpo-20261115 (configs/eval-ctxv2-avg3-eagle3-k3/, oal-think-exp
branch), the run that measured avg3 on this set. Changed: the model; the eval set by name
(asr/context-correction-zh@v2-20261117, published from the same selection 09 used); the code (here this experiment's code-v3, main
15eacbf), whose TS server needs vLLM 0.32 (env vllm-serving-032); the draft is the checkpoint's own joint-MTP mtp/
with EAGLE k=2 (AGENTS.md "MTP"); GPUs, ports, served names, output root and attempts dirs.

Model names and output roots are ctx-grpo-20261120-<name>; serve records go to tasks/03-eval/.
usage: make_eval_configs.py <name> <checkpoint> <asr gpu> <ts gpu> <asr port>   (TS port = asr port + 1)
"""
import json
import sys
from pathlib import Path

import yaml

EXP = Path(__file__).resolve().parents[1]
SRC = Path("/workspace/workspace/mingdong/repo/oal-think-exp/runs/zh-spelling-think-grpo-20261115/configs/eval-ctxv2-avg3-eagle3-k3")
OLD = "ctxv2-avg3-eagle3-k3"
EVAL_SET = "asr/context-correction-zh@v2-20261117"
SERVING = "/workspace/workspace/mingdong/miniconda3/envs/vllm-serving-032/bin/python"
CODE_SRC = "../../shared/code-v3/src"
OLD_DRAFT = ("    --speculative-config:\n      method: eagle3\n"
             "      model: /workspace/workspace/mingdong/repo/oal-think-exp/runs/zh-spelling-think-grpo-20261115/shared/draft-avg3-eagle3-10k\n"
             "      num_speculative_tokens: 3\n")


def sub(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def main():
    name, checkpoint, gpus, port = sys.argv[1], sys.argv[2], (int(sys.argv[3]), int(sys.argv[4])), int(sys.argv[5])
    assert (Path(checkpoint) / "mtp" / "model.safetensors").exists(), checkpoint
    dst = EXP / "configs" / f"eval-{name}"
    dst.mkdir(exist_ok=True)
    handoff = json.loads((SRC / "handoff.json").read_text())
    handoff.update(checkpoint=checkpoint, training_run=str(Path(checkpoint).parent))
    (dst / "handoff.json").write_text(json.dumps(handoff))
    for m, gpu, p in (("asr", gpus[0], port), ("tsasr", gpus[1], port + 1)):
        s = (SRC / f"serve-{m}.yaml").read_text()
        s = sub(s, s.splitlines()[0] + "\n",
                f"# 由 scripts/make_eval_configs.py（抄自 zh-ctx-clue-target-20261117）从 zh-spelling-think-grpo-20261115 的 configs/{SRC.name}/serve-{m}.yaml 生成："
                "换模型、草稿（自带 mtp/，EAGLE k=2）、代码、vLLM 0.32 环境、GPU、端口和记录目录。\n")
        cfg = yaml.safe_load(s)
        s = sub(s, f"    path: {cfg['runtime']['python']['path']}\n", f"    path: {SERVING}\n")
        # 08's eval configs point at its own snapshot, also named code-v3; the same relative path here is this
        # experiment's code-v3 (main 15eacbf), so the text is already right. CODE_SRC is what it must point to.
        s = s.replace("../../shared/code-v3/src", CODE_SRC)
        s = sub(s, f"  gpus:\n  - {cfg['runtime']['gpus'][0]}\n", f"  gpus:\n  - {gpu}\n")
        s = sub(s, f"    path: {cfg['task']['model']['path']}\n", f"    path: {checkpoint}\n")
        s = sub(s, f"    --port: {cfg['task']['arguments']['--port']}\n", f"    --port: {p}\n")
        s = sub(s, f"--served-model-name: {OLD}-{m}\n", f"--served-model-name: {name}-{m}\n")
        s = sub(s, OLD_DRAFT, f"    --speculative-config:\n      method: eagle\n      model: {checkpoint}/mtp\n"
                              "      num_speculative_tokens: 2\n")
        s = sub(s, f"    path: {cfg['recording']['attempts_dir']['path']}\n",
                f"    path: ../../tasks/03-eval/serve-{name}-{m}/attempts\n")
        assert "oal-think-exp" not in s, s
        paths = [p["path"] for p in yaml.safe_load(s)["runtime"]["pythonpath"]]
        assert paths == [CODE_SRC], paths
        (dst / f"serve-{m}.yaml").write_text(s)
    for think in ("", "-think"):
        s = (SRC / f"after-training-context-correction-v2{think}.yaml").read_text()
        s = sub(s, f"model_name: {OLD}{think}\n", f"model_name: ctx-grpo-{name}{think}\n")
        s = sub(s, "data_config: ../ctx-v2-data.yaml\n", f"eval_set: {EVAL_SET}\n")
        s = sub(s, "server_url: http://127.0.0.1:19012\n", f"server_url: http://127.0.0.1:{port}\n")
        s = sub(s, "tsasr_server_url: http://127.0.0.1:19013\n", f"tsasr_server_url: http://127.0.0.1:{port + 1}\n")
        s = sub(s, f"served_model: {OLD}-asr\n", f"served_model: {name}-asr\n")
        s = sub(s, f"tsasr_served_model: {OLD}-tsasr\n", f"tsasr_served_model: {name}-tsasr\n")
        s = sub(s, f"spelling-think-grpo-20261115-{OLD}{think}\n", f"ctx-grpo-20261120-{name}{think}\n")
        (dst / f"after-training-context-correction-v2{think}.yaml").write_text(s)
    print(dst)


if __name__ == "__main__":
    main()
