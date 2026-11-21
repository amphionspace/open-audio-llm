"""Scene-TS configs for one checkpoint: configs/eval-scene-<name>/ with one TS server and tdtech/scene-ts@v1-20261028.

Why: 08-scene-ts-diag (runs/zh-spelling-think-grpo-20261115, oal-think-exp) traced the delivery blocker of the
context-correction line, the scene-TS regression (CER 3.35 -> 9.93 at round 2's checkpoint-3000), to round 2's
additions; the main suspect is the synthetic train_ts (one clean speaker, target always present). The implicit data
adds more of the same kind, so template (round-2 recipe) and template-implicit-noexplain are both measured here.

The TS server is this experiment's (make_eval_configs.py output: vLLM 0.32, code-v1, the checkpoint's own mtp/ with
EAGLE k=2, greedy so the draft does not change outputs), copied from configs/eval-<from>/serve-tsasr.yaml with model,
GPU, port, served name and attempts dir changed. The suite is 08's gate-scene-ts config (ae 0.14: the new TS prompt,
as 08's "-new" rows), with model name, server and output root changed.
usage: make_scene_ts_configs.py <name> <checkpoint> <ts gpu> <ts port>
"""
import json
import sys
from pathlib import Path

import yaml

EXP = Path(__file__).resolve().parents[1]
FROM = EXP / "configs/eval-clue-target-implicit-noexplain"
GATE = Path("/workspace/workspace/mingdong/repo/oal-think-exp/runs/zh-spelling-think-grpo-20261115/configs/"
            "eval-diag-scene-ts-r1-think-new/after-training-gate-scene-ts.yaml")


def sub(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def write_serve(model, checkpoint, gpu, port, header):
    """configs/eval-<model>/serve-tsasr.yaml and handoff.json for the checkpoint; returns the directory."""
    assert (Path(checkpoint) / "mtp" / "model.safetensors").exists(), checkpoint
    dst = EXP / "configs" / f"eval-{model}"
    dst.mkdir(exist_ok=True)
    s = (FROM / "serve-tsasr.yaml").read_text()
    cfg = yaml.safe_load(s)
    s = sub(s, s.splitlines()[0] + "\n", header)
    old_model = cfg["task"]["model"]["path"]
    s = sub(s, f"    path: {old_model}\n", f"    path: {checkpoint}\n")
    s = sub(s, f"      model: {old_model}/mtp\n", f"      model: {checkpoint}/mtp\n")
    s = sub(s, f"  gpus:\n  - {cfg['runtime']['gpus'][0]}\n", f"  gpus:\n  - {gpu}\n")
    s = sub(s, f"    --port: {cfg['task']['arguments']['--port']}\n", f"    --port: {port}\n")
    s = sub(s, f"--served-model-name: {cfg['task']['arguments']['--served-model-name']}\n",
            f"--served-model-name: {model}-tsasr\n")
    s = sub(s, f"    path: {cfg['recording']['attempts_dir']['path']}\n",
            f"    path: ../../tasks/02-eval/serve-{model}-tsasr/attempts\n")
    (dst / "serve-tsasr.yaml").write_text(s)
    handoff = json.loads((FROM / "handoff.json").read_text())
    handoff.update(checkpoint=checkpoint, training_run=str(Path(checkpoint).parent))
    (dst / "handoff.json").write_text(json.dumps(handoff))
    return dst


def main():
    name, checkpoint, gpu, port = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    model = f"scene-{name}"
    dst = write_serve(model, checkpoint, gpu, port,
                      "# 由 scripts/make_scene_ts_configs.py 从 configs/eval-clue-target-implicit-noexplain/serve-tsasr.yaml"
                      " 生成：只换模型（草稿随之换成它自带的 mtp/）、GPU、端口、服务名和记录目录。\n")
    g = GATE.read_text()
    g = sub(g, g.splitlines()[0] + "\n", "# 场景 TS（think 关）：08-scene-ts-diag 的 r1-think-new 配置，只换模型、服务与输出目录"
                                        "（scripts/make_scene_ts_configs.py）。\n")
    g = sub(g, "model_name: diag-scene-ts-r1-think-new\n", f"model_name: ctx-clue-target-{model}\n")
    g = sub(g, "tsasr_server_url: http://127.0.0.1:18994\n", f"tsasr_server_url: http://127.0.0.1:{port}\n")
    g = sub(g, "tsasr_served_model: diag-scene-ts-r1-think-new-tsasr\n", f"tsasr_served_model: {model}-tsasr\n")
    g = sub(g, "output_root: /workspace/data/ae-runs/spelling-think-grpo-20261115-diag-scene-ts-r1-think-new\n",
            f"output_root: /workspace/data/ae-runs/ctx-clue-target-20261117-{model}\n")
    (dst / "after-training-gate-scene-ts.yaml").write_text(g)
    print(dst)


if __name__ == "__main__":
    main()
