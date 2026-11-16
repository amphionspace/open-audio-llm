"""08-scene-ts-diag: configs/eval-diag-scene-ts-<name>/, one TS server per model on node GPUs 4-7, no MTP.

Each dir is configs/eval-grpo-avg3/serve-tsasr.yaml (avg3's 03 TS server) plus 05's gate-scene-ts config, with only the
model, GPU, port, served name, output root and attempts dir changed. Which prompt is sent depends on the ae used:
amphion-eval 0.14 sends the new TS prompt, the frozen 0.13.3-oldprompt env the old one (scripts/run_eval.sh <ae>).
"""
import json
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
CACHE = "/workspace/data/ae-state/model-cache/entries"
MODELS = {  # name: (checkpoint, GPU, port)
    "avg3-new": (f"{CACHE}/model-377c0d630417ca1fc281e847", 4, 18990),
    "avg3-old": (f"{CACHE}/model-377c0d630417ca1fc281e847", 5, 18991),
    "release-new": ("/workspace/data/ae-runs/amphionasr2-grpo-v10interp050/checkpoint", 6, 18992),
    "r2-new": (f"{CACHE}/checkpoint-3000-927844461eb1", 7, 18993),
}


def sub(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def main():
    serve = (EXP / "configs/eval-grpo-avg3/serve-tsasr.yaml").read_text()
    gate = (EXP / "configs/eval-grpo-avg3-mtp-d-ts/after-training-gate-scene-ts.yaml").read_text()
    handoff = json.loads((EXP / "configs/eval-grpo-avg3/handoff.json").read_text())
    for name, (ckpt, gpu, port) in MODELS.items():
        model = f"diag-scene-ts-{name}"
        dst = EXP / f"configs/eval-{model}"
        dst.mkdir(exist_ok=True)
        s = "# 08-scene-ts-diag：由 scripts/make_scene_ts_diag_configs.py 从 configs/eval-grpo-avg3/serve-tsasr.yaml 生成，只换模型、GPU、端口和记录目录，不开 MTP。\n" + serve
        s = sub(s, "  gpus:\n  - 1\n", f"  gpus:\n  - {gpu}\n")
        s = sub(s, f"    path: {CACHE}/model-377c0d630417ca1fc281e847\n", f"    path: {ckpt}\n")
        s = sub(s, "    --port: 18967\n", f"    --port: {port}\n")
        s = sub(s, "--served-model-name: grpo-avg3-tsasr\n", f"--served-model-name: {model}-tsasr\n")
        s = sub(s, "../../tasks/03-eval/grpo-avg3/serve-tsasr/attempts", f"../../tasks/08-scene-ts-diag/serve-{name}/attempts")
        (dst / "serve-tsasr.yaml").write_text(s)
        g = sub(gate, "model_name: grpo-avg3-mtp\n", f"model_name: {model}\n")
        g = sub(g, "tsasr_server_url: http://127.0.0.1:18977\n", f"tsasr_server_url: http://127.0.0.1:{port}\n")
        g = sub(g, "tsasr_served_model: grpo-avg3-mtp-d-tsasr\n", f"tsasr_served_model: {model}-tsasr\n")
        g = sub(g, "spelling-think-grpo-20261115-grpo-avg3-mtp\n", f"spelling-think-grpo-20261115-{model}\n")
        (dst / "after-training-gate-scene-ts.yaml").write_text(g)
        (dst / "handoff.json").write_text(json.dumps(dict(handoff, checkpoint=ckpt, training_run=str(Path(ckpt).parent))) + "\n")
        print(dst)


if __name__ == "__main__":
    main()
