"""09-ctx-v2-eval: the 11.06 delivery model and avg3 on the context-correction benchmark v2.

- configs/ctx-v2-data.yaml: the v2 eval-set definition (audio-data-contract scripts/context_correction/eval_set.py:
  v1's three splits plus the v2 splits), resolved against the audio-data-contract catalog at the commit that
  registers context_correction_zh@v2-20261116 (PR #251), until the eval set is published;
- configs/eval-ctxv2-release/: the delivery model, servers as configs/eval-grpo-avg3/ (no draft: it has none);
- configs/eval-ctxv2-avg3-eagle3-k3/: avg3 with its EAGLE-3 draft head at k=3 as configs/eval-grpo-avg3-eagle3-k3/
  (07), think off and on.
Only the model, data, GPUs, ports, served names, output roots and attempts dirs change.
usage: make_ctx_v2_configs.py <eval-set data.yaml> <audio-data-contract catalog dir>
"""
import sys
from pathlib import Path

import yaml

EXP = Path(__file__).resolve().parents[1]
RELEASE = "/workspace/data/ae-runs/amphionasr2-grpo-v10interp050/checkpoint"
LANES = {
    "release": {"src": "eval-grpo-avg3", "name": "grpo-avg3", "suites": [""], "asr": (0, 19010), "tsasr": (1, 19011)},
    "avg3-eagle3-k3": {"src": "eval-grpo-avg3-eagle3-k3", "name": "grpo-avg3-eagle3-k3", "suites": ["", "-think"],
                       "asr": (2, 19012), "tsasr": (3, 19013)},
}


def sub(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def main():
    data = yaml.safe_load(Path(sys.argv[1]).read_text())
    data = {"catalog": str(Path(sys.argv[2]).resolve()), "roots": "/workspace/data/registry/roots.json", **data}
    (EXP / "configs/ctx-v2-data.yaml").write_text(
        "# 09-ctx-v2-eval：上下文纠偏评测集 v2 的定义（audio-data-contract eval_set.py 生成），catalog 用登记 v2 的提交；"
        "由 scripts/make_ctx_v2_configs.py 生成\n"
        + yaml.safe_dump(data, allow_unicode=True, sort_keys=False, width=200))
    for lane, spec in LANES.items():
        src, old = EXP / "configs" / spec["src"], spec["name"]
        name = f"ctxv2-{lane}"
        dst = EXP / f"configs/eval-{name}"
        dst.mkdir(exist_ok=True)
        handoff = (src / "handoff.json").read_text()
        if lane == "release":
            handoff = (EXP / "configs/eval-diag-scene-ts-release-new/handoff.json").read_text()
        (dst / "handoff.json").write_text(handoff)
        for m in ("asr", "tsasr"):
            gpu, port = spec[m]
            s = (src / f"serve-{m}.yaml").read_text()
            header = (f"# 09-ctx-v2-eval：由 scripts/make_ctx_v2_configs.py 从 configs/{spec['src']}/serve-{m}.yaml 生成，"
                      "只换模型、GPU、端口和记录目录。\n")
            s = header + s if not s.startswith("#") else sub(s, s.splitlines()[0] + "\n", header)
            served = yaml.safe_load(s)["task"]
            s = sub(s, f"  gpus:\n  - {yaml.safe_load(s)['runtime']['gpus'][0]}\n", f"  gpus:\n  - {gpu}\n")
            s = sub(s, f"    --port: {served['arguments']['--port']}\n", f"    --port: {port}\n")
            s = sub(s, f"--served-model-name: {old}-{m}\n", f"--served-model-name: {name}-{m}\n")
            if lane == "release":
                s = sub(s, f"    path: {served['model']['path']}\n", f"    path: {RELEASE}\n")
            attempts = yaml.safe_load(s)["recording"]["attempts_dir"]["path"]
            s = sub(s, f"    path: {attempts}\n", f"    path: ../../tasks/09-ctx-v2-eval/serve-{lane}-{m}/attempts\n")
            (dst / f"serve-{m}.yaml").write_text(s)
        for think in spec["suites"]:
            s = (src / f"after-training-context-correction{think}.yaml").read_text()
            s = sub(s, f"model_name: {old}{think}\n", f"model_name: {name}{think}\n")
            s = sub(s, "eval_set: open-audio-llm/context-correction-zh@v1-20261018\n", "data_config: ../ctx-v2-data.yaml\n")
            for m, key in (("asr", "server_url"), ("tsasr", "tsasr_server_url")):
                url = yaml.safe_load(s)[key]
                s = sub(s, f"{key}: {url}\n", f"{key}: http://127.0.0.1:{spec[m][1]}\n")
            s = sub(s, f"served_model: {old}-asr\n", f"served_model: {name}-asr\n")
            s = sub(s, f"tsasr_served_model: {old}-tsasr\n", f"tsasr_served_model: {name}-tsasr\n")
            s = sub(s, f"spelling-think-grpo-20261115-{old}{think}\n", f"spelling-think-grpo-20261115-{name}{think}\n")
            (dst / f"after-training-context-correction-v2{think}.yaml").write_text(s)
        print(dst)


if __name__ == "__main__":
    main()
