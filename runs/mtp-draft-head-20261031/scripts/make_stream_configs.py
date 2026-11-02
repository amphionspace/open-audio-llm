"""Write stream_bench lane configs from the 003 configs: one variant per lane, one attempt.

Usage: make_stream_configs.py ATTEMPT PREFIX PACK_DIR_NAME VARIANT_PREFIX VARIANT...
VARIANT is base, suffix or eagle<1|3>-k<N>[@CHECKPOINT_DIR]; lanes take GPUs 4-7 in order.
"""
import copy
import sys
from pathlib import Path

import yaml

EXP = Path(__file__).resolve().parents[1]
PY = "/workspace/workspace/mingdong/miniconda3/envs/vllm-serving/bin/python"
PY_SPEC = "/workspace/workspace/mingdong/miniconda3/envs/vllm-serving-spec/bin/python"
DRAFTS = {"1": "tasks/01-validate-draft/attempts/002/artifacts/training/checkpoint-2500/eagle",
          "3": "tasks/01-validate-draft/attempts/005/artifacts/training/checkpoint-2500/eagle"}


def variant(spec):
    if spec == "base":
        return {"name": spec, "python": PY}
    if spec == "suffix":
        return {"name": spec, "python": PY_SPEC, "arguments": ["--speculative-config", '{"method": "suffix"}']}
    name, _, draft = spec.partition("@")
    arch, k = name.removeprefix("eagle").split("-k")
    method = "eagle3" if arch == "3" else "eagle"
    model = draft or str(EXP / DRAFTS[arch])
    return {"name": name, "python": PY, "arguments": ["--speculative-config",
            '{"method": "%s", "model": "%s", "num_speculative_tokens": %s}' % (method, model, k)]}


attempt, prefix, pack, variant_prefix, *specs = sys.argv[1:]
source = yaml.safe_load((EXP / "configs/stream-bench-lane0.yaml").read_text())["parameters"]
for lane, spec in enumerate(specs):
    p = copy.deepcopy(source)
    v = variant(spec)
    v["name"] = variant_prefix + v["name"]
    p.update(output=str(EXP / f"tasks/02-serve-bench/attempts/{attempt}/artifacts/lane{lane}"), pack=str(EXP / "shared" / pack),
             gpu=4 + lane, port=18600 + 10 * lane, variants=[v])
    p["wandb"] = {"enabled": True, "mode": "online", "entity": "1016097967-amphion", "project": "open-audio-llm",
                  "name": f"mtp-draft-head-20261031-a{attempt}-stream"}
    head = (f"# 流式回放（参数同 stream-bench-lane0.yaml / tsasr-latency-20261031），02 的 {attempt}，lane {lane}：{v['name']}，交付包 shared/{pack}。\n")
    (EXP / f"configs/{prefix}-lane{lane}.yaml").write_text(
        head + yaml.safe_dump({"version": 1, "parameters": p}, allow_unicode=True, sort_keys=False, width=250))
    print(f"configs/{prefix}-lane{lane}.yaml", v["name"])
