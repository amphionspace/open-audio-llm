"""tasks/16: after-training configs for a candidate on every suite 11.17 was evaluated on for its release (the 27 runs
of whitebox-merge-v18a-es116inc on 2026-11-17, from each run's own run.json config), so the candidate compares with
11.17 suite by suite. Only the model name, output root and server fields change; each config names lane A (ASR server)
and B (TS server) ports as __ASR_PORT__ / __TS_PORT__, filled per lane by run_candidate_evals.sh.
usage: make_candidate_configs.py MODEL_NAME TAG     (writes configs/eval-cand-TAG/after-training-<suite>.yaml + suites.txt)
"""
import glob
import json
import re
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parents[1]
SRC = "/workspace/data/ae-runs/whitebox-merge-v18a-es116inc/runs/run-20261117T0[3-4]*"
# Already run for every tasks/12-15 candidate by eval_lanes.sh (same suites): Chinese TS v12, English TS, plain ASR, customer TS.
DONE = {"ts-asr-benchmark@v12-20261022", "ts-asr-benchmark-en@v1-20261031", "asr-benchmark-data", "customer-ts@v4-20261104"}


def suite_name(config):
    ref = config.get("eval_set") or config["data_config"]
    name = ref.rsplit("/", 1)[-1]
    if name.endswith(".yaml"):
        name = name[:-len(".yaml")]
        if name in ("eval-set",):
            name = ref.rsplit("/", 2)[-2]
    return name


def main():
    model, tag = sys.argv[1], sys.argv[2]
    out = HERE / f"configs/eval-cand-{tag}"
    out.mkdir(parents=True, exist_ok=True)
    names = []
    for run in sorted(glob.glob(SRC)):
        r = json.load(open(f"{run}/run.json"))
        if r.get("state") != "complete":
            continue
        c = dict(r["config"])
        name = suite_name(c)
        if name in DONE:
            continue
        slug = re.sub(r"[^A-Za-z0-9.-]+", "-", name)
        c["model_name"] = model
        c["output_root"] = f"/workspace/data/ae-runs/{model}"
        if "server_url" in c:
            c["server_url"], c["served_model"] = "http://127.0.0.1:__ASR_PORT__", f"{tag}-asr-__ASR_PORT__"
        if "tsasr_server_url" in c:
            c["tsasr_server_url"], c["tsasr_served_model"] = "http://127.0.0.1:__TS_PORT__", f"{tag}-tsasr-__TS_PORT__"
        header = (f"# tasks/16：由 scripts/make_candidate_configs.py 从 11.17 发布评测 {Path(run).name} 的配置生成，"
                  "只改模型名、输出目录与服务地址。\n")
        (out / f"after-training-{slug}.yaml").write_text(header + yaml.safe_dump(c, allow_unicode=True, sort_keys=False))
        names.append(f"{slug} {Path(run).name}")
    (out / "suites.txt").write_text("".join(n + "\n" for n in names))
    print(len(names), "suites")


if __name__ == "__main__":
    main()
