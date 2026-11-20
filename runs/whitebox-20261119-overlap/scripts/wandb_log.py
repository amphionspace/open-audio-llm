"""Log the white-box summaries (no per-sample rows) to W&B and print the run URL.

Usage: wandb_log.py NAME analysis.json [decompose.json ...]
W&B credentials come from the environment (WANDB_API_KEY); entity/project follow AGENTS.md.
"""
import json
import sys

import wandb

ENTITY, PROJECT = "1016097967-amphion", "open-audio-llm"


def flatten(prefix, value, out):
    if isinstance(value, dict):
        for k, v in value.items():
            flatten(f"{prefix}/{k}" if prefix else str(k), v, out)
    elif isinstance(value, list):
        if all(isinstance(x, (int, float)) for x in value) and len(value) <= 32:
            for i, x in enumerate(value):
                out[f"{prefix}/{i}"] = x
    elif isinstance(value, (int, float)) and value is not None:
        out[prefix] = value


def main():
    name, files = sys.argv[1], sys.argv[2:]
    run = wandb.init(entity=ENTITY, project=PROJECT, name=name, job_type="whitebox",
                     config={"inputs": files, "doc": "docs/experiments/2026-11-19-ts-retro-overlap/README.md"})
    summary = {}
    for f in files:
        data = json.load(open(f))
        if "bins" in data and "total" not in data:  # analyze_ref_score.py output
            flatten("refscore", data, summary)
        else:  # decompose_errors.py output: {model: {total, bins}}
            for model, v in data.items():
                flatten(f"decompose/{model}/total", v["total"], summary)
                for b in ("overlap=0", "overlap=<0.9", "overlap=>=0.9", "n_interf=1", "n_interf=2", "n_interf=4"):
                    if b in v["bins"]:
                        flatten(f"decompose/{model}/{b}", v["bins"][b], summary)
    run.summary.update(summary)
    print(run.url, len(summary), "summary keys", flush=True)
    run.finish()


if __name__ == "__main__":
    main()
