"""Log the numbers of summary JSON files (any nesting; short numeric lists by index) to one W&B run's summary.

Keys are "<file stem>/<path>". Used for comparisons whose JSON is not one of the retrospective's formats
(e.g. v12_refscore_compare.py, layer_selectivity.py, enroll_len_paired-style tables saved as JSON).
usage: log_summary_wandb.py RUN_NAME JOB_TYPE FILE.json [FILE.json ...]   (oal-train python, WANDB_API_KEY in env)
"""
import json
import sys
from pathlib import Path

import wandb


def flatten(prefix, value, out):
    if isinstance(value, dict):
        for k, v in value.items():
            flatten(f"{prefix}/{k}", v, out)
    elif isinstance(value, list):
        if len(value) <= 64 and all(isinstance(x, (int, float)) or x is None for x in value):
            for i, x in enumerate(value):
                if x is not None:
                    out[f"{prefix}/{i}"] = x
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        out[prefix] = value


def main():
    name, job, files = sys.argv[1], sys.argv[2], sys.argv[3:]
    summary = {}
    for f in files:
        flatten(Path(f).stem, json.load(open(f)), summary)
    run = wandb.init(entity="1016097967-amphion", project="open-audio-llm", name=name, job_type=job,
                     config={"inputs": [str(Path(f).resolve()) for f in files]})
    run.summary.update(summary)
    print(run.url, len(summary), "summary keys", flush=True)
    run.finish()


if __name__ == "__main__":
    main()
