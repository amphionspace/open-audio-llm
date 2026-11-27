"""Score AmphionEval ts-asr runs with the installed native scoring version from their frozen predictions.

`ae open-audio-llm rescore` only takes published runs, and the gate records (11.17, task 28/29) are unpublished, so this
writes the summary aside instead of deriving a run: OUT_DIR/<run>.json = {"protocol", "metrics", "rescored_from"},
the same rows `ae open-audio-llm rescore` computes for ts-asr (scoring.summarize). The source runs are not touched, and a
run already written to OUT_DIR is not scored again. summary_compare.py reads these files in place of run IDs.
All runs of one call go to one W&B run (WANDB_NAME): per source and (task, language), the error rate, false alarms and
misses (summed as in summary_compare.py).
Run with amphion-eval 0.15.0 (NATIVE_SCORING_VERSION 5) first on PYTHONPATH; the shared env stays on 0.13.3 / v3.
usage: rescore_v5.py OUT_DIR LABEL=RUN_ID [LABEL=RUN_ID ...]   (runs are looked up under /workspace/data/ae-runs/*/runs/)
"""
import collections
import glob
import json
import os
import sys

from amphion_eval.integrations.open_audio_llm.scoring import protocol, summarize
from amphion_eval.results import NATIVE_SCORING_VERSION

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
logged = {}
for arg in sys.argv[2:]:
    label, run = arg.split("=")
    path = f"{out}/{run}.json"
    if not os.path.exists(path):
        src = glob.glob(f"/workspace/data/ae-runs/*/runs/{run}")[0]
        previous = json.load(open(f"{src}/summary.json"))
        rows = [json.loads(line) for line in open(f"{src}/predictions.jsonl") if line.strip()]
        metrics = summarize(rows)
        if {k: v["utterances"] for k, v in metrics.items()} != {k: v["utterances"] for k, v in previous["metrics"].items()}:
            raise ValueError(f"{run}: predictions do not match its frozen summary")
        json.dump({"protocol": {**previous["protocol"], **protocol()}, "metrics": metrics, "rescored_from": run},
                  open(path, "w"), ensure_ascii=False, indent=1)
    summary = json.load(open(path))
    assert summary["protocol"]["version"] == NATIVE_SCORING_VERSION, path
    total = collections.defaultdict(collections.Counter)
    for key, m in summary["metrics"].items():
        _, task, language = key.split("/")
        t, negative = total[f"{task}_{language}"], m.get("negative_utterances") or 0
        t["err"] += m.get("errors") or 0
        t["ref"] += m.get("reference_units") or 0
        t["fa"] += round(negative * (m.get("false_alarm_rate") or 0))
        t["miss"] += round((m["utterances"] - negative) * (m.get("miss_rate") or 0))
    for key, t in total.items():
        logged[f"{label}/{key}/error_rate"] = 100 * t["err"] / max(t["ref"], 1)
        logged[f"{label}/{key}/false_alarms"] = t["fa"]
        logged[f"{label}/{key}/misses"] = t["miss"]
    print(label, run, "->", path)

import wandb

with wandb.init(entity="1016097967-amphion", project="open-audio-llm", name=os.environ["WANDB_NAME"], job_type="rescore",
                config={"scoring_version": NATIVE_SCORING_VERSION, "amphion_eval": "0.15.0 (db56886)",
                        "sources": dict(a.split("=") for a in sys.argv[2:])}) as run:
    run.log(logged)  # summary.update alone left the remote summary empty (W&B 9631gan0)
    print("wandb", run.url)
