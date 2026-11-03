"""Check W&B runs on the server (entity/project of this experiment) and write the evidence file.

Usage: wandb_verify.py OUTPUT.json NAME_OR_ID...  Exits non-zero unless every run is finished with metrics.
"""
import json
import sys
from pathlib import Path

import wandb

PATH = "1016097967-amphion/open-audio-llm"
api = wandb.Api(timeout=60)
rows, failed = [], False
for key in sys.argv[2:]:
    runs = [api.run(f"{PATH}/{key}")] if not key.startswith("mtp-") else list(api.runs(PATH, filters={"display_name": key}))
    if not runs:
        rows.append({"name": key, "state": "missing"})
        failed = True
    for run in runs:
        metrics = [k for k in run.summary.keys() if not k.startswith("_")]
        rows.append({"id": run.id, "name": run.name, "state": run.state, "url": run.url, "summary_keys": len(metrics),
                     "last_step": run.summary.get("_step")})
        failed |= run.state != "finished" or not metrics
Path(sys.argv[1]).write_text(json.dumps(rows, indent=1, ensure_ascii=False))
for row in rows:
    print(row)
sys.exit(1 if failed else 0)
