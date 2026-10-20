"""Backfill a hotword-bench run's metrics to its W&B run.

The 34479 v3 run was scored against the first v3 build, where ContextASR's
3,957 free-form domains were a breakdown label; its summary (96k keys, 12 MB)
passed W&B's 10 MB row limit and after-training's run.log() was dropped. This
logs the same keys with the same prefix (after_training._numeric_metrics)
without that breakdown.

    python backfill_wandb.py <run dir> <run id>
"""

import json
import re
import sys

import wandb
from amphion_eval.integrations.open_audio_llm.after_training import _numeric_metrics

FREE_FORM_DOMAIN = re.compile(
    r"^evaluation/hotword_bench/breakdowns/contextasr-speech-zh/[^/]+/domain/"
)


def main(run_dir, run_id):
    with open(f"{run_dir}/summary.json") as stream:
        metrics = _numeric_metrics(json.load(stream))
    kept = {k: v for k, v in metrics.items() if not FREE_FORM_DOMAIN.match(k)}
    print(f"{len(metrics)} keys, {len(kept)} kept, {len(json.dumps(kept))} bytes")
    run = wandb.init(
        entity="1016097967-amphion",
        project="open-audio-llm",
        id=run_id,
        resume="must",
        dir=run_dir,
    )
    run.log(kept)
    run.summary["evaluation_state"] = "complete"
    run.summary["evaluation_note"] = (
        "metrics backfilled from summary.json; ContextASR free-form domain "
        "breakdown omitted (W&B 10 MB limit)"
    )
    run.finish()


if __name__ == "__main__":
    main(*sys.argv[1:3])
