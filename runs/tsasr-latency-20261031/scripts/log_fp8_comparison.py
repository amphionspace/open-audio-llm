"""Record the task 07 paired comparisons as one W&B run.

The per-set runs carry their own metrics, but the ASR full set was split into two runs (one of them
ends as failed over a single refused item), and the paired deltas exist only in results/. This run
holds the pooled numbers, so W&B has the full ASR result and every comparison in one place.
"""
import json
import sys
from pathlib import Path

import wandb

RESULTS = Path(__file__).resolve().parent.parent / "tasks/07-fp8-h20-emu/results"
FIELDS = ("items", "changed_predictions", "pooled_bf16", "pooled_fp8", "macro_bf16", "macro_fp8",
          "sources_worse", "sources_better")


def main():
    summary = {}
    for name in ("compare_small_with_noise", "compare_part2", "compare_hot_subset", "compare_asrfull"):
        for pair, row in json.loads((RESULTS / f"{name}.json").read_text()).items():
            summary.update({f"{pair}/{field}": row[field] for field in FIELDS})
            summary[f"{pair}/pooled_delta_lo"], summary[f"{pair}/pooled_delta_hi"] = row["pooled_delta_95"]
    for name, row in json.loads((RESULTS / "v12_compare.json").read_text()).items():
        summary.update({f"v12/{name}/{k}": v for k, v in row.items() if isinstance(v, (int, float))})
    run = wandb.init(entity="1016097967-amphion", project="open-audio-llm", job_type="comparison",
                     name=sys.argv[1], id=sys.argv[1], config={"task": "tsasr-latency-20261031/07-fp8-h20-emu"})
    run.summary.update(summary)
    run.finish()
    print(len(summary), "values")


if __name__ == "__main__":
    main()
