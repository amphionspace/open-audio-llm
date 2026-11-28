"""tasks/13: the ts-hard GRPO plan, the ts-transcribe plan of tasks/12 restricted to prompts whose samples differ a lot.

tasks/12's ts-transcribe plan kept every TS prompt whose 16 samples split on reaching the group's best accuracy; 53% of
its 4,138 prompts had a mean accuracy >= 0.95 and 48% a best-minus-worst spread < 0.1 (one or two characters), and
GRPO's group-normalised advantage pushes such near-ties as hard as real failures. It moved v12 overlap >= 0.9
transcription by 0.1 point. ts-hard keeps a TS prompt only when the target speaker is present, every sample writes
text, presence and hotword rewards do not split (those prompts go to ts-present, as in grpo_mining_plans.py), and the
accuracy rewards spread by at least MIN_SPREAD.
usage: grpo_hard_plan.py OUT_DIR RUN_NAME TS_LANE_DIR [TS_LANE_DIR ...]   (lanes must draw different epochs)
"""
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

MIN_MINORITY = 2
# 0.3 = a sample missing or mis-writing about a third of the reference against another that gets it; 21.6% of
# tasks/12's ts-transcribe prompts (6.0% of its target-present TS prompts with all samples writing).
MIN_SPREAD = 0.3


def minority(values):
    return len(values) - Counter(values).most_common(1)[0][1]


def hard(row):
    if row.get("enroll_seconds") is None or not row["solution"].split("<asr_text>", 1)[-1].strip():
        return False
    writes = [bool(o.split("<asr_text>", 1)[-1].strip()) for o in row["outputs"]]
    if not all(writes):
        return False
    hotword = [r["hotword"] for r in row["rewards"] if r["hotword"] is not None]
    if len(hotword) == len(row["outputs"]) and minority(hotword) >= MIN_MINORITY:
        return False
    accuracy = [r["accuracy"] for r in row["rewards"]]
    return max(accuracy) - min(accuracy) >= MIN_SPREAD


out, run_name, lanes = Path(sys.argv[1]), sys.argv[2], [Path(p) for p in sys.argv[3:]]
out.mkdir(parents=True, exist_ok=True)
seen, kept, stats, means = set(), [], defaultdict(Counter), []
for lane in lanes:
    for line in open(lane / "kept.all.jsonl"):
        row = json.loads(line)
        key = (row["source"], tuple(row["index"]))
        if key in seen:
            raise ValueError(f"{key} appears in two lanes: lanes must draw different epochs")
        seen.add(key)
        stats[row["source"]]["scored"] += 1
        if hard(row):
            stats[row["source"]]["ts-hard"] += 1
            kept.append({k: row[k] for k in ("source", "index", "record_id")})
            means.append(statistics.mean(r["accuracy"] for r in row["rewards"]))
with open(out / "plan-ts-hard.jsonl", "w") as f:
    for row in kept:
        f.write(json.dumps(row) + "\n")
summary = {"lanes": [str(p) for p in lanes], "min_spread": MIN_SPREAD, "scored": len(seen), "ts-hard": len(kept),
           "mean_accuracy_of_kept": statistics.mean(means) if means else None,
           "sources": {s: dict(c) for s, c in sorted(stats.items())}}
(out / "summary-ts-hard.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
print(json.dumps({k: summary[k] for k in ("scored", "ts-hard", "mean_accuracy_of_kept")}), flush=True)

import wandb

run = wandb.init(name=run_name, job_type="grpo-mining", config={"lanes": summary["lanes"], "min_spread": MIN_SPREAD})
run.summary.update({"plan/ts-hard": len(kept), "scored": len(seen), "mean_accuracy_of_kept": summary["mean_accuracy_of_kept"]})
for source, counts in summary["sources"].items():
    for key, value in counts.items():
        run.summary[f"{source}/{key}"] = value
print("wandb", run.url, flush=True)
run.finish()
