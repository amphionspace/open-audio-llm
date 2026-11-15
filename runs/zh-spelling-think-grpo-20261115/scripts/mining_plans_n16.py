"""Build direction-split GRPO plans from n=16 grpo_mining lanes (tasks/13-grpo-v6-prep: one n=8 pass kept many
prompts that give no signal in training).

usage: mining_plans_n16.py OUT_DIR RUN_NAME LANE_DIR [LANE_DIR ...]
A prompt is kept only when the minority side of a decision holds at least MIN_MINORITY of the samples, so a fresh
group of 8 is likely to disagree again. Plans, by the direction their signal pushes:
  nonspeech   plain ASR, empty target, some samples write text            -> toward empty output
  miss        plain ASR, text target, some samples write nothing          -> toward writing
  hotword     plain ASR, text target, presence agreed, hotword reward split
  ts-absent   TS-ASR, target speaker absent, some samples write text      -> toward empty output
  ts-present  TS-ASR, target speaker present, some samples write nothing  -> toward writing
              (or presence agreed and the hotword reward split)
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

MIN_MINORITY = 2


def minority(values):
    return len(values) - Counter(values).most_common(1)[0][1]


def plan_of(row):
    outputs = row["outputs"]
    target_empty = not row["solution"].split("<asr_text>", 1)[-1].strip()
    writes = [bool(o.split("<asr_text>", 1)[-1].strip()) for o in outputs]
    presence_split = minority(writes) >= MIN_MINORITY
    hotword = [r["hotword"] for r in row["rewards"] if r["hotword"] is not None]
    hotword_split = len(hotword) == len(outputs) and minority(hotword) >= MIN_MINORITY
    ts = row.get("enroll_seconds") is not None
    if target_empty:
        return ("ts-absent" if ts else "nonspeech") if presence_split else None
    if presence_split:
        return "ts-present" if ts else "miss"
    if hotword_split:
        return "ts-present" if ts else "hotword"
    return None


out, run_name, lanes = Path(sys.argv[1]), sys.argv[2], [Path(p) for p in sys.argv[3:]]
out.mkdir(parents=True, exist_ok=True)
plans = defaultdict(list)
stats = defaultdict(Counter)
seen = set()
for lane in lanes:
    for line in open(lane / "kept.all.jsonl"):
        row = json.loads(line)
        key = (row["source"], tuple(row["index"]))
        if key in seen:
            raise ValueError(f"{key} appears in two lanes: lanes must draw different epochs")
        seen.add(key)
        stats[row["source"]]["scored"] += 1
        kind = plan_of(row)
        if kind:
            stats[row["source"]][kind] += 1
            plans[kind].append({k: row[k] for k in ("source", "index", "record_id")})
for kind, rows in plans.items():
    with open(out / f"plan-{kind}.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
summary = {"lanes": [str(p) for p in lanes], "min_minority": MIN_MINORITY,
           "plans": {k: len(v) for k, v in sorted(plans.items())},
           "sources": {s: dict(c) for s, c in sorted(stats.items())}}
(out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
print(json.dumps(summary["plans"]), flush=True)

import wandb
run = wandb.init(name=run_name, job_type="grpo-mining", config={"lanes": summary["lanes"], "min_minority": MIN_MINORITY})
run.summary.update({f"plan/{k}": v for k, v in summary["plans"].items()})
for source, counts in summary["sources"].items():
    for key, value in counts.items():
        run.summary[f"{source}/{key}"] = value
print("wandb", run.url, flush=True)
run.finish()
