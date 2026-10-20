"""Compare two training attempts that consumed the same batches, over a step window.

Checks first that each logging interval saw the same samples and sources, then reports
seconds per update, loss alignment, token padding, peak memory and GPU utilization.
"""
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def performance(attempt):
    rows = defaultdict(dict)
    for path in sorted((attempt / "artifacts/training").glob("performance-rank*.jsonl")):
        for line in path.open():
            row = json.loads(line)
            rows[row["step"]][row["rank"]] = row
    return rows


def losses(attempt):
    out = {}
    for line in (attempt / "artifacts/training/logging.jsonl").open():
        row = json.loads(line)
        step = row.get("global_step/max_steps", "").split("/")[0] if "global_step/max_steps" in row else row.get("step")
        if "loss" in row and step not in (None, ""):
            out[int(step)] = row["loss"]
    return out


def gpu_util(attempt, start, end):
    path = attempt / "artifacts/training/gpu.csv"
    values = []
    with path.open() as stream:
        for row in csv.reader(stream):
            if len(row) > 3 and row[3].strip().isdigit():
                values.append(int(row[3]))
    # gpu.csv has no step column; the middle half of the samples avoids startup and teardown.
    return statistics.fmean(values[len(values) // 4: 3 * len(values) // 4]) if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--start", type=int, default=101)
    parser.add_argument("--end", type=int, default=300)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    base, cand = performance(args.baseline), performance(args.candidate)
    steps = sorted(s for s in set(base) & set(cand) if args.start <= s <= args.end)
    same_batches = all(
        sum(r["samples"] for r in base[s].values()) == sum(r["samples"] for r in cand[s].values())
        and {k: sum(r["sources"].get(k, 0) for r in base[s].values()) for k in
             {k for r in base[s].values() for k in r["sources"]}}
        == {k: sum(r["sources"].get(k, 0) for r in cand[s].values()) for k in
            {k for r in cand[s].values() for k in r["sources"]}}
        for s in steps)
    seconds = lambda rows: [rows[s][0]["interval_wall_s"] / rows[s][0]["updates"] for s in steps]
    padding = lambda rows: 1 - sum(r["tokens"] for s in steps for r in rows[s].values()) / sum(
        r["tokens_capacity"] for s in steps for r in rows[s].values())
    peak = lambda rows: max(r["cuda_peak_allocated_gib"] for s in steps for r in rows[s].values())
    base_loss, cand_loss = losses(args.baseline), losses(args.candidate)
    loss_steps = sorted(s for s in set(base_loss) & set(cand_loss) if args.start <= s <= args.end)
    relative = [(cand_loss[s] - base_loss[s]) / base_loss[s] for s in loss_steps]
    base_s, cand_s = seconds(base), seconds(cand)
    report = {
        "window": [args.start, args.end], "intervals": len(steps), "same_batches": same_batches,
        "seconds_per_update": {"baseline_mean": statistics.fmean(base_s), "candidate_mean": statistics.fmean(cand_s),
                               "baseline_median": statistics.median(base_s),
                               "candidate_median": statistics.median(cand_s)},
        "speedup_mean": statistics.fmean(base_s) / statistics.fmean(cand_s),
        "speedup_median": statistics.median(base_s) / statistics.median(cand_s),
        "token_padding": {"baseline": padding(base), "candidate": padding(cand)},
        "peak_allocated_gib": {"baseline": peak(base), "candidate": peak(cand)},
        "gpu_util_mid_run": {"baseline": gpu_util(args.baseline, args.start, args.end),
                             "candidate": gpu_util(args.candidate, args.start, args.end)},
        "loss": {"steps": len(loss_steps),
                 "mean_baseline": statistics.fmean(base_loss[s] for s in loss_steps) if loss_steps else None,
                 "mean_candidate": statistics.fmean(cand_loss[s] for s in loss_steps) if loss_steps else None,
                 "relative_diff_mean": statistics.fmean(relative) if relative else None,
                 "relative_diff_abs_max": max(map(abs, relative)) if relative else None},
    }
    print(json.dumps(report, indent=2))
    with open(args.output, "w") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
