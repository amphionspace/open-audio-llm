"""Re-batch realized samples under alternative sort keys and report token padding.

Batches are cut as packing: sorted does (per slot count, ascending key, at most 12
samples, padded cost within max_duration); padding is measured on realized tokens.
"""
import argparse
import json
from collections import defaultdict

import numpy as np


def family(row):
    if row["online_mix"]:
        return "ts_online"
    if row["dataset_id"] in {"hw_sft", "hard_hw_sft"}:
        return "hotword"
    if row["dataset_id"].startswith("ts_") or "spk" in row["split"] or "neg" in row["split"]:
        return "ts_record"
    return "asr"


def batches(rows, key, max_samples, max_duration):
    out = []
    for slots in sorted({r["slots"] for r in rows}):
        group = sorted((r for r in rows if r["slots"] == slots), key=key)
        batch, cost = [], 0.0
        for row in group:
            if batch and (len(batch) >= max_samples or max(cost, row["cost"]) * (len(batch) + 1) > max_duration):
                out.append(batch)
                batch, cost = [], 0.0
            batch.append(row)
            cost = max(cost, row["cost"])
        if batch:
            out.append(batch)
    return out


def padding(plan, field):
    real = sum(r[field] for b in plan for r in b)
    capacity = sum(len(b) * max(r[field] for r in b) for b in plan)
    return 1 - real / capacity, capacity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", required=True)
    parser.add_argument("--max_samples", type=int, default=12)
    parser.add_argument("--max_duration", type=float, default=480)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in open(args.lengths)]
    errors = [r for r in rows if "error" in r]
    rows = sorted((r for r in rows if "error" not in r), key=lambda r: r["number"])
    for r in rows:
        r["family"] = family(r)
        r["text_tokens"] = r["tokens"] - r["audio_tokens"]
    keys = {
        "current_cost": lambda r: r["cost"],
        "realized_audio_tokens": lambda r: r["audio_tokens"],
        "realized_tokens_oracle": lambda r: r["tokens"],
    }
    report = {"samples": len(rows), "errors": len(errors), "error_examples": errors[:5], "windows": {}}
    for window in (2000, 8000):
        report["windows"][window] = {}
        for name, key in keys.items():
            plan = [b for start in range(0, len(rows) - window + 1, window)
                    for b in batches(rows[start:start + window], key, args.max_samples, args.max_duration)]
            token_padding, capacity = padding(plan, "tokens")
            waste = defaultdict(float)
            for b in plan:
                top = max(r["tokens"] for r in b)
                for r in b:
                    waste[r["family"]] += top - r["tokens"]
            total_waste = sum(waste.values())
            report["windows"][window][name] = {
                "batches": len(plan), "token_padding": token_padding, "token_capacity": capacity,
                "audio_token_padding": padding(plan, "audio_tokens")[0],
                "waste_share_by_family": {k: v / total_waste for k, v in sorted(waste.items())},
            }
    by_family = defaultdict(list)
    for r in rows:
        by_family[r["family"]].append(r)
    report["families"] = {}
    for name, group in sorted(by_family.items()):
        cost = np.array([r["cost"] for r in group])
        audio = np.array([r["audio_tokens"] for r in group])
        text = np.array([r["text_tokens"] for r in group])
        rate = audio / cost  # audio tokens per second of sort key
        report["families"][name] = {
            "share": len(group) / len(rows),
            "audio_tokens_per_key_second_p10_p50_p90": np.percentile(rate, [10, 50, 90]).round(2).tolist(),
            "text_tokens_p10_p50_p90": np.percentile(text, [10, 50, 90]).round(1).tolist(),
            "text_share_of_tokens": float(text.sum() / (text.sum() + audio.sum())),
        }
    print(json.dumps(report, indent=2))
    with open(args.output, "w") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
