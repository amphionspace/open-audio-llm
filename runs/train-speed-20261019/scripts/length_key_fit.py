"""Out-of-sample check of sort keys the sampler could compute without decoding audio.

Per-source text-token estimates are fitted on the first half of the windows and the keys
are scored on the second half, against the current key and the realized-length oracle.
Updates (world_size x accumulation batches) are formed as packing: sorted does, and their
task mix is compared with the global mix.
"""
import argparse
import json
from collections import Counter, defaultdict

import numpy as np

from length_key_sim import batches, family, padding

AUDIO_TOKENS_PER_SECOND = 13.05  # Measured median for records whose sort key is their audio length.


def updates(plan, key, size):
    cost = lambda b: len(b) * max(key(r) for r in b)
    ordered = sorted(plan, key=cost)
    return [ordered[i:i + size] for i in range(0, len(ordered) - size + 1, size)]


def mixing(groups, share):
    distances, dominated = [], 0
    for group in groups:
        counts = Counter(r["family"] for b in group for r in b)
        total = sum(counts.values())
        local = {k: counts.get(k, 0) / total for k in share}
        distances.append(0.5 * sum(abs(local[k] - share[k]) for k in share))
        dominated += max(local.values()) > 0.9
    return float(np.mean(distances)), dominated / len(groups)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", required=True)
    parser.add_argument("--window", type=int, default=2000)
    parser.add_argument("--update_batches", type=int, default=16)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = sorted((json.loads(line) for line in open(args.lengths)), key=lambda r: r["number"])
    for r in rows:
        r["family"] = family(r)
        r["text_tokens"] = r["tokens"] - r["audio_tokens"]
    half = len(rows) // 2 // args.window * args.window
    fit, test = rows[:half], rows[half:]
    constant = defaultdict(list)
    slope = defaultdict(list)
    for r in fit:
        constant[r["source"]].append(r["text_tokens"])
        slope[r["source"]].append((r["cost"], r["text_tokens"]))
    fallback = float(np.median([r["text_tokens"] for r in fit]))
    text_median = {s: float(np.median(v)) for s, v in constant.items()}
    affine = {}
    for s, pairs in slope.items():
        x, y = np.array(pairs).T
        affine[s] = np.polyfit(x, y, 1) if len(pairs) >= 20 and np.ptp(x) > 0 else (0.0, text_median[s])
    audio = lambda r: AUDIO_TOKENS_PER_SECOND * r["cost"]
    keys = {
        "current_cost": lambda r: r["cost"],
        "audio_plus_source_text_median": lambda r: audio(r) + text_median.get(r["source"], fallback),
        "audio_plus_source_text_affine": lambda r: audio(r) + float(np.polyval(
            affine.get(r["source"], (0.0, fallback)), r["cost"])),
        "realized_audio_plus_source_text_median": lambda r: r["audio_tokens"] + text_median.get(
            r["source"], fallback),
        "realized_tokens_oracle": lambda r: r["tokens"],
    }
    total = Counter(r["family"] for r in test)
    share = {k: v / len(test) for k, v in total.items()}
    report = {"fit_samples": len(fit), "test_samples": len(test), "window": args.window,
              "sources_fitted": len(text_median), "keys": {}}
    for name, key in keys.items():
        plan = [b for start in range(0, len(test), args.window)
                for b in batches(test[start:start + args.window], key, 12, 480)]
        token_padding, capacity = padding(plan, "tokens")
        report["keys"][name] = {"token_padding": token_padding, "token_capacity": capacity,
                                "batches": len(plan)}
        report["keys"][name]["capacity_vs_current"] = capacity / report["keys"]["current_cost"]["token_capacity"]
        window_groups = []
        for start in range(0, len(test), args.window):
            window = test[start:start + args.window]
            window_plan = batches(window, key, 12, 480)
            window_groups += updates(window_plan, key, args.update_batches)
        tv, dominated = mixing(window_groups, share)
        report["keys"][name].update(update_family_tv_mean=tv, updates_one_family_over_90pct=dominated,
                                    updates=len(window_groups))
        print(name, json.dumps(report["keys"][name]), flush=True)
    with open(args.output, "w") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
