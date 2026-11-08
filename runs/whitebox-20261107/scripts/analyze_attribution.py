"""Tables for grad_attribution.py: rollout signal, and how each gradient part / realised update moves the margin.

m = log p(None) - max language log-prob at the decision; a step helps when it raises m on target-absent
samples (toward empty output) and lowers or keeps it on target-present ones. For a direction u the
first-order change is grad m . u; ``discrimination`` = mean change on hard negatives - mean change on
positives, ``common`` = mean change on positives (the part that only moves the threshold when it has
the same sign as on negatives).
Usage: analyze_attribution.py DIR  (writes DIR/analysis.json and prints the tables)
"""
import json
import sys
from pathlib import Path

import numpy as np


def rollout_table(groups):
    from open_audio_llm.integrations.ms_swift.rewards import qwen3_asr

    rows = {}
    for g in groups:
        r = rows.setdefault(g["kind"], {"prompts": 0, "samples": 0, "writes_text": 0, "signal_groups": 0,
                                        "contrast": 0, "all_correct_presence": 0, "lookalike_interferers": 0})
        speech = bool(qwen3_asr.units(qwen3_asr.transcript(g["solution"])))
        writes = [bool(qwen3_asr.units(qwen3_asr.transcript(o))) for o in g["outputs"]]
        r["prompts"] += 1
        r["samples"] += len(writes)
        r["writes_text"] += sum(writes)
        r["signal_groups"] += float(np.abs(g["advantage"]).max() > 1e-6)
        r["contrast"] += g["contrast"]
        r["all_correct_presence"] += all(w == speech for w in writes)
        r["lookalike_interferers"] += any(i.get("lookalike") for i in g["mix_info"].get("mixture", {}).get("interferers", []))
    return rows


def main():
    out = Path(sys.argv[1])
    data = np.load(out / "attribution.npz")
    dots, comps, groups, role = data["dots"], list(data["components"]), list(data["module_groups"]), data["role"]
    family = data["family"]
    total = dots.sum(-1)  # (samples, components)
    result = {"rollouts": rollout_table([json.loads(line) for line in open(out / "rollouts.jsonl")])}
    print("rollouts:", json.dumps(result["rollouts"], indent=1))

    roles = ("hard-neg", "easy-neg", "pos")
    table = {}
    print(f"\n{'component':34s}" + "".join(f"{r:>11s}" for r in roles) + f"{'discrim':>10s}{'pos/hard':>10s}")
    for c, name in enumerate(comps):
        means = {r: float(total[role == r, c].mean()) for r in roles}
        disc = means["hard-neg"] - means["pos"]
        ratio = means["pos"] / means["hard-neg"] if abs(means["hard-neg"]) > 1e-12 else float("nan")
        table[name] = {**means, "discrimination": disc, "pos_over_hardneg": ratio,
                       "by_module": {g: {r: float(dots[role == r, c, gi].mean()) for r in roles} for gi, g in enumerate(groups)}}
        print(f"{name:34s}" + "".join(f"{means[r]:11.3g}" for r in roles) + f"{disc:10.3g}{ratio:10.2f}")
    result["components"] = table

    print("\nby module (hard-neg / pos) for each component:")
    print(f"{'component':34s}" + "".join(f"{g:>20s}" for g in groups))
    for name in comps:
        bm = table[name]["by_module"]
        print(f"{name:34s}" + "".join(f"{bm[g]['hard-neg']:9.2g}/{bm[g]['pos']:<9.2g} " for g in groups))

    actual = {}
    for key in data.files:
        if key.startswith("margin_") and key != "margin_S":
            name = key[len("margin_"):]
            change = data[key] - data["margin_S"]
            pred = total[:, comps.index(f"delta/{name}")]
            actual[name] = {r: float(change[role == r].mean()) for r in roles}
            actual[name]["first_order_corr"] = float(np.corrcoef(pred, change)[0, 1])
            actual[name]["first_order_slope"] = float(np.polyfit(pred, change, 1)[0])
            actual[name]["by_family"] = {f: float(change[family == f].mean()) for f in sorted(set(family))}
    result["actual_margin_change"] = actual
    print("\nactual margin change S -> checkpoint:", json.dumps({k: {r: v[r] for r in (*roles, "first_order_corr")}
                                                               for k, v in actual.items()}, indent=1))
    (out / "analysis.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
