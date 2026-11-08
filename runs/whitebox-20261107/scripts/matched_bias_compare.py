"""Compare a model with a control shifted by a None bias to the same total false alarms, family by family.

Both runs are decision states (``extract_states.py``) on the same v12 samples. The control's decision with bias b is
None iff margin + b > 0 (as ``none_bias_curve.py``); b is chosen so its total FA matches the model's (b = 0). A
model that only moved the threshold then matches the control in every family; a model that separates look-alikes
has fewer FA on hard negatives and/or fewer misses at the same total FA.
Usage: matched_bias_compare.py MODEL.npz CONTROL.npz OUT.json
"""
import json
import sys

import numpy as np

HARD = ("short-hard-neg", "abs-hard", "abs-hard-k1", "realreg3-hard-neg", "enroll-short1-neg", "enroll-contam0-neg",
        "enroll-len2-neg", "enroll-contam10-neg", "realreg3-mid2-neg")


def load(path):
    z = np.load(path, allow_pickle=True)
    lp = z["lens_logprob"][:, -1]
    margin = lp[:, 0] - np.maximum(lp[:, 1], lp[:, 2])
    return {str(i): (m, bool(a), str(s)) for i, m, a, s in zip(z["id"], margin, z["reference_empty"], z["split"])}


def counts(rows, bias):
    out = {"fa": 0, "miss": 0, "hard_fa": 0, "families": {}}
    for margin, absent, split in rows:
        none = margin + bias > 0
        f = out["families"].setdefault(split, {"fa": 0, "miss": 0, "n": 0})
        f["n"] += 1
        if absent and not none:
            out["fa"] += 1
            f["fa"] += 1
            out["hard_fa"] += split in HARD
        if not absent and none:
            out["miss"] += 1
            f["miss"] += 1
    return out


def main():
    model, control = load(sys.argv[1]), load(sys.argv[2])
    shared = sorted(set(model) & set(control))
    m_rows, c_rows = [model[i] for i in shared], [control[i] for i in shared]
    target = counts(m_rows, 0.0)
    grid = np.round(np.arange(-4, 6.001, 0.01), 2)
    bias = min(grid, key=lambda b: abs(counts(c_rows, b)["fa"] - target["fa"]))
    shifted, plain = counts(c_rows, float(bias)), counts(c_rows, 0.0)
    families = {f: {"model_fa": target["families"][f]["fa"], "control_shifted_fa": shifted["families"][f]["fa"],
                    "model_miss": target["families"][f]["miss"], "control_shifted_miss": shifted["families"][f]["miss"],
                    "n": target["families"][f]["n"]} for f in target["families"]}
    worst = sorted(families.items(), key=lambda kv: (kv[1]["model_fa"] - kv[1]["control_shifted_fa"]) +
                   (kv[1]["model_miss"] - kv[1]["control_shifted_miss"]))
    result = {"samples": len(shared), "control_bias": float(bias),
              "model": {k: target[k] for k in ("fa", "miss", "hard_fa")},
              "control_at_bias": {k: shifted[k] for k in ("fa", "miss", "hard_fa")},
              "control_plain": {k: plain[k] for k in ("fa", "miss", "hard_fa")},
              "model_better_families": dict(worst[:10]), "model_worse_families": dict(worst[-10:]), "families": families}
    json.dump(result, open(sys.argv[3], "w"), indent=1)
    print(json.dumps({k: result[k] for k in ("samples", "control_bias", "model", "control_at_bias", "control_plain")}, indent=1))
    print("model better (FA+miss vs shifted control):", {f: (v["model_fa"] - v["control_shifted_fa"], v["model_miss"] - v["control_shifted_miss"]) for f, v in worst[:8]})
    print("model worse:", {f: (v["model_fa"] - v["control_shifted_fa"], v["model_miss"] - v["control_shifted_miss"]) for f, v in worst[-8:]})


if __name__ == "__main__":
    main()
