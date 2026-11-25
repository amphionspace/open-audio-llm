"""README "TS 回退的原因" static check: sizes of the weight increments behind the pair merges, and the cosine between the
control and noclean increments (both from 11.06, 3000 steps, data differing only by the 说字清洗 versions and upsampling).
The G / J / H checkpoints have left the model cache, so their increments are read back from the merges (merged - 11.17,
bf16 rounding only); pm-clean itself is written root-only on the node and computed from its inputs instead.
Writes tasks/02-delta-norms/delta-norms.json.
usage: python delta_norms.py   (oal-train env; CPU, reads one tensor at a time)
"""
import json
from pathlib import Path

from safetensors import safe_open

M = "/workspace/data/ae-runs/ctx-grpo-k-20261122-pairmerge/"
P = {"1106": "/workspace/data/ae-runs/amphionasr2-grpo-v10interp050/checkpoint",
     "1117": "/workspace/data/packages/amphionasr-1.7b-v3-v18aes-mtp-candidate/model",
     "control": M + "control-3000", "noclean": M + "noclean-3000",
     "pm-real": M + "1117-plus-g-minus-control", "pm-jg": M + "1117-plus-j-minus-g", "pm-hg": M + "1117-plus-h-minus-g"}
PAIRS = {"control-noclean": ("control", "noclean"), "g-control": ("pm-real", "1117"), "j-g": ("pm-jg", "1117"),
         "h-g": ("pm-hg", "1117"), "control-1106": ("control", "1106"), "noclean-1106": ("noclean", "1106"),
         "1117-1106": ("1117", "1106")}
OUT = Path(__file__).resolve().parents[1] / "tasks/02-delta-norms/delta-norms.json"


def main():
    f = {k: safe_open(v + "/model.safetensors", "pt") for k, v in P.items()}
    keys = [k for k in f["1106"].keys() if all(k in f[m].keys() for m in f)]
    sq = dict.fromkeys(PAIRS, 0.0)
    dot = base = 0.0
    for n in keys:
        t = {m: f[m].get_tensor(n).float() for m in f}
        base += t["1106"].pow(2).sum().item()
        d = {k: t[a] - t[b] for k, (a, b) in PAIRS.items()}
        for k in PAIRS:
            sq[k] += d[k].pow(2).sum().item()
        dot += (d["control-1106"] * d["noclean-1106"]).sum().item()
    res = {"tensors": len(keys), "norm_1106": base ** .5, "norms": {k: v ** .5 for k, v in sq.items()},
           "cos_control_noclean_increments": dot / (sq["control-1106"] * sq["noclean-1106"]) ** .5, "paths": P}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: res[k] for k in ("norms", "cos_control_noclean_increments")}, indent=1))


if __name__ == "__main__":
    main()
