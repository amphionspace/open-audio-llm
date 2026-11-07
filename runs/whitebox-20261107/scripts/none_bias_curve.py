"""Operating curve of a fixed logit bias on ' None' at the language decision (TS-ASR, decision only).
Decision with bias b: None if logprob(None) + b > max(logprob(Chinese), logprob(English)) at the final layer
(lens_logprob[:, -1], which equals the model's logits). FA = target absent & not None; miss = target present & None.
Usage: none_bias_curve.py STATES.npz OUT.json"""
import json
import sys

import numpy as np

z = np.load(sys.argv[1], allow_pickle=True)
keep = z["final_argmax"] == z["emitted_token"]
lp = z["lens_logprob"][keep][:, -1]
margin = lp[:, 0] - np.maximum(lp[:, 1], lp[:, 2])
absent = z["reference_empty"][keep]
rows = []
for b in np.round(np.arange(-3, 3.01, 0.25), 2):
    none = margin + b > 0
    rows.append({"bias": float(b), "fa": int((absent & ~none).sum()), "miss": int((~absent & none).sum()),
                 "fa_rate": float((absent & ~none).mean() / absent.mean()), "miss_rate": float((~absent & none).sum() / (~absent).sum())})
json.dump({"n": int(keep.sum()), "absent": int(absent.sum()), "present": int((~absent).sum()), "rows": rows}, open(sys.argv[2], "w"), indent=1)
for r in rows:
    print(f'bias {r["bias"]:+.2f}  FA {r["fa"]:4}  miss {r["miss"]:4} ({100*r["miss_rate"]:.2f}%)')
