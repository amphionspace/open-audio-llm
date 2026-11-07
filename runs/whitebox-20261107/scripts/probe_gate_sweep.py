"""Offline estimate of a probe gate: when the condition-held-out truth probe at one LLM layer gives P(present) < t,
answer empty. Counts false alarms removed and hits turned into misses on ts-asr-benchmark v12 (v16 avg3).
Usage: probe_gate_sweep.py NPZ LAYER OUT.json"""
import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
npz, layer, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
sys.argv = sys.argv[:1] + [npz, "/dev/null"]
import probe_control as pc  # noqa: E402  (reuses its folds and out-of-fold probe)

p = pc.oof(pc.z["residual"][pc.keep][:, layer].astype(np.float32), pc.present.astype(int))
hit, fa = pc.present & ~pc.empty, pc.absent & ~pc.empty
rows = []
for t in (0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01):
    gate = p < t
    rows.append({"threshold": t, "fa_before": int(fa.sum()), "fa_removed": int((fa & gate).sum()),
                 "hits_lost": int((hit & gate).sum()), "hits": int(hit.sum())})
    print(rows[-1])
json.dump({"layer": layer, "rows": rows}, open(out, "w"), indent=1)
