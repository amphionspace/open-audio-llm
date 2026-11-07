"""Control for probe_analysis.py: folds by acoustic condition, so a condition's present and absent items are always
tested together and a probe cannot win by recognising the condition. Condition = split name without -pos / -neg /
-enrollmix. Conditions with only one class (e.g. abs-*, dyn-*) still count; whole conditions are held out.
Usage: probe_control.py NPZ OUT.json"""
import json
import re
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.argv, (npz, out) = sys.argv[:1], sys.argv[1:]
z = np.load(npz, allow_pickle=True)
keep = z["final_argmax"] == z["emitted_token"]
absent, empty = z["reference_empty"][keep], z["prediction_empty"][keep]
present = ~absent
cond = np.array([re.sub(r"-(pos|neg)(-enrollmix)?$", "", s) for s in z["split"][keep]])
names = sorted(set(cond))
rng = np.random.default_rng(0)
fold_of = {c: i % 5 for i, c in enumerate(rng.permutation(names))}
fold = np.array([fold_of[c] for c in cond])
paired = np.array([((cond == c) & present).any() and ((cond == c) & absent).any() for c in cond])
dev = "cuda"


def oof(x, y):
    x = torch.as_tensor(x, dtype=torch.float32, device=dev)
    yt = torch.as_tensor(y, dtype=torch.float32, device=dev)
    p = np.zeros(len(y))
    for i in range(5):
        tr, te = torch.as_tensor(fold != i, device=dev), torch.as_tensor(fold == i, device=dev)
        mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-4
        xs = (x - mu) / sd
        w = torch.zeros(x.shape[1], device=dev, requires_grad=True)
        b = torch.zeros((), device=dev, requires_grad=True)
        pos = yt[tr].mean()
        weight = torch.where(yt[tr] > 0, 0.5 / pos, 0.5 / (1 - pos))
        opt = torch.optim.LBFGS([w, b], max_iter=300, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = (weight * torch.nn.functional.binary_cross_entropy_with_logits(xs[tr] @ w + b, yt[tr], reduction="none")).mean() + 1e-3 * (w ** 2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        with torch.no_grad():
            p[fold == i] = torch.sigmoid(xs[te] @ w + b).cpu().numpy()
    return p


fa, miss = absent & ~empty, present & empty
rows = []
for name, key, layers in [("llm", "residual", [4, 8, 12, 16, 20, 24, 28]), ("audio_mix", "audio_mix_mean", [12, 16, 24])]:
    feats = z[key][keep]
    for l in layers:
        p = oof(feats[:, l].astype(np.float32), present.astype(int))
        rows.append({"features": name, "layer": l, "truth_auc_heldout_conditions": roc_auc_score(present, p),
                     "truth_auc_paired_conditions_only": roc_auc_score(present[paired], p[paired]),
                     "fa_called_absent": float((p[fa] < 0.5).mean()), "miss_called_present": float((p[miss] >= 0.5).mean()),
                     "fa_called_absent_paired": float((p[fa & paired] < 0.5).mean())})
        print(rows[-1], flush=True)
json.dump({"conditions": len(names), "paired_items": int(paired.sum()), "fa_paired": int((fa & paired).sum()), "rows": rows},
          open(out, "w"), indent=1)
