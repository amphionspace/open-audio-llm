"""Transfer of the presence probe: fit on all of ts-asr-benchmark v12 (one LLM layer, decision position), apply
unchanged to other TS sets, and count what a gate P(present) < t would do there.
Usage: probe_transfer.py TRAIN_NPZ LAYER OUT.json NAME=NPZ..."""
import json
import sys

import numpy as np
import torch

train, layer, out, *targets = sys.argv[1:]
layer = int(layer)
dev = "cuda"


def load(path):
    z = np.load(path, allow_pickle=True)
    keep = z["final_argmax"] == z["emitted_token"]
    return (z["residual"][keep][:, layer].astype(np.float32), ~z["reference_empty"][keep], z["prediction_empty"][keep],
            z["split"][keep])


x, present, _, _ = load(train)
xt = torch.as_tensor(x, device=dev)
yt = torch.as_tensor(present, dtype=torch.float32, device=dev)
mu, sd = xt.mean(0), xt.std(0) + 1e-4
w = torch.zeros(x.shape[1], device=dev, requires_grad=True)
b = torch.zeros((), device=dev, requires_grad=True)
weight = torch.where(yt > 0, 0.5 / yt.mean(), 0.5 / (1 - yt.mean()))
opt = torch.optim.LBFGS([w, b], max_iter=300, line_search_fn="strong_wolfe")


def closure():
    opt.zero_grad()
    loss = (weight * torch.nn.functional.binary_cross_entropy_with_logits(((xt - mu) / sd) @ w + b, yt, reduction="none")).mean() + 1e-3 * (w ** 2).sum()
    loss.backward()
    return loss


opt.step(closure)
res = {"layer": layer, "train": train, "sets": {}}
for spec in targets:
    name, path = spec.split("=", 1)
    x, present, empty, split = load(path)
    with torch.no_grad():
        p = torch.sigmoid(((torch.as_tensor(x, device=dev) - mu) / sd) @ w + b).cpu().numpy()
    hit, fa = present & ~empty, ~present & ~empty
    rows = [{"threshold": t, "fa_removed": int((fa & (p < t)).sum()), "hits_lost": int((hit & (p < t)).sum())}
            for t in (0.2, 0.1, 0.05, 0.02)]
    res["sets"][name] = {"n": len(p), "present": int(present.sum()), "fa": int(fa.sum()), "hits": int(hit.sum()),
                         "miss": int((present & empty).sum()), "gate": rows}
    print(name, res["sets"][name], flush=True)
json.dump(res, open(out, "w"), indent=1)
