"""Layer-wise probes on the TS-ASR decision position (Glazer et al. 2508.15882; Aparin et al. 2606.07473).

From states-full.npz (extract_states.py) on ts-asr-benchmark v12 with v16 avg3:
- truth probe: is the target speaker present (reference non-empty)? Out-of-fold predictions let us ask whether
  the model's own states say "absent" on its false alarms (absent, output text) and "present" on its misses;
- behaviour probe on negatives: false alarm vs correctly empty;
- speaker signal in the audio tower: cosine(enrollment mean, mixture mean) per encoder layer, AUC for presence;
- attention mass of the decision position on the enrollment, per LLM layer, by outcome group;
- logit lens: per layer, logprob(' None') - logprob(emitted language) by outcome group.
Probe = L2 logistic regression on standardized features, 5 folds stratified by label, GPU. Usage: probe_analysis.py NPZ OUT.json"""
import json
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

npz, out = sys.argv[1:]
z = np.load(npz, allow_pickle=True)
keep = z["final_argmax"] == z["emitted_token"]  # drop bf16 near-ties (33 of 26,750)
absent = z["reference_empty"][keep]
empty = z["prediction_empty"][keep]
present = ~absent
groups = {"hit": present & ~empty, "miss": present & empty, "fa": absent & ~empty, "reject": absent & empty}
print({k: int(v.sum()) for k, v in groups.items()})
dev = "cuda"
rng = np.random.default_rng(0)


def folds(y, k=5):
    idx = np.arange(len(y))
    f = np.zeros(len(y), int)
    for c in (0, 1):
        sel = rng.permutation(idx[y == c])
        f[sel] = np.arange(len(sel)) % k
    return f


def probe(x, y, k=5, l2=1e-3, steps=300):
    """Out-of-fold P(y=1)."""
    x = torch.as_tensor(x, dtype=torch.float32, device=dev)
    yt = torch.as_tensor(y, dtype=torch.float32, device=dev)
    f = folds(y, k)
    p = np.zeros(len(y))
    for i in range(k):
        tr, te = torch.as_tensor(f != i, device=dev), torch.as_tensor(f == i, device=dev)
        mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-4
        xs = (x - mu) / sd
        w = torch.zeros(x.shape[1], device=dev, requires_grad=True)
        b = torch.zeros((), device=dev, requires_grad=True)
        pos = yt[tr].mean()
        weight = torch.where(yt[tr] > 0, 0.5 / pos, 0.5 / (1 - pos))  # balanced classes
        opt = torch.optim.LBFGS([w, b], max_iter=steps, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            logit = xs[tr] @ w + b
            loss = (weight * torch.nn.functional.binary_cross_entropy_with_logits(logit, yt[tr], reduction="none")).mean() + l2 * (w ** 2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        with torch.no_grad():
            p[f == i] = torch.sigmoid(xs[te] @ w + b).cpu().numpy()
    return p


res = {"counts": {k: int(v.sum()) for k, v in groups.items()}, "layers": {}}
residual = z["residual"][keep]
for name, feats in [("llm", residual), ("audio_mix", z["audio_mix_mean"][keep]), ("audio_sep", z["audio_sep"][keep])]:
    rows = []
    for l in range(feats.shape[1]):
        x = feats[:, l].astype(np.float32)
        p_truth = probe(x, present.astype(int))
        neg = absent
        p_fa = probe(x[neg], (~empty[neg]).astype(int))
        rows.append({
            "layer": l,
            "truth_auc": roc_auc_score(present, p_truth),
            # among false alarms, share the truth probe calls "absent" (p<0.5); same for misses -> "present"
            "fa_called_absent": float((p_truth[groups["fa"]] < 0.5).mean()),
            "miss_called_present": float((p_truth[groups["miss"]] >= 0.5).mean()),
            "reject_called_absent": float((p_truth[groups["reject"]] < 0.5).mean()),
            "hit_called_present": float((p_truth[groups["hit"]] >= 0.5).mean()),
            "fa_vs_reject_auc": roc_auc_score(~empty[neg], p_fa),
        })
        print(name, rows[-1], flush=True)
    res["layers"][name] = rows

# speaker signal: cosine between enrollment and mixture means, per encoder layer
e, m = z["audio_enroll_mean"][keep].astype(np.float32), z["audio_mix_mean"][keep].astype(np.float32)
cos = (e * m).sum(-1) / (np.linalg.norm(e, axis=-1) * np.linalg.norm(m, axis=-1) + 1e-6)
res["speaker_cos"] = [{"layer": l, "presence_auc": roc_auc_score(present, cos[:, l]),
                       **{f"mean_{g}": float(cos[v, l].mean()) for g, v in groups.items()}} for l in range(cos.shape[1])]
att = z["attn_mass"][keep]  # (S, 28, 5): enrollment, sep, mixture, system, other
res["attention"] = [{"layer": l, **{f"enroll_{g}": float(att[v, l, 0].mean()) for g, v in groups.items()},
                     **{f"mix_{g}": float(att[v, l, 2].mean()) for g, v in groups.items()}} for l in range(att.shape[1])]
lens = z["lens_logprob"][keep]  # (S, 29, 3): None, Chinese, English
margin = lens[:, :, 0] - np.maximum(lens[:, :, 1], lens[:, :, 2])
res["lens_none_margin"] = [{"layer": l, **{g: float(np.median(margin[v, l])) for g, v in groups.items()}}
                           for l in range(margin.shape[1])]
json.dump(res, open(out, "w"), indent=1)
print("saved", out)
