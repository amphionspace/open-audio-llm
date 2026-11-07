"""Weight-space view of what each training stage changed (Mukherjee et al. 2025, arXiv 2505.11711; 2606.18521).

Per tensor: relative update norm |dW|/|W|, fraction of bf16 entries that changed at all, cosine between two
stages' updates; per sampled matrix: rank holding 90% of the update's energy. Usage:
weight_delta.py OUT.json BASE_A TRAINED_A BASE_B TRAINED_B [SFT_BASE SFT_TRAINED]
(stage A, stage B, optional SFT stage; each a model.safetensors)."""
import json
import re
import sys

import torch
from safetensors import safe_open

out, a0, a1, b0, b1, *sft = sys.argv[1:]
files = {k: safe_open(p, "pt") for k, p in zip(("a0", "a1", "b0", "b1", "s0", "s1"), [a0, a1, b0, b1, *sft])}
SAMPLE = re.compile(r"\.layers\.(0|13|27)\.(self_attn\.q_proj|self_attn\.o_proj|mlp\.down_proj)\.weight$")


def energy_rank(d, frac=0.9):
    s = torch.linalg.svdvals(d.float())
    e = (s ** 2).cumsum(0) / (s ** 2).sum()
    return int((e < frac).sum()) + 1, min(d.shape)


rows = []
for name in files["a0"].keys():
    t = {k: f.get_tensor(name) for k, f in files.items()}
    if not t["a0"].is_floating_point():
        continue
    da, db = t["a1"].float() - t["a0"].float(), t["b1"].float() - t["b0"].float()
    w = t["a0"].float().norm().item() or 1.0
    row = {"name": name, "numel": t["a0"].numel(),
           "rel_a": da.norm().item() / w, "rel_b": db.norm().item() / w,
           "changed_a": (t["a1"] != t["a0"]).float().mean().item(),
           "changed_b": (t["b1"] != t["b0"]).float().mean().item(),
           "cos_ab": torch.nn.functional.cosine_similarity(da.flatten(), db.flatten(), dim=0).item()}
    if "s0" in t:
        ds = t["s1"].float() - t["s0"].float()
        row["rel_s"] = ds.norm().item() / w
        row["cos_sa"] = torch.nn.functional.cosine_similarity(ds.flatten(), da.flatten(), dim=0).item()
    if SAMPLE.search(name) and da.dim() == 2:
        row["rank90_a"], row["full_rank"] = energy_rank(da)
        row["rank90_b"], _ = energy_rank(db)
    rows.append(row)
json.dump(rows, open(out, "w"), indent=0)
print(len(rows), "tensors")
