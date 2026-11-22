"""De-interference map at one 11.17 audio-tower layer: from the layer's features on [enrollment | mixture] predict the
same layer's features on [enrollment | clean] (target + noise, interferers removed; same frames), trained on the
generated v12-like scenes. Residual 4-layer Transformer (d 768) with a zero-initialised output, so it starts as the
identity. Loss: MSE on features standardised by the clean features' per-dimension mean / std, over all frames.
Reports held-out (5% of scenes) MSE of the identity vs the map every 1,000 steps and keeps the best.
usage: train_adapter.py LAYER TRAIN_FEAT_DIR OUT_PT [STEPS]"""
import glob, json, math, random, sys
import numpy as np, torch, torch.nn as nn
layer, trd, out_pt = sys.argv[1:4]
steps = int(sys.argv[4]) if len(sys.argv) > 4 else 6000
torch.manual_seed(0); random.seed(0)
items, mm = [], {}
for f in sorted(glob.glob(f"{trd}/index-*.json")):
    shard = f.rsplit("-", 1)[1][:-5]; idx = json.load(open(f)); dim = idx["dims"][layer]
    mm[shard] = np.memmap(f"{trd}/{layer}-{shard}.f16", dtype=np.float16, mode="r").reshape(-1, dim)
    items += [dict(it, shard=shard) for it in idx["items"]]
pairs = {}
for it in items: pairs.setdefault(it["key"], {})[it["cond"]] = it
keys = sorted(k for k, v in pairs.items() if "mix" in v and "clean" in v and v["mix"]["length"] == v["clean"]["length"])
random.Random(1).shuffle(keys); dev, train = keys[: len(keys) // 20], keys[len(keys) // 20:]
def arr(it): return torch.from_numpy(np.asarray(mm[it["shard"]][it["offset"]: it["offset"] + it["length"]], dtype=np.float32))
sample = torch.cat([arr(pairs[k]["clean"]) for k in train[:500]])
mean, std = sample.mean(0).cuda(), sample.std(0).clamp_min(1e-3).cuda()

class Adapter(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim)); self.register_buffer("std", torch.ones(dim))
        self.inp = nn.Linear(dim, 768); self.pos = nn.Embedding(4096, 768)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(768, 12, 3072, 0.1, batch_first=True, norm_first=True), 4)
        self.norm = nn.LayerNorm(768); self.out = nn.Linear(768, dim)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
    def forward(self, x, pad=None):  # x: (B, T, dim) raw features -> raw features
        z = (x - self.mean) / self.std
        h = self.inp(z) + self.pos(torch.arange(x.shape[1], device=x.device))[None]
        return x + self.out(self.norm(self.enc(h, src_key_padding_mask=pad))) * self.std

def batch(ks):
    xs = [arr(pairs[k]["mix"]) for k in ks]; ys = [arr(pairs[k]["clean"]) for k in ks]; T = max(len(x) for x in xs)
    x = torch.zeros(len(ks), T, dim); y = torch.zeros(len(ks), T, dim); pad = torch.ones(len(ks), T, dtype=torch.bool)
    for i, (a, b) in enumerate(zip(xs, ys)): x[i, :len(a)] = a; y[i, :len(b)] = b; pad[i, :len(a)] = False
    return x.cuda(), y.cuda(), pad.cuda()
def mse(pred, y, pad):
    d = ((pred - y) / std) ** 2; m = (~pad).unsqueeze(-1).float()
    return (d * m).sum() / (m.sum() * dim)
model = Adapter(dim).cuda(); model.mean.copy_(mean); model.std.copy_(std)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 500) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
@torch.no_grad()
def evaluate():
    model.eval(); a = b = 0.0; n = 0
    for i in range(0, len(dev), 32):
        x, y, pad = batch(dev[i:i + 32])
        with torch.autocast("cuda", dtype=torch.bfloat16): p = model(x, pad).float()
        a += mse(x, y, pad).item(); b += mse(p, y, pad).item(); n += 1
    model.train(); return round(a / n, 4), round(b / n, 4)
best, log = 1e9, []
for step in range(1, steps + 1):
    x, y, pad = batch(random.sample(train, 32))
    with torch.autocast("cuda", dtype=torch.bfloat16): p = model(x, pad).float()
    loss = mse(p, y, pad); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
    if step % 1000 == 0 or step == steps:
        ident, mapped = evaluate(); log.append({"step": step, "loss": round(loss.item(), 4), "dev_identity_mse": ident, "dev_map_mse": mapped})
        print(log[-1], flush=True)
        if mapped < best: best = mapped; torch.save({"layer": layer, "dim": dim, "state": model.state_dict(), "log": log}, out_pt)
print("best dev mse", best)
