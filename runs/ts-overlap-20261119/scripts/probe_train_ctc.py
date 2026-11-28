"""CTC probe on frozen 11.17 audio-tower features (one layer): 4-layer Transformer (d 512) + CTC over AmphionEval zh
scoring units, trained on generated v12-like scenes (train speakers) in two conditions, mixture and clean (interferers
removed), 50/50. Picks the step with the lowest held-out error (5% of scenes), then reports the error on the v12 test
subset for every condition: (S+D+I) / reference units, greedy CTC, by interferer count.
usage: train_probe.py LAYER TRAIN_FEAT_DIR TEST_FEAT_DIR TOKENS_JSON OUT_JSON [STEPS]"""
import glob, json, math, random, sys
import numpy as np, torch, torch.nn as nn
layer, trd, ted, tok_path, out_path = sys.argv[1:6]
steps = int(sys.argv[6]) if len(sys.argv) > 6 else 8000
torch.manual_seed(0); random.seed(0)
tok = json.load(open(tok_path)); vocab = {u: i + 1 for i, u in enumerate(tok["vocab"])}

def load(d):
    items, mm = [], {}
    for f in sorted(glob.glob(f"{d}/index-*.json")):
        shard = f.rsplit("-", 1)[1][:-5]; idx = json.load(open(f)); dim = idx["dims"][layer]
        mm[shard] = np.memmap(f"{d}/{layer}-{shard}.f16", dtype=np.float16, mode="r").reshape(-1, dim)
        items += [dict(it, shard=shard) for it in idx["items"]]
    return items, mm, dim
tr_items, tr_mm, dim = load(trd); te_items, te_mm, _ = load(ted)
keys = sorted({it["key"] for it in tr_items}); random.Random(1).shuffle(keys)
dev_keys = set(keys[: len(keys) // 20])
train = [it for it in tr_items if it["key"] not in dev_keys and tok["train"].get(it["key"])]
dev = [it for it in tr_items if it["key"] in dev_keys and tok["train"].get(it["key"])]
def feats(it, mm): return torch.from_numpy(np.asarray(mm[it["shard"]][it["offset"]: it["offset"] + it["length"]], dtype=np.float32))

class Probe(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 512))
        self.pos = nn.Embedding(2048, 512)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(512, 8, 2048, 0.1, batch_first=True, norm_first=True), 4)
        self.head = nn.Linear(512, len(vocab) + 1)
    def forward(self, x, pad):
        h = self.inp(x) + self.pos(torch.arange(x.shape[1], device=x.device))[None]
        return self.head(self.enc(h, src_key_padding_mask=pad))

def batchify(its, mm, labels=True):
    xs = [feats(it, mm) for it in its]; T = max(len(x) for x in xs)
    x = torch.zeros(len(xs), T, dim); pad = torch.ones(len(xs), T, dtype=torch.bool)
    for i, f in enumerate(xs): x[i, :len(f)] = f; pad[i, :len(f)] = False
    return x.cuda(), pad.cuda(), torch.tensor([len(f) for f in xs])

def edits(r, h):
    prev = list(range(len(h) + 1))
    for i, a in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, b in enumerate(h, 1): cur[j] = min(prev[j-1] + (a != b), prev[j] + 1, cur[j-1] + 1)
        prev = cur
    return prev[-1]
inv = {i: u for u, i in vocab.items()}
@torch.no_grad()
def decode(model, its, mm):
    model.eval(); hyps = []
    for b in range(0, len(its), 64):
        x, pad, lens = batchify(its[b:b + 64], mm); ids = model(x, pad).argmax(-1).cpu()
        for row, n in zip(ids, lens):
            prev, h = 0, []
            for t in row[:n].tolist():
                if t != prev and t != 0: h.append(inv[t])
                prev = t
            hyps.append(h)
    model.train(); return hyps

model = Probe().cuda(); opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 1000) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
ctc = nn.CTCLoss(blank=0, zero_infinity=True)
best, best_state, log = 1e9, None, []
for step in range(1, steps + 1):
    its = random.sample(train, 32)
    x, pad, lens = batchify(its, tr_mm)
    tgt = [torch.tensor([vocab[u] for u in tok["train"][it["key"]]]) for it in its]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(x, pad)
    lp = logits.float().log_softmax(-1).transpose(0, 1)
    loss = ctc(lp, torch.cat(tgt), lens, torch.tensor([len(t) for t in tgt]))
    opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
    if step % 1000 == 0 or step == steps:
        hyps = decode(model, dev, tr_mm)
        err = {c: [0, 0] for c in ("mix", "clean")}
        for it, h in zip(dev, hyps):
            r = tok["train"][it["key"]]; err[it["cond"]][0] += edits(r, h); err[it["cond"]][1] += len(r)
        d = {c: round(100 * e / n, 2) for c, (e, n) in err.items()}
        log.append({"step": step, "loss": round(loss.item(), 4), "dev": d}); print(log[-1], flush=True)
        score = (d["mix"] + d["clean"]) / 2
        if score < best: best, best_state = score, {k: v.detach().clone() for k, v in model.state_dict().items()}
model.load_state_dict(best_state)
te = [it for it in te_items if it["key"] in tok["test"]]
hyps = decode(model, te, te_mm)
agg = {}
for it, h in zip(te, hyps):
    t = tok["test"][it["key"]]; r = t["tokens"]
    for g in ("all", f"interferers_{t['n_interferers']}"):
        a = agg.setdefault(g, {}).setdefault(it["cond"], [0, 0]); a[0] += edits(r, h); a[1] += len(r)
res = {"layer": layer, "steps": steps, "train_sequences": len(train), "log": log,
       "test": {g: {c: round(100 * e / n, 2) for c, (e, n) in v.items()} for g, v in agg.items()}}
json.dump(res, open(out_path, "w"), ensure_ascii=False, indent=1); print(json.dumps(res["test"]))
