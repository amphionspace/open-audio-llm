"""MossFormer2_SS_16K (ClearerVoice-Studio 6b3774d) two-speaker separation of the exported mixtures.
Segmented decoding: 4 s windows, 3 s stride; the 1 s overlap aligns the two outputs' order with the previous window
(correlation) and cross-fades. Samples with 2 interferers are separated again stream by stream (4 more candidates).
Each candidate is rescaled to the mixture's RMS. usage: separate.py SUBSET_DIR CKPT [SHARD NUM_SHARDS]"""
import json, sys, types
from pathlib import Path
import numpy as np, soundfile as sf, torch
sub, ckpt = Path(sys.argv[1]), sys.argv[2]
shard, shards = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (0, 1)
from mossformer2_ss.mossformer2 import MossFormer2_SS_16K
args = types.SimpleNamespace(encoder_embedding_dim=512, mossformer_sequence_dim=512, num_mossformer_layer=24,
                             encoder_kernel_size=16, num_spks=2)
model = MossFormer2_SS_16K(args).model
state = torch.load(ckpt, map_location="cpu"); state = state.get("model", state)
own = model.state_dict(); missing = []
for k in own:
    for cand in (k, k.replace("module.", ""), "module." + k):
        if cand in state and state[cand].shape == own[k].shape:
            own[k] = state[cand]; break
    else:
        missing.append(k)
assert not missing, missing[:5]
model.load_state_dict(own); model = model.cuda().eval()
SR, W, H = 16000, 4 * 16000, 3 * 16000
O = W - H

@torch.no_grad()
def sep(x):
    n = len(x)
    if n <= W:
        pad = np.pad(x, (0, W - n))
        outs = model(torch.from_numpy(pad[None].astype(np.float32)).cuda())
        return [o[0, :n].float().cpu().numpy() for o in outs]
    total = W + int(np.ceil((n - W) / H)) * H
    xp = np.pad(x, (0, total - n))
    acc = np.zeros((2, total), np.float32)
    fade = np.linspace(0, 1, O, dtype=np.float32)
    for start in range(0, total - W + 1, H):
        outs = model(torch.from_numpy(xp[None, start:start + W].astype(np.float32)).cuda())
        seg = np.stack([o[0].float().cpu().numpy() for o in outs])
        if start:
            prev = acc[:, start:start + O]; cur = seg[:, :O]
            same = abs(np.dot(prev[0], cur[0])) + abs(np.dot(prev[1], cur[1]))
            swap = abs(np.dot(prev[0], cur[1])) + abs(np.dot(prev[1], cur[0]))
            if swap > same: seg = seg[::-1]
            acc[:, start:start + O] = prev * (1 - fade) + seg[:, :O] * fade
            acc[:, start + O:start + W] = seg[:, O:]
        else:
            acc[:, :W] = seg
    return [acc[0, :n], acc[1, :n]]

def rms(a): return float(np.sqrt(np.mean(a ** 2)) + 1e-9)
rows = [json.loads(l) for l in open(sub / "manifest.jsonl", encoding="utf-8")][shard::shards]
(sub / "sep").mkdir(exist_ok=True)
for r in rows:
    mix, _ = sf.read(sub / "wav" / f"{r['key']}.mix.wav", dtype="float32")
    level = rms(mix)
    cands = {}
    a, b = sep(mix); cands["s1a"], cands["s1b"] = a, b
    if r["n_interferers"] >= 2:
        for name in ("s1a", "s1b"):
            c, d = sep(cands[name] / rms(cands[name]) * level); cands[name + "-a"], cands[name + "-b"] = c, d
    for name, w in cands.items():
        sf.write(sub / "sep" / f"{r['key']}.{name}.wav", (w / rms(w) * level).astype(np.float32), SR, subtype="FLOAT")
print(f"shard {shard}: {len(rows)} done")
