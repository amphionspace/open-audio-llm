"""Speaker check of the fills: frozen CAM++ (zh-cn) cosine between each filled 5 s enrollment and the native 5 s
enrollment of the same sample, and the AUC of same-sample vs other-sample cosines (does the fill keep who is speaking).
usage: fill_speaker_sim.py WAV_DIR OUT_JSON   (oal-train python)"""
import collections, glob, json, random, sys
from pathlib import Path
import numpy as np, soundfile as sf, torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tse"))
from spk_embed import SpeakerEmbed
wav, out = Path(sys.argv[1]), sys.argv[2]
m = SpeakerEmbed().cuda(); emb = collections.defaultdict(dict)
for p in sorted(glob.glob(str(wav / "*.wav"))):
    key, cond = Path(p).name[:-4].rsplit(".", 1)
    with torch.no_grad():
        emb[key][cond] = torch.nn.functional.normalize(m(torch.from_numpy(sf.read(p, dtype="float32")[0])[None].cuda()), dim=-1)[0].cpu()
keys = sorted(k for k in emb if "native5" in emb[k]); conds = sorted({c for k in keys for c in emb[k]} - {"native5"})
rng = random.Random(0); res = {}
for c in conds:
    same = [float(emb[k][c] @ emb[k]["native5"]) for k in keys if c in emb[k]]
    other = [float(emb[k][c] @ emb[keys[rng.randrange(len(keys))]]["native5"]) for k in keys if c in emb[k]]
    auc = float(np.mean([s > o for s, o in zip(same, rng.sample(other, len(other)))]))
    res[c] = {"cos_to_native5_mean": round(float(np.mean(same)), 4), "cos_other_speaker_mean": round(float(np.mean(other)), 4), "auc_pairs": round(auc, 4), "n": len(same)}
    print(c, res[c], flush=True)
json.dump(res, open(out, "w"), indent=1)
