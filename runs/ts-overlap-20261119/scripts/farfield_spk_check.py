"""CAM++ speaker check of the downstream far-field target-absent pack: for each request, the cosine between the
target's enrollment (first 5 s, as sent) and each MOSS turn of the segment; same-speaker reference = turns of the
target's own speech. usage: spk_check.py PACK_DIR OUT_JSON"""
import collections, json, sys
from pathlib import Path
import numpy as np, soundfile as sf, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from spk_embed import SpeakerEmbed
pack, out = Path(sys.argv[1]), sys.argv[2]
m = SpeakerEmbed()
def emb(x):
    return torch.nn.functional.normalize(m(torch.from_numpy(x.astype(np.float32))[None]), dim=-1)[0]
cases = [json.loads(l) for l in open(pack / "cases.jsonl", encoding="utf-8")]
E = {}
for c in cases:
    f = c["target"]["enroll"]
    if f not in E:
        w, sr = sf.read(pack / f, dtype="float32"); assert sr == 16000
        E[f] = emb(w[:80000])
short = lambda s: s[-6:]
own = collections.defaultdict(set)  # speaker -> enroll files used for it
for c in cases: own[short(c["target"]["speaker"])].add(c["target"]["enroll"])
rows = []
for c in cases:
    w, sr = sf.read(pack / c["mix"], dtype="float32"); t0 = c["segment"][0]
    tgt = short(c["target"]["speaker"]); te = E[c["target"]["enroll"]]
    for t in c["moss_turns"]:
        a, b = int(max(0, t["start"] - t0) * sr), int(max(0, t["end"] - t0) * sr)
        if b - a < sr: continue  # < 1 s: too short for a stable embedding
        e = emb(w[a:b]); spk = t["speaker"]
        rows.append({"id": c["id"], "type": c["type"], "target": tgt, "turn_speaker": spk, "is_target": spk == tgt,
                     "dur": round((b - a) / sr, 2), "cos_target_enroll": round(float(e @ te), 3),
                     "cos_own_enroll": round(max(float(e @ E[f]) for f in own[spk]), 3) if spk in own else None})
same = [r["cos_target_enroll"] for r in rows if r["is_target"]]
diff = [r["cos_target_enroll"] for r in rows if not r["is_target"]]
auc = float(np.mean([s > d for s in same for d in diff])) if same and diff else None
res = {"enroll_files": len(E), "speakers": {k: sorted(v) for k, v in own.items()},
       "enroll_cos": {f"{Path(a).stem}~{Path(b).stem}": round(float(E[a] @ E[b]), 3) for a in E for b in E if a < b},
       "turns": len(rows), "same_speaker_cos_mean": round(float(np.mean(same)), 3), "n_same": len(same),
       "other_speaker_cos_mean": round(float(np.mean(diff)), 3), "n_other": len(diff), "auc_same_vs_other": auc}
by = collections.defaultdict(list)
for r in rows:
    if not r["is_target"]: by[r["type"]].append(r)
for k, v in by.items():
    known = [r for r in v if r["cos_own_enroll"] is not None]
    res[f"other_turns/{k}"] = {"n": len(v), "cos_target_enroll_mean": round(float(np.mean([r["cos_target_enroll"] for r in v])), 3),
        "n_registered": len(known), "cos_own_enroll_mean": round(float(np.mean([r["cos_own_enroll"] for r in known])), 3) if known else None,
        "target_closer_than_own": sum(r["cos_target_enroll"] >= r["cos_own_enroll"] for r in known)}
res["rows"] = rows
json.dump(res, open(out, "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in res.items() if k not in ("rows", "enroll_cos")}, ensure_ascii=False, indent=1))
