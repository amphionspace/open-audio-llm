"""Voice-matched foreign enrollments for the far-field target-absent cases (CAM++ zh-cn cosine).
For each absent case, the interferer audio is the segment's non-target MOSS turns (the whole segment if they are
under 1 s). From a pool of unrelated speakers (KeSpeech phone, AISHELL-4 far-field from the 11-16 pack) it picks
  match_max  the pool enrollment most similar to the interferer;
  match_eq   the pool enrollment whose similarity is closest to, and not below, the in-meeting target enrollment's,
and writes the requests (5 s enrollment, then the segment) for both. It also splits every in-meeting enrollment into
its source pieces (enroll_ranges) and compares each piece with every speaker's other enrollments (contamination).
usage: voice_match.py PACK A4_ENROLL_DIR OUT_DIR"""
import glob, json, random, sys
from pathlib import Path
import numpy as np, soundfile as sf, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from spk_embed import SpeakerEmbed
pack, a4dir, out = map(Path, sys.argv[1:4])
REF = 80000; m = SpeakerEmbed()
emb = lambda x: torch.nn.functional.normalize(m(torch.from_numpy(np.asarray(x, np.float32))[None]), dim=-1)[0]
K = Path("/workspace/data/datasets/kespeech/source/KeSpeech-f4259034/KeSpeech/Audio")
rng = random.Random(0); spks = sorted(p.name for p in K.iterdir()); rng.shuffle(spks)
pool = []
for s in spks[:800]:
    w, n = [], 0
    for p in sorted(glob.glob(str(K / s / "*" / "*.wav"))):
        x = sf.read(p, dtype="int16")[0]; w.append(x); n += len(x)
        if n >= REF: break
    if n >= REF: pool.append((f"kespeech/{s}", np.concatenate(w)[:REF]))
for p in sorted(a4dir.glob("*.wav")):
    pool.append((f"aishell4/{p.stem}", sf.read(p, dtype="int16")[0][:REF]))
P = torch.stack([emb(x / 32768.0) for _, x in pool]); print("pool", len(pool), flush=True)
cases = [json.loads(l) for l in open(pack / "cases.jsonl", encoding="utf-8")]
rows = []
for c in cases:
    if c["type"] not in ("absent_dup", "absent_other"): continue
    mix = sf.read(pack / c["mix"], dtype="int16")[0]; t0 = c["segment"][0]; tgt = c["target"]["speaker"][-6:]
    parts = [mix[int(max(0, t["start"] - t0) * 16000): int(max(0, t["end"] - t0) * 16000)] for t in c["moss_turns"] if t["speaker"] != tgt]
    inter = np.concatenate(parts) if parts and sum(map(len, parts)) >= 16000 else mix
    ie = emb(inter / 32768.0); te = emb(sf.read(pack / c["target"]["enroll"], dtype="int16")[0][:REF] / 32768.0)
    c_in = float(ie @ te); sims = (P @ ie).numpy()
    i_max = int(sims.argmax()); above = [i for i in np.argsort(sims) if sims[i] >= c_in]
    i_eq = int(above[0]) if above else i_max
    rows.append({"id": c["id"], "type": c["type"], "cos_in_meeting": round(c_in, 3), "max": pool[i_max][0], "cos_max": round(float(sims[i_max]), 3),
                 "eq": pool[i_eq][0], "cos_eq": round(float(sims[i_eq]), 3), "pool_rank_of_in_meeting": int((sims > c_in).sum())})
    for var, i in (("match_max", i_max), ("match_eq", i_eq)):
        (out / var).mkdir(parents=True, exist_ok=True)
        sf.write(out / var / f'{c["id"]}.wav', np.concatenate([pool[i][1], mix]), 16000, subtype="PCM_16")
# contamination: each source piece of each in-meeting enrollment vs every speaker's enrollments
enrolls = {}
for c in cases: enrolls[c["target"]["enroll"]] = (c["target"]["speaker"][-6:], c["target"]["enroll_ranges"])
E = {f: emb(sf.read(pack / f, dtype="int16")[0][:REF] / 32768.0) for f in enrolls}
pieces = []
for f, (spk, ranges) in enrolls.items():
    w = sf.read(pack / f, dtype="int16")[0]; o = 0
    for a, b in ranges:
        n = int(round((b - a) * 16000)); seg = w[o:o + n]; o += n
        if len(seg) < 16000: continue
        e = emb(seg / 32768.0)
        own = max(float(e @ E[g]) for g, (s, _) in enrolls.items() if s == spk and g != f) if sum(s == spk for s, _ in enrolls.values()) > 1 else None
        other = max((float(e @ E[g]), s) for g, (s, _) in enrolls.items() if s != spk)
        pieces.append({"enroll": f, "speaker": spk, "range": [a, b], "cos_own_other_enrolls": None if own is None else round(own, 3),
                       "cos_best_other_speaker": round(other[0], 3), "best_other_speaker": other[1]})
res = {"pool": len(pool), "cases": len(rows),
       "cos_in_meeting_median": float(np.median([r["cos_in_meeting"] for r in rows])),
       "cos_max_median": float(np.median([r["cos_max"] for r in rows])),
       "cos_eq_median": float(np.median([r["cos_eq"] for r in rows])),
       "pool_rank_of_in_meeting_median": float(np.median([r["pool_rank_of_in_meeting"] for r in rows])),
       "rows": rows, "enroll_pieces": pieces}
json.dump(res, open(out / "voice_match.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in res.items() if k not in ("rows", "enroll_pieces")}))
for p in pieces: print(p)
