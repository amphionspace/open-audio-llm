"""ECAPA (the benchmark's speaker model) on the downstream phone-meeting pack: cosine between the registered people's
enrollments (first 5 s), and per case the cosine of the target's enrollment with the profile (mean of MOSS turns >= 2 s
over the whole pack) of each other talker in the segment; the max is the case's impostor_sim, as in the benchmark."""
import collections, json, sys
from pathlib import Path
import numpy as np, soundfile as sf
sys.path.insert(0, "/workspace/workspace/mingdong/repo/adc-meeting-fa/scripts/ts_real")
import voice_check as voice
pack = Path(sys.argv[1])
emb = voice.Embedder(Path("/workspace/workspace/mingdong/miniconda3/envs/k2/bin/python"),
                     Path("/workspace/model/wespeaker-ecapa-tdnn512-LM/voxceleb_ECAPA512_LM.onnx"), threads=8)
cases = [json.loads(l) for l in open(pack / "cases.jsonl", encoding="utf-8")]
short = lambda s: s[-6:]
E = {}
for c in cases:
    f = c["target"]["enroll"]
    if f not in E:
        w, sr = sf.read(pack / f, dtype="float32"); E[f] = (short(c["target"]["speaker"]), emb(w[:80000]))
turns = collections.defaultdict(list)
seen = set()
for c in cases:
    if c["mix"] in seen: continue
    seen.add(c["mix"]); w, sr = sf.read(pack / c["mix"], dtype="float32"); t0 = c["segment"][0]
    for t in c["moss_turns"]:
        a, b = int(max(0, t["start"] - t0) * sr), int(max(0, t["end"] - t0) * sr)
        if b - a >= 2 * sr and t["speaker"]: turns[short(t["speaker"])].append(emb(w[a:b]))
prof = {s: (np.mean(v, 0) / np.linalg.norm(np.mean(v, 0))) for s, v in turns.items()}
print("profiles", {s: len(v) for s, v in turns.items()})
spk = sorted({s for s, _ in E.values()})
by = collections.defaultdict(list)
for f, (s, v) in E.items(): by[s].append(v)
mean = {s: np.mean(v, 0) / np.linalg.norm(np.mean(v, 0)) for s, v in by.items()}
print("enrollment-enrollment cosine (mean over versions)")
for i, a in enumerate(spk):
    print(" ", a, {b: round(float(mean[a] @ mean[b]), 3) for b in spk[i + 1:]})
print("enrollment vs other people's profiles")
for a in spk:
    print(" ", a, {b: round(float(mean[a] @ prof[b]), 3) for b in sorted(prof) if b != a})
rows = []
for c in cases:
    tgt, v = E[c["target"]["enroll"]]
    others = [s for s, sec in (c.get("talk") or {}).items() if short(s) != tgt and sec >= 0.5 and short(s) in prof]
    sim = max((float(v @ prof[short(s)]) for s in others), default=None)
    rows.append((c["type"] if "type" in c else c.get("case_type"), sim))
for k in sorted({r[0] for r in rows}, key=str):
    xs = [s for t, s in rows if t == k and s is not None]
    if xs: print(k, len(xs), "impostor_sim quartiles", [round(float(q), 3) for q in np.percentile(xs, [25, 50, 75])])
emb.close()
