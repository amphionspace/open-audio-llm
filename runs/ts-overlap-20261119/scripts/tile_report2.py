"""tasks/24 attempt 002 report. Present samples: error (S+D+I over reference chars, empty = all deleted), whole-utterance
misses, by overlap bin; absent samples: false alarms. Paired bootstrap (2,000) of each fill vs pad (same L) and vs
native5, for the error and for the false-alarm rate. usage: tile_report2.py TILE2_JSONL OUT_JSON   (amphion-eval python)"""
import collections, json, random, sys
from amphion_eval.integrations.open_audio_llm import scoring
def edits(r, h):
    prev = list(range(len(h) + 1))
    for i, a in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, b in enumerate(h, 1): cur[j] = min(prev[j-1] + (a != b), prev[j] + 1, cur[j-1] + 1)
        prev = cur
    return prev[-1]
d = collections.defaultdict(dict); meta = {}
for l in open(sys.argv[1], encoding="utf-8"):
    o = json.loads(l); d[o["id"]][o["cond"]] = (o["text"] or "").strip(); meta[o["id"]] = o
conds = ["native5"] + [f"{k}{L}" for L in (1, 2) for k in ("pad", "noisepad", "tile", "tilexf", "tilegap")]
ids = [i for i in d if all(c in d[i] for c in conds)]
pos = [i for i in ids if scoring.tokens(meta[i]["reference"] or "", "zh")]; neg = [i for i in ids if i not in pos]
N = {i: len(scoring.tokens(meta[i]["reference"], "zh")) for i in pos}
E = {i: {c: edits(scoring.tokens(meta[i]["reference"], "zh"), scoring.tokens(d[i][c], "zh")) for c in conds} for i in pos}
F = {i: {c: int(bool(d[i][c])) for c in conds} for i in neg}
def ob(i):
    ov = meta[i]["overlap"] or 0
    return "0" if ov == 0 else "<0.6" if ov < 0.6 else "0.6-0.9" if ov < 0.9 else ">=0.9"
rng = random.Random(0)
def boot_err(xs, a, b):
    v = []
    for _ in range(2000):
        s = [xs[rng.randrange(len(xs))] for _ in xs]; v.append(100 * sum(E[i][a] - E[i][b] for i in s) / sum(N[i] for i in s))
    v.sort(); return [round(100 * sum(E[i][a] - E[i][b] for i in xs) / sum(N[i] for i in xs), 2), round(v[50], 2), round(v[1949], 2)]
def boot_fa(a, b):
    v = []
    for _ in range(2000):
        s = [neg[rng.randrange(len(neg))] for _ in neg]; v.append(100 * sum(F[i][a] - F[i][b] for i in s) / len(s))
    v.sort(); return [round(100 * sum(F[i][a] - F[i][b] for i in neg) / len(neg), 2), round(v[50], 2), round(v[1949], 2)]
res = {"present": len(pos), "absent": len(neg), "conds": {}}
for c in conds:
    r = {"error": round(100 * sum(E[i][c] for i in pos) / sum(N.values()), 2), "misses": sum(1 for i in pos if not d[i][c]),
         "fa_rate": round(100 * sum(F[i][c] for i in neg) / len(neg), 2)}
    for b in ("0", "<0.6", "0.6-0.9", ">=0.9"):
        xs = [i for i in pos if ob(i) == b]; r[f"error_ov{b}"] = round(100 * sum(E[i][c] for i in xs) / sum(N[i] for i in xs), 2)
    if c != "native5":
        L = c[-1]; r["error_minus_native5"] = boot_err(pos, c, "native5"); r["fa_minus_native5"] = boot_fa(c, "native5")
        if not c.startswith("pad"):
            r["error_minus_pad"] = boot_err(pos, c, f"pad{L}"); r["fa_minus_pad"] = boot_fa(c, f"pad{L}")
    res["conds"][c] = r; print(c, json.dumps(r, ensure_ascii=False))
json.dump(res, open(sys.argv[2], "w"), indent=1, ensure_ascii=False)
