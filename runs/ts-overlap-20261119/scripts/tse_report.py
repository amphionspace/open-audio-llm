"""tasks/19 report. Heavy-overlap set (615, stems): 11.17 TS error on the mixture vs the extracted target, by interferer
count, paired bootstrap; with the oracle-mask rows of task 18. Side-effect set: error on target-present samples
without / with light overlap, and false-alarm rate (non-empty output) on target-absent samples, mixture vs extracted.
usage: tse_report.py SUBSET_DIR ASR_HEAVY ORACLE_JSONL SAFETY_DIR ASR_SAFETY SISDR_JSON OUT_JSON   (amphion-eval python)"""
import collections, json, random, sys
import numpy as np
from amphion_eval.integrations.open_audio_llm import scoring
sub, asr_heavy, orc, saf, asr_saf, sis, out = sys.argv[1:8]
def edits(r, h):
    prev = list(range(len(h) + 1))
    for i, a in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, b in enumerate(h, 1): cur[j] = min(prev[j-1] + (a != b), prev[j] + 1, cur[j-1] + 1)
        prev = cur
    return prev[-1]
def read(paths):
    d = collections.defaultdict(dict)
    for p in paths:
        for l in open(p, encoding="utf-8"):
            o = json.loads(l); d[o["id"]][o["cand"]] = o["text"] or ""
            if o["cand"] == "mix": d[o["id"]]["orig"] = o["text"] or ""  # the mixture decoded in the same pass
    return d
rng = random.Random(0)
def boot(items, a, b):
    ds = []
    for _ in range(2000):
        s = [items[rng.randrange(len(items))] for _ in items]
        ds.append(100 * sum(x["e"][a] - x["e"][b] for x in s) / sum(x["n"] for x in s))
    ds.sort(); return [round(ds[50], 2), round(ds[1949], 2)]
res = {}
rows = {json.loads(l)["id"]: json.loads(l) for l in open(f"{sub}/manifest.jsonl", encoding="utf-8")}
stats = json.load(open(f"{sub}/oracle-stats.json")); H = read([orc, asr_heavy])
items = []
for sid in stats:
    ref = scoring.tokens(rows[sid]["reference"], "zh"); h = H[sid]
    items.append({"n": len(ref), "k": rows[sid]["n_interferers"],
                  "e": {c: edits(ref, scoring.tokens(h[c], "zh")) for c in ("orig", "tse", "remix", "clean", "ibm", "irm")}})
for g, xs in [("all", items)] + [(f"interferers_{k}", [x for x in items if x["k"] == k]) for k in (1, 2)]:
    n = sum(x["n"] for x in xs)
    res[f"heavy/{g}"] = {**{c: round(100 * sum(x["e"][c] for x in xs) / n, 2) for c in xs[0]["e"]}, "samples": len(xs),
                         **{f"{c}_minus_orig": [round(100 * sum(x["e"][c] - x["e"]["orig"] for x in xs) / n, 2), *boot(xs, c, "orig")] for c in ("tse", "remix")}}
    print(g, json.dumps(res[f"heavy/{g}"]))
sd = json.load(open(sis))["sisdr"]
res["heavy/sisdr"] = {"mixture": round(float(np.mean([v["mix"] for v in sd.values()])), 2), "extracted": round(float(np.mean([v["tse"] for v in sd.values()])), 2)}
print("SI-SDR", res["heavy/sisdr"])
srows = {json.loads(l)["id"]: json.loads(l) for l in open(f"{saf}/manifest.jsonl", encoding="utf-8")}; Sd = read([asr_saf])
for g in ("present_ov0", "present_ov_lt0.6", "absent"):
    ids = [i for i, r in srows.items() if r["group"] == g and i in Sd]
    if g == "absent":
        res[f"safety/{g}"] = {"samples": len(ids), **{c: round(100 * sum(bool(Sd[i][c].strip()) for i in ids) / len(ids), 2) for c in ("mix", "tse", "remix")}}
    else:
        xs = []
        for i in ids:
            ref = scoring.tokens(srows[i]["reference"], "zh")
            xs.append({"n": len(ref), "e": {c: edits(ref, scoring.tokens(Sd[i][c], "zh")) for c in ("mix", "tse", "remix")}})
        n = sum(x["n"] for x in xs)
        res[f"safety/{g}"] = {"samples": len(xs), **{c: round(100 * sum(x["e"][c] for x in xs) / n, 2) for c in ("mix", "tse", "remix")},
                              **{f"{c}_minus_mix": [round(100 * sum(x["e"][c] - x["e"]["mix"] for x in xs) / n, 2), *boot(xs, c, "mix")] for c in ("tse", "remix")}}
    print(g, json.dumps(res[f"safety/{g}"]))
json.dump(res, open(out, "w"), indent=1)
