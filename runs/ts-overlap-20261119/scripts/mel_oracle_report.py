"""Score the mel-domain oracle decodes next to task 18's STFT-domain rows (vLLM): error by interferer count, paired
bootstrap of melibm / melirm vs melcheck, and the target energy share dominated in mel bands vs STFT bins.
usage: mel_oracle_report.py SUBSET_DIR MEL_GLOB ORACLE_JSONL ASR_JSONL OUT_JSON   (amphion-eval python)"""
import collections, glob, json, random, sys
import numpy as np
from amphion_eval.integrations.open_audio_llm import scoring
sub, mel_glob, orc, asr, out = sys.argv[1:6]
def edits(r, h):
    prev = list(range(len(h) + 1))
    for i, a in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, b in enumerate(h, 1): cur[j] = min(prev[j-1] + (a != b), prev[j] + 1, cur[j-1] + 1)
        prev = cur
    return prev[-1]
rows = {json.loads(l)["id"]: json.loads(l) for l in open(f"{sub}/manifest.jsonl", encoding="utf-8")}
stats = json.load(open(f"{sub}/oracle-stats.json")); T = collections.defaultdict(dict); share = {}
for p in glob.glob(mel_glob):
    for l in open(p, encoding="utf-8"):
        o = json.loads(l); T[o["id"]][o["cand"]] = o["text"] or ""; share[o["id"]] = o["mel_dominated_energy_share"]
for p in (orc, asr):
    for l in open(p, encoding="utf-8"):
        o = json.loads(l)
        if o["cand"] in ("orig", "clean", "ibm", "irm"): T[o["id"]]["stft_" + o["cand"]] = o["text"] or ""
conds = ("stft_orig", "melcheck", "melibm", "melirm", "stft_ibm", "stft_irm", "stft_clean")
items = []
for i in stats:
    if not all(c in T[i] for c in conds): continue
    ref = scoring.tokens(rows[i]["reference"], "zh")
    items.append({"n": len(ref), "k": rows[i]["n_interferers"], "e": {c: edits(ref, scoring.tokens(T[i][c], "zh")) for c in conds}})
rng = random.Random(0); res = {"samples": len(items),
       "dominated_energy_share": {"mel": round(float(np.mean([share[i] for i in stats if i in share])), 4),
                                  "stft": round(float(np.mean([stats[i]["dominated_energy_share"] for i in stats])), 4)}}
for g, xs in [("all", items)] + [(f"interferers_{k}", [x for x in items if x["k"] == k]) for k in (1, 2)]:
    n = sum(x["n"] for x in xs); r = {c: round(100 * sum(x["e"][c] for x in xs) / n, 2) for c in conds}
    for c in ("melibm", "melirm"):
        d = []
        for _ in range(2000):
            s = [xs[rng.randrange(len(xs))] for _ in xs]; d.append(100 * sum(x["e"][c] - x["e"]["melcheck"] for x in s) / sum(x["n"] for x in s))
        d.sort(); r[f"{c}_minus_melcheck"] = [round(r[c] - r["melcheck"], 2), round(d[50], 2), round(d[1949], 2)]
    res[g] = r; print(g, json.dumps(r))
print("dominated energy share", res["dominated_energy_share"])
json.dump(res, open(out, "w"), indent=1)
