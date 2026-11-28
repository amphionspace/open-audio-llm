"""Score decode_adapter.py outputs on the v12 test subset: (S+D+I) / reference units, empty = all deleted, AmphionEval
zh tokens; by interferer count; paired bootstrap (2,000) against the no-map HF reproduction.
usage: score_decode.py MANIFEST NAME=GLOB [NAME=GLOB ...] (first NAME is the baseline)   (amphion-eval python)"""
import collections, glob, json, random, sys
from amphion_eval.integrations.open_audio_llm import scoring
def edits(r, h):
    prev = list(range(len(h) + 1))
    for i, a in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, b in enumerate(h, 1): cur[j] = min(prev[j-1] + (a != b), prev[j] + 1, cur[j-1] + 1)
        prev = cur
    return prev[-1]
rows = {json.loads(l)["id"]: json.loads(l) for l in open(sys.argv[1], encoding="utf-8")}
runs = {}
for a in sys.argv[2:]:
    name, pat = a.split("=", 1); out = {}
    for f in glob.glob(pat):
        for l in open(f, encoding="utf-8"):
            o = json.loads(l); out[o["id"]] = o["text"]
    runs[name] = out
names = list(runs); base = names[0]
ids = sorted(set.intersection(*[set(v) for v in runs.values()]))
err = {n: {i: edits(scoring.tokens(rows[i]["reference"], "zh"), scoring.tokens(runs[n][i] or "", "zh")) for i in ids} for n in names}
ln = {i: len(scoring.tokens(rows[i]["reference"], "zh")) for i in ids}
rng = random.Random(0); res = {}
for g, sel in [("all", ids)] + [(f"interferers_{k}", [i for i in ids if rows[i]["n_interferers"] == k]) for k in (1, 2)]:
    N = sum(ln[i] for i in sel); res[g] = {"samples": len(sel)}
    for n in names:
        res[g][n] = round(100 * sum(err[n][i] for i in sel) / N, 2)
        if n != base:
            d = []
            for _ in range(2000):
                s = [sel[rng.randrange(len(sel))] for _ in sel]
                d.append(100 * sum(err[n][i] - err[base][i] for i in s) / sum(ln[i] for i in s))
            d.sort(); res[g][f"{n}-{base}"] = [round(res[g][n] - res[g][base], 2), round(d[50], 2), round(d[1949], 2)]
    print(g, json.dumps(res[g]))
json.dump(res, open("score-last.json", "w"), indent=1)
