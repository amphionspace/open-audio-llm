"""tasks/21 report: 11.17 vs the end-to-end arms (HF decoding throughout). Heavy-overlap set: error on the mixture and on
the clean version by interferer count; side-effect set: error on target-present groups and false alarms on absent;
paired bootstrap (2,000) against 11.17. usage: e2e_report.py SUBSET_DIR SAFETY_MANIFEST T18_DIR E2E_DIR OUT_JSON ARM [ARM ...]"""
import collections, glob, json, random, sys
from amphion_eval.integrations.open_audio_llm import scoring
sub, saf_manifest, t18, e2e, out, arms = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6:]
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
            o = json.loads(l); d[o["id"]][o.get("cond", o.get("cand"))] = (o["text"] or "").strip()
    return d
rows = {json.loads(l)["id"]: json.loads(l) for l in open(f"{sub}/manifest.jsonl", encoding="utf-8")}; st = json.load(open(f"{sub}/oracle-stats.json"))
srows = {json.loads(l)["id"]: json.loads(l) for l in open(saf_manifest, encoding="utf-8")}
H = {"1117": read(glob.glob(f"{t18}/decode-none-[01].jsonl") + [f"{t18}/decode-none-clean.jsonl"])}
Sd = {"1117": read([f"{e2e}/decode-1117-safety.jsonl"])}
for a in arms:
    H[a] = read(glob.glob(f"{e2e}/decode-{a}-heavy-*.jsonl")); Sd[a] = read([f"{e2e}/decode-{a}-safety.jsonl"])
names = ["1117"] + arms; rng = random.Random(0); res = {}
def table(items, conds, key):
    for grp, xs in items.items():
        n = sum(x["n"] for x in xs); r = {"samples": len(xs)}
        for nm in names:
            for c in conds: r[f"{nm}/{c}"] = round(100 * sum(x["e"][nm][c] for x in xs) / n, 2)
        for nm in arms:
            for c in conds:
                d = []
                for _ in range(2000):
                    s = [xs[rng.randrange(len(xs))] for _ in xs]; d.append(100 * sum(x["e"][nm][c] - x["e"]["1117"][c] for x in s) / sum(x["n"] for x in s))
                d.sort(); r[f"{nm}/{c}_minus_1117"] = [round(r[f"{nm}/{c}"] - r[f"1117/{c}"], 2), round(d[50], 2), round(d[1949], 2)]
        res[f"{key}/{grp}"] = r; print(key, grp, json.dumps(r, ensure_ascii=False))
groups = collections.defaultdict(list)
for i in st:
    ref = scoring.tokens(rows[i]["reference"], "zh")
    x = {"n": len(ref), "e": {nm: {c: edits(ref, scoring.tokens(H[nm][i].get(c, ""), "zh")) for c in ("orig", "clean")} for nm in names}}
    groups["all"].append(x); groups[f"interferers_{rows[i]['n_interferers']}"].append(x)
table(groups, ("orig", "clean"), "heavy")
groups = collections.defaultdict(list); fa = collections.Counter()
for i, r in srows.items():
    ref = scoring.tokens(r["reference"], "zh")
    if not ref:
        fa["n"] += 1
        for nm in names: fa[nm] += bool(Sd[nm][i].get("mix", ""))
        continue
    x = {"n": len(ref), "e": {nm: {"mix": edits(ref, scoring.tokens(Sd[nm][i].get("mix", ""), "zh"))} for nm in names}}
    groups[r["group"]].append(x); groups["present_all"].append(x)
table(groups, ("mix",), "safety")
res["safety/absent_fa"] = {nm: round(100 * fa[nm] / fa["n"], 1) for nm in names}; print("absent FA%", res["safety/absent_fa"])
json.dump(res, open(out, "w"), indent=1)
