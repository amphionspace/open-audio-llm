"""Phase A report: is the target recoverable from the mixture? On samples with exact stems, 11.17 TS error
(S+D+I over ref chars, empty = all deleted) for: orig mixture, clean (interferers removed exactly, noise kept),
IBM and IRM oracle masks on the mixture; by interferer count and by the share of target energy that interferers +
noise dominate. Decomposes the interference cost orig - clean into ibm - clean (left even after ideal time-frequency
selection) and orig - ibm (removed by ideal selection). Paired bootstrap 2,000.
usage: oracle_report.py SUBSET_DIR ASR_JSONL ORACLE_JSONL OUT_JSON   (amphion-eval python)"""
import collections, json, random, sys
from amphion_eval.integrations.open_audio_llm import scoring
sub, asr, orc, out = sys.argv[1:5]
def edits(r, h):
    prev = list(range(len(h) + 1))
    for i, a in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, b in enumerate(h, 1): cur[j] = min(prev[j-1] + (a != b), prev[j] + 1, cur[j-1] + 1)
        prev = cur
    return prev[-1]
rows = {json.loads(l)["id"]: json.loads(l) for l in open(f"{sub}/manifest.jsonl", encoding="utf-8")}
stats = json.load(open(f"{sub}/oracle-stats.json"))
texts = collections.defaultdict(dict)
for path in (asr, orc):
    for l in open(path, encoding="utf-8"):
        o = json.loads(l)
        if o["cand"] in ("orig", "clean", "ibm", "irm"): texts[o["id"]][o["cand"]] = o["text"]
conds = ("orig", "clean", "ibm", "irm")
items = []
for sid, st in stats.items():
    r = rows[sid]; ref = scoring.tokens(r["reference"], "zh")
    e = {c: edits(ref, scoring.tokens(texts[sid].get(c) or "", "zh")) for c in conds}
    items.append({"id": sid, "n": len(ref), "k": r["n_interferers"], "e": e, **st})
dom = sorted(x["dominated_energy_share"] for x in items)
cuts = [dom[len(dom) // 3], dom[2 * len(dom) // 3]]
def group(xs):
    n = sum(x["n"] for x in xs)
    return {"samples": len(xs), "ref_chars": n, **{c: round(100 * sum(x["e"][c] for x in xs) / n, 2) for c in conds},
            "dominated_energy_share": round(sum(x["dominated_energy_share"] for x in xs) / len(xs), 3),
            "glimpse_proportion": round(sum(x["glimpse_proportion"] for x in xs) / len(xs), 3)}
rng = random.Random(0)
def boot(xs, a, b):
    d = []
    for _ in range(2000):
        s = [xs[rng.randrange(len(xs))] for _ in xs]
        d.append(100 * sum(x["e"][a] - x["e"][b] for x in s) / sum(x["n"] for x in s))
    d.sort(); return [round(d[50], 2), round(d[1949], 2)]
res = {"cuts_dominated_energy_share": [round(c, 3) for c in cuts]}
for name, xs in [("all", items)] + [(f"interferers_{k}", [x for x in items if x["k"] == k]) for k in (1, 2)] + \
        [(f"masking_{i}", [x for x in items if (x["dominated_energy_share"] >= ([0] + cuts)[i]) and
                           (i == 2 or x["dominated_energy_share"] < cuts[i])]) for i in range(3)]:
    g = group(xs)
    g["ci_orig_minus_ibm"] = boot(xs, "orig", "ibm"); g["ci_ibm_minus_clean"] = boot(xs, "ibm", "clean")
    hard = [x for x in xs if x["e"]["orig"] / x["n"] >= 0.3]
    g["orig_hard"] = group(hard) if hard else None
    res[name] = g
    print(name, json.dumps({k: v for k, v in g.items() if k != "orig_hard"}))
    if hard: print("   orig>=30%:", json.dumps(g["orig_hard"]))
json.dump(res, open(out, "w"), indent=1)
