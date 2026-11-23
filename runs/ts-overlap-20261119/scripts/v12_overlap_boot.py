"""v12 served runs, paired bootstrap (2,000) vs 11.17 by target overlap bin: error (S+D+I, empty = all deleted) on
target-present samples, and false-alarm count on absent ones. usage: v12_boot.py SAMPLES BASE_RUN NAME=RUN ...  (amphion-eval python)"""
import collections, json, random, sys
from amphion_eval.integrations.open_audio_llm import scoring
def edits(r, h):
    prev = list(range(len(h) + 1))
    for i, a in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, b in enumerate(h, 1): cur[j] = min(prev[j-1] + (a != b), prev[j] + 1, cur[j-1] + 1)
        prev = cur
    return prev[-1]
ov = {}
for l in open(sys.argv[1], encoding="utf-8"):
    o = json.loads(l); ov[o["id"]] = ((o.get("record") or {}).get("labels") or {}).get("target_overlap_ratio")
def load(d): return {json.loads(l)["id"]: json.loads(l) for l in open(d + "/predictions.jsonl", encoding="utf-8")}
base = load(sys.argv[2]); arms = [(a.split("=")[0], load(a.split("=")[1])) for a in sys.argv[3:]]
def b(x): return "absent" if x is None else "0" if x == 0 else "<0.6" if x < 0.6 else "0.6-0.9" if x < 0.9 else ">=0.9"
rng = random.Random(0)
for nm, P in arms:
    g = collections.defaultdict(list)
    for i, o in base.items():
        ref = scoring.tokens(o["reference"] or "", "zh")
        if not ref: g["absent"].append((1, int(bool((o.get("prediction") or "").strip())), int(bool((P[i].get("prediction") or "").strip())))); continue
        x = (len(ref), edits(ref, scoring.tokens(o.get("prediction") or "", "zh")), edits(ref, scoring.tokens(P[i].get("prediction") or "", "zh")))
        g[b(ov.get(i))].append(x); g["present"].append(x)
    for k in ("present", "0", "<0.6", "0.6-0.9", ">=0.9", "absent"):
        xs = g[k]; v = []
        for _ in range(2000):
            s = [xs[rng.randrange(len(xs))] for _ in xs]; v.append(100 * sum(c - a for _, a, c in s) / sum(n for n, _, _ in s))
        v.sort(); n = sum(x[0] for x in xs)
        print(f"{nm:8s} {k:8s} n={len(xs):5d} 1117 {100*sum(x[1] for x in xs)/n:6.2f} {nm} {100*sum(x[2] for x in xs)/n:6.2f} diff {100*sum(x[2]-x[1] for x in xs)/n:+6.2f} [{v[50]:+.2f}, {v[1949]:+.2f}]")
