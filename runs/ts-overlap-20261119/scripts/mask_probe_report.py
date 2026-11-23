"""tasks/22 report: per probed model, error of its plain decode, its decode with its own predicted mel mask, and 11.17's
decode with that mask, on the 615 heavy-overlap samples by interferer count, with paired bootstrap of masked vs plain
and the dev mask MSE. usage: mask_probe_report.py SUBSET_DIR PROBE_DIR OUT_JSON NAME [NAME ...]   (amphion-eval python)"""
import collections, glob, json, random, sys
from amphion_eval.integrations.open_audio_llm import scoring
sub, pd, out, names = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
def edits(r, h):
    prev = list(range(len(h) + 1))
    for i, a in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, b in enumerate(h, 1): cur[j] = min(prev[j-1] + (a != b), prev[j] + 1, cur[j-1] + 1)
        prev = cur
    return prev[-1]
rows = {json.loads(l)["id"]: json.loads(l) for l in open(f"{sub}/manifest.jsonl", encoding="utf-8")}
rng = random.Random(0); res = {}
for nm in names:
    d = {}
    for f in glob.glob(f"{pd}/{nm}/decode-*.jsonl"):
        for l in open(f, encoding="utf-8"):
            o = json.loads(l); d[o["id"]] = o
    groups = collections.defaultdict(list)
    for i, o in d.items():
        ref = scoring.tokens(rows[i]["reference"], "zh")
        x = {"n": len(ref), **{c: edits(ref, scoring.tokens(o[c] or "", "zh")) for c in ("self_plain", "self_masked", "base_masked")}}
        groups["all"].append(x); groups[f"interferers_{rows[i]['n_interferers']}"].append(x)
    r = {"dev_mask_mse": json.load(open(f"{pd}/{nm}/dev_mse.json"))}
    for g, xs in sorted(groups.items()):
        n = sum(x["n"] for x in xs); q = {"samples": len(xs), **{c: round(100 * sum(x[c] for x in xs) / n, 2) for c in ("self_plain", "self_masked", "base_masked")}}
        for c in ("self_masked", "base_masked"):
            ds = []
            for _ in range(2000):
                s = [xs[rng.randrange(len(xs))] for _ in xs]; ds.append(100 * sum(x[c] - x["self_plain"] for x in s) / sum(x["n"] for x in s))
            ds.sort(); q[f"{c}_minus_self_plain"] = [round(q[c] - q["self_plain"], 2), round(ds[50], 2), round(ds[1949], 2)]
        r[g] = q
    res[nm] = r; print(nm, json.dumps(r, ensure_ascii=False))
json.dump(res, open(out, "w"), indent=1)
