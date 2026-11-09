"""Plain-ASR change of a bias variant vs the baseline, only on utterances whose prediction changed.
Split dev (md5(id) % 10 < 3) / test. Errors: amphion_eval tokenize_text + token Levenshtein (AmphionEval v3 units).
Reference units per dataset come from the baseline's AmphionEval summary (for CER deltas)."""
import glob, hashlib, json, sys, collections
from amphion_eval.normalize import tokenize_text
from rapidfuzz.distance import Levenshtein

def load(model):
    rows, units = {}, collections.Counter()
    for run in sorted(glob.glob(f"/workspace/data/ae-runs/{model}/runs/*")):
        try:
            preds = [json.loads(l) for l in open(run + "/predictions.jsonl")]
        except FileNotFoundError:
            continue
        for p in preds:
            if p.get("task") == "asr" and p.get("status") == "success":
                rows[p["id"]] = p
    return rows

def summary_units(model):
    units = {}
    for path in sorted(glob.glob(f"/workspace/data/ae-runs/{model}/runs/*/summary.json")):
        m = json.load(open(path)).get("metrics") or {}
        if isinstance(m, dict):
            for k, v in m.items():
                units[k] = v.get("reference_units") or 0
    per = collections.Counter()
    for k, n in units.items():
        per[k.split(":")[0]] += n
    return per

def score(p):
    lang = p.get("language") or ""
    ref, hyp = tokenize_text(p["reference"], lang), tokenize_text(p.get("prediction", ""), lang)
    return Levenshtein.distance(ref, hyp), (not ref) and bool(hyp), bool(ref) and not hyp

base, var = load(sys.argv[1]), load(sys.argv[2])
units = summary_units(sys.argv[1])
shared = set(base) & set(var)
agg = collections.defaultdict(lambda: [0] * 8)   # changed, d_err, fa_b, fa_v, drop_b, drop_v, utts, ...
for i in shared:
    ds = base[i]["source"].split(":")[0]
    part = "dev" if int(hashlib.md5(i.encode()).hexdigest(), 16) % 10 < 3 else "test"
    r = agg[(ds, part)]; r[6] += 1
    if base[i].get("prediction", "") == var[i].get("prediction", ""):
        continue
    eb, fab, db = score(base[i]); ev, fav, dv = score(var[i])
    r[0] += 1; r[1] += ev - eb; r[2] += fab; r[3] += fav; r[4] += db; r[5] += dv
result = {}
for part, share in (("dev", 0.3), ("test", 0.7)):
    print(f"\n== {part}  dataset | utts | changed | d_errors (d_CER pts) | FA base->var | drops base->var")
    tot = [0] * 8
    for (ds, pt), r in sorted(agg.items(), key=lambda kv: kv[0]):
        if pt != part: continue
        tot = [x + y for x, y in zip(tot, r)]
        u = units.get(ds, 0) * share
        print(f"{ds[:46]:46s} {r[6]:6d} {r[0]:5d}  {r[1]:+6d} ({100*r[1]/max(u,1):+.3f})  {r[2]:4d}->{r[3]:4d}  {r[4]:4d}->{r[5]:4d}")
        result[f"{part}/{ds}"] = r + [u]
    print(f"{'TOTAL':46s} {tot[6]:6d} {tot[0]:5d}  {tot[1]:+6d}            {tot[2]:4d}->{tot[3]:4d}  {tot[4]:4d}->{tot[5]:4d}")
json.dump(result, open(sys.argv[3], "w"))
