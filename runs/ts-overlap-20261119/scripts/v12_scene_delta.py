"""Per scene type (id prefix) of v12: error (S+D+I, empty = all deleted) on target-present samples and false alarms on
absent ones, 11.17 vs the e2e model, both served runs; plus the same on the 600 safety ids and the 615 heavy ids."""
import collections, json, sys
from amphion_eval.integrations.open_audio_llm import scoring
base, new, safety, heavy = sys.argv[1:5]
def edits(r, h):
    prev = list(range(len(h) + 1))
    for i, a in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, b in enumerate(h, 1): cur[j] = min(prev[j-1] + (a != b), prev[j] + 1, cur[j-1] + 1)
        prev = cur
    return prev[-1]
P = [{json.loads(l)["id"]: json.loads(l) for l in open(f + "/predictions.jsonl", encoding="utf-8")} for f in (base, new)]
sids = {json.loads(l)["id"] for l in open(safety, encoding="utf-8")}; hids = {json.loads(l)["id"] for l in open(heavy, encoding="utf-8")}
agg = collections.defaultdict(lambda: [0, 0, 0, 0, 0])  # n_ref, e_base, e_new, fa_base, fa_new / absent count in n_ref slot for absent rows
for i, o in P[0].items():
    if i not in P[1]: continue
    ref = scoring.tokens(o["reference"] or "", "zh"); hb = scoring.tokens(o.get("prediction") or "", "zh"); hn = scoring.tokens(P[1][i].get("prediction") or "", "zh")
    keys = [i.split(":")[0].rsplit("-", 1)[0] if i.split(":")[0].endswith(("-pos", "-neg")) else i.split(":")[0].split("-")[0], "ALL"]
    keys = [i.split(":")[0].split("-")[0], "ALL"] + (["SAFETY600"] if i in sids else []) + (["HEAVY615"] if i in hids else [])
    for k in keys:
        if ref:
            a = agg[(k, "present")]; a[0] += len(ref); a[1] += edits(ref, hb); a[2] += edits(ref, hn)
        else:
            a = agg[(k, "absent")]; a[0] += 1; a[3] += bool(hb); a[4] += bool(hn)
for (k, t), a in sorted(agg.items(), key=lambda x: (x[0][1], x[0][0])):
    if t == "present": print(f"{k:12s} present chars {a[0]:7d}  err 1117 {100*a[1]/a[0]:6.2f}  e2e {100*a[2]/a[0]:6.2f}  delta {100*(a[2]-a[1])/a[0]:+6.2f}")
    else: print(f"{k:12s} absent  utts  {a[0]:7d}  FA  1117 {a[3]:5d}  e2e {a[4]:5d}")
