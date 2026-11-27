"""Presence-decision flips of the delivery TS suites (task READMEs 06/09): for each suite's compare JSON
(runs/zh-ctx-grpo-20261120/tasks/04-delivery/compare/<A>-vs-<B>/cmp-<suite>.json), the same samples of run_a and run_b
compared on empty / non-empty prediction. Target absent = empty reference: new text (FA) / new empty;
target present = non-empty reference: new empty (miss) / new text.
A>B (two candidates, no compare of their own) takes run_b of r1117-vs-A and r1117-vs-B.
usage: presence_flips.py A-vs-B|A>B[,...] SUITE [SUITE ...]"""
import json, sys
G = "/workspace/workspace/mingdong/repo/oal-ctx-grpo/runs/zh-ctx-grpo-20261120/tasks/04-delivery/compare"
def preds(run):
    out = {}
    for l in open(run + "/predictions.jsonl"):
        r = json.loads(l); out[r["id"]] = (bool((r.get("reference") or "").strip()), bool((r.get("prediction") or "").strip()))
    return out
for pair in sys.argv[1].split(","):
    print("==", pair)
    for suite in sys.argv[2:]:
        if ">" in pair:
            ra, rb = (json.load(open(f"{G}/r1117-vs-{m}/cmp-{suite}.json"))["run_b"] for m in pair.split(">"))
        else:
            c = json.load(open(f"{G}/{pair}/cmp-{suite}.json")); ra, rb = c["run_a"], c["run_b"]
        a, b = preds(ra), preds(rb)
        ids = a.keys() & b.keys(); assert len(ids) == len(a) == len(b), (suite, len(a), len(b), len(ids))
        absent = [i for i in ids if not a[i][0]]; present = [i for i in ids if a[i][0]]
        fa_new = sum(b[i][1] and not a[i][1] for i in absent); fa_gone = sum(a[i][1] and not b[i][1] for i in absent)
        miss_new = sum(a[i][1] and not b[i][1] for i in present); miss_gone = sum(b[i][1] and not a[i][1] for i in present)
        print(f"  {suite}: absent {len(absent)} new text {fa_new} / new empty {fa_gone}; present {len(present)} new empty {miss_new} / new text {miss_gone}")
