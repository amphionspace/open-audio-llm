"""ctx-v2 TS-mode items (task ts_asr, sources *_ts; the target speaker is always present) answered `language None`: count per
model, and exact McNemar of new-None / new-text between two models (task READMEs 06/09)."""
import json, sys, glob, math
def run(name):
    return sorted(glob.glob(f"/workspace/data/ae-runs/ctx-grpo-k-20261122-{name}/runs/run-*"))[-1]
def none_set(name):
    rows = [json.loads(l) for l in open(run(name) + "/predictions.jsonl")]
    ts = {r["id"]: r["raw_text"].startswith("language None") for r in rows if r["task"] == "ts_asr"}
    return ts
def mcnemar(b, c):
    n = b + c; k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
models = sys.argv[1].split(",")
sets = {m: none_set(m) for m in models}
for m, s in sets.items():
    print(f"{m}: {sum(s.values())} / {len(s)} ({100 * sum(s.values()) / len(s):.1f}%)")
for pair in sys.argv[2:]:
    a, b = pair.split(":"); A, B = sets[a], sets[b]; assert A.keys() == B.keys()
    new_none = sum(B[i] and not A[i] for i in A); new_text = sum(A[i] and not B[i] for i in A)
    print(f"{a} -> {b}: new None {new_none}, new text {new_text}, p = {mcnemar(new_none, new_text):.2g}")
