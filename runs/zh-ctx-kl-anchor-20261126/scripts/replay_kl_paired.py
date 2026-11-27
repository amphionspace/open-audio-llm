"""Row-paired replay KL of two attempts that saw the same replay samples (same seed): per logged window, the task
sets must match; compares ts_negative token KL (x4 = bound on the decision KL in the arm that did not log it)."""
import json, sys, statistics as st
a, b = ([json.loads(l) for l in open(p + "/replay-kl-by-task.jsonl")] for p in sys.argv[1:3])
assert [r["step"] for r in a] == [r["step"] for r in b]
same = sum({k for k in x if not k.endswith("_decision")} == {k for k in y if not k.endswith("_decision")} for x, y in zip(a, b))
print("rows", len(a), "rows with the same task set", same)
for part, lo, hi in (("first half", 0, len(a) // 2), ("second half", len(a) // 2, len(a)), ("last 10%", len(a) - len(a) // 10, len(a))):
    pa = [(x["ts_negative"], y["ts_negative"], y.get("ts_negative_decision")) for x, y in zip(a[lo:hi], b[lo:hi]) if "ts_negative" in x and "ts_negative" in y]
    up = sum(y > x for x, y, _ in pa)
    print(f"{part}: windows {len(pa)}, B>A {up}; ts_neg tok mean A {st.mean(x for x,_,_ in pa):.4f} B {st.mean(y for _,y,_ in pa):.4f}"
          f"; median A {st.median(x for x,_,_ in pa):.4f} B {st.median(y for _,y,_ in pa):.4f}; B decision mean {st.mean(d for *_, d in pa):.4f} median {st.median(d for *_, d in pa):.4f}")
