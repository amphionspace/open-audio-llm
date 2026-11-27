"""Replay KL of several attempts over the logged windows with step in [LO, HI] (README "后续 3" 生效核查 2: windows 300-500).
The attempts share the CE and replay samples (same seed), so the windows are paired; prints the window means of the
ts_negative token-mean KL (x4 = the bound on its decision KL for an arm that did not log the decision), its decision KL,
and the ts_positive decision KL, and how many windows the first attempt is above the second on the ts_negative decision
(the second's token KL x4 when it has no decision).
usage: replay_kl_window.py LO HI NAME=ATTEMPT_DIR [NAME=ATTEMPT_DIR ...]   (ATTEMPT_DIR/artifacts/training/replay-kl-by-task.jsonl)
"""
import json, sys
lo, hi = int(sys.argv[1]), int(sys.argv[2])
runs = {}
for arg in sys.argv[3:]:
    name, path = arg.split("=", 1)
    rows = [json.loads(l) for l in open(path + "/artifacts/training/replay-kl-by-task.jsonl")]
    runs[name] = {r["step"]: r for r in rows if lo <= r["step"] <= hi}
steps = sorted(set.intersection(*(set(r) for r in runs.values())))
print(f"windows {lo}-{hi} logged by all: {len(steps)}")
def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")
for name, r in runs.items():
    tok = [r[s].get("ts_negative") for s in steps]
    print(f"  {name}: ts_neg tok {mean(tok):.4f} (x4 {4 * mean(tok):.4f}); ts_neg decision {mean(r[s].get('ts_negative_decision') for s in steps):.4f}; "
          f"ts_pos decision {mean(r[s].get('ts_positive_decision') for s in steps):.4f}; "
          f"decision batch ts_neg {mean(r[s].get('ts_negative_decision_batch') for s in steps):.4f} ts_pos {mean(r[s].get('ts_positive_decision_batch') for s in steps):.4f}")
names = list(runs)
for other in names[1:]:
    a, b = runs[names[0]], runs[other]
    pairs = [(a[s].get("ts_negative_decision"), b[s].get("ts_negative_decision", None if b[s].get("ts_negative") is None else 4 * b[s]["ts_negative"]))
             for s in steps if "ts_negative" in a[s] and "ts_negative" in b[s]]
    print(f"  {names[0]} > {other} on ts_neg decision: {sum(x > y for x, y in pairs)} of {len(pairs)} windows")
