"""CE / replay-KL statistics of a training attempt (task READMEs 07/08): first/last 10%, mid, per-task last 10%.
usage: replay_kl_stats.py ATTEMPT/artifacts/training [...]"""
import json, sys
for d in sys.argv[1:]:
    log = [json.loads(l) for l in open(d + "/logging.jsonl")]
    log = [r for r in log if "ce_asr" in r]
    n = len(log); k = max(1, n // 10)
    m = lambda rs, key: sum(r[key] for r in rs if key in r) / max(1, sum(key in r for r in rs))
    mid = log[n // 2 - k // 2: n // 2 + k // 2]
    print(d.split("/tasks/")[1].split("/")[0], "rows", n)
    for key in ("ce_asr", "replay_kl", "replay_kl_decision"):
        if key in log[-1]:
            print(f"  {key}: first10 {m(log[:k], key):.4f} mid {m(mid, key):.4f} last10 {m(log[-k:], key):.4f}")
    by = [json.loads(l) for l in open(d + "/replay-kl-by-task.jsonl")]
    nb = len(by); kb = max(1, nb // 10); last = by[-kb:]
    keys = sorted({x for r in last for x in r if x != "step"})
    print("  by-task last10 (rows %d..%d):" % (last[0]["step"], last[-1]["step"]), " ".join(f"{x}={m(last, x):.4f}" for x in keys))
