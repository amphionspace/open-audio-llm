import json, re, sys
from collections import defaultdict
import numpy as np


def norm(s):
    return re.sub(r"[^\w]", "", s or "").replace("_", "")


def edit(a, b):
    if not a:
        return len(b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


rows = [json.loads(l) for l in open(sys.argv[1])]
g = defaultdict(list)
for r in rows:
    g[(r["split"], r["cond"])].append(r)
conds = []
for (_, c) in g:
    if c not in conds:
        conds.append(c)
print(f"{'split':14s} {'cond':12s} {'n':>4s} | {'CER%':>6s} {'empty%':>7s} {'wrongspk%':>9s} | {'reject%':>7s}")
for sp in ["test_1spk", "test_2mix", "test_3mix", "test_2mix_neg", "test_3mix_neg"]:
    for c in conds:
        rs = g.get((sp, c))
        if not rs:
            continue
        if sp.endswith("neg"):
            rej = np.mean([norm(r["hyp"]) == "" for r in rs]) * 100
            print(f"{sp:14s} {c:12s} {len(rs):4d} | {'':>6s} {'':>7s} {'':>9s} | {rej:7.1f}")
            continue
        err = sum(edit(norm(r["hyp"]), norm(r["ref"])) for r in rs)
        ln = sum(len(norm(r["ref"])) for r in rs)
        empty = np.mean([norm(r["hyp"]) == "" for r in rs]) * 100
        ws = []
        for r in rs:
            h = norm(r["hyp"])
            if not h or not r.get("others"):
                continue
            de = edit(h, norm(r["ref"]))
            do = min(edit(h, norm(o)) for o in r["others"])
            ws.append(do < de)
        w = (np.mean(ws) * 100) if ws else float("nan")
        print(f"{sp:14s} {c:12s} {len(rs):4d} | {err / ln * 100:6.2f} {empty:7.1f} {w:9.1f} | {'':>7s}")
