"""Pair AmphionEval runs of two decode settings on the same eval set and print the deltas.

Matches runs by (evaluator, eval_set or data-config basename), then reports per-source
error rate, false alarm / miss where the set has them, and degeneration counts
(loop, max_tokens finish, empty) read back from predictions.jsonl.
"""
import json, os, re, sys, collections

LOOP = re.compile(r"(.{1,4}?)\1{7,}")
STRIP = re.compile(r"[\s，。？！、,.?!]")
# AmphionEval's own criterion, for comparison with the 2026-10-09 decode-loop study
AE_LOOP = [re.compile(r"(.)\1{29,}"), re.compile(r"(.{2,10}?)\1{19,}")]

def key_of(cfg):
    return (cfg.get("evaluator"), cfg.get("eval_set") or os.path.basename(str(cfg.get("data_config", ""))))

def runs(root):
    out = {}
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name, "run.json")
        if not os.path.exists(p):
            continue
        cfg = json.load(open(p)).get("config", {})
        out.setdefault(key_of(cfg), []).append(os.path.join(root, name))
    return out

def degeneration(run_dir):
    c = collections.Counter()
    loops = set()
    path = os.path.join(run_dir, "predictions.jsonl")
    if not os.path.exists(path):
        return c, loops
    for line in open(path):
        r = json.loads(line)
        t = (r.get("prediction") or "")
        s = STRIP.sub("", t)
        c["n"] += 1
        if LOOP.search(s):
            c["loop"] += 1
            loops.add(r["id"])
        if any(x.search(s) for x in AE_LOOP):
            c["ae_loop"] += 1
        if r.get("finish_reason") == "length":
            c["length"] += 1
        if not t.strip():
            c["empty"] += 1
    return c, loops

def metrics(run_dir):
    path = os.path.join(run_dir, "summary.json")
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    if "metrics" in d:
        return {k: v for k, v in d["metrics"].items()}
    if "hotwords" in d:
        return {k: v for k, v in d["hotwords"].items()}
    return {}

base_root, rp_root = sys.argv[1], sys.argv[2]
base, rp = runs(base_root), runs(rp_root)
for k in sorted(rp, key=lambda k: str(k)):
    if k not in base:
        print(f"\n## {k[1]} ({k[0]}): no baseline run under {base_root}")
        continue
    b, r = base[k][-1], rp[k][-1]
    mb, mr = metrics(b), metrics(r)
    if mb is None or mr is None:
        print(f"\n## {k[1]} ({k[0]}): 还没打分完（{os.path.basename(r)}）")
        continue
    cb, lb = degeneration(b)
    cr, lr = degeneration(r)
    print(f"\n## {k[1]}  ({k[0]})\n   baseline {os.path.basename(b)}  rp1.1 {os.path.basename(r)}")
    print(f"   退化计数  loop {cb['loop']}→{cr['loop']}  (AE 判定 {cb['ae_loop']}→{cr['ae_loop']})  "
          f"length {cb['length']}→{cr['length']}  empty {cb['empty']}→{cr['empty']}  n={cb['n']}/{cr['n']}")
    fixed, new = lb - lr, lr - lb
    if fixed or new:
        print(f"   复读样本 修复 {len(fixed)} / 新增 {len(new)}")
    rows, fa_rows = [], []
    for src in sorted(set(mb) & set(mr)):
        a, c = mb[src], mr[src]
        if not isinstance(a, dict):
            continue
        if a.get("error_rate") is None:
            # background-only / negative sources: false output rate instead of CER
            fa_rows.append((src, a.get("false_alarm_rate"), c.get("false_alarm_rate"), a.get("utterances"),
                            a.get("negative_utterances")))
            continue
        rows.append((src, a.get("metric"), 100 * a["error_rate"], 100 * c["error_rate"],
                     a.get("false_alarm_rate"), c.get("false_alarm_rate"),
                     a.get("miss_rate"), c.get("miss_rate"), a.get("utterances")))
    for src, fa, fc, n, neg in sorted(fa_rows, key=lambda x: -(x[2] or 0)):
        print(f"     {src[:70]:70} 误输出率 n={neg or n or 0:6} {100*(fa or 0):7.2f}% → {100*(fc or 0):7.2f}% ({100*((fc or 0)-(fa or 0)):+.2f})")
    if rows:
        worse = sum(1 for x in rows if x[3] > x[2] + 1e-9)
        tot_b = sum(x[2] for x in rows) / len(rows)
        tot_r = sum(x[3] for x in rows) / len(rows)
        print(f"   {len(rows)} 个切片：宏平均 {tot_b:.2f} → {tot_r:.2f} ({tot_r - tot_b:+.2f} 点)，变差 {worse}/{len(rows)}")
        for src, m, a, c, fa, fc, ma, mc, n in sorted(rows, key=lambda x: x[2] - x[3]):
            extra = ""
            if fa is not None or fc is not None:
                extra = f"  FA {100*(fa or 0):.2f}→{100*(fc or 0):.2f}"
            if ma is not None or mc is not None:
                extra += f"  miss {100*(ma or 0):.2f}→{100*(mc or 0):.2f}"
            print(f"     {src[:70]:70} {m or '':4} n={n or 0:6} {a:7.2f} → {c:7.2f} ({c-a:+.2f}){extra}")
