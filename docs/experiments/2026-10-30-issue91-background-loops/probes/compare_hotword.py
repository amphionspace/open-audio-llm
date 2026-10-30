"""热词 benchmark 两种解码设置的对照：总错误率、热词处错误率、热词命中率，以及逐切片变化。

用法：compare_hotword.py <baseline run dir> <rp run dir>
summary.json 的 hotword_bench.metrics 以「数据集/热词列表条件」为键。
"""
import json, sys

def load(run_dir):
    return json.load(open(f"{run_dir}/summary.json"))["hotword_bench"]["metrics"]

B, R = load(sys.argv[1]), load(sys.argv[2])
common = sorted(set(B) & set(R))
print(f"共同切片 {len(common)}（baseline {len(B)} / rp {len(R)}）")

def pool(m, cond):
    e = u = be = bu = h = g = 0
    for k, v in m.items():
        if not cond(k):
            continue
        e += v["errors"]; u += v["units"]; be += v["b_errors"]; bu += v["b_units"]
        h += v.get("hits", 0); g += v.get("gold_total", 0)
    return (100 * e / u if u else 0, 100 * be / bu if bu else 0, 100 * h / g if g else 0)

conds = [("全部条件", lambda k: True),
         ("无热词 none", lambda k: k.endswith("/none")),
         ("gold 列表", lambda k: k.endswith("/gold")),
         ("pool50", lambda k: "/pool50" in k)]
print(f"{'口径':12} {'总错误率':>22} {'热词处错误率(B)':>22} {'热词命中率':>22}")
for name, c in conds:
    a, b = pool(B, c), pool(R, c)
    print(f"{name:12} {a[0]:6.3f}→{b[0]:6.3f} ({b[0]-a[0]:+.3f})  "
          f"{a[1]:6.3f}→{b[1]:6.3f} ({b[1]-a[1]:+.3f})  {a[2]:6.2f}→{b[2]:6.2f} ({b[2]-a[2]:+.2f})")

rows = sorted(((100 * (R[k]["error_rate"] - B[k]["error_rate"]), k,
                100 * B[k]["error_rate"], 100 * R[k]["error_rate"],
                100 * (B[k].get("recall") or 0), 100 * (R[k].get("recall") or 0)) for k in common), reverse=True)
print(f"\n变差 {sum(1 for r in rows if r[0] > 1e-9)}/{len(rows)} 个切片")
for d, k, a, b, ra, rb in rows:
    print(f"  {k[:52]:52} {a:7.3f} → {b:7.3f} ({d:+.3f})   热词召回 {ra:5.1f} → {rb:5.1f}")
