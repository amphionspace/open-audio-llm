"""Summarize the confidence-rejection probe (items.jsonl from confprobe.py).

For each confidence feature: AUC of "should reject" against "must keep" (items with
output only; an empty output needs no gate), then operating points where the gate rejects a
fixed share of scene-asr outputs, with the effect on every set scored by AmphionEval's own
v2 scoring (rejected output = empty prediction). Prints aggregates only, no transcripts.

    python confprobe_report.py items.jsonl [--json summary.json]
"""
import argparse, json, math, re
from collections import defaultdict

from amphion_eval.integrations.open_audio_llm.scoring import summarize
from amphion_eval.legacy.open_audio_llm.qwen3_asr import normalize as presence_text

FRR = [0.005, 0.01, 0.02, 0.05, 0.10]
LOOP = re.compile(r"(.{1,4}?)\1{7,}")


def features(row):
    lp = row["lp"]
    if not lp:
        return {}
    eos = row["eos_first"] if row["eos_first"] is not None else row["first_top_min"]
    worst = sorted(lp)[: max(1, len(lp) // 4)]
    return {
        "avg_lp": sum(lp) / len(lp),                      # mean token log-probability
        "q25_lp": sum(worst) / len(worst),                # mean of the least confident quarter
        "min_lp": min(lp),
        "head16_lp": sum(lp[:16]) / len(lp[:16]),        # before a repetition loop takes over
        "no_eos_first": -math.exp(eos),                   # P(empty) at the first transcript step, negated
        "lang_lp": min(row["header_lp"]) if row["header_lp"] else 0.0,
    }


def auc(pos, neg):
    """P(score of a reject item < score of a keep item); ties count half."""
    if not pos or not neg:
        return None
    ranked = sorted([(s, 0) for s in pos] + [(s, 1) for s in neg])
    rank, i, r_pos = 1, 0, 0.0
    while i < len(ranked):
        j = i
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        mean = (rank + rank + (j - i) - 1) / 2
        r_pos += mean * sum(1 for k in range(i, j) if ranked[k][1] == 0)
        rank += j - i
        i = j
    u = r_pos - len(pos) * (len(pos) + 1) / 2
    return 1 - u / (len(pos) * len(neg))


def scored(rows, gate=None):
    out = []
    for r in rows:
        pred = r["text"]
        if gate and gate(r):
            pred = ""
        out.append({"source": r["source"], "language": r["language"], "task": "asr",
                    "reference": r["reference"], "prediction": pred})
    return summarize(out)


def pooled(metrics):
    err = sum(m["errors"] for m in metrics.values())
    units = sum(m["reference_units"] for m in metrics.values())
    rates = [m["error_rate"] for m in metrics.values() if m["error_rate"] is not None]
    neg = sum(m["negative_utterances"] for m in metrics.values())
    fa = sum((m["false_alarm_rate"] or 0) * m["negative_utterances"] for m in metrics.values())
    return {"weighted": 100 * err / units if units else None, "macro": 100 * sum(rates) / len(rates) if rates else None,
            "fa": 100 * fa / neg if neg else None}


def level(source):
    split = source.split(":")[1].split("/")[0]
    if split.startswith("far-"):
        return "far"
    if split.startswith("dev-"):
        return "dev"
    return next((x for x in ("quiet", "light", "medium", "heavy", "extreme") if f"-{x}-" in split), "other")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("items")
    ap.add_argument("--json")
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.items)]
    for r in rows:
        r["has_out"] = bool(presence_text(r["text"]))
        r["f"] = features(r)
        r["nchar"] = len(re.sub(r"\s", "", presence_text(r["text"])))
    by = defaultdict(list)
    for r in rows:
        key = r["set"] if r["set"] != "babble" else f"babble-{r['label']}"
        key = key if r["set"] != "issue91" else ("issue91-bg" if r["label"] == "reject" else "issue91-control")
        by[key].append(r)
    summary = {"counts": {k: len(v) for k, v in by.items()}}
    print("## 样本与输出")
    for k, v in by.items():
        loops = sum(bool(LOOP.search(re.sub(r"\W", "", x["text"]))) for x in v)
        length = sum(x["finish"] == "length" for x in v)
        print(f"  {k:16s} n={len(v):5d}  有输出 {sum(x['has_out'] for x in v):5d}  复读 {loops}  撞上限 {length}  输出字数 {sum(x['nchar'] for x in v)}")
    reject_sets = ["issue91-bg", "babble-reject"]
    keep_sets = ["scene-asr", "wenet-meeting", "babble-keep", "issue91-control"]
    names = list(next(r["f"] for r in rows if r["f"]).keys())
    print("\n## AUC（只算有输出的样本；1 = 完全分开，0.5 = 随机）")
    print("  特征           " + "  ".join(f"{a}↔{b}" for a in reject_sets for b in keep_sets[:3]))
    summary["auc"] = {}
    for f in names:
        cells = []
        for a in reject_sets:
            for b in keep_sets[:3]:
                v = auc([r["f"][f] for r in by[a] if r["has_out"]], [r["f"][f] for r in by[b] if r["has_out"]])
                summary["auc"][f"{f}/{a}/{b}"] = v
                cells.append("  n/a " if v is None else f"{v:.3f}")
        print(f"  {f:14s} " + "  ".join(f"{c:>{len(a) + len(b) + 1}s}" for c, (a, b) in zip(cells, [(a, b) for a in reject_sets for b in keep_sets[:3]])))
    base = {k: pooled(scored(v)) for k, v in by.items()}
    summary["baseline"] = base
    summary["points"] = {}
    for f in names:
        ref = sorted(r["f"][f] for r in by["scene-asr"] if r["has_out"])
        print(f"\n## 特征 {f}：按 scene-asr 有输出样本的误拒率定阈值（得分低于阈值 → 输出置空）")
        print("  误拒率  阈值      客户背景段拒掉  背景段字数去掉  对照留下  babble纯背景误输出  scene-asr CER(加权/宏)  wenet会议 拒/CER  babble语音 拒/错误率")
        b = base
        print(f"  不拒    —         —               —               4/4       {b['babble-reject']['fa']:.1f}%            "
              f"{b['scene-asr']['weighted']:.2f}/{b['scene-asr']['macro']:.2f}           0%/{b['wenet-meeting']['weighted']:.2f}      0%/{b['babble-keep']['macro']:.2f}")
        for frr in FRR:
            tau = ref[min(len(ref) - 1, int(frr * len(ref)))]
            gate = lambda r, f=f, tau=tau: r["has_out"] and r["f"] and r["f"][f] < tau
            res = {k: pooled(scored(v, gate)) for k, v in by.items()}
            bg = by["issue91-bg"]
            rej_bg = sum(gate(r) for r in bg if r["has_out"])
            chars = sum(r["nchar"] for r in bg)
            chars_cut = sum(r["nchar"] for r in bg if gate(r))
            ctrl = sum(not gate(r) for r in by["issue91-control"])
            share = lambda k: 100 * sum(gate(r) for r in by[k]) / len(by[k])
            point = {"tau": tau, "issue91_rejected": rej_bg, "issue91_with_output": sum(r["has_out"] for r in bg),
                     "issue91_chars_removed_pct": 100 * chars_cut / chars if chars else None, "controls_kept": ctrl,
                     "babble_bg_fa": res["babble-reject"]["fa"], "scene_weighted": res["scene-asr"]["weighted"],
                     "scene_macro": res["scene-asr"]["macro"], "wenet_rejected_pct": share("wenet-meeting"),
                     "wenet_weighted": res["wenet-meeting"]["weighted"], "babble_keep_rejected_pct": share("babble-keep"),
                     "babble_keep_macro": res["babble-keep"]["macro"]}
            summary["points"][f"{f}@{frr}"] = point
            print(f"  {100 * frr:4.1f}%  {tau:8.3f}  {rej_bg:2d}/{point['issue91_with_output']:<2d}           {point['issue91_chars_removed_pct']:5.1f}%          "
                  f"{ctrl}/4       {point['babble_bg_fa']:.1f}%            {point['scene_weighted']:.2f}/{point['scene_macro']:.2f}           "
                  f"{point['wenet_rejected_pct']:.1f}%/{point['wenet_weighted']:.2f}    {point['babble_keep_rejected_pct']:.1f}%/{point['babble_keep_macro']:.2f}")
        # where the scene-asr false rejections fall, at the 2% point
        tau = ref[min(len(ref) - 1, int(0.02 * len(ref)))]
        lv = defaultdict(lambda: [0, 0])
        for r in by["scene-asr"]:
            lv[level(r["source"])][0] += bool(r["has_out"] and r["f"] and r["f"][f] < tau)
            lv[level(r["source"])][1] += 1
        print("  2% 档 scene-asr 误拒按场景噪声等级：" + "  ".join(f"{k} {a}/{n}" for k, (a, n) in sorted(lv.items())))
    if args.json:
        json.dump(summary, open(args.json, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
