"""Gates of README "判定" (written before the evaluation): kgrpo (K's round, with context correction) against k0grpo
(this experiment, without), think off, paired bootstrap 95% CI, diff = kgrpo - k0grpo in percentage points.

1. gain: context correction v2, real recordings that need a correction with an explicit explanation (Emilia, 516 items,
   the grouping of ctx_v2_mined_groups.py); passes when the CI low > 0. The other ctx groups are recorded.
2. regressions: the 27 delivery suites. ts-asr suites from runs/zh-ctx-grpo-20261120 compare_delivery.py's
   compare/k0grpo-vs-kgrpo/ (ae compare, paired by utterance); hotword suites bootstrapped here from hotword-scores.jsonl
   (pooled CER and recall as compare_delivery.py reports them, resampled by utterance). A regression is significantly
   worse and worse by more than 0.5 points; a TS false-alarm / miss pair moving in opposite directions with the sum not
   worse by more than 0.5 points is a threshold move, not a regression.
usage: gate.py   (after eval_after_training.sh and compare_delivery.py k0grpo kgrpo)
"""
import gzip
import json
import numpy as np
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
G = HERE.parents[0] / "zh-ctx-grpo-20261120"
sys.path.insert(0, str(G / "scripts"))
from make_delivery_configs import SUITES, run_dir  # noqa: E402

RUNS = Path("/workspace/data/ae-runs")
CTX = {"kgrpo": RUNS / "ctx-grpo-k-20261122-kgrpo-avg3", "k0grpo": RUNS / "ctx-mainline-pair-20261127-k0grpo-avg3"}
MINED = [f"/workspace/data/datasets/context_correction_zh/versions/{v}/records/mined_real.jsonl.gz"
         for v in ("v1-20261018", "v2-20261116")]  # podcast items are v1's, Emilia v2's (ctx_v2_mined_groups.py)
MODELS = ("k0grpo", "kgrpo")  # baseline, candidate (delivery model names of acp-delivery-one.sh)
CMP = G / "tasks/04-delivery/compare" / f"{MODELS[0]}-vs-{MODELS[1]}"
B, SEED, MARGIN = 2000, 20261127, 0.5  # resamples and seed as ae compare's 2000; margin: README 判定 2
HIGHER_BETTER = {"term_hit_rate", "recall", "correction_rate", "keep_rate", "accuracy"}


def bootstrap(an, ad, bn, bd):
    """Paired bootstrap of a pooled ratio sum(num) / sum(den), one entry per resampling unit (a mean: den = 1).
    Returns a, b, diff and the percentile 95% CI of b - a, in points."""
    an, ad, bn, bd = (np.asarray(x, dtype=float) for x in (an, ad, bn, bd))
    rng, diffs = np.random.default_rng(SEED), []
    for _ in range(B // 200):  # 200 resamples at a time keeps the index matrix small
        idx = rng.integers(0, len(an), (200, len(an)))
        diffs.append(bn[idx].sum(1) / bd[idx].sum(1) - an[idx].sum(1) / ad[idx].sum(1))
    lo, hi = np.percentile(np.concatenate(diffs), [2.5, 97.5])
    a, b = an.sum() / ad.sum(), bn.sum() / bd.sum()
    return 100 * a, 100 * b, 100 * (b - a), (100 * lo, 100 * hi)


def identity(config):  # compare_delivery.py's (it reads sys.argv on import, so copied)
    return config.get("eval_set") or str(config.get("data_config"))


def latest(model, ident):  # compare_delivery.py's: the newest complete delivery run of MODEL on that suite
    found = []
    for run in sorted((RUNS / f"ctx-grpo-20261120-delivery-{model}/runs").glob("run-*")):
        c = json.loads((run / "run.json").read_text())["config"]
        cov = json.loads((run / "coverage.json").read_text()) if (run / "coverage.json").exists() else {}
        if identity(c) == ident and cov.get("complete") and cov.get("successful") == cov.get("total"):
            found.append(run)
    if not found:
        raise SystemExit(f"no complete run of {model} for {ident}")
    return found[-1]


def last_run(root):
    runs = sorted((root / "runs").glob("run-*"))
    assert runs, root
    return runs[-1]


def ctx_gate():
    need = {}
    for path in MINED:
        for line in gzip.open(path, "rt"):
            r = json.loads(line)
            need[r["id"]] = not r["metadata"]["corpus_transcript_correct"]
    rows = {}
    for model, root in CTX.items():
        run = last_run(root)
        rows[model] = {(s["id"], s["mode"]): s for s in map(json.loads, open(run / "context-correction-scores.jsonl"))}
        print(f"{model}: {run}")
    assert rows["kgrpo"].keys() == rows["k0grpo"].keys()
    groups = defaultdict(list)
    for key, s in rows["k0grpo"].items():
        src = s["source"].split(":")[-1]
        if s["kind"] == "mined":
            g = ("需要纠正" if need[s["id"]] else "原转写已对") + " · " + s["category"]
            groups[(src, g, "target_ok")].append(key)
        else:  # synthetic: correction rate on correct items, keep and over-correction on confirm items
            for metric in (["target_ok"] if s["kind"] == "correct" else ["target_ok", "target_alternative"]):
                groups[(f"{src} {s['mode']}", f"{s['kind']} · {s['category']}", metric)].append(key)
    out = []
    for (src, g, metric), keys in sorted(groups.items()):
        ones = [1] * len(keys)
        a, b, d, ci = bootstrap([rows["k0grpo"][k][metric] for k in keys], ones, [rows["kgrpo"][k][metric] for k in keys], ones)
        out.append({"source": src, "group": g, "metric": metric, "items": len(keys), "k0grpo": a, "kgrpo": b,
                    "diff": d, "ci": ci})
    gate = [x for x in out if x["source"] == "emilia_zh" and x["group"] == "需要纠正 · explicit"]
    assert len(gate) == 1 and gate[0]["items"] == 516, gate
    return out, bool(gate[0]["ci"][0] > 0)


def hotword(model, suffix):
    base = json.loads((run_dir(suffix) / "run.json").read_text())["config"]
    run = latest(model, identity(base))
    return {(s.get("label"), s["id"], s["condition"]): s for s in map(json.loads, open(run / "hotword-scores.jsonl"))}


def delivery_gate():
    out = []
    for key, suffix, _ in SUITES:
        base = json.loads((run_dir(suffix) / "run.json").read_text())["config"]
        if base.get("evaluator") == "hotwords":
            a, b = (hotword(m, suffix) for m in MODELS)
            assert a.keys() == b.keys(), key
            by_utt = defaultdict(list)  # resample utterances: every condition of one recording together
            for k in a:
                by_utt[(k[0], k[1])].append(k)
            units = list(by_utt.values())
            total = lambda run, field, listed=False: [sum(run[k][field] for k in ks if not listed or k[2] != "none")  # noqa: E731
                                                      for ks in units]
            metrics = [("CER", "error_rate", "errors", "units", False)]  # pooled as compare_delivery.py's pooled()
            if sum(total(a, "gold_total", True)):
                metrics.append(("CER", "recall", "hits", "gold_total", True))
            for group, name, num, den, listed in metrics:
                av, bv, d, ci = bootstrap(total(a, num, listed), total(a, den, listed), total(b, num, listed),
                                          total(b, den, listed))
                out.append({"suite": key, "group": group, "metric": name, "k0grpo": av, "kgrpo": bv, "diff": d,
                            "ci": ci})
            continue
        overall = json.loads((CMP / f"cmp-{key}.json").read_text())["overall"]
        for group, ms in overall.items():
            for name, x in ms.items():
                if x.get("a") is None:
                    continue
                out.append({"suite": key, "group": group, "metric": name, "k0grpo": 100 * x["a"], "kgrpo": 100 * x["b"],
                            "diff": 100 * x["diff"], "ci": (100 * x["ci"][0], 100 * x["ci"][1])})
    for x in out:
        hb = x["metric"] in HIGHER_BETTER
        x["worse"] = bool(x["ci"][1] < 0 if hb else x["ci"][0] > 0)
        x["better"] = bool(x["ci"][0] > 0 if hb else x["ci"][1] < 0)
        x["material"] = bool(abs(x["diff"]) > MARGIN)
    regressions, moves = [], []
    for x in out:
        if not (x["worse"] and x["material"]):
            continue
        if x["metric"] in ("false_alarm_rate", "miss_rate"):
            other = next((y for y in out if y["suite"] == x["suite"] and y["group"] == x["group"]
                          and y["metric"] == ({"false_alarm_rate", "miss_rate"} - {x["metric"]}).pop()), None)
            if other and other["diff"] < 0 and x["diff"] + other["diff"] <= MARGIN:
                moves.append((x, other))
                continue
        regressions.append(x)
    return out, regressions, moves


def fmt(x, label):
    return (f"| {label} | {x['k0grpo']:.2f} | {x['kgrpo']:.2f} | {x['diff']:+.2f} [{x['ci'][0]:+.2f}, {x['ci'][1]:+.2f}]"
            f"{' *' if x['ci'][0] > 0 or x['ci'][1] < 0 else ''} |")


def pipes():
    """Recorded only: outputs with a '|' the reference lacks, per delivery suite and model. The synthetic data's
    protected-word expansion left "反应|反映" in 74 targets per ctx split and 159 of train_asr_dialog (both arms have
    train_asr_dialog; README "执行"), and K's models learned it (11.17 has none)."""
    out = {}
    for key, suffix, _ in SUITES:
        ident = identity(json.loads((run_dir(suffix) / "run.json").read_text())["config"])
        for m in MODELS:
            n = 0
            for line in open(latest(m, ident) / "predictions.jsonl"):
                r = json.loads(line)
                n += "|" in (r.get("prediction") or "") and "|" not in (r.get("reference") or "")
            out.setdefault(key, {})[m] = n
    return out


def main():
    ctx, gain = ctx_gate()
    dl, regressions, moves = delivery_gate()
    bars = pipes()
    lines = ["# 判定：kgrpo（带纠偏）对 k0grpo（不带），think 关，配对 bootstrap 95% CI，差 = kgrpo − k0grpo（百分点）", "",
             "## 1. 收益：纠偏 v2", "", "| 来源 · 组 · 指标（条数） | k0grpo | kgrpo | 差 [95% CI] |", "|---|---:|---:|---|"]
    for x in ctx:
        lines.append(fmt(x, f"{x['source']} · {x['group']} · {x['metric']}（{x['items']}）"))
    lines += ["", f"判定 1（Emilia 需要纠正 · explicit，516 条，CI 下限 > 0）：{'成立' if gain else '不成立'}", "",
              "## 2. 回退：交付 27 组", "", "| 项目 · 指标 | k0grpo | kgrpo | 差 [95% CI] |", "|---|---:|---:|---|"]
    for x in dl:
        tag = "（回退）" if x in regressions else ("（门槛移动）" if any(x is m[0] for m in moves) else "")
        lines.append(fmt(x, f"{x['suite']} · {x['group']} {x['metric']}{tag}"))
    lines += ["", "显著更差且差 > 0.5 个点（回退）：" + ("；".join(f"{x['suite']} {x['group']} {x['metric']} {x['diff']:+.2f}"
                                                         for x in regressions) or "无"),
              "门槛移动（TS 误触发和漏识别一升一降、和不超过 +0.5）：" + ("；".join(
                  f"{x['suite']} {x['metric']} {x['diff']:+.2f} / {y['metric']} {y['diff']:+.2f}" for x, y in moves) or "无"),
              "显著更差但差 ≤ 0.5 个点（只记录）：" + ("；".join(f"{x['suite']} {x['group']} {x['metric']} {x['diff']:+.2f}"
                                                         for x in dl if x["worse"] and not x["material"]) or "无"),
              "显著更好：" + ("；".join(f"{x['suite']} {x['group']} {x['metric']} {x['diff']:+.2f}" for x in dl if x["better"]) or "无"),
              "", "只记录：输出里带竖线（参考里没有）的条数，" + " / ".join(MODELS) + "：" + "；".join(
                  f"{k} {v[MODELS[0]]}/{v[MODELS[1]]}" for k, v in bars.items() if any(v.values())) + f"（合计 "
              + " / ".join(str(sum(v[m] for v in bars.values())) for m in MODELS) + "）",
              "", "## 3. 结论规则", "",
              ("1 成立且 2 没有回退：纠偏这一包在主线一轮里有净收益，可以进主线。" if gain and not regressions else
               "1 成立但 2 有回退：收益和回退的全表交给用户取舍。" if gain else
               "1 不成立：在一轮 SFT → GRPO 里纠偏这一包拿不到明确说明类的收益，不进主线。")]
    out = HERE / "tasks/04-eval/gate.md"
    out.write_text("\n".join(lines) + "\n")
    (HERE / "tasks/04-eval/gate.json").write_text(json.dumps({"ctx": ctx, "delivery": dl, "gain": gain,
        "regressions": regressions, "pipes": bars, "moves": [[x, y] for x, y in moves]}, ensure_ascii=False, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
