"""Question 9 gate (README "判定 · 第 9 问"), think off, paired bootstrap 95% CI. Inputs:
- K against 11.17 on the delivery suites: GRPO experiment tasks/04-delivery/compare/r1117-vs-k/ (compare_delivery.py r1117 k);
- the scene-asr long-dialogue drops: GRPO experiment scripts/scene_asr_drops.py, run here on the latest K run;
- K against J on the context-correction eval v2: tasks/02-eval/compare/cmp-synthtstwins-k-off.json (compare_q9.sh).
Criteria:
1. scene-asr CER not significantly worse than 11.17 (CI low <= 0); 45 s+ coverage >= 0.90; at most 60 dialogues with
   one speaker mostly dropped;
2. es-asr, ts-es-v1, ts-es-v2, babble-es, crosslingual-es WER not significantly worse (CI low <= 0);
3. delivery ctx-v1 CER significantly better (CI high < 0) and <= 4.64; on ctx-v2 against J, Emilia accuracy and the
   overall keep rate with CI low >= -1 point;
4. scene-ts CER and false-alarm rate, v12 and first-other miss and false-alarm rates not significantly worse.
usage: python3 gate_q9.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
G = Path("/workspace/workspace/mingdong/repo/oal-ctx-grpo/runs/zh-ctx-grpo-20261120")
D = G / "tasks/04-delivery/compare/r1117-vs-k"
sys.path.insert(0, str(G / "scripts"))
from scene_asr_drops import latest, measure  # noqa: E402

COVERAGE_MIN, DROPPED_MAX, CTX_V1_MAX, POINT = 0.90, 60, 0.0464, 0.01


def metric(suite, name):
    overall = json.loads((D / f"cmp-{suite}.json").read_text())["overall"]
    (group,) = overall
    return overall[group][name]


def fmt(x):
    return (f"{100 * x['a']:.2f} → {100 * x['b']:.2f}，{100 * x['diff']:+.2f} "
            f"[{100 * x['ci'][0]:+.2f}, {100 * x['ci'][1]:+.2f}]{'*' if x['significant'] else ''}")


def check(label, ok, detail):
    print(f"  {'通过' if ok else '不通过'}  {label}：{detail}")
    return ok


def main():
    results = []
    print("1. 长对话不漏人")
    x = metric("scene-asr", "error_rate")
    cov, dlg = measure(latest("k"))
    results.append(all([check("scene-asr CER", x["ci"][0] <= 0, fmt(x)),
                        check("≥45 秒覆盖率", cov["45+"] >= COVERAGE_MIN, f"{cov['45+']:.3f}"),
                        check("一人基本漏掉的对话", dlg["one speaker mostly dropped"] <= DROPPED_MAX,
                              f"{dlg['one speaker mostly dropped']}/{dlg['dialogues']}")]))
    print("2. 西语不差")
    results.append(all([check(f"{s} WER", metric(s, "error_rate")["ci"][0] <= 0, fmt(metric(s, "error_rate")))
                        for s in ("es-asr", "ts-es-v1", "ts-es-v2", "babble-es", "crosslingual-es")]))
    print("3. 纠偏保住")
    x = metric("ctx-v1", "error_rate")
    cc = json.loads((HERE / "tasks/02-eval/compare/cmp-synthtstwins-k-off.json").read_text())["context_correction"]
    emilia, keep = cc["by"]["source"]["mined:emilia_zh"]["mined_accuracy"], cc["overall"]["keep_rate"]
    results.append(all([check("ctx-v1 CER", x["ci"][1] < 0 and x["b"] <= CTX_V1_MAX, fmt(x)),
                        check("纠偏 v2 对 J · Emilia 准确率", emilia["ci"][0] >= -POINT, fmt(emilia)),
                        check("纠偏 v2 对 J · 全部保持率", keep["ci"][0] >= -POINT, fmt(keep))]))
    print("4. TS 不差")
    results.append(all([check(f"{s} {n}", metric(s, n)["ci"][0] <= 0, fmt(metric(s, n)))
                        for s, n in (("scene-ts", "error_rate"), ("scene-ts", "false_alarm_rate"),
                                     ("v12", "miss_rate"), ("v12", "false_alarm_rate"),
                                     ("first-other", "miss_rate"), ("first-other", "false_alarm_rate"))]))
    failed = [str(i + 1) for i, ok in enumerate(results) if not ok]
    print("第 9 问结论：" + ("1–4 都满足，K 作为上下文纠偏 GRPO 的起点" if not failed
                         else f"第 {'、'.join(failed)} 条不满足，按 README 写定的分支记录，不定配方"))


if __name__ == "__main__":
    main()
