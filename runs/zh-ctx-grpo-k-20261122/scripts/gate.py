"""This experiment's gate (README "判定"), avg3 (kgrpo), think off, paired bootstrap 95% CI. Inputs:
- avg3 against 11.17 on the delivery suites: runs/zh-ctx-grpo-20261120 tasks/04-delivery/compare/r1117-vs-kgrpo/
  (compare_delivery.py r1117 kgrpo) and its scripts/scene_asr_drops.py, run here on the latest kgrpo run;
- avg3 against the start K on the context-correction eval v2: tasks/04-gate/cmp-start-k-avg3-off.json (compare.sh).
1. Question 9's four criteria (runs/zh-ctx-clue-target-20261117 scripts/gate_q9.py), with ctx-v2 against K instead of J;
2. ts-es-v1, ts-es-v2 and crosslingual-es false-alarm rates not significantly worse than 11.17.
The 拆分诊断 hybrid kgrpo-llm is judged the same way: gate.py kgrpollm cmp-start-k-kgrpo-llm-off.json.
usage: python3 gate.py [DELIVERY_NAME CTX_V2_COMPARE]   (default kgrpo cmp-start-k-avg3-off.json)
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
G = Path("/workspace/workspace/mingdong/repo/oal-ctx-grpo/runs/zh-ctx-grpo-20261120")
NAME, CTX = (sys.argv[1:3] or ["kgrpo", "cmp-start-k-avg3-off.json"])
D = G / f"tasks/04-delivery/compare/r1117-vs-{NAME}"
sys.path.insert(0, str(G / "scripts"))
from scene_asr_drops import latest, measure  # noqa: E402

COVERAGE_MIN, DROPPED_MAX, CTX_V1_MAX, POINT = 0.90, 60, 0.0464, 0.01  # as question 9


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


def not_worse(suite, name):
    x = metric(suite, name)
    return check(f"{suite} {name}", x["ci"][0] <= 0, fmt(x))


def main():
    print("1. 第 9 问的四条")
    x = metric("scene-asr", "error_rate")
    cov, dlg = measure(latest(NAME))
    long_ok = all([check("scene-asr CER", x["ci"][0] <= 0, fmt(x)),
                   check("≥45 秒覆盖率", cov["45+"] >= COVERAGE_MIN, f"{cov['45+']:.3f}"),
                   check("一人基本漏掉的对话", dlg["one speaker mostly dropped"] <= DROPPED_MAX,
                         f"{dlg['one speaker mostly dropped']}/{dlg['dialogues']}")])
    es_ok = all([not_worse(s, "error_rate") for s in ("es-asr", "ts-es-v1", "ts-es-v2", "babble-es", "crosslingual-es")])
    x = metric("ctx-v1", "error_rate")
    cc = json.loads((HERE / "tasks/04-gate" / CTX).read_text())["context_correction"]
    emilia, keep = cc["by"]["source"]["mined:emilia_zh"]["mined_accuracy"], cc["overall"]["keep_rate"]
    ctx_ok = all([check("ctx-v1 CER", x["ci"][1] < 0 and x["b"] <= CTX_V1_MAX, fmt(x)),
                  check("纠偏 v2 对 K · Emilia 准确率", emilia["ci"][0] >= -POINT, fmt(emilia)),
                  check("纠偏 v2 对 K · 全部保持率", keep["ci"][0] >= -POINT, fmt(keep))])
    ts_ok = all([not_worse(s, n) for s, n in (("scene-ts", "error_rate"), ("scene-ts", "false_alarm_rate"),
                                              ("v12", "miss_rate"), ("v12", "false_alarm_rate"),
                                              ("first-other", "miss_rate"), ("first-other", "false_alarm_rate"))])
    print("2. 西语 TS 误触发")
    fa_ok = all([not_worse(s, "false_alarm_rate") for s in ("ts-es-v1", "ts-es-v2", "crosslingual-es")])
    results = {"1·长对话": long_ok, "1·西语 WER": es_ok, "1·纠偏": ctx_ok, "1·TS": ts_ok, "2·西语 TS 误触发": fa_ok}
    failed = [k for k, ok in results.items() if not ok]
    print("结论：" + (f"1、2 都满足，{NAME} 作为替换 11.17 的候选" if not failed
                    else f"不满足：{'、'.join(failed)}，按 README 写定的分支记录，不定配方"))


if __name__ == "__main__":
    main()
