"""Question 3 gate (README "判定 · 第 3 问") from compare_q3.sh's outputs in tasks/02-eval/compare/.

Main: template-implicit-noexplain vs template, think on, implicit accuracy up with the CI above 0 on the independent
synthetic group and on Emilia. Must not get worse: Emilia accuracy and overall keep rate (CI low >= -1 point),
overall over-correction (CI high <= +1 point), think-off overall correction rate (CI low >= -1 point); Emilia items
whose original transcript was already right (no paired CI) within 1 point of template. Prints one line per check,
then the reference comparison with clue-target-implicit-noexplain.
usage: python3 gate_q3.py
"""
import json
import re
from pathlib import Path

C = Path(__file__).resolve().parents[1] / "tasks/02-eval/compare"
POINT = 0.01  # the README's "1 个点"


def load(name):
    return json.loads((C / f"cmp-{name}.json").read_text())["context_correction"]


def fmt(x):
    return (f"{x['a'] * 100:.1f} → {x['b'] * 100:.1f}，{x['diff'] * 100:+.1f} "
            f"[{x['ci'][0] * 100:+.1f}, {x['ci'][1] * 100:+.1f}]{'*' if x['significant'] else ''}")


def emilia_already_right():
    """template-on and tplimplicit-on accuracy on Emilia's 原转写已对 row of mined-groups-tplimplicit.md."""
    for line in (C / "mined-groups-tplimplicit.md").read_text().splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if cells[:2] == ["emilia_zh", "原转写已对"]:
            return float(cells[3]), float(cells[4])
    raise ValueError("no Emilia 原转写已对 row")


def main():
    on, off = load("template-tplimplicit-on"), load("template-tplimplicit-off")
    src = on["by"]["source"]
    checks = [
        ("主判定 · 独立合成组 隐式题准确率", src["synthetic_indep"]["implicit_accuracy"], lambda x: x["ci"][0] > 0),
        ("主判定 · Emilia 隐式题准确率", src["mined:emilia_zh"]["implicit_accuracy"], lambda x: x["ci"][0] > 0),
        ("不变差 · Emilia 准确率", src["mined:emilia_zh"]["mined_accuracy"], lambda x: x["ci"][0] >= -POINT),
        ("不变差 · 全部保持率", on["overall"]["keep_rate"], lambda x: x["ci"][0] >= -POINT),
        ("不变差 · 全部误改率", on["overall"]["over_correction_rate"], lambda x: x["ci"][1] <= POINT),
        ("不变差 · think 关 全部纠正成功率", off["overall"]["correction_rate"], lambda x: x["ci"][0] >= -POINT),
    ]
    passed = True
    for label, x, ok in checks:
        good = ok(x)
        passed &= good
        print(f"{'通过' if good else '不通过'}  {label}：{fmt(x)}")
    a, b = emilia_already_right()
    good = b >= a - POINT * 100
    passed &= good
    print(f"{'通过' if good else '不通过'}  不变差 · Emilia 原转写已对（703 条，点估计）：{a:.1f} → {b:.1f}")
    print(f"第 3 问判定：{'都满足' if passed else '不满足'}")
    ref = load("noexplain-tplimplicit-on")["by"]["source"]["mined:emilia_zh"]
    print("参考（与 clue-target-implicit-noexplain 比，think 开）："
          f"Emilia 准确率 {fmt(ref['mined_accuracy'])}；Emilia 隐式题准确率 {fmt(ref['implicit_accuracy'])}")


if __name__ == "__main__":
    main()
