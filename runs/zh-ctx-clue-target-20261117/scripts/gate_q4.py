"""Question 4 gate (README "判定 · 第 4 问") from compare_q4.sh's outputs in tasks/02-eval/compare/.

Each arm against template on scene TS (think off): recovered = CER and false-alarm rate both significantly lower
(CI high < 0) with CER <= 5.0 and at most 30 of the 70 target-absent items with output; not recovered = CER not
significantly different; partly otherwise. Then the conclusion for the pair, and F's cost on the context-correction
eval v2 (overall correction rate, overall keep rate, Emilia accuracy; CI low >= -1 point, think off and on).
usage: python3 gate_q4.py
"""
import json
from pathlib import Path

C = Path(__file__).resolve().parents[1] / "tasks/02-eval/compare"
CER_MAX, FA_MAX, ABSENT = 0.05, 30, 70  # README: CER <= 5.0, <= 30 of the 70 target-absent items
POINT = 0.01


def fmt(x, scale=100):
    return (f"{x['a'] * scale:.1f} → {x['b'] * scale:.1f}，{x['diff'] * scale:+.1f} "
            f"[{x['ci'][0] * scale:+.1f}, {x['ci'][1] * scale:+.1f}]{'*' if x['significant'] else ''}")


def scene(name):
    s = json.loads((C / f"cmp-scene-template-{name}.json").read_text())["overall"]["all/CER"]
    cer, fa = s["error_rate"], s["false_alarm_rate"]
    fa_items = round(fa["b"] * ABSENT)
    if cer["ci"][1] < 0 and fa["ci"][1] < 0 and cer["b"] <= CER_MAX and fa_items <= FA_MAX:
        verdict = "恢复"
    elif not (cer["ci"][1] < 0 or cer["ci"][0] > 0):
        verdict = "没恢复"
    else:
        verdict = "部分恢复"
    print(f"{name}：CER {fmt(cer)}；误出字率 {fmt(fa)}（{fa_items}/{ABSENT} 条）→ {verdict}")
    return verdict


def main():
    f, g = scene("noempty"), scene("nosynth")
    conclusion = {("恢复", "没恢复"): "原因是 #158/#159 的 think 分配；修法：配方把 empty_prob、uncovered_prob 设为 0（看下面的代价）",
                  ("没恢复", "恢复"): "原因是合成说字数据；下一步改合成 TS 数据的构造",
                  ("恢复", "恢复"): "两项同时存在才退化；去掉代价小的那项",
                  ("没恢复", "没恢复"): "两项各自足以造成退化，或另有差别（MTP）；补“第一轮数据 + MTP、不开 #158/#159”的对照"}
    print("第 4 问结论：" + conclusion.get((f, g), f"F {f}、G {g}：按数字对照最接近的一条解释"))
    for mode in ("off", "on"):
        cc = json.loads((C / f"cmp-template-noempty-{mode}.json").read_text())["context_correction"]
        rows = [("全部纠正成功率", cc["overall"]["correction_rate"]), ("全部保持率", cc["overall"]["keep_rate"]),
                ("Emilia 准确率", cc["by"]["source"]["mined:emilia_zh"]["mined_accuracy"])]
        for label, x in rows:
            print(f"{'通过' if x['ci'][0] >= -POINT else '不通过'}  F 的代价 · think {'关' if mode == 'off' else '开'} · {label}：{fmt(x)}")


if __name__ == "__main__":
    main()
