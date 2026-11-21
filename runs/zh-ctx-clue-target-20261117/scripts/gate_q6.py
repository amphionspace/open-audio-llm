"""Question 6 gate (README "判定 · 第 6 问") from compare_q6.sh's outputs in tasks/02-eval/compare/.

H (template-no-synth-ts) and I (template-synth-ts-mix), each against template, think off:
- scene TS recovered as in question 4: CER and false-alarm rate both significantly lower (CI high < 0), CER <= 5.0,
  at most 30 of the 70 target-absent items with output;
- context correction kept: Emilia accuracy and the overall keep rate with CI low >= -1 point.
An arm passes with both. Both pass: I when its think-off TS-mode correction rate beats H's (I vs H, CI low > 0),
else H (one synthetic source fewer). Reported only: the TS-mode and synthetic correction rates and think on.
usage: python3 gate_q6.py
"""
import json
from pathlib import Path

C = Path(__file__).resolve().parents[1] / "tasks/02-eval/compare"
CER_MAX, FA_MAX, ABSENT, POINT = 0.05, 30, 70, 0.01  # as question 4 (gate_q4.py)


def fmt(x, scale=100):
    return (f"{x['a'] * scale:.1f} → {x['b'] * scale:.1f}，{x['diff'] * scale:+.1f} "
            f"[{x['ci'][0] * scale:+.1f}, {x['ci'][1] * scale:+.1f}]{'*' if x['significant'] else ''}")


def load(name):
    return json.loads((C / f"cmp-{name}.json").read_text())


def arm(label, tag):
    s = load(f"scene-template-{tag}")["overall"]["all/CER"]
    cer, fa = s["error_rate"], s["false_alarm_rate"]
    items = round(fa["b"] * ABSENT)
    scene = cer["ci"][1] < 0 and fa["ci"][1] < 0 and cer["b"] <= CER_MAX and items <= FA_MAX
    print(f"{label} 场景 TS：CER {fmt(cer)}；误出字率 {fmt(fa)}（{items}/{ABSENT} 条）→ {'恢复' if scene else '没恢复'}")
    ok = scene
    for mode in ("off", "on"):
        cc = load(f"template-{tag}-{mode}")["context_correction"]
        rows = [("Emilia 准确率", cc["by"]["source"]["mined:emilia_zh"]["mined_accuracy"], True),
                ("全部保持率", cc["overall"]["keep_rate"], True),
                ("TS 题纠正成功率", cc["by"]["mode"]["ts_asr"]["correction_rate"], False),
                ("合成题纠正成功率", cc["by"]["source"]["synthetic"]["correction_rate"], False)]
        for name, x, gated in rows:
            gate = gated and mode == "off"
            passed = x["ci"][0] >= -POINT
            ok = ok and (passed or not gate)
            mark = ("通过" if passed else "不通过") if gate else "参考"
            print(f"  {mark}  think {'关' if mode == 'off' else '开'} · {name}：{fmt(x)}")
    return ok


def main():
    h, i = arm("H 去掉合成 train_ts", "nosynthts"), arm("I 合成 train_ts 在线混音", "synthtsmix")
    x = load("nosynthts-synthtsmix-off")["context_correction"]["by"]["mode"]["ts_asr"]["correction_rate"]
    print(f"I 对 H · think 关 · TS 题纠正成功率：{fmt(x)}")
    if h and i:
        pick = "I" if x["ci"][0] > 0 else "H"
    else:
        pick = "H" if h else "I" if i else None
    names = {"H": "template-no-synth-ts", "I": "template-synth-ts-mix"}
    print("第 6 问结论：" + (f"{pick} 通过，{names[pick]} 的数据配置作为上下文纠偏线的 SFT 配方，它的 checkpoint-3000 作为 GRPO 起点"
                          if pick else "两组都不通过，按数字记录最接近的解释，不定配方"))


if __name__ == "__main__":
    main()
