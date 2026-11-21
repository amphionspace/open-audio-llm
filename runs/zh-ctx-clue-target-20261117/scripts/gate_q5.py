"""Question 5 gate (README "判定 · 第 5 问") from compare_q5.sh's outputs in tasks/02-eval/compare/.

G (template-no-synth) against template on the context-correction eval v2, think off (the delivery and GRPO mode):
Emilia accuracy and the overall keep rate must have CI low >= -1 point. Reported only: WenetSpeech podcast accuracy,
the synthetic correction rates (synthetic items share the format of the dropped training data, so they are expected
to drop) and everything with think on.
usage: python3 gate_q5.py
"""
import json
from pathlib import Path

C = Path(__file__).resolve().parents[1] / "tasks/02-eval/compare"
POINT = 0.01


def fmt(x, scale=100):
    return (f"{x['a'] * scale:.1f} → {x['b'] * scale:.1f}，{x['diff'] * scale:+.1f} "
            f"[{x['ci'][0] * scale:+.1f}, {x['ci'][1] * scale:+.1f}]{'*' if x['significant'] else ''}")


def rows(cc):
    src = cc["by"]["source"]
    return [("Emilia 准确率", src["mined:emilia_zh"]["mined_accuracy"], True),
            ("全部保持率", cc["overall"]["keep_rate"], True),
            ("WenetSpeech 播客准确率", src["mined:wenetspeech_podcast"]["mined_accuracy"], False),
            ("合成题纠正成功率", src["synthetic"]["correction_rate"], False),
            ("独立合成组纠正成功率", src["synthetic_indep"]["correction_rate"], False),
            ("全部误改率", cc["overall"]["over_correction_rate"], False)]


def main():
    failed = []
    for mode in ("off", "on"):
        cc = json.loads((C / f"cmp-template-nosynth-{mode}.json").read_text())["context_correction"]
        for label, x, gated in rows(cc):
            gate = gated and mode == "off"
            mark = ("通过" if x["ci"][0] >= -POINT else "不通过") if gate else "参考"
            if gate and x["ci"][0] < -POINT:
                failed.append(label)
            print(f"{mark}  think {'关' if mode == 'off' else '开'} · {label}：{fmt(x)}")
    if not failed:
        print("第 5 问结论：通过。G 作为上下文纠偏线的 SFT 配方和 GRPO 起点")
    elif "Emilia 准确率" in failed:
        print("第 5 问结论：Emilia 变差，合成数据在 think 关的真实录音上有用；下一步只去掉合成 train_ts（第 H 组）")
    else:
        print("第 5 问结论：只有保持率变差。G 仍作起点（场景 TS 是交付卡点），GRPO 上下文纠偏计划保留保持题，GRPO 判定看保持率")


if __name__ == "__main__":
    main()
