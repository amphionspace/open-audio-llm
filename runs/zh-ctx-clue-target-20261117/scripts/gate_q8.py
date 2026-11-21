"""Question 8 gate (README "判定 · 第 8 问") from compare_q8.sh's outputs in tasks/02-eval/compare/.

J (template-synth-ts-twins) against template, think off, on the real-speech TS sets ts-asr-benchmark@v12-20261022
and ts-asr-first-other@v3-20261105: miss rate, false-alarm rate and CER each not significantly worse (95% CI low
<= 0), the 11.06 delivery gate's standard. Passes with all six.
usage: python3 gate_q8.py
"""
import json
from pathlib import Path

C = Path(__file__).resolve().parents[1] / "tasks/02-eval/compare"
SETS = (("v12", "中文 TS-ASR 测试集 v12"), ("first-other", "跟错人 first-other v3"))
METRICS = (("miss_rate", "漏识别率"), ("false_alarm_rate", "误触发率"), ("error_rate", "CER"))


def fmt(x):
    return (f"{x['a'] * 100:.2f} → {x['b'] * 100:.2f}，{x['diff'] * 100:+.2f} "
            f"[{x['ci'][0] * 100:+.2f}, {x['ci'][1] * 100:+.2f}]{'*' if x['significant'] else ''}")


def main():
    ok, miss_worse = True, False
    for key, label in SETS:
        overall = json.loads((C / f"cmp-tsreal-template-synthtstwins-{key}.json").read_text())["overall"]["all/CER"]
        for metric, name in METRICS:
            x = overall[metric]
            worse = x["ci"][0] > 0
            ok = ok and not worse
            miss_worse = miss_worse or (worse and metric == "miss_rate")
            print(f"  {'不通过' if worse else '通过'}  {label} · {name}：{fmt(x)}")
    print("第 8 问结论：" + (
        "J 通过，template-synth-ts-twins 的数据配置作为上下文纠偏线的 SFT 配方，它的 checkpoint-3000 作为 GRPO 起点" if ok else
        "J 在真人语音上漏识别更多，按数字记录，不定配方，汇报" if miss_worse else
        "J 漏识别不变，但 CER 或误触发变差，按数字记录、按来源定位，不定配方，汇报"))


if __name__ == "__main__":
    main()
