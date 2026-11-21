"""Question 7 gate (README "判定 · 第 7 问") from compare_q7.sh's outputs in tasks/02-eval/compare/.

J (template-synth-ts-twins) against template, think off, with question 6's criteria (gate_q6.arm):
- scene TS recovered: CER and false-alarm rate both significantly lower (CI high < 0), CER <= 5.0, at most 30 of the
  70 target-absent items with output;
- context correction kept: Emilia accuracy and the overall keep rate with CI low >= -1 point.
Reported only: TS-mode items of the context-correction eval left empty (template 70, H 943), the TS-mode and
synthetic correction rates and think on.
usage: python3 gate_q7.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate_q6 import arm  # noqa: E402

RUNS = Path("/workspace/data/ae-runs")


def empty_ts(name):
    """TS-mode items of the latest context-correction run of ``name`` answered with nothing."""
    run = sorted((RUNS / f"ctx-clue-target-20261117-{name}/runs").glob("run-*"))[-1]
    rows = [json.loads(line) for line in open(run / "predictions.jsonl", encoding="utf-8")]
    ts = [r for r in rows if r["task"] == "ts_asr"]
    return sum(not r["prediction"].strip() for r in ts), len(ts)


def main():
    ok = arm("J 合成音色 TS 场景（带孪生）", "synthtstwins")
    for name in ("template", "template-no-synth-ts", "template-synth-ts-twins"):
        print("上下文纠偏 v2 TS 题输出空（think 关）：%s %d/%d" % (name, *empty_ts(name)))
    print("第 7 问结论：" + ("J 通过，template-synth-ts-twins 的数据配置作为上下文纠偏线的 SFT 配方，它的 checkpoint-3000 作为 GRPO 起点"
                          if ok else "J 不通过，按数字记录，不定配方，汇报后再定"))


if __name__ == "__main__":
    main()
