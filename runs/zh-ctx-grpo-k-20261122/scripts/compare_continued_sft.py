"""README "TS 回退的来源": continued SFT on a GRPO model, with and without the context-correction data, against its
start, on ts-asr-benchmark v12 (only existing runs, all vLLM 0.18, enroll 5 s, 26,750/26,750 samples).
- 10.28 = ts-robust 054 avg3 (SFT only, the GRPO line's start); 11.06 = GRPO v10interp050 (package run of 11-06).
- control / think = runs/zh-spelling-think-20261113: SFT from 11.06, 3000 steps, lr 3.55e-6, delivery recipe v7set34 with
  the ctx-cleaned versions, no synthetic context-correction data (think only adds the think section).
- template / J = context-correction experiment (zh-ctx-clue-target-20261117) arms from 11.06, same steps and lr, with the
  synthetic context-correction data (J adds ts-twins); runs of its question 8 (tsreal).
Paired bootstrap 95% CI (ae open-audio-llm compare, AmphionEval !158). Writes tasks/06-continued-sft/<a>-vs-<b>.json and
prints CER / false-alarm / miss rates.
usage: python3 compare_continued_sft.py
"""
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
RUNS = Path("/workspace/data/ae-runs")
PY = "/workspace/workspace/mingdong/.local-tools/amphion-eval-0.14.0-eedcf15/bin/python"
CMP_SRC = "/workspace/workspace/mingdong/repo/amphion-eval-ctx-compare/src"
V12 = {
    "r1028": "ts-robust-054-avg3/runs/run-20261104T085008Z-67006e6c",
    "r1106": "amphionasr-1.7b-v3-grpo-v10interp050-pkg/runs/run-20261106T041911Z-bd4fda03",
    "control": "spelling-think-20261113-control-3000/runs/run-20261114T121919Z-7dd5a306",
    "think": "spelling-think-20261113-think-3000/runs/run-20261114T110535Z-bf5b1393",
    "template": "ctx-clue-target-20261117-tsreal-template/runs/run-20261121T070648Z-d7557475",
    "j": "ctx-clue-target-20261117-tsreal-template-synth-ts-twins/runs/run-20261121T070652Z-3a75dc95",
}
PAIRS = [("r1106", "r1028"), ("r1106", "control"), ("r1106", "think"), ("r1106", "template"), ("r1106", "j"),
         ("r1028", "control"), ("control", "template"), ("control", "j")]


def compare(pair):
    a, b = pair
    out = HERE / "tasks/06-continued-sft" / f"{a}-vs-{b}.json"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([PY, "-m", "amphion_eval.cli", "open-audio-llm", "compare", str(RUNS / V12[a]), str(RUNS / V12[b]),
                        "--output", str(out)], check=True, env={**os.environ, "PYTHONPATH": CMP_SRC}, stdout=subprocess.DEVNULL)
    return pair, json.loads(out.read_text())["overall"]["all/CER"]


def main():
    with ThreadPoolExecutor(len(PAIRS)) as pool:
        for (a, b), o in pool.map(compare, PAIRS):
            cells = []
            for m in ("error_rate", "false_alarm_rate", "miss_rate"):
                x = o[m]
                cells.append(f"{m.split('_')[0]} {100 * x['a']:.2f} -> {100 * x['b']:.2f}{'*' if x['significant'] else ''}"
                             f" [{100 * x['ci'][0]:+.2f}, {100 * x['ci'][1]:+.2f}]")
            print(f"{a} -> {b}: " + "; ".join(cells))


if __name__ == "__main__":
    main()
