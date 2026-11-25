"""README "合成数据拆分诊断": the merges on 11.17 add one piece of synthetic data at a time, r1117 -> pm-train (11.17 +
(H - G), synthetic ``train``) -> pm-train-ts (11.17 + (template - G), adds ``train_ts``) -> pairmerge-g (11.17 + (J - G),
adds the twins). Paired bootstrap 95% CI of each step on the TS suites (ae open-audio-llm compare, AmphionEval !158) over
the delivery runs of runs/zh-ctx-grpo-20261120; writes tasks/09-datasplit/<suite>-<a>-vs-<b>.json and prints error /
false-alarm / miss rates.
usage: python3 split_synth_1117.py
"""
import glob
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "tasks/09-datasplit"
RUNS = "/workspace/data/ae-runs/ctx-grpo-20261120-delivery-{}/runs"
PY = "/workspace/workspace/mingdong/.local-tools/amphion-eval-0.14.0-eedcf15/bin/python"
CMP_SRC = "/workspace/workspace/mingdong/repo/amphion-eval-ctx-compare/src"
SUITES = ("ts-asr-benchmark@v12", "ts-asr-first-other@", "ts-asr-benchmark-en@", "ts-asr-crosslingual@")
PAIRS = (("r1117", "pm-train"), ("pm-train", "pm-train-ts"), ("pm-train-ts", "pairmerge-g"))


def run(model, suite):
    return [r for r in sorted(glob.glob(RUNS.format(model) + "/run-*"))
            if suite in (json.load(open(r + "/run.json"))["config"].get("eval_set") or "")
            and json.load(open(r + "/coverage.json")).get("complete")][-1]


def compare(job):
    suite, (a, b) = job
    out = OUT / f"{suite.split('@')[0]}-{a}-vs-{b}.json"
    if not out.exists():
        subprocess.run([PY, "-m", "amphion_eval.cli", "open-audio-llm", "compare", run(a, suite), run(b, suite),
                        "--output", str(out)], check=True, env={**os.environ, "PYTHONPATH": CMP_SRC},
                       stdout=subprocess.DEVNULL)
    return job, json.loads(out.read_text())["overall"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(s, p) for s in SUITES for p in PAIRS]
    with ThreadPoolExecutor(len(jobs)) as pool:
        for (suite, (a, b)), overall in pool.map(compare, jobs):
            o = overall.get("all/CER") or overall.get("all/WER") or next(iter(overall.values()))
            cells = []
            for m in ("error_rate", "false_alarm_rate", "miss_rate"):
                x = o[m]
                cells.append(f"{m.split('_')[0]} {100 * x['a']:.2f} -> {100 * x['b']:.2f}{'*' if x['significant'] else ''}"
                             f" [{100 * x['ci'][0]:+.2f}, {100 * x['ci'][1]:+.2f}]")
            print(f"{suite} {a} -> {b}: " + "; ".join(cells))


if __name__ == "__main__":
    main()
