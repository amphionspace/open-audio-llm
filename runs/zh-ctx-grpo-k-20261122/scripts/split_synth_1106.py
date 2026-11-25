"""README "数据增量诊断" split of the synthetic data at 11.06 on the two TS suites where the merged models miss more:
G (template-no-synth, delivery run g1106) -> template (adds the synthetic spelling data ``train`` + ``train_ts``) -> J
(adds ``ts-twins-v1``). template and J only have the context-correction experiment's question 8 (tsreal) runs, so
J's delivery run against J's question-8 run measures the client difference. Paired bootstrap 95% CI (ae open-audio-llm
compare, AmphionEval !158); writes tasks/08-datainc/<suite>-<a>-vs-<b>.json and prints error / false-alarm / miss rates.
usage: python3 split_synth_1106.py
"""
import glob
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "tasks/08-datainc"
RUNS = "/workspace/data/ae-runs"
PY = "/workspace/workspace/mingdong/.local-tools/amphion-eval-0.14.0-eedcf15/bin/python"
CMP_SRC = "/workspace/workspace/mingdong/repo/amphion-eval-ctx-compare/src"
MODELS = {"g": "ctx-grpo-20261120-delivery-g1106", "jdeliv": "ctx-grpo-20261120-delivery-startj",
          "template": "ctx-clue-target-20261117-tsreal-template",
          "jq8": "ctx-clue-target-20261117-tsreal-template-synth-ts-twins"}
SUITES = ("ts-asr-benchmark@v12", "ts-asr-first-other")
PAIRS = (("g", "template"), ("template", "jq8"), ("jdeliv", "jq8"))


def run(model, suite):
    return [r for r in sorted(glob.glob(f"{RUNS}/{MODELS[model]}/runs/run-*"))
            if suite in (json.load(open(r + "/run.json"))["config"].get("eval_set") or "")
            and json.load(open(r + "/coverage.json")).get("complete")][-1]


def compare(job):
    suite, (a, b) = job
    out = OUT / f"{suite.split('@')[0]}-{a}-vs-{b}.json"
    if not out.exists():
        subprocess.run([PY, "-m", "amphion_eval.cli", "open-audio-llm", "compare", run(a, suite), run(b, suite),
                        "--output", str(out)], check=True, env={**os.environ, "PYTHONPATH": CMP_SRC},
                       stdout=subprocess.DEVNULL)
    return job, json.loads(out.read_text())["overall"]["all/CER"]


def main():
    jobs = [(s, p) for s in SUITES for p in PAIRS]
    with ThreadPoolExecutor(len(jobs)) as pool:
        for (suite, (a, b)), o in pool.map(compare, jobs):
            cells = []
            for m in ("error_rate", "false_alarm_rate", "miss_rate"):
                x = o[m]
                cells.append(f"{m.split('_')[0]} {100 * x['a']:.2f} -> {100 * x['b']:.2f}{'*' if x['significant'] else ''}"
                             f" [{100 * x['ci'][0]:+.2f}, {100 * x['ci'][1]:+.2f}]")
            print(f"{suite} {a} -> {b}: " + "; ".join(cells))


if __name__ == "__main__":
    main()
