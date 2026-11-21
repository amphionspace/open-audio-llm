"""Latest complete AmphionEval run of one model on one suite (scripts/eval_after_training.sh's outputs).

Suites share an output root (/workspace/data/ae-runs/ctx-grpo-20261120-<model>); a run is told apart by its recorded
config. Think-on context correction has its own root (<model>-think).
usage: python3 eval_runs.py MODEL SUITE   (prints the run directory)
"""
import json
import sys
from pathlib import Path

RUNS = Path("/workspace/data/ae-runs")
SUITES = {
    "ctx-off": ("", lambda c: c.get("eval_set") == "asr/context-correction-zh@v2-20261117"),
    "ctx-on": ("-think", lambda c: c.get("eval_set") == "asr/context-correction-zh@v2-20261117"),
    "v12": ("", lambda c: c.get("eval_set") == "open-audio-llm/ts-asr-benchmark@v12-20261022"),
    "first-other": ("", lambda c: c.get("eval_set") == "open-audio-llm/ts-asr-first-other@v3-20261105"),
    "scene-ts": ("", lambda c: c.get("eval_set") == "tdtech/scene-ts@v1-20261028"),
    "asr-benchmark": ("", lambda c: str(c.get("data_config", "")).endswith("eval-ckpt15000/asr-benchmark-data.yaml")),
    "police": ("", lambda c: str(c.get("data_config", "")).endswith("police-full/police-data.yaml")),
}


def latest(model, suite):
    suffix, match = SUITES[suite]
    found = []
    for run in sorted((RUNS / f"ctx-grpo-20261120-{model}{suffix}/runs").glob("run-*")):
        config = json.loads((run / "run.json").read_text())["config"]
        coverage = json.loads((run / "coverage.json").read_text()) if (run / "coverage.json").exists() else {}
        if match(config) and coverage.get("complete") and coverage.get("successful") == coverage.get("total"):
            found.append(run)
    if not found:
        raise SystemExit(f"no complete {suite} run of {model}")
    return found[-1]


if __name__ == "__main__":
    print(latest(sys.argv[1], sys.argv[2]))
