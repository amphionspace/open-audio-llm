"""README "GRPO 配方对照": the two GRPO rounds already run from 11.17 (GRPO experiment amphionasr2-grpo-20261101, worktree
oal-grpo: v26 = round-9 recipe, configs/train-grpo-v26.yaml; v27 = the same with es116's Spanish replay, train-grpo-v27.yaml),
avg3 of each against 11.17's own runs of 2026-11-17 (whitebox-merge-v18a-es116inc, same vLLM 0.18 stack, enroll 5 s;
11.17's later enrollment-length runs are excluded), paired bootstrap 95% CI (ae open-audio-llm compare, AmphionEval !158),
every ts-asr eval set both have. Writes tasks/05-recipe/<model>/<eval set>.json and prints error/FA/miss rates.
usage: python3 compare_grpo_from_1117.py
"""
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
RUNS = Path("/workspace/data/ae-runs")
PY = "/workspace/workspace/mingdong/.local-tools/amphion-eval-0.14.0-eedcf15/bin/python"
CMP_SRC = "/workspace/workspace/mingdong/repo/amphion-eval-ctx-compare/src"
BASE, BEFORE = "whitebox-merge-v18a-es116inc", "2026-11-18"
MODELS = {"v26avg3": "amphionasr2-grpo-v26avg3-ckpt400", "v27avg3": "amphionasr2-grpo-v27avg3-ckpt400"}
ENROLL_SECONDS = 5.0  # the serving contract (AMPHION_TSASR_ENROLL_SEC)


def runs(model, before=None):
    found = {}
    for run in sorted((RUNS / model / "runs").glob("run-*")):
        r = json.loads((run / "run.json").read_text())
        c, s = r["config"], json.loads((run / "settings.json").read_text())["effective"]
        cov = json.loads((run / "coverage.json").read_text()) if (run / "coverage.json").exists() else {}
        if before and r["created_at"] >= before:
            continue
        if c.get("evaluator") == "ts-asr" and s.get("enroll_seconds") == ENROLL_SECONDS and cov.get("complete"):
            found[c.get("eval_set") or str(c.get("data_config"))] = run
    return found


def main():
    base = runs(BASE, BEFORE)
    for name, model in MODELS.items():
        for ident, run in runs(model).items():
            out = HERE / "tasks/05-recipe" / name / (ident.split("/")[-1] + ".json")
            out.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run([PY, "-m", "amphion_eval.cli", "open-audio-llm", "compare", str(base[ident]), str(run),
                            "--output", str(out)], env={"PYTHONPATH": CMP_SRC, "PATH": "/usr/bin:/bin"},
                           stdout=subprocess.DEVNULL, check=True)
            ((_, ms),) = json.loads(out.read_text())["overall"].items()
            cells = [f"{n} {100 * x['a']:.2f}→{100 * x['b']:.2f} [{100 * x['ci'][0]:+.2f},{100 * x['ci'][1]:+.2f}]"
                     f"{'*' if x['significant'] else ''}" for n, x in ms.items()
                     if n in ("error_rate", "false_alarm_rate", "miss_rate")]
            print(name, ident.split("/")[-1], "|", "  ".join(cells))


if __name__ == "__main__":
    main()
