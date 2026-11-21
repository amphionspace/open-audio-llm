#!/usr/bin/env bash
# Question 5 analysis on the dev machine, once acp-90-nosynth-ctx-eval.sh is done: run completeness, then G's
# context-correction eval v2 against template, think off and on. Outputs go to tasks/02-eval/compare/.
# usage: bash scripts/compare_q5.sh
set -u
E=/workspace/workspace/mingdong/repo/oal-clue-target/runs/zh-ctx-clue-target-20261117; C=$E/tasks/02-eval/compare; cd $E
PY=/workspace/workspace/mingdong/.local-tools/amphion-eval-0.14.0-eedcf15/bin/python
CMP_SRC=/workspace/workspace/mingdong/repo/amphion-eval-ctx-compare/src  # AmphionEval !158 (beb1f99), not in 0.14/0.15
r() { ls -td /workspace/data/ae-runs/ctx-clue-target-20261117-$1/runs/run-* | head -n 1; }
cmp() { PYTHONPATH=$CMP_SRC $PY -m amphion_eval.cli open-audio-llm compare "$(r $1)" "$(r $2)" --output $C/cmp-$3.json \
          >/dev/null 2>$C/cmp-$3.log; echo "cmp $3 exit $?"; }
rc=0
for d in template-no-synth template-no-synth-think; do
  python3 - "$(r $d)" "$d" <<'PYEOF' || rc=1
import json, sys
run, name = sys.argv[1], sys.argv[2]
c = json.load(open(f"{run}/coverage.json")); j = json.load(open(f"{run}/run.json"))
print(name, run.rsplit("/", 1)[-1], c["successful"], "/", c["total"], "complete" if c["complete"] else "INCOMPLETE",
      j.get("wandb_url", ""))
sys.exit(0 if c["complete"] and c["successful"] == c["total"] else 1)
PYEOF
done
[ $rc = 0 ] || { echo "eval runs incomplete"; exit 1; }
cmp template template-no-synth template-nosynth-off &
cmp template-think template-no-synth-think template-nosynth-on &
wait
