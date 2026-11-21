#!/usr/bin/env bash
# Question 7 analysis on the dev machine, once acp-97-synth-ts-twins.sh is done: run completeness, then J against
# template on scene TS and on the context-correction eval v2 (think off and on). Outputs go to tasks/02-eval/compare/.
# usage: bash scripts/compare_q7.sh
set -u
E=/workspace/workspace/mingdong/repo/oal-clue-target/runs/zh-ctx-clue-target-20261117; C=$E/tasks/02-eval/compare; cd $E
PY=/workspace/workspace/mingdong/.local-tools/amphion-eval-0.14.0-eedcf15/bin/python
CMP_SRC=/workspace/workspace/mingdong/repo/amphion-eval-ctx-compare/src  # AmphionEval !158 (beb1f99), not in 0.14/0.15
r() { ls -td /workspace/data/ae-runs/ctx-clue-target-20261117-$1/runs/run-* | head -n 1; }
cmp() { PYTHONPATH=$CMP_SRC $PY -m amphion_eval.cli open-audio-llm compare "$(r $1)" "$(r $2)" --output $C/cmp-$3.json \
          >/dev/null 2>$C/cmp-$3.log; echo "cmp $3 exit $?"; }
rc=0
for d in template-synth-ts-twins template-synth-ts-twins-think scene-template-synth-ts-twins; do
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
cmp scene-template scene-template-synth-ts-twins scene-template-synthtstwins &
cmp template template-synth-ts-twins template-synthtstwins-off &
cmp template-think template-synth-ts-twins-think template-synthtstwins-on &
wait
