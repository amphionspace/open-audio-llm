#!/usr/bin/env bash
# Question 9 analysis on the dev machine, once acp-99-k.sh is done: run completeness, then K against J on the
# context-correction eval v2 (think off and on). Outputs go to tasks/02-eval/compare/. The delivery suites (K against
# 11.17) are compared in the GRPO experiment: python3 scripts/compare_delivery.py r1117 k; scene_asr_drops.py r1117 j k.
# usage: bash scripts/compare_q9.sh
set -u
E=/workspace/workspace/mingdong/repo/oal-clue-target/runs/zh-ctx-clue-target-20261117; C=$E/tasks/02-eval/compare; cd $E
PY=/workspace/workspace/mingdong/.local-tools/amphion-eval-0.14.0-eedcf15/bin/python
CMP_SRC=/workspace/workspace/mingdong/repo/amphion-eval-ctx-compare/src  # AmphionEval !158 (beb1f99), not in 0.14/0.15
r() { ls -td /workspace/data/ae-runs/ctx-clue-target-20261117-$1/runs/run-* | head -n 1; }
cmp() { PYTHONPATH=$CMP_SRC $PY -m amphion_eval.cli open-audio-llm compare "$(r $1)" "$(r $2)" --output $C/cmp-$3.json \
          >/dev/null 2>$C/cmp-$3.log; echo "cmp $3 exit $?"; }
rc=0
for d in template-twins-dialog-final-1117 template-twins-dialog-final-1117-think; do
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
cmp template-synth-ts-twins template-twins-dialog-final-1117 synthtstwins-k-off &
cmp template-synth-ts-twins-think template-twins-dialog-final-1117-think synthtstwins-k-on &
wait
