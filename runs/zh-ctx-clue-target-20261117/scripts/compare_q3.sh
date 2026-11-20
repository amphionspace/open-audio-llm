#!/usr/bin/env bash
# Question 3 analysis on the dev machine, once template-implicit-noexplain's two eval runs are done (same tools as
# questions 1-2): run completeness, paired bootstrap against template (the gate) and clue-target-implicit-noexplain
# (reference), scene TS against template (acp-60-scene-ts.sh), Emilia split by whether the original transcript
# needed correcting, and the think diagnostic.
# Outputs go to tasks/02-eval/compare/. usage: bash scripts/compare_q3.sh
set -u
E=/workspace/workspace/mingdong/repo/oal-clue-target/runs/zh-ctx-clue-target-20261117; C=$E/tasks/02-eval/compare; cd $E
PY=/workspace/workspace/mingdong/.local-tools/amphion-eval-0.14.0-eedcf15/bin/python
CMP_SRC=/workspace/workspace/mingdong/repo/amphion-eval-ctx-compare/src  # AmphionEval !158 (beb1f99), not in 0.14/0.15
r() { ls -td /workspace/data/ae-runs/ctx-clue-target-20261117-$1/runs/run-* | head -n 1; }
cmp() { PYTHONPATH=$CMP_SRC $PY -m amphion_eval.cli open-audio-llm compare "$(r $1)" "$(r $2)" --output $C/cmp-$3.json \
          >/dev/null 2>$C/cmp-$3.log; echo "cmp $3 exit $?"; }
rc=0
for d in template-implicit-noexplain template-implicit-noexplain-think scene-template scene-template-implicit-noexplain; do
  python3 - "$(r $d)" "$d" <<'EOF' || rc=1
import json, sys
run, name = sys.argv[1], sys.argv[2]
c = json.load(open(f"{run}/coverage.json")); j = json.load(open(f"{run}/run.json"))
print(name, run.rsplit("/", 1)[-1], c["successful"], "/", c["total"], "complete" if c["complete"] else "INCOMPLETE",
      j.get("wandb_url", ""))
sys.exit(0 if c["complete"] and c["successful"] == c["total"] else 1)
EOF
done
[ $rc = 0 ] || { echo "eval runs incomplete"; exit 1; }
cmp template-think template-implicit-noexplain-think template-tplimplicit-on &
cmp template template-implicit-noexplain template-tplimplicit-off &
cmp clue-target-implicit-noexplain-think template-implicit-noexplain-think noexplain-tplimplicit-on &
cmp clue-target-implicit-noexplain template-implicit-noexplain noexplain-tplimplicit-off &
cmp scene-template scene-template-implicit-noexplain scene-template-tplimplicit &  # scene TS, think off
wait
$PY scripts/ctx_v2_mined_groups.py template-on="$(r template-think)" tplimplicit-on="$(r template-implicit-noexplain-think)" \
  noexplain-on="$(r clue-target-implicit-noexplain-think)" > $C/mined-groups-tplimplicit.md 2>&1; echo "mined exit $?"
$PY scripts/ctx_v2_think_clue_target.py "$(r template-implicit-noexplain-think)" "$(r template-implicit-noexplain)" \
  > $C/think-clue-target-template-implicit-noexplain.md 2>$C/think-clue-target-template-implicit-noexplain.log
echo "think exit $?"
