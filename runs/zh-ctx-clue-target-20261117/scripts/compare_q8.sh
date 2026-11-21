#!/usr/bin/env bash
# Question 8 analysis on the dev machine, once acp-98-ts-real.sh is done: run completeness, then J against template on
# the two real-speech TS sets (think off). Outputs go to tasks/02-eval/compare/cmp-tsreal-template-synthtstwins-*.json.
# usage: bash scripts/compare_q8.sh
set -u
E=/workspace/workspace/mingdong/repo/oal-clue-target/runs/zh-ctx-clue-target-20261117; C=$E/tasks/02-eval/compare; cd $E
PY=/workspace/workspace/mingdong/.local-tools/amphion-eval-0.14.0-eedcf15/bin/python
CMP_SRC=/workspace/workspace/mingdong/repo/amphion-eval-ctx-compare/src  # AmphionEval !158 (beb1f99), not in 0.14/0.15
# latest run of model <name> on eval set <set> (both suites share an output root)
r() { for d in $(ls -td /workspace/data/ae-runs/ctx-clue-target-20261117-tsreal-$1/runs/run-*); do
        grep -q "\"eval_set\": \"$2\"" $d/run.json && { echo $d; return; }; done; }
rc=0
for n in template template-synth-ts-twins; do
  for s in open-audio-llm/ts-asr-benchmark@v12-20261022 open-audio-llm/ts-asr-first-other@v3-20261105; do
    python3 - "$(r $n $s)" "$n $s" <<'PYEOF' || rc=1
import json, sys
run, name = sys.argv[1], sys.argv[2]
c = json.load(open(f"{run}/coverage.json")); j = json.load(open(f"{run}/run.json"))
print(name, run.rsplit("/", 1)[-1], c["successful"], "/", c["total"], "complete" if c["complete"] else "INCOMPLETE",
      j.get("wandb_url", ""))
sys.exit(0 if c["complete"] and c["successful"] == c["total"] else 1)
PYEOF
  done
done
[ $rc = 0 ] || { echo "eval runs incomplete"; exit 1; }
for s in v12:open-audio-llm/ts-asr-benchmark@v12-20261022 first-other:open-audio-llm/ts-asr-first-other@v3-20261105; do
  k=${s%%:*}; set_=${s#*:}
  PYTHONPATH=$CMP_SRC $PY -m amphion_eval.cli open-audio-llm compare "$(r template $set_)" "$(r template-synth-ts-twins $set_)" \
    --output $C/cmp-tsreal-template-synthtstwins-$k.json >/dev/null 2>$C/cmp-tsreal-template-synthtstwins-$k.log
  echo "cmp $k exit $?"
done
