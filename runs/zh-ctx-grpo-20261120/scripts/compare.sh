#!/usr/bin/env bash
# README "判定" inputs: each candidate (grpo-avg3, grpo-interp050) against the start (start-j), paired bootstrap
# (ae open-audio-llm compare, AmphionEval !158 as in zh-ctx-clue-target-20261117) on every gated suite.
# Outputs tasks/03-eval/compare/cmp-<candidate>-<suite>.json. Run on the dev machine after eval_after_training.sh.
set -u
E=/workspace/workspace/mingdong/repo/oal-ctx-grpo/runs/zh-ctx-grpo-20261120; C=$E/tasks/03-eval/compare; mkdir -p $C; cd $E
PY=/workspace/workspace/mingdong/.local-tools/amphion-eval-0.14.0-eedcf15/bin/python
CMP_SRC=/workspace/workspace/mingdong/repo/amphion-eval-ctx-compare/src  # AmphionEval !158 (beb1f99), not in 0.14/0.15
rc=0
for cand in grpo-avg3 grpo-interp050; do
  for s in ctx-off ctx-on v12 first-other scene-ts asr-benchmark police; do
    a=$(python3 scripts/eval_runs.py start-j $s) && b=$(python3 scripts/eval_runs.py $cand $s) || { rc=1; continue; }
    PYTHONPATH=$CMP_SRC $PY -m amphion_eval.cli open-audio-llm compare "$a" "$b" --output $C/cmp-$cand-$s.json \
      > /dev/null 2> $C/cmp-$cand-$s.log &
  done
  wait
done
for f in $C/cmp-*.json; do [ -s "$f" ] || { echo "missing $f"; rc=1; }; done
echo "compare exit $rc"; exit $rc
