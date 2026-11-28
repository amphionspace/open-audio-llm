#!/usr/bin/env bash
# tasks/30 language-ID localization: ar-asr-test (v6, pinned venv) on the three module-swap models, dev GPUs 4-7.
set -u; umask 000
E=/workspace/workspace/mingdong/repo/oal-clean-teacher/runs/ts-overlap-20261119; D=/workspace/data/ts-overlap-20261119
L=$D/task30-ar-lid/logs; cd $E; unset no_proxy NO_PROXY
export AE_ENV=$D/ar-baseline/ae-ca7d87f-venv TEMPLATES=$E/configs/ar-templates WB_PROXY=http://127.0.0.1:7890
run() { bash scripts/eval_lanes.sh ar-lid-swap-orig$1-on1117 arlid-orig$1 $2 $3 "ar/arasr" > $L/eval-$1.log 2>&1; echo "$1 rc=$? $(date -u +%FT%TZ)" >> $L/eval.txt; }
echo "start $(date -u +%FT%TZ)" > $L/eval.txt
run enc 4 19970 & a=$!
run llm 6 19980 & b=$!
wait $a; run all 4 19970; wait $b; echo "done $(date -u +%FT%TZ)" >> $L/eval.txt
