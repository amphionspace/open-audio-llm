#!/usr/bin/env bash
# Delivery-line check of the leak 45 found on the board line (TS training makes plain ASR drop the second speaker):
# scene-asr (11.17 delivery template) on the task 29 step-30k candidate, dev GPUs 6-7. Model is already cached.
set -u; umask 000
E=/workspace/workspace/mingdong/repo/oal-clean-teacher/runs/ts-overlap-20261119; L=/workspace/data/ts-overlap-20261119/task29-simneg-asranchor/logs
cd $E; unset no_proxy NO_PROXY; export TEMPLATES=$E/configs WB_PROXY=http://127.0.0.1:7890
bash scripts/eval_lanes.sh ts-overlap-t29simnegasr-s30000 t29s30k-sceneasr 6 19980 "eval-cand-soup/scene-asr-asronly-data" > $L/eval-sceneasr-s30000.log 2>&1
echo "sceneasr rc=$? $(date -u +%FT%TZ)" >> $L/eval-sceneasr-s30000.log
