#!/usr/bin/env bash
# Delivery-line check of the leak 45 found on the board line (TS training makes plain ASR drop the second speaker):
# scene-asr (11.17 delivery template) on task 28 attempt 002 (attribution: task 29 vs the shared recipe), dev GPUs 6-7. Model is already cached.
set -u; umask 000
E=/workspace/workspace/mingdong/repo/oal-clean-teacher/runs/ts-overlap-20261119; L=/workspace/data/ts-overlap-20261119/task29-simneg-asranchor/logs
cd $E; unset no_proxy NO_PROXY; export TEMPLATES=$E/configs WB_PROXY=http://127.0.0.1:7890
bash scripts/eval_lanes.sh ts-overlap-meeting002-s30000 a002-sceneasr 6 19980 "eval-cand-soup/scene-asr-asronly-data" > $L/eval-sceneasr-a002.log 2>&1
echo "sceneasr rc=$? $(date -u +%FT%TZ)" >> $L/eval-sceneasr-a002.log
