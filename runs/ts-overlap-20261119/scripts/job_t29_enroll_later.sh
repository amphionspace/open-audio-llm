#!/usr/bin/env bash
# tasks/29 enrollment-swap causal test (README "判定细则"): the local eval variant ts_asr_real@meeting-enroll-later-20261128
# (start-of-recording enrollments swapped for later utterances of the same speaker, plus a control; built by
# amphiondata-58, not published) on 11.17 / attempt 002 / step 30k, one after another on dev GPUs 6-7 (GPUs 4-5 held by the data side's
# Arabic voice-identity embedding at launch). All three models are cached.
set -u; umask 000
E=/workspace/workspace/mingdong/repo/oal-clean-teacher/runs/ts-overlap-20261119; X=/workspace/workspace/mingdong/data/amphiondata/ownsim-20261128/meeting-enroll-later
L=/workspace/data/ts-overlap-20261119/task29-simneg-asranchor/logs; cd $E; unset no_proxy NO_PROXY
export TEMPLATES=$E/configs/meeting-enroll-later CATALOG=$X/catalog ROOTS=$X/roots.json WB_PROXY=http://127.0.0.1:7890
run() { bash scripts/eval_lanes.sh $1 $2-enrlater $3 $4 "meeting/enrlater" > $L/eval-enrlater-$2.log 2>&1; echo "$2 rc=$? $(date -u +%FT%TZ)" >> $L/eval-enrlater.txt; }
echo "start $(date -u +%FT%TZ)" > $L/eval-enrlater.txt
run ts-overlap-t29simnegasr-s30000 s30k 6 19980
run ts-overlap-meeting002-s30000 a002 6 19980
run whitebox-merge-v18a-es116inc r1117 6 19980
echo "done $(date -u +%FT%TZ)" >> $L/eval-enrlater.txt
