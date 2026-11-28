#!/usr/bin/env bash
# tasks/29 first-speaker causal test (README "因果验证：是不是最先听到的声音被当成目标"): the local scene-asr variants
# police_scene_bench@scene-asr-cut-20261128 (cut-first / cut-same, built by amphiondata-58, not published) on step 30k /
# attempt 002 / 11.17. Two lanes on dev GPUs 4-7 if free, else one lane on 6-7. All three models are cached.
set -u; umask 000
E=/workspace/workspace/mingdong/repo/oal-clean-teacher/runs/ts-overlap-20261119; X=/workspace/workspace/mingdong/data/amphiondata/ownsim-20261128/scene-asr-cut
L=/workspace/data/ts-overlap-20261119/task29-simneg-asranchor/logs; cd $E; unset no_proxy NO_PROXY
export TEMPLATES=$E/configs/scene-asr-cut CATALOG=$X/catalog ROOTS=$X/roots.json WB_PROXY=http://127.0.0.1:7890
run() { bash scripts/eval_lanes.sh $1 $2-scenecut $3 $4 "scene/cutfirst scene/cutsame" > $L/eval-scenecut-$2.log 2>&1; echo "$2 rc=$? $(date -u +%FT%TZ)" >> $L/eval-scenecut.txt; }
echo "start $(date -u +%FT%TZ) gpus ${GPUS:-67}" > $L/eval-scenecut.txt
if [ "${GPUS:-67}" = 4567 ]; then
  ( run ts-overlap-t29simnegasr-s30000 s30k 4 19970; run whitebox-merge-v18a-es116inc r1117 4 19970 ) & a=$!
  run ts-overlap-meeting002-s30000 a002 6 19980 & b=$!; wait $a $b
else
  run ts-overlap-t29simnegasr-s30000 s30k 6 19980; run ts-overlap-meeting002-s30000 a002 6 19980; run whitebox-merge-v18a-es116inc r1117 6 19980
fi
echo "done $(date -u +%FT%TZ)" >> $L/eval-scenecut.txt
