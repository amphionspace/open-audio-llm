#!/usr/bin/env bash
# scene-asr for four context-correction SFT arms (scripts/make_scene_asr_arm_configs.sh), one ASR server each on
# GPUs 4-7, beside J's delivery run (scripts/acp-delivery-startj.sh) on 0-3.
set -u; umask 002
E=/workspace/workspace/mingdong/repo/oal-ctx-grpo/runs/zh-ctx-grpo-20261120; cd $E
AE=/workspace/workspace/mingdong/.local-tools/amphion-eval-0.14.0-eedcf15/bin/ae
PROXY=http://10.119.17.252:17920
pids=(); names=()
for d in sft-template sft-template-no-synth sft-template-no-empty-think sft-clue-target; do
  bash scripts/run_eval.sh configs/eval-delivery-$d-a $PROXY $AE delivery-scene-asr & pids+=($!); names+=($d)
  sleep 30
done
rc=0; for i in "${!pids[@]}"; do wait ${pids[$i]}; x=$?; echo "${names[$i]} exit $x $(date -u +%FT%TZ)"; [ $x -ne 0 ] && rc=$x; done
exit $rc
