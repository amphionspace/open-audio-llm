#!/usr/bin/env bash
# scene-asr (鼎桥场景普通识别) for the context-correction SFT arms of runs/zh-ctx-clue-target-20261117, to find which
# part of J's recipe drops spans of 30 s+ recordings (README "结果：交付评估"). Each arm gets one ASR server on its own
# GPU (4-7, beside J's delivery run on 0-3); the configs are make_delivery_configs.py's pair a, pruned to scene-asr.
# usage: bash scripts/make_scene_asr_arm_configs.sh
set -eu
cd "$(dirname "$0")/.."
C=/workspace/data/ae-state/model-cache/entries
i=0
for arm in template:184eb425f4e5 template-no-synth:90c9d6f4fa35 template-no-empty-think:c4ee45e12c3c clue-target:c821c0d481b4; do
  name=sft-${arm%%:*}; gpu=$((4 + i)); port=$((18910 + 2 * i))
  python3 scripts/make_delivery_configs.py $name $C/checkpoint-3000-${arm##*:} $gpu $gpu $port $gpu $gpu $((port + 40)) > /dev/null
  rm -r configs/eval-delivery-$name-b
  find configs/eval-delivery-$name-a -maxdepth 1 -type f ! -name handoff.json ! -name serve-asr.yaml \
    ! -name after-training-delivery-scene-asr.yaml -delete
  i=$((i + 1))
done
