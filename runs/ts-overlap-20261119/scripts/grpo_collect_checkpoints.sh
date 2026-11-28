#!/usr/bin/env bash
# Copy each saved GRPO checkpoint's weights (no optimizer/RNG/DeepSpeed state) for evaluation before the uploader removes
# the local copy; print one line per copy and exit when training stops. The copy goes into the ae model cache (amphion-eval
# docs/model-cache.md: uploaded to whai, evicted when idle) and /workspace/data/ae-runs/<name>/checkpoint links to it;
# `ae-models.sh fetch ts-overlap-grpo-<name>-ckpt<step>` brings an evicted one back.
# usage: grpo_collect_checkpoints.sh <attempt> [task dir under tasks/, default 12-grpo-transcribe] [name, default the attempt]
# Copied from runs/amphionasr2-grpo-20261101/scripts/collect_checkpoints.sh (oal-grpo), names prefixed ts-overlap-grpo-.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd); ATTEMPT=$1; TASK=${2:-12-grpo-transcribe}; NAME=${3:-$ATTEMPT}
TRAIN="$ROOT/tasks/$TASK/attempts/$ATTEMPT/artifacts/training"
STATUS="$ROOT/tasks/$TASK/attempts/$ATTEMPT/status.json"
M=/workspace/workspace/mingdong/scratch/acp-dev/ae-models.sh
while true; do
  for ckpt in "$TRAIN"/checkpoint-*; do
    [ -f "$ckpt/trainer_state.json" ] || continue
    step=${ckpt##*-}; name=ts-overlap-grpo-$NAME-ckpt$step; dest=/workspace/data/ae-runs/$name/checkpoint
    [ -L "$dest" ] || [ -f "$dest/.complete" ] && continue
    stage=$($M stage) || { echo "FAILED staging checkpoint-$step"; continue; }
    copied=1
    for file in "$ckpt"/*; do
      case "${file##*/}" in global_step*|optimizer*|rng_state*|scheduler.pt|zero_to_*.py) continue ;; esac
      cp -r "$file" "$stage/" || copied=0
    done
    if [ "$copied" = 1 ] && touch "$stage/.complete" && entry=$($M put "$stage" --name $name); then
      mkdir -p "$(dirname "$dest")" && ln -sfn "$entry" "$dest"; echo "copied checkpoint-$step -> $dest -> $entry"
    else
      echo "FAILED copying checkpoint-$step"
    fi
  done
  state=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['execution'])" "$STATUS" 2>/dev/null)
  if [ "$state" != running ] && [ "$state" != starting ]; then echo "training $state"; exit 0; fi
  sleep 60
done
