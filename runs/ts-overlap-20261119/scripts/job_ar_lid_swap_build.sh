#!/usr/bin/env bash
# tasks/30 language-ID localization: build the three module-swap models (11.17 format) and put them in the ae model cache.
set -u
E=/workspace/workspace/mingdong/repo/oal-clean-teacher/runs/ts-overlap-20261119; M=/workspace/workspace/mingdong/scratch/acp-dev/ae-models.sh
BASE=/workspace/data/ae-state/model-cache/entries/model-afea989da22b77adc0ce9365; DONOR=/workspace/model/Qwen3-ASR-1.7B
L=/workspace/data/ts-overlap-20261119/task30-ar-lid/logs; mkdir -p $L
for g in enc llm all; do
  W=$(bash $M stage) && echo "stage $W" > $L/build-$g.log && \
  env -u PYTHONPATH /workspace/workspace/mingdong/miniconda3/envs/oal-train/bin/python -I $E/scripts/swap_modules.py $BASE $DONOR $W $g >> $L/build-$g.log 2>&1 && \
  rm -f $W/.complete && bash $M put $W --name ar-lid-swap-orig$g-on1117 >> $L/build-$g.log 2>&1; echo "$g rc=$? $(date -u +%FT%TZ)" | tee -a $L/build-$g.log
done
