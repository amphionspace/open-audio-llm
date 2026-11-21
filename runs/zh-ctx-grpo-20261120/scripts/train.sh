#!/usr/bin/env bash
# 02-grpo, after 01-mining (scripts/mining.sh) on the same node: checks the six plans are non-empty, waits for free GPUs,
# then runs the grpo task (configs/train-grpo.yaml, 8 GPUs). Checkpoints 100-400 are uploaded to whai by the trainer.
# Copied from 08's scripts/train.sh (runs/zh-spelling-think-grpo-20261115, oal-think-exp) without its wait for a mining
# job on another node, with the ctx plan and this experiment's code.
# usage: train.sh
set -u; umask 002
R=/workspace/workspace/mingdong/repo/oal-ctx-grpo; E=$R/runs/zh-ctx-grpo-20261120
for p in hotword nonspeech miss ts-absent ts-present ctx; do [ -s $E/tasks/01-mining/plan-$p.jsonl ] || { echo "plan $p empty"; exit 2; }; done
wc -l $E/tasks/01-mining/plan-*.jsonl
for i in $(seq 1 60); do
  busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2>2000' | wc -l)
  [ "$busy" = 0 ] && break; sleep 10
done
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader; echo "GPUs busy: $busy"
cd $R
S=runs/zh-ctx-grpo-20261120/shared/code-v2
echo "start grpo $(date -u +%FT%TZ)"
env -u PYTHONPATH /workspace/workspace/mingdong/repo/open-audio-llm/third_party/amphionkeys/load.sh sensecore -- \
  env PYTHONPATH=$S/src:$S/audio-data-contract-src /workspace/workspace/mingdong/miniconda3/envs/oal-train/bin/python \
  -m open_audio_llm.cli experiment run --config runs/zh-ctx-grpo-20261120/experiment.yaml --task grpo
rc=$?; echo "grpo exit $rc $(date -u +%FT%TZ)"; exit $rc
