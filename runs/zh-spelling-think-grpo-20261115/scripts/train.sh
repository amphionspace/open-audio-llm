#!/usr/bin/env bash
# 02-grpo: waits for 01-mining (runner exit file, five non-empty plans) and free GPUs, then runs the grpo task
# (configs/train-grpo.yaml, 8 GPUs, port 29719). Checkpoints 100-400 are uploaded to whai by the trainer.
# usage: train.sh <runner dir> <mining job name>
set -u; umask 002
D=$1; J=$2
E=/workspace/workspace/mingdong/repo/oal-think-exp/runs/zh-spelling-think-grpo-20261115
until [ -f $D/done/$J.exit ]; do sleep 30; done
[ "$(cat $D/done/$J.exit)" = 0 ] || { echo "mining failed"; exit 2; }
for p in hotword nonspeech miss ts-absent ts-present; do [ -s $E/tasks/01-mining/plan-$p.jsonl ] || { echo "plan $p empty"; exit 2; }; done
wc -l $E/tasks/01-mining/plan-*.jsonl
for i in $(seq 1 60); do
  busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2>2000' | wc -l)
  [ "$busy" = 0 ] && break; sleep 10
done
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader; echo "GPUs busy: $busy"
cd /workspace/workspace/mingdong/repo/oal-think-exp
S=runs/zh-spelling-think-grpo-20261115/shared/code-v2
echo "start grpo $(date -u +%FT%TZ)"
env -u PYTHONPATH /workspace/workspace/mingdong/repo/open-audio-llm/third_party/amphionkeys/load.sh sensecore -- \
  env PYTHONPATH=$S/src:$S/audio-data-contract-src /workspace/workspace/mingdong/miniconda3/envs/oal-train/bin/python \
  -m open_audio_llm.cli experiment run --config runs/zh-spelling-think-grpo-20261115/experiment.yaml --task grpo
echo "grpo exit $? $(date -u +%FT%TZ)"
