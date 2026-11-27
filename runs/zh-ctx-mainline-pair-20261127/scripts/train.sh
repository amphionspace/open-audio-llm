#!/usr/bin/env bash
# 02-grpo, after 01-mining (scripts/mining.sh) on the same node: checks the six plans are non-empty, waits for free GPUs,
# then runs the grpo task (configs/train-grpo.yaml, 8 GPUs). Checkpoints 100-400 are uploaded to whai by the trainer.
# Copied from runs/zh-ctx-grpo-20261120/scripts/train.sh (J's round), only this experiment's paths changed; that
# one was copied from 08's scripts/train.sh (runs/zh-spelling-think-grpo-20261115, oal-think-exp) without its wait for a mining
# job on another node, with the ctx plan and this experiment's code.
# Arm k0s2 (second seed): plans from tasks/06-mining-k0s2, task grpo-k0s2.
# usage: train.sh [k0s2]
set -u; umask 002
R=/workspace/workspace/mingdong/repo/oal-ctx-grpo; E=$R/runs/zh-ctx-mainline-pair-20261127
case ${1:-k0} in k0) M=02-mining T=grpo;; k0s2) M=06-mining-k0s2 T=grpo-k0s2;; *) echo "unknown arm $1"; exit 2;; esac
for p in hotword nonspeech miss ts-absent ts-present; do [ -s $E/tasks/$M/plan-$p.jsonl ] || { echo "plan $p empty"; exit 2; }; done
wc -l $E/tasks/$M/plan-*.jsonl
for i in $(seq 1 60); do
  busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2>2000' | wc -l)
  [ "$busy" = 0 ] && break; sleep 10
done
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader; echo "GPUs busy: $busy"
cd $R
S=runs/zh-ctx-mainline-pair-20261127/shared/code-v3
echo "start grpo $(date -u +%FT%TZ)"
env -u PYTHONPATH /workspace/workspace/mingdong/repo/open-audio-llm/third_party/amphionkeys/load.sh sensecore -- \
  env PYTHONPATH=$S/src:$S/audio-data-contract-src /workspace/workspace/mingdong/miniconda3/envs/oal-train/bin/python \
  -m open_audio_llm.cli experiment run --config runs/zh-ctx-mainline-pair-20261127/experiment.yaml --task $T
rc=$?; echo "grpo exit $rc $(date -u +%FT%TZ)"; exit $rc
