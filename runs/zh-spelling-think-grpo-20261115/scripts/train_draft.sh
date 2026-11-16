#!/usr/bin/env bash
# 06-train-draft on the delivery node: waits until the 05 eval groups (runner exit files) are done and the GPUs are
# free, then trains avg3's EAGLE-3 draft head (configs/train-draft.yaml, 8 GPUs, backbone frozen).
# usage: train_draft.sh <runner dir> <05 job name> [...]
set -u; umask 002
D=$1; shift
E=/workspace/workspace/mingdong/repo/oal-think-exp/runs/zh-spelling-think-grpo-20261115
for j in "$@"; do until [ -f $D/done/$j.exit ]; do sleep 30; done; echo "$j exit $(cat $D/done/$j.exit)"; done
for i in $(seq 1 60); do
  busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2>2000' | wc -l)
  [ "$busy" = 0 ] && break; sleep 10
done
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader; echo "GPUs busy: $busy"
export WANDB_HTTP_PROXY=http://10.119.17.252:17920 WANDB_HTTPS_PROXY=http://10.119.17.252:17920
cd $E; S=shared/code-v3
echo "start draft $(date -u +%FT%TZ)"
env -u PYTHONPATH PYTHONPATH=$S/src:$S/audio-data-contract-src /workspace/workspace/mingdong/miniconda3/envs/oal-train/bin/python \
  -m open_audio_llm.cli train --config configs/train-draft.yaml
echo "draft exit $? $(date -u +%FT%TZ)"
