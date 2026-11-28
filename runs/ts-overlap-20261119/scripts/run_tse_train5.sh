#!/usr/bin/env bash
# tasks/19 v5 (target-selection training: uniform and wrong-enrollment crops), warm start from v4 final.
set -u; T=/workspace/workspace/mingdong/.claude/tmp/claude-1001/-workspace-workspace-mingdong-repo-open-audio-llm/ca9aa6d0-4ecc-4cfc-b526-5322bf6c9de5/scratchpad/ovdiag/tse; W=/workspace/data/ts-overlap-20261119/task19-tse; mkdir -p $W/logs
export INIT=$W/ckpt4/final-best.pt OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 AUDIO_DATA_ROOTS_FILE=/workspace/data/registry/roots.json
source /workspace/workspace/mingdong/.config/open-audio-llm/wandb-cluster.sh >/dev/null 2>&1
export WANDB_HTTP_PROXY=http://10.119.17.252:17922 WANDB_HTTPS_PROXY=http://10.119.17.252:17922 WANDB_DIR=$W
/workspace/workspace/mingdong/miniconda3/envs/oal-train/bin/torchrun --nproc_per_node 8 --master_port 29805 $T/tse_train5.py $W/ckpt5 $W/dev 6000 2 14 1500 > $W/logs/train5.log 2>&1
echo "tse train5 rc=$? $(date -u +%FT%TZ)"
