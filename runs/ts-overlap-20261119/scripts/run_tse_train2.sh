#!/usr/bin/env bash
# tasks/19 v2 (CAM++ speaker vector, SI-SDR loss): train the mask estimator on 8 GPUs (on-the-fly scenes, 15 generator workers per GPU).
set -u; T=/workspace/workspace/mingdong/.claude/tmp/claude-1001/-workspace-workspace-mingdong-repo-open-audio-llm/ca9aa6d0-4ecc-4cfc-b526-5322bf6c9de5/scratchpad/ovdiag/tse; W=/workspace/data/ts-overlap-20261119/task19-tse; mkdir -p $W/logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 AUDIO_DATA_ROOTS_FILE=/workspace/data/registry/roots.json
source /workspace/workspace/mingdong/.config/open-audio-llm/wandb-cluster.sh >/dev/null 2>&1
export WANDB_HTTP_PROXY=http://10.119.17.252:17922 WANDB_HTTPS_PROXY=http://10.119.17.252:17922 WANDB_DIR=$W
/workspace/workspace/mingdong/miniconda3/envs/oal-train/bin/torchrun --nproc_per_node 8 --master_port 29802 $T/tse_train2.py $W/ckpt2 $W/dev 20000 8 15 2000 > $W/logs/train2.log 2>&1
echo "tse train2 rc=$? $(date -u +%FT%TZ)"
