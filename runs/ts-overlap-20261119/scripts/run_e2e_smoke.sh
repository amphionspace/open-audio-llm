#!/usr/bin/env bash
# tasks/21 smoke: both arms, 3 steps, micro-batch 8, one GPU each.
set -u; mkdir -p /workspace/data/ts-overlap-20261119/task21-e2e/logs
export OMP_NUM_THREADS=2 AUDIO_DATA_ROOTS_FILE=/workspace/data/registry/roots.json WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/workspace/workspace/mingdong/repo/oal-clean-teacher/runs/ts-overlap-20261119/shared/code-v4/src
PY=/workspace/workspace/mingdong/miniconda3/envs/oal-train/bin/python
CUDA_VISIBLE_DEVICES=0 $PY /workspace/workspace/mingdong/.claude/tmp/claude-1001/-workspace-workspace-mingdong-repo-open-audio-llm/ca9aa6d0-4ecc-4cfc-b526-5322bf6c9de5/scratchpad/ovdiag/probe/ft_e2e.py ce /workspace/data/ts-overlap-20261119/task21-e2e/smoke-ce 3 8 4 2.0 smoke 1000 > /workspace/data/ts-overlap-20261119/task21-e2e/logs/smoke-ce.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY /workspace/workspace/mingdong/.claude/tmp/claude-1001/-workspace-workspace-mingdong-repo-open-audio-llm/ca9aa6d0-4ecc-4cfc-b526-5322bf6c9de5/scratchpad/ovdiag/probe/ft_e2e.py aux /workspace/data/ts-overlap-20261119/task21-e2e/smoke-aux 3 8 4 2.0 smoke 1000 > /workspace/data/ts-overlap-20261119/task21-e2e/logs/smoke-aux.log 2>&1 &
wait; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -2
grep -h -E "arm |^step|saved|Error|Traceback" /workspace/data/ts-overlap-20261119/task21-e2e/logs/smoke-*.log; rm -rf /workspace/data/ts-overlap-20261119/task21-e2e/smoke-ce /workspace/data/ts-overlap-20261119/task21-e2e/smoke-aux
