#!/usr/bin/env bash
# tasks/23 smoke on one node: 2 GPUs, 4 steps, warm start from task-21 aux step 8000, save at step 3 (heads included).
set -u; mkdir -p /workspace/data/ts-overlap-20261119/task23-e2e-scale/logs
export INIT=/workspace/data/ts-overlap-20261119/task21-e2e/aux/step8000.pt HEAD_LAYERS=5,11 OMP_NUM_THREADS=1 AUDIO_DATA_ROOTS_FILE=/workspace/data/registry/roots.json WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/workspace/workspace/mingdong/repo/oal-clean-teacher/runs/ts-overlap-20261119/shared/code-v4/src NCCL_IB_TIMEOUT=22 NCCL_IB_RETRY_CNT=13 NCCL_IB_AR_THRESHOLD=0
CUDA_VISIBLE_DEVICES=0,1 /workspace/workspace/mingdong/miniconda3/envs/oal-train/bin/torchrun --nproc_per_node 2 --master_port 29911 /workspace/workspace/mingdong/.claude/tmp/claude-1001/-workspace-workspace-mingdong-repo-open-audio-llm/ca9aa6d0-4ecc-4cfc-b526-5322bf6c9de5/scratchpad/ovdiag/probe/ft_e2e2.py aux /workspace/data/ts-overlap-20261119/task23-e2e-scale/smoke 4 4 4 4.0 smoke 3 > /workspace/data/ts-overlap-20261119/task23-e2e-scale/logs/smoke.log 2>&1
echo "smoke rc=$?"; grep -E "arm |^step|saved|Error|Traceback" /workspace/data/ts-overlap-20261119/task23-e2e-scale/logs/smoke.log
/workspace/workspace/mingdong/miniconda3/envs/oal-train/bin/python -c "import torch; c=torch.load('/workspace/data/ts-overlap-20261119/task23-e2e-scale/smoke/step3.pt', map_location='cpu'); print('heads', sorted(c['heads']), 'init_step', c['init_step'], 'tensors', len(c['state']))"
rm -rf /workspace/data/ts-overlap-20261119/task23-e2e-scale/smoke
