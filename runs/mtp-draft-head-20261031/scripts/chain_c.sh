#!/usr/bin/env bash
# Self-distillation A/B: fine-tune the 5000-step EAGLE-3 head 500 steps on the same fresh batches,
# reference transcripts vs the frozen model's greedy outputs.
T=/workspace/workspace/mingdong/repo/oal-mtp/runs/mtp-draft-head-20261031/scripts/train_draft_acp.sh
A=/workspace/workspace/mingdong/repo/oal-mtp/runs/mtp-draft-head-20261031/tasks/01-validate-draft/attempts
I=/workspace/workspace/mingdong/repo/oal-mtp/runs/mtp-draft-head-20261031/tasks/01-validate-draft/attempts/007/artifacts/training/checkpoint-5000/model.safetensors
DATA=/workspace/workspace/mingdong/repo/oal-mtp/runs/mtp-draft-head-20261031/configs/train-data-ft.yaml
bash $T $A/008/artifacts/training 500 100 16 eagle3 mtp-draft-head-20261031-ft-eagle3-reference reference $I 3e-5 20 $DATA; echo "ft-reference exit $?"
bash $T $A/009/artifacts/training 500 100 16 eagle3 mtp-draft-head-20261031-ft-eagle3-greedy greedy $I 3e-5 20 $DATA; echo "ft-greedy exit $?"
