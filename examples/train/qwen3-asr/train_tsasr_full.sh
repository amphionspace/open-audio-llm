#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DATA_CONFIG="${DATA_CONFIG:-$SCRIPT_DIR/../../configs/data/qwen3_asr_ts_full.yaml}"
export OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/../../../runs/qwen3-asr-ts-full}"
export MAX_STEPS="${MAX_STEPS:-12000}"
exec bash "$SCRIPT_DIR/train_tsasr.sh" \
  --tuner_type full --learning_rate 5e-6 \
  --retention_teacher "${RETENTION_TEACHER:-${MODEL:?Set MODEL to pretrained Qwen3-ASR}}" \
  --warmup_steps 300 --warmup_ratio 0 \
  --optim adamw_torch_fused --weight_decay 0.01 --max_grad_norm 1.0 \
  --gradient_checkpointing true --vit_gradient_checkpointing false \
  --use_logits_to_keep false --prediction_loss_only true \
  --save_steps 500 --eval_steps 500 --save_total_limit 4 \
  "$@"
