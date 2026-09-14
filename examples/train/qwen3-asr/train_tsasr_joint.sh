#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/../../../runs/qwen3-asr-ts-joint}"
# At roughly 100 examples/update, 80k updates approach the reference's 7.6M examples.
export MAX_STEPS="${MAX_STEPS:-80000}"
exec bash "$SCRIPT_DIR/train_tsasr_full.sh" \
  --freeze_vit false --freeze_aligner false --freeze_llm false \
  --learning_rate "${LLM_LR:-5e-6}" \
  --vit_lr "${ENCODER_LR:-1e-6}" --aligner_lr "${ALIGNER_LR:-5e-6}" \
  --audio_encoder_parallel false --vit_gradient_checkpointing true \
  --gradient_accumulation_steps 4 --ddp_timeout 7200 \
  --save_steps 1000 --eval_steps 1000 --save_only_model true \
  "$@"
