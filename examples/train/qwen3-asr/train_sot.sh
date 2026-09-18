#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/../../../runs/qwen3-asr-sot}"
export DATA_CONFIG="${DATA_CONFIG:?Run prepare_sot.py and set DATA_CONFIG to train-data.yaml}"
export MAX_STEPS="${MAX_STEPS:-60000}"
export ENCODER_LR="${ENCODER_LR:-2e-5}"
export ALIGNER_LR="${ALIGNER_LR:-2e-5}"
export LLM_LR="${LLM_LR:-1e-5}"
exec bash "$SCRIPT_DIR/train_tsasr_joint.sh" \
  --audio_encoder_batching true --torch_dtype float32 --bf16 true --fp16 false \
  --max_length 4096 --gradient_accumulation_steps 2 \
  --per_device_eval_batch_size 1 --save_only_model true \
  --save_steps 500 --eval_steps 1000 --save_total_limit 3 \
  "$@"
