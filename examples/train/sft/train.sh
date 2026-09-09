#!/usr/bin/env bash
# Smoke or short-run SFT recipe for Open Audio-LLM via ms-swift.

set -euo pipefail
umask 0000

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
export MASTER_PORT="${MASTER_PORT:-29501}"

MODEL="${MODEL:?Set MODEL to an Open Audio-LLM HF checkpoint}"
DATA_CONFIG="${DATA_CONFIG:?Set DATA_CONFIG to a Catalog training YAML config}"
export AUDIO_DATA_CONTRACT_ROOT="${AUDIO_DATA_CONTRACT_ROOT:-$(dirname "$REPO_ROOT")/audio-data-contract}"
export AUDIO_DATA_CATALOG="${AUDIO_DATA_CATALOG:-$AUDIO_DATA_CONTRACT_ROOT/catalog}"
export AUDIO_DATA_ROOTS_FILE="${AUDIO_DATA_ROOTS_FILE:-$AUDIO_DATA_CONTRACT_ROOT/roots.json}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/open_audio_llm_sft_smoke}"
DEEPSPEED="${DEEPSPEED:-zero2}"
MAX_STEPS="${MAX_STEPS:-1}"
resume_args=()
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  resume_args=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi

python -m torch.distributed.run \
  --nproc_per_node "$NPROC_PER_NODE" --master_port "$MASTER_PORT" \
  --module open_audio_llm.integrations.ms_swift.train sft \
  --model "$MODEL" \
  --external_plugins "$REPO_ROOT/src/open_audio_llm/integrations/ms_swift/register_audio_llm.py" \
  --data_config "$DATA_CONFIG" \
  "${resume_args[@]}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-0}" \
  --freeze_vit "${FREEZE_VIT:-true}" \
  --freeze_aligner "${FREEZE_ALIGNER:-false}" \
  --freeze_llm "${FREEZE_LLM:-false}" \
  --tuner_type "${TUNER_TYPE:-lora}" \
  --deepspeed "$DEEPSPEED" \
  --torch_dtype "${TORCH_DTYPE:-bfloat16}" \
  --lora_rank "${LORA_RANK:-64}" \
  --lora_alpha "${LORA_ALPHA:-128}" \
  --target_modules "${TARGET_MODULES:-all-linear}" \
  --max_length "${MAX_LENGTH:-2048}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-1}" \
  --learning_rate "${LEARNING_RATE:-1e-5}" \
  --max_steps "$MAX_STEPS" \
  --lr_scheduler_type "${LR_SCHEDULER_TYPE:-cosine}" \
  --warmup_ratio "${WARMUP_RATIO:-0.05}" \
  --gradient_checkpointing "${GRADIENT_CHECKPOINTING:-true}" \
  --save_steps "${SAVE_STEPS:-1}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT:-2}" \
  --eval_steps "${EVAL_STEPS:-1}" \
  --logging_steps "${LOGGING_STEPS:-1}" \
  --output_dir "$OUTPUT_DIR" \
  --ddp_timeout "${DDP_TIMEOUT:-180000000}"
