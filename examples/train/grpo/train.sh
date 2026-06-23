#!/usr/bin/env bash
# Smoke or short-run GRPO recipe for Open Audio-LLM via ms-swift.

set -euo pipefail
umask 0000

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
export MASTER_PORT="${MASTER_PORT:-29501}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"

MODEL="${MODEL:?Set MODEL to an Open Audio-LLM HF checkpoint}"
DATASET="${DATASET:?Set DATASET to a ShareGPT GRPO JSONL file}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/open_audio_llm_grpo_smoke}"
MAX_STEPS="${MAX_STEPS:-1}"

VLLM_ARGS=()
if [[ "${USE_VLLM:-false}" == "true" ]]; then
  VLLM_ARGS=(
    --use_vllm true
    --vllm_mode server
    --vllm_server_host "${VLLM_SERVER_HOST:-127.0.0.1}"
    --vllm_server_port "${VLLM_SERVER_PORT:-8006}"
    --vllm_server_timeout "${VLLM_SERVER_TIMEOUT:-600}"
  )
fi

swift rlhf \
  --rlhf_type grpo \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --output_dir "$OUTPUT_DIR" \
  --external_plugins \
    "$REPO_ROOT/src/open_audio_llm/integrations/ms_swift/register_audio_llm.py" \
    "$REPO_ROOT/src/open_audio_llm/integrations/ms_swift/rewards/plugin.py" \
  --reward_funcs asr_format_reward asr_accuracy_reward hotword_reward \
  --freeze_vit "${FREEZE_VIT:-true}" \
  --lora_rank "${LORA_RANK:-64}" \
  --lora_alpha "${LORA_ALPHA:-128}" \
  --target_modules "${TARGET_MODULES:-all-linear}" \
  --torch_dtype "${TORCH_DTYPE:-bfloat16}" \
  --num_generations "${NUM_GENERATIONS:-2}" \
  --generation_batch_size "${GENERATION_BATCH_SIZE:-${NUM_GENERATIONS:-2}}" \
  --temperature "${TEMPERATURE:-0.6}" \
  --top_k "${TOP_K:-50}" \
  --max_completion_length "${MAX_COMPLETION_LENGTH:-256}" \
  --deepspeed "${DEEPSPEED:-zero2}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-1}" \
  --learning_rate "${LEARNING_RATE:-1e-6}" \
  --max_steps "$MAX_STEPS" \
  --warmup_ratio "${WARMUP_RATIO:-0.05}" \
  --save_steps "${SAVE_STEPS:-1}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT:-2}" \
  --eval_steps "${EVAL_STEPS:-1}" \
  --logging_steps "${LOGGING_STEPS:-1}" \
  --log_completions "${LOG_COMPLETIONS:-true}" \
  --ddp_timeout "${DDP_TIMEOUT:-180000000}" \
  "${VLLM_ARGS[@]}"
