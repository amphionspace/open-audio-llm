#!/usr/bin/env bash
# Native Qwen3-ASR LoRA with Catalog audio decoded and augmented on demand.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
export AUDIO_DATA_CONTRACT_ROOT="${AUDIO_DATA_CONTRACT_ROOT:-$(dirname "$REPO_ROOT")/audio-data-contract}"
export AUDIO_DATA_CATALOG="${AUDIO_DATA_CATALOG:-$AUDIO_DATA_CONTRACT_ROOT/catalog}"
export AUDIO_DATA_ROOTS_FILE="${AUDIO_DATA_ROOTS_FILE:-$AUDIO_DATA_CONTRACT_ROOT/roots.json}"
export PYTHONPATH="$REPO_ROOT/src:$AUDIO_DATA_CONTRACT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
"${PYTHON:-python}" -m torch.distributed.run \
  --nproc_per_node "${NPROC_PER_NODE:-2}" --master_port "${MASTER_PORT:-29523}" \
  --module open_audio_llm.integrations.ms_swift.train sft \
  --model "${MODEL:?Set MODEL to native Qwen3-ASR-1.7B weights}" \
  --model_type amphion_asr_1.7b \
  --external_plugins "$REPO_ROOT/src/open_audio_llm/integrations/ms_swift/register_qwen3_asr.py" \
  --data_config "${DATA_CONFIG:-$REPO_ROOT/examples/configs/data/qwen3_asr_hotwords.yaml}" \
  --tuner_type lora --lora_rank 32 --lora_alpha 64 --target_modules all-linear \
  --freeze_vit true --freeze_aligner true --freeze_llm false \
  --torch_dtype bfloat16 --attn_impl sdpa \
  --max_length 1024 --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 4 --gradient_accumulation_steps 1 \
  --learning_rate 5e-5 --max_steps "${MAX_STEPS:-500}" \
  --lr_scheduler_type cosine --warmup_ratio 0.05 \
  --gradient_checkpointing false --vit_gradient_checkpointing false \
  --ddp_find_unused_parameters false --dataloader_num_workers 2 \
  --performance_logging "${PERFORMANCE_LOGGING:-true}" \
  --save_steps 250 --save_total_limit 2 --eval_steps 250 \
  --logging_steps 10 --report_to none --seed 42 \
  --output_dir "${OUTPUT_DIR:-$REPO_ROOT/runs/qwen3-asr-hotwords}" \
  "$@"
