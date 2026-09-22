#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DATA_CONFIG="${DATA_CONFIG:-$SCRIPT_DIR/../../configs/data/qwen3_asr_ts_replay.yaml}"
export OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/../../../runs/qwen3-asr-ts-replay}"
export MAX_STEPS="${MAX_STEPS:-2000}"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
export AUDIO_DATA_CONTRACT_ROOT="${AUDIO_DATA_CONTRACT_ROOT:-$(dirname "$REPO_ROOT")/audio-data-contract}"
export AUDIO_DATA_CATALOG="${AUDIO_DATA_CATALOG:-$AUDIO_DATA_CONTRACT_ROOT/catalog}"
export AUDIO_DATA_ROOTS_FILE="${AUDIO_DATA_ROOTS_FILE:-$AUDIO_DATA_CONTRACT_ROOT/roots.json}"
export AUDIO_DATA_METADATA_CACHE="${AUDIO_DATA_METADATA_CACHE:-$REPO_ROOT/runs/catalog-metadata-cache}"
export PYTHONPATH="$REPO_ROOT/src:$AUDIO_DATA_CONTRACT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
# Match train.sh's default and the final caller override.
preflight_batch_size=8
previous_arg=""
for arg in "$@"; do
  if [[ "$previous_arg" == "--per_device_train_batch_size" ]]; then
    preflight_batch_size="$arg"
  elif [[ "$arg" == --per_device_train_batch_size=* ]]; then
    preflight_batch_size="${arg#*=}"
  fi
  previous_arg="$arg"
done
reuse_args=()
if [[ -n "${PREVIOUS_DATA_CONFIG:-}" ]]; then
  reuse_args=(--reuse-config "$PREVIOUS_DATA_CONFIG")
fi
# Finish metadata, sampler quotas and audio reads before allocating DDP GPUs.
"${PYTHON:-python}" -m open_audio_llm.data.catalog_cache \
  --data_config "$DATA_CONFIG" --workers "${METADATA_WORKERS:-4}" \
  --preflight-report "$OUTPUT_DIR/data-preflight.json" \
  --batch-size "$preflight_batch_size" --world-size "${NPROC_PER_NODE:-2}" \
  "${reuse_args[@]}"
exec bash "$SCRIPT_DIR/train.sh" \
  --lora_rank 64 --lora_alpha 128 --learning_rate 2e-5 \
  --max_length 2048 --gradient_accumulation_steps 2 \
  --audio_encoder_parallel true \
  --dataloader_num_workers 4 --dataloader_persistent_workers true \
  --freeze_vit true --freeze_aligner true --freeze_llm false \
  --eval_strategy steps --save_steps 250 --eval_steps 250 \
  --save_total_limit 8 \
  "$@"
