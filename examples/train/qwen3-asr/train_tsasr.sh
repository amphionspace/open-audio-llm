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
# Build once before starting DDP, so metadata parsing cannot time out its collectives.
"${PYTHON:-python}" -m open_audio_llm.data.catalog_cache \
  --data_config "$DATA_CONFIG" --workers "${METADATA_WORKERS:-4}"
exec bash "$SCRIPT_DIR/train.sh" \
  --lora_rank 64 --lora_alpha 128 --learning_rate 2e-5 \
  --max_length 2048 --gradient_accumulation_steps 2 \
  --audio_encoder_parallel true \
  --dataloader_num_workers 4 --dataloader_persistent_workers true \
  --freeze_vit true --freeze_aligner true --freeze_llm false \
  --eval_strategy steps --save_steps 250 --eval_steps 250 \
  --save_total_limit 8 \
  "$@"
