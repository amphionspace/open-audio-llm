#!/usr/bin/env bash
set -eo pipefail
source "$HOME/.bashrc" >/dev/null 2>&1
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${MODEL:?Set MODEL to a frozen native SOT checkpoint}"
: "${RETENTION_TEACHER:?Set RETENTION_TEACHER to the original Qwen3-ASR checkpoint}"
: "${DATA_CONFIG:?Set DATA_CONFIG to a recipe with annotated enrollment sources}"
: "${OUTPUT_DIR:?Set a new OUTPUT_DIR for this experiment}"
: "${CUDA_VISIBLE_DEVICES:?Select GPUs allocated to this new experiment}"
: "${NPROC_PER_NODE:?Set the number of allocated GPUs}"
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "OUTPUT_DIR already exists; select a new experiment directory" >&2
  exit 1
fi
export WANDB_ENTITY=1016097967-amphion
export WANDB_PROJECT=open-audio-llm
export WANDB_MODE=online
exec bash "$SCRIPT_DIR/train_sot.sh" \
  --max_length 16384 --truncation_strategy delete \
  --report_to wandb --predict_with_generate false \
  "$@"
