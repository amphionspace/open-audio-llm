#!/usr/bin/env bash
# Merge an ms-swift LoRA adapter into its base model.
#
# Usage:
#   bash examples/model/merge_lora.sh -c <checkpoint_dir> [-b base_model_dir] [-o save_dir]
#   bash examples/model/merge_lora.sh <checkpoint_dir> [base_model_dir]
#
# If base_model_dir is omitted, it is read from adapter_config.json. When
# CUDA_VISIBLE_DEVICES is unset and nvidia-smi is available, the script picks
# the single GPU with the most free memory to avoid accidental busy-GPU OOMs.

set -euo pipefail

CHECKPOINT_DIR=""
BASE_MODEL_DIR=""
SAVE_DIR=""

usage() { sed -n '2,10p' "$0"; exit "${1:-0}"; }

while getopts ":c:b:o:h" opt; do
  case "$opt" in
    c) CHECKPOINT_DIR="$OPTARG" ;;
    b) BASE_MODEL_DIR="$OPTARG" ;;
    o) SAVE_DIR="$OPTARG" ;;
    h) usage 0 ;;
    \?) echo "[merge_lora] unknown option: -$OPTARG" >&2; usage 1 ;;
    :) echo "[merge_lora] option -$OPTARG requires an argument" >&2; usage 1 ;;
  esac
done
shift $((OPTIND - 1))

if [[ -z "$CHECKPOINT_DIR" && $# -gt 0 ]]; then
  CHECKPOINT_DIR="$1"
  shift
fi
if [[ -z "$BASE_MODEL_DIR" && $# -gt 0 ]]; then
  BASE_MODEL_DIR="$1"
  shift
fi
if [[ -z "$CHECKPOINT_DIR" ]]; then
  echo "[merge_lora] ERROR: checkpoint_dir is required" >&2
  usage 1
fi

if [[ -z "$BASE_MODEL_DIR" ]]; then
  BASE_MODEL_DIR="$(python3 - "$CHECKPOINT_DIR/adapter_config.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    print(json.load(handle)["base_model_name_or_path"])
PY
)"
fi
SAVE_DIR="${SAVE_DIR:-${CHECKPOINT_DIR%/}_merged}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  export CUDA_VISIBLE_DEVICES="$(
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
      | sort -t',' -k2 -n -r \
      | awk -F',' 'NR == 1 {gsub(/ /, "", $1); print $1}'
  )"
  echo "[merge_lora] auto-selected CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi

swift export \
  --model "$BASE_MODEL_DIR" \
  --adapters "$CHECKPOINT_DIR" \
  --merge_lora true \
  --output_dir "$SAVE_DIR"

echo "Merged model saved to: $SAVE_DIR"
