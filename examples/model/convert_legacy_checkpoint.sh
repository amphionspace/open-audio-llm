#!/usr/bin/env bash
# Convert a legacy AmphionASR checkpoint into an Open Audio-LLM HF directory.
#
# Usage:
#   bash examples/model/convert_legacy_checkpoint.sh \
#       -c /path/to/checkpoint.pt \
#       -l /path/to/base_llm \
#       -o /path/to/output_hf
#
# Options:
#   -c  legacy .pt checkpoint (required)
#   -l  base HF LLM directory (required unless --legacy-hf-dir is passed after --)
#   -o  output HF model directory (required)
#   -t  audio tower / encoder type (default: qwen3asr)
#   -w  standalone encoder weights
#   -C  encoder config JSON
#   -d  connector downsample rate (default: 1)
#   -L  do not merge LoRA deltas
#   -h  print help
#
# Extra arguments after "--" are passed to `python -m open_audio_llm.scripts.convert_to_hf`.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

AMPHION_CKPT=""
LLM_PATH=""
OUTPUT_DIR=""
ENCODER_TYPE="qwen3asr"
ENCODER_WEIGHTS=""
ENCODER_CONFIG=""
DOWNSAMPLE_RATE="1"
MERGE_LORA="true"

usage() { sed -n '2,22p' "$0"; exit "${1:-0}"; }

while getopts ":c:l:o:t:w:C:d:Lh" opt; do
  case "$opt" in
    c) AMPHION_CKPT="$OPTARG" ;;
    l) LLM_PATH="$OPTARG" ;;
    o) OUTPUT_DIR="$OPTARG" ;;
    t) ENCODER_TYPE="$OPTARG" ;;
    w) ENCODER_WEIGHTS="$OPTARG" ;;
    C) ENCODER_CONFIG="$OPTARG" ;;
    d) DOWNSAMPLE_RATE="$OPTARG" ;;
    L) MERGE_LORA="false" ;;
    h) usage 0 ;;
    \?) echo "[convert_legacy_checkpoint] unknown option: -$OPTARG" >&2; usage 1 ;;
    :) echo "[convert_legacy_checkpoint] option -$OPTARG requires an argument" >&2; usage 1 ;;
  esac
done
shift $((OPTIND - 1))
EXTRA_ARGS=("$@")

if [[ -z "$AMPHION_CKPT" || -z "$OUTPUT_DIR" ]]; then
  echo "[convert_legacy_checkpoint] ERROR: -c and -o are required" >&2
  usage 1
fi
if [[ -z "$LLM_PATH" ]]; then
  has_legacy_hf_dir="0"
  for arg in "${EXTRA_ARGS[@]}"; do
    [[ "$arg" == "--legacy-hf-dir" ]] && has_legacy_hf_dir="1"
  done
  if [[ "$has_legacy_hf_dir" != "1" ]]; then
    echo "[convert_legacy_checkpoint] ERROR: -l is required unless --legacy-hf-dir is passed after --" >&2
    usage 1
  fi
fi

ARGS=(
  --amphion-ckpt "$AMPHION_CKPT"
  --output-dir "$OUTPUT_DIR"
  --encoder-type "$ENCODER_TYPE"
  --ds-rate "$DOWNSAMPLE_RATE"
)
[[ -n "$LLM_PATH" ]] && ARGS+=(--llm-path "$LLM_PATH")

if [[ -n "$ENCODER_WEIGHTS" ]]; then
  ARGS+=(--encoder-weights "$ENCODER_WEIGHTS")
fi
if [[ -n "$ENCODER_CONFIG" ]]; then
  ARGS+=(--encoder-config "$ENCODER_CONFIG")
fi
if [[ "$MERGE_LORA" != "true" ]]; then
  ARGS+=(--no-merge-lora)
fi
ARGS+=("${EXTRA_ARGS[@]}")

PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
  python -m open_audio_llm.scripts.convert_to_hf "${ARGS[@]}"
