#!/usr/bin/env bash
# Legacy AmphionASR HF converter wrapper kept for old checkpoint formats.
#
# Prefer examples/model/convert_legacy_checkpoint.sh for Open Audio-LLM output.
#
# Usage:
#   bash examples/model/convert_amphionasr_checkpoint.sh \
#       -c exp/xxx/checkpoint-48000.pt \
#       -o models/qwen3omni-hf \
#       -- --llm-path /local/copy/of/Qwen3-4B-Instruct-2507 --no-merge-lora
#
# Options:
#   -c  AmphionASR checkpoint path (required)
#   -o  output directory (optional; converter derives one when omitted)
#   -h  print help
#
# Extra arguments after "--" are passed to the legacy converter.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

AMPHION_CKPT=""
OUTPUT_DIR=""

usage() { sed -n '2,17p' "$0"; exit "${1:-0}"; }

while getopts ":c:o:h" opt; do
  case "$opt" in
    c) AMPHION_CKPT="$OPTARG" ;;
    o) OUTPUT_DIR="$OPTARG" ;;
    h) usage 0 ;;
    \?) echo "[convert_amphionasr_checkpoint] unknown option: -$OPTARG" >&2; usage 1 ;;
    :) echo "[convert_amphionasr_checkpoint] option -$OPTARG requires an argument" >&2; usage 1 ;;
  esac
done
shift $((OPTIND - 1))
EXTRA_ARGS=("$@")

if [[ -z "$AMPHION_CKPT" ]]; then
  echo "[convert_amphionasr_checkpoint] ERROR: -c is required" >&2
  usage 1
fi

CMD=(
  python -m open_audio_llm.integrations.hf.convert_amphion_to_hf
  --amphion-ckpt "$AMPHION_CKPT"
)
[[ -n "$OUTPUT_DIR" ]] && CMD+=(--output-dir "$OUTPUT_DIR")
CMD+=("${EXTRA_ARGS[@]}")

PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "${CMD[@]}"
echo "Done!"
