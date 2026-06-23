#!/usr/bin/env bash
# Convert Lhotse manifests to ShareGPT JSONL for ms-swift training.
#
# Usage:
#   bash examples/data/convert_sharegpt.sh -c config.json -o output/sharegpt_data
#   bash examples/data/convert_sharegpt.sh config.json output/sharegpt_data
#
# Options:
#   -c  converter config JSON
#       (default: src/open_audio_llm/integrations/ms_swift/configs/unified_template.json)
#   -o  output directory (default: output/sharegpt_data)
#   -h  print help

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

CONFIG="$REPO_ROOT/src/open_audio_llm/integrations/ms_swift/configs/unified_template.json"
OUT_DIR="output/sharegpt_data"

usage() { sed -n '2,13p' "$0"; exit "${1:-0}"; }

while getopts ":c:o:h" opt; do
  case "$opt" in
    c) CONFIG="$OPTARG" ;;
    o) OUT_DIR="$OPTARG" ;;
    h) usage 0 ;;
    \?) echo "[convert_sharegpt] unknown option: -$OPTARG" >&2; usage 1 ;;
    :) echo "[convert_sharegpt] option -$OPTARG requires an argument" >&2; usage 1 ;;
  esac
done
shift $((OPTIND - 1))

if [[ $# -gt 0 ]]; then
  CONFIG="$1"
  shift
fi
if [[ $# -gt 0 ]]; then
  OUT_DIR="$1"
  shift
fi

echo "[$(date)] Converting Lhotse -> ShareGPT JSONL"
echo "  Config:  $CONFIG"
echo "  Out dir: $OUT_DIR"

PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
  python -m open_audio_llm.integrations.ms_swift.data.convert \
    --config "$CONFIG" \
    --out-dir "$OUT_DIR"

echo "[$(date)] Done."
