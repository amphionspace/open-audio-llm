#!/usr/bin/env bash
# Plain and enrolled test sets for one checkpoint. Does not touch training caches.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
export AUDIO_DATA_CONTRACT_ROOT="${AUDIO_DATA_CONTRACT_ROOT:-$(dirname "$REPO_ROOT")/audio-data-contract}"
export PYTHONPATH="$REPO_ROOT/src:$AUDIO_DATA_CONTRACT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
unset AUDIO_DATA_CATALOG AUDIO_DATA_METADATA_CACHE
PYTHON="${PYTHON:-/ai_sds_wuzz/MODELS/miniconda3/envs/amphionft/bin/python}"
MODEL="${MODEL:-$REPO_ROOT/runs/target-sot-recipe-20261004/v7-20261004-145321/checkpoint-37000}"
OUT="${OUT:-$REPO_ROOT/runs/target-sot-recipe-20261004/v7-20261004-145321/eval-checkpoint-37000}"
mkdir -p "$OUT/logs"
"$PYTHON" -m open_audio_llm.eval.recipe_test prepare --output "$OUT"
pids=()
for rank in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES="$rank" "$PYTHON" -m open_audio_llm.eval.recipe_test work \
    --output "$OUT" --rank "$rank" --model "$MODEL" --batch-size "${BATCH_SIZE:-16}" \
    > "$OUT/logs/rank${rank}.log" 2>&1 &
  pids+=("$!")
done
fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done
"$PYTHON" -m open_audio_llm.eval.recipe_test score --output "$OUT"
exit "$fail"
