#!/usr/bin/env bash
# Three-stage SFT, stage 3: joint LoRA over encoder + aligner + LLM.
#
# Usage:
#   bash examples/train/sft/stage3_joint.sh \
#       -d /path/to/train.jsonl \
#       -2 exp/staged_sft/stage2_llm \
#       -o exp/staged_sft/stage3_joint
#
# Pass -m to override automatic use of stage2 best checkpoint merged output.

set -euo pipefail
umask 0000

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

MODEL=""
STAGE2_OUTPUT_DIR="exp/staged_sft/stage2_llm"
DATASETS=()
VAL_DATASETS=()
OUTPUT_DIR="exp/staged_sft/stage3_joint"
GPUS=""
NPROC_PER_NODE_VALUE=""
MASTER_PORT_VALUE="29503"
REGISTER_PLUGIN="$REPO_ROOT/src/open_audio_llm/integrations/ms_swift/register_audio_llm.py"
DEEPSPEED="zero1"
NUM_EPOCHS="2"
SPLIT_DATASET_RATIO="0.01"
RESUME=""

usage() { sed -n '2,11p' "$0"; exit "${1:-0}"; }

latest_best_checkpoint() {
  python - "$1" <<'PY'
import json
import sys
from pathlib import Path

versions = sorted(Path(sys.argv[1]).glob("v*-*"))
if not versions:
    raise SystemExit(f"no version dir found in {sys.argv[1]}")
log_path = versions[-1] / "logging.jsonl"
best = None
with log_path.open(encoding="utf-8") as handle:
    for line in handle:
        row = json.loads(line)
        best = row.get("best_model_checkpoint", best)
if not best:
    raise SystemExit(f"best_model_checkpoint not found in {log_path}")
print(best)
PY
}

while getopts ":m:2:d:v:o:g:n:P:r:D:e:s:R:h" opt; do
  case "$opt" in
    m) MODEL="$OPTARG" ;;
    2) STAGE2_OUTPUT_DIR="$OPTARG" ;;
    d) DATASETS+=("$OPTARG") ;;
    v) VAL_DATASETS+=("$OPTARG") ;;
    o) OUTPUT_DIR="$OPTARG" ;;
    g) GPUS="$OPTARG" ;;
    n) NPROC_PER_NODE_VALUE="$OPTARG" ;;
    P) MASTER_PORT_VALUE="$OPTARG" ;;
    r) REGISTER_PLUGIN="$OPTARG" ;;
    D) DEEPSPEED="$OPTARG" ;;
    e) NUM_EPOCHS="$OPTARG" ;;
    s) SPLIT_DATASET_RATIO="$OPTARG" ;;
    R) RESUME="$OPTARG" ;;
    h) usage 0 ;;
    \?) echo "[stage3] unknown option: -$OPTARG" >&2; usage 1 ;;
    :) echo "[stage3] option -$OPTARG requires an argument" >&2; usage 1 ;;
  esac
done
shift $((OPTIND - 1))
EXTRA_ARGS=("$@")

if [[ ${#DATASETS[@]} -eq 0 ]]; then
  echo "[stage3] ERROR: at least one -d is required" >&2
  usage 1
fi
if [[ -z "$MODEL" ]]; then
  MODEL="$(latest_best_checkpoint "$STAGE2_OUTPUT_DIR")_merged"
fi

if [[ -n "$GPUS" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPUS"
  if [[ -z "$NPROC_PER_NODE_VALUE" ]]; then
    NPROC_PER_NODE_VALUE="$(python - "$GPUS" <<'PY'
import sys
print(len([item for item in sys.argv[1].split(",") if item.strip()]))
PY
)"
  fi
fi
export NPROC_PER_NODE="${NPROC_PER_NODE_VALUE:-1}"
export MASTER_PORT="$MASTER_PORT_VALUE"

VAL_ARGS=()
if [[ ${#VAL_DATASETS[@]} -gt 0 ]]; then
  VAL_ARGS=(--val_dataset "${VAL_DATASETS[@]}")
else
  VAL_ARGS=(--split_dataset_ratio "$SPLIT_DATASET_RATIO")
fi
RESUME_ARGS=()
[[ -n "$RESUME" ]] && RESUME_ARGS=(--resume_from_checkpoint "$RESUME")

mkdir -p "$REPO_ROOT/log"
LOG_FILE="$REPO_ROOT/log/run_sft_stage3_$(date +%Y%m%d%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[stage3] logging to $LOG_FILE"
echo "[stage3] using model: $MODEL"

swift sft \
  --model "$MODEL" \
  --external_plugins "$REGISTER_PLUGIN" \
  --dataset "${DATASETS[@]}" \
  "${VAL_ARGS[@]}" \
  "${RESUME_ARGS[@]}" \
  --tuner_type lora \
  --freeze_vit false \
  --freeze_aligner false \
  --target_modules all-linear \
  --deepspeed "$DEEPSPEED" \
  --torch_dtype bfloat16 \
  --lora_rank 64 \
  --lora_alpha 128 \
  --max_length 2048 \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 4 \
  --learning_rate 5e-6 \
  --num_train_epochs "$NUM_EPOCHS" \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.05 \
  --gradient_checkpointing true \
  --dataloader_num_workers 4 \
  --save_steps 500 \
  --save_total_limit 3 \
  --eval_steps 500 \
  --eval_strategy steps \
  --logging_steps 10 \
  --output_dir "$OUTPUT_DIR" \
  --ddp_timeout 180000000 \
  "${EXTRA_ARGS[@]}"

BEST_CKPT="$(latest_best_checkpoint "$OUTPUT_DIR")"
echo "[stage3] done. final LoRA checkpoint -> $OUTPUT_DIR"
echo "[stage3] merge best checkpoint with:"
echo "  bash examples/model/merge_lora.sh \"$BEST_CKPT\""
