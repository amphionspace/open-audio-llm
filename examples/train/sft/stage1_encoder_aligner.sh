#!/usr/bin/env bash
# Three-stage SFT, stage 1: train encoder + aligner LoRA while freezing the LLM.
#
# Usage:
#   bash examples/train/sft/stage1_encoder_aligner.sh \
#       -m /path/to/base_model \
#       -d /path/to/train.jsonl \
#       -o exp/staged_sft/stage1_encoder_aligner \
#       -g 0,1,2 -n 3
#
# Options mirror production_swift.sh where possible. Use repeated -d/-v for
# multiple train/validation datasets. Extra arguments after "--" are passed
# through to `swift sft`.

set -euo pipefail
umask 0000

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

MODEL=""
DATASETS=()
VAL_DATASETS=()
OUTPUT_DIR="exp/staged_sft/stage1_encoder_aligner"
GPUS=""
NPROC_PER_NODE_VALUE=""
MASTER_PORT_VALUE="29501"
REGISTER_PLUGIN="$REPO_ROOT/src/open_audio_llm/integrations/ms_swift/register_audio_llm.py"
DEEPSPEED="zero1"
NUM_EPOCHS="1"
SPLIT_DATASET_RATIO="0.01"
RESUME=""

usage() { sed -n '2,13p' "$0"; exit "${1:-0}"; }

while getopts ":m:d:v:o:g:n:P:r:D:e:s:R:h" opt; do
  case "$opt" in
    m) MODEL="$OPTARG" ;;
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
    \?) echo "[stage1] unknown option: -$OPTARG" >&2; usage 1 ;;
    :) echo "[stage1] option -$OPTARG requires an argument" >&2; usage 1 ;;
  esac
done
shift $((OPTIND - 1))
EXTRA_ARGS=("$@")

if [[ -z "$MODEL" || ${#DATASETS[@]} -eq 0 ]]; then
  echo "[stage1] ERROR: -m and at least one -d are required" >&2
  usage 1
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
LOG_FILE="$REPO_ROOT/log/run_sft_stage1_$(date +%Y%m%d%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[stage1] logging to $LOG_FILE"

swift sft \
  --model "$MODEL" \
  --external_plugins "$REGISTER_PLUGIN" \
  --dataset "${DATASETS[@]}" \
  "${VAL_ARGS[@]}" \
  "${RESUME_ARGS[@]}" \
  --tuner_type lora \
  --freeze_vit false \
  --freeze_aligner false \
  --freeze_llm true \
  --target_modules all-linear \
  --deepspeed "$DEEPSPEED" \
  --torch_dtype bfloat16 \
  --lora_rank 64 \
  --lora_alpha 128 \
  --max_length 2048 \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-5 \
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

BEST_CKPT="$(python - "$OUTPUT_DIR" <<'PY'
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
)"

echo "[stage1] best checkpoint: $BEST_CKPT"
unset CUDA_VISIBLE_DEVICES
bash "$REPO_ROOT/examples/model/merge_lora.sh" "$BEST_CKPT"
echo "[stage1] done. merged model -> ${BEST_CKPT}_merged"
