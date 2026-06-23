#!/usr/bin/env bash
# Production-style SFT recipe for Open Audio-LLM via ms-swift.
#
# Usage:
#   bash examples/train/sft/production_swift.sh \
#       -m /path/to/model \
#       -d /path/to/train_a.jsonl \
#       -d /path/to/train_b.jsonl \
#       -o exp/open_audio_llm_sft \
#       -g 0,1,2,3 -n 4
#
# Options:
#   -m  HF model directory (required)
#   -d  ShareGPT SFT JSONL dataset; repeat for multiple datasets (required)
#   -v  validation dataset; repeat for multiple validation datasets
#   -o  output directory (default: exp/open_audio_llm_sft)
#   -g  CUDA_VISIBLE_DEVICES (default: unchanged)
#   -n  NPROC_PER_NODE (default: number of GPUs from -g, else 1)
#   -P  MASTER_PORT (default: 29501)
#   -r  ms-swift external plugin path
#       (default: src/open_audio_llm/integrations/ms_swift/register_audio_llm.py)
#   -D  DeepSpeed config/name (default: zero1)
#   -e  num train epochs (default: 1)
#   -s  split_dataset_ratio when -v is omitted (default: 0.01)
#   -R  resume_from_checkpoint
#   -h  print help
#
# Extra arguments after "--" are passed through to `swift sft`.

set -euo pipefail
umask 0000

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

MODEL=""
DATASETS=()
VAL_DATASETS=()
OUTPUT_DIR="exp/open_audio_llm_sft"
GPUS=""
NPROC_PER_NODE_VALUE=""
MASTER_PORT_VALUE="29501"
REGISTER_PLUGIN="$REPO_ROOT/src/open_audio_llm/integrations/ms_swift/register_audio_llm.py"
DEEPSPEED="zero1"
NUM_EPOCHS="1"
SPLIT_DATASET_RATIO="0.01"
RESUME=""

usage() { sed -n '2,31p' "$0"; exit "${1:-0}"; }

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
    \?) echo "[sft production] unknown option: -$OPTARG" >&2; usage 1 ;;
    :) echo "[sft production] option -$OPTARG requires an argument" >&2; usage 1 ;;
  esac
done
shift $((OPTIND - 1))
EXTRA_ARGS=("$@")

if [[ -z "$MODEL" || ${#DATASETS[@]} -eq 0 ]]; then
  echo "[sft production] ERROR: -m and at least one -d are required" >&2
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
if [[ -n "$RESUME" ]]; then
  RESUME_ARGS=(--resume_from_checkpoint "$RESUME")
fi

mkdir -p "$REPO_ROOT/log"
LOG_FILE="$REPO_ROOT/log/run_sft_$(date +%Y%m%d%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[sft production] logging to $LOG_FILE"
echo "[sft production] model=$MODEL output=$OUTPUT_DIR nproc=$NPROC_PER_NODE master_port=$MASTER_PORT deepspeed=$DEEPSPEED epochs=$NUM_EPOCHS"
echo "[sft production] datasets=${DATASETS[*]}"
[[ ${#VAL_DATASETS[@]} -gt 0 ]] && echo "[sft production] val_datasets=${VAL_DATASETS[*]}"
[[ ${#EXTRA_ARGS[@]} -gt 0 ]] && echo "[sft production] extra=${EXTRA_ARGS[*]}"

swift sft \
  --model "$MODEL" \
  --external_plugins "$REGISTER_PLUGIN" \
  --dataset "${DATASETS[@]}" \
  "${VAL_ARGS[@]}" \
  "${RESUME_ARGS[@]}" \
  --tuner_type lora \
  --freeze_vit true \
  --freeze_aligner false \
  --deepspeed "$DEEPSPEED" \
  --torch_dtype bfloat16 \
  --lora_rank 64 \
  --lora_alpha 128 \
  --target_modules all-linear \
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
  --save_total_limit 5 \
  --eval_steps 500 \
  --eval_strategy steps \
  --logging_steps 10 \
  --output_dir "$OUTPUT_DIR" \
  --ddp_timeout 180000000 \
  "${EXTRA_ARGS[@]}"
