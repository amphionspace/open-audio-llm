#!/usr/bin/env bash
# Production-style GRPO recipe for Open Audio-LLM via ms-swift.
#
# Usage:
#   bash examples/train/grpo/production_swift.sh \
#       -m /path/to/sft_merged_model \
#       -d /path/to/grpo_sharegpt.jsonl \
#       -o exp/open_audio_llm_grpo \
#       -g 1,2,3 -n 3 \
#       -V -H 127.0.0.1 -p 8006
#
# Options:
#   -m  HF model directory (required)
#   -d  ShareGPT GRPO JSONL dataset (required)
#   -o  output directory (default: exp/open_audio_llm_grpo)
#   -g  CUDA_VISIBLE_DEVICES (default: unchanged)
#   -n  NPROC_PER_NODE (default: number of GPUs from -g, else 1)
#   -P  MASTER_PORT (default: 29501)
#   -r  ms-swift model registration plugin
#       (default: src/open_audio_llm/integrations/ms_swift/register_audio_llm.py)
#   -R  reward plugin
#       (default: src/open_audio_llm/integrations/ms_swift/rewards/plugin.py)
#   -V  enable vLLM server mode
#   -H  vLLM server host (default: 127.0.0.1)
#   -p  vLLM server port (default: 8006)
#   -h  print help
#
# Extra arguments after "--" are passed through to `swift rlhf`.

set -euo pipefail
umask 0000

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

MODEL=""
DATASET=""
OUTPUT_DIR="exp/open_audio_llm_grpo"
GPUS=""
NPROC_PER_NODE_VALUE=""
MASTER_PORT_VALUE="29501"
REGISTER_PLUGIN="$REPO_ROOT/src/open_audio_llm/integrations/ms_swift/register_audio_llm.py"
REWARD_PLUGIN="$REPO_ROOT/src/open_audio_llm/integrations/ms_swift/rewards/plugin.py"
USE_VLLM="0"
VLLM_SERVER_HOST="127.0.0.1"
VLLM_SERVER_PORT="8006"

usage() { sed -n '2,29p' "$0"; exit "${1:-0}"; }

while getopts ":m:d:o:g:n:P:r:R:VH:p:h" opt; do
  case "$opt" in
    m) MODEL="$OPTARG" ;;
    d) DATASET="$OPTARG" ;;
    o) OUTPUT_DIR="$OPTARG" ;;
    g) GPUS="$OPTARG" ;;
    n) NPROC_PER_NODE_VALUE="$OPTARG" ;;
    P) MASTER_PORT_VALUE="$OPTARG" ;;
    r) REGISTER_PLUGIN="$OPTARG" ;;
    R) REWARD_PLUGIN="$OPTARG" ;;
    V) USE_VLLM="1" ;;
    H) VLLM_SERVER_HOST="$OPTARG" ;;
    p) VLLM_SERVER_PORT="$OPTARG" ;;
    h) usage 0 ;;
    \?) echo "[grpo production] unknown option: -$OPTARG" >&2; usage 1 ;;
    :) echo "[grpo production] option -$OPTARG requires an argument" >&2; usage 1 ;;
  esac
done
shift $((OPTIND - 1))
EXTRA_ARGS=("$@")

if [[ -z "$MODEL" || -z "$DATASET" ]]; then
  echo "[grpo production] ERROR: -m and -d are required" >&2
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
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"

VLLM_ARGS=()
if [[ "$USE_VLLM" == "1" ]]; then
  VLLM_ARGS=(
    --use_vllm true
    --vllm_mode server
    --vllm_server_host "$VLLM_SERVER_HOST"
    --vllm_server_port "$VLLM_SERVER_PORT"
    --vllm_server_timeout 600
  )
fi

mkdir -p "$REPO_ROOT/log"
LOG_FILE="$REPO_ROOT/log/run_grpo_$(date +%Y%m%d%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[grpo production] logging to $LOG_FILE"
echo "[grpo production] model=$MODEL dataset=$DATASET output=$OUTPUT_DIR nproc=$NPROC_PER_NODE use_vllm=$USE_VLLM"
[[ ${#EXTRA_ARGS[@]} -gt 0 ]] && echo "[grpo production] extra=${EXTRA_ARGS[*]}"

swift rlhf \
  --rlhf_type grpo \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --output_dir "$OUTPUT_DIR" \
  --external_plugins "$REGISTER_PLUGIN" "$REWARD_PLUGIN" \
  --reward_funcs asr_accuracy_reward hotword_reward asr_format_reward \
  --reward_weights 1.0 0.3 0.1 \
  --freeze_vit true \
  --lora_rank 64 \
  --lora_alpha 128 \
  --target_modules all-linear \
  --torch_dtype bfloat16 \
  --num_generations 16 \
  --temperature 0.6 \
  --top_k 50 \
  --max_completion_length 1024 \
  --deepspeed zero2 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --learning_rate 1e-6 \
  --num_train_epochs 1 \
  --warmup_ratio 0.05 \
  --save_steps 100 \
  --save_total_limit 3 \
  --eval_steps 100 \
  --logging_steps 1 \
  --log_completions true \
  --ddp_timeout 180000000 \
  "${VLLM_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
