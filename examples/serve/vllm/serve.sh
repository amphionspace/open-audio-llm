#!/usr/bin/env bash
# Serve an Open Audio-LLM or compatible HF checkpoint through vLLM.
#
# Usage:
#   bash examples/serve/vllm/serve.sh \
#       -m /path/to/model \
#       -p 8000 \
#       -n open-audio-llm \
#       -t 1 -d 1 -a 4
#
# Options:
#   -m  model path (required; nested ModelScope layouts are resolved automatically)
#   -p  port (default: 8000)
#   -g  CUDA_VISIBLE_DEVICES (optional)
#   -n  served-model-name (default: vLLM default)
#   -t  tensor-parallel-size (default: 1)
#   -d  data-parallel-size (default: 1)
#   -a  max audio items per prompt (default: 4)
#   -u  gpu-memory-utilization (optional)
#   -l  max-model-len (optional)
#   -e  pass --enable-mm-embeds to vLLM
#   -q  enable Open Audio-LLM Qwen3-ASR audio_embeds plugin override
#   -h  print help
#
# The same knobs can be set through matching VLLM_* environment variables.
# CLI flags take precedence. Extra arguments after "--" are passed through to
# `vllm serve`.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

MODEL="${OPEN_AUDIO_LLM_MODEL:-}"
PORT="${VLLM_PORT:-8000}"
GPUS="${CUDA_VISIBLE_DEVICES:-}"
SERVED_NAME="${VLLM_SERVED_MODEL_NAME:-}"
TP_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
DP_SIZE="${VLLM_DATA_PARALLEL_SIZE:-1}"
MAX_AUDIO="${VLLM_MAX_AUDIO:-4}"
GPU_UTIL="${VLLM_GPU_MEMORY_UTILIZATION:-}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-}"
ENABLE_MM_EMBEDS="${VLLM_ENABLE_MM_EMBEDS:-0}"
ENABLE_QWEN3_ASR_EMBEDS="${OPEN_AUDIO_LLM_ENABLE_QWEN3_ASR_EMBEDS:-${VLLM_ENABLE_QWEN3_ASR_EMBEDS:-0}}"

usage() { sed -n '2,26p' "$0"; exit "${1:-0}"; }

while getopts ":m:p:g:n:t:d:a:u:l:eqh" opt; do
  case "$opt" in
    m) MODEL="$OPTARG" ;;
    p) PORT="$OPTARG" ;;
    g) GPUS="$OPTARG" ;;
    n) SERVED_NAME="$OPTARG" ;;
    t) TP_SIZE="$OPTARG" ;;
    d) DP_SIZE="$OPTARG" ;;
    a) MAX_AUDIO="$OPTARG" ;;
    u) GPU_UTIL="$OPTARG" ;;
    l) MAX_MODEL_LEN="$OPTARG" ;;
    e) ENABLE_MM_EMBEDS="1" ;;
    q) ENABLE_QWEN3_ASR_EMBEDS="1" ;;
    h) usage 0 ;;
    \?) echo "[open-audio-llm serve] unknown option: -$OPTARG" >&2; usage 1 ;;
    :) echo "[open-audio-llm serve] option -$OPTARG requires an argument" >&2; usage 1 ;;
  esac
done
shift $((OPTIND - 1))
EXTRA_ARGS=("$@")

if [[ -z "$MODEL" ]]; then
  echo "[open-audio-llm serve] ERROR: model path is required, use -m /path/to/model" >&2
  usage 1
fi
MODEL="${MODEL%/}"

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export OPEN_AUDIO_LLM_ENABLE_QWEN3_ASR_EMBEDS="$ENABLE_QWEN3_ASR_EMBEDS"
if [[ -n "$GPUS" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPUS"
fi

resolve_model_dir() {
  local base="$1"
  python - "$base" <<'PY'
from pathlib import Path
import sys

base = Path(sys.argv[1]).expanduser()
if not base.is_dir() or (base / "config.json").is_file():
    print(base)
    raise SystemExit

root_depth = len(base.parts)
for path in sorted(base.rglob("config.json")):
    if ".lock" in path.parts:
        continue
    if len(path.parent.parts) - root_depth <= 3:
        print(path.parent)
        raise SystemExit
print(base)
PY
}

RESOLVED_MODEL="$(resolve_model_dir "$MODEL")"
if [[ "$RESOLVED_MODEL" != "$MODEL" ]]; then
  echo "[open-audio-llm serve] resolved nested model dir: $MODEL -> $RESOLVED_MODEL"
  MODEL="$RESOLVED_MODEL"
fi

if [[ ! -f "$MODEL/config.json" ]]; then
  echo "[open-audio-llm serve] ERROR: config.json not found under model path: $MODEL" >&2
  exit 2
fi

ARCH="$(python - "$MODEL/config.json" <<'PY' 2>/dev/null || true
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    cfg = json.load(handle)
archs = cfg.get("architectures") or []
print(archs[0] if archs else cfg.get("model_type", ""))
PY
)"

EXTRA_ARGS_DEFAULT=()
case "$ARCH" in
  MoonshotKimiaForCausalLM)
    # vLLM prefix caching can reuse text KV cache across Kimi-Audio samples
    # while audio embeddings differ, which has produced empty generations at
    # high concurrency. Keep this model-specific safety flag by default.
    EXTRA_ARGS_DEFAULT+=(--no-enable-prefix-caching)
    ;;
esac

VLLM_ARGS=(
  --trust-remote-code
  --port "$PORT"
  --data-parallel-size "$DP_SIZE"
  --tensor-parallel-size "$TP_SIZE"
  --limit-mm-per-prompt "{\"audio\": $MAX_AUDIO}"
)

if [[ -n "$SERVED_NAME" ]]; then
  VLLM_ARGS+=(--served-model-name "$SERVED_NAME")
fi
if [[ "$ENABLE_MM_EMBEDS" == "1" ]]; then
  VLLM_ARGS+=(--enable-mm-embeds)
fi
if [[ -n "$GPU_UTIL" ]]; then
  VLLM_ARGS+=(--gpu-memory-utilization "$GPU_UTIL")
fi
if [[ -n "$MAX_MODEL_LEN" ]]; then
  VLLM_ARGS+=(--max-model-len "$MAX_MODEL_LEN")
fi

unset \
  VLLM_PORT \
  VLLM_SERVED_MODEL_NAME \
  VLLM_TENSOR_PARALLEL_SIZE \
  VLLM_DATA_PARALLEL_SIZE \
  VLLM_MAX_AUDIO \
  VLLM_GPU_MEMORY_UTILIZATION \
  VLLM_MAX_MODEL_LEN \
  VLLM_ENABLE_MM_EMBEDS \
  VLLM_ENABLE_QWEN3_ASR_EMBEDS

if [[ "${AMPHION_TSASR_INSERT_SEP:-0}" =~ ^(1|true|yes|on)$ ]]; then
  HF_OVERRIDES="$(python - <<'PY'
import json
from open_audio_llm.tsasr.global_audio_attn import vllm_hf_overrides
print(json.dumps(vllm_hf_overrides()))
PY
)"
  if [[ -n "$HF_OVERRIDES" && "$HF_OVERRIDES" != "{}" ]]; then
    EXTRA_ARGS_DEFAULT+=(--hf-overrides "$HF_OVERRIDES")
  fi
fi

echo "[open-audio-llm serve] arch=${ARCH:-<unknown>} model=$MODEL port=$PORT gpus=${GPUS:-<unchanged>} served_name=${SERVED_NAME:-<default>} tp=$TP_SIZE dp=$DP_SIZE max_audio=$MAX_AUDIO util=${GPU_UTIL:-<vllm-default>} max_model_len=${MAX_MODEL_LEN:-<model-default>} enable_mm_embeds=$ENABLE_MM_EMBEDS enable_qwen3_asr_embeds=$ENABLE_QWEN3_ASR_EMBEDS ts_sep=${AMPHION_TSASR_INSERT_SEP:-0}"
[[ ${#EXTRA_ARGS[@]} -gt 0 ]] && echo "[open-audio-llm serve] extra: ${EXTRA_ARGS[*]}"
vllm serve "$MODEL" "${VLLM_ARGS[@]}" "${EXTRA_ARGS_DEFAULT[@]}" "${EXTRA_ARGS[@]}"
