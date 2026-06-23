#!/usr/bin/env bash
# vLLM evaluation entry point for an already running OpenAI-compatible server.
#
# Prerequisites:
#   bash examples/serve/vllm/serve.sh -m <model> -p 8000
#
# Usage:
#   bash examples/eval/vllm/eval.sh
#   bash examples/eval/vllm/eval.sh -p 8001 -m open-audio-llm
#   bash examples/eval/vllm/eval.sh -f configs/eval_plans/mix_all.yaml -j 32 -n my_run
#   bash examples/eval/vllm/eval.sh -- --no-language -n 30
#
# Options:
#   -p  service port (default: 8000)
#   -m  served-model-name (default: qwen3-omni; auto-switches to /v1/models when needed)
#   -f  test plan yaml (default: configs/eval_plans/ts_hw_test_only.yaml)
#   -o  output root relative to repo root (default: exp/eval_vllm)
#   -n  run-name subdirectory (default: <model>-<plan>-<timestamp>)
#   -j  concurrency (default: 64)
#   -l  default language fallback (default: zh-cn)
#   -w  hotwords pad-to count, 0 disables random padding (default: 10)
#   -R  hotword mode {none,random,retrieve,tower_retrieve} (default: none)
#   -K  retrieve top-k (default: 10)
#   -C  retrieve cache dir; empty uses Python default
#   -W  retrieve workers (default: 8)
#   -s  prompt style {swift,train,qwen3_asr} (default: swift; auto qwen3_asr for Qwen3-ASR names)
#   -b  backend {auto,chat,transcription} (default: auto)
#   -E  encoder source {vllm,triton} (default: vllm)
#   -U  Triton HTTP address (default: localhost:8000)
#   -V  Triton model name (default: rag_asr_retrieve)
#   -Q  Triton TOP_K (default: 0)
#   -M  manifest-dir
#   -S  SER/SEC manifest-dir
#   -T  TS-ASR test manifest-dir
#   -h  print help
#
# Extra arguments after "--" are passed to the Python evaluation driver.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

PORT="8000"
MODEL="qwen3-omni"
TEST_PLAN_FILE="configs/eval_plans/ts_hw_test_only.yaml"
OUTPUT_DIR="exp/eval_vllm"
RUN_NAME=""
CONCURRENCY=64
DEFAULT_LANG="zh-cn"
HOTWORDS_PAD_TO=10
HOTWORD_MODE="none"
RETRIEVE_TOP_K=10
RETRIEVE_CACHE_DIR=""
RETRIEVE_WORKERS=8
PROMPT_STYLE="swift"
PROMPT_STYLE_EXPLICIT=0
BACKEND="auto"
ENCODER_SOURCE="vllm"
TRITON_URL="localhost:8000"
TRITON_MODEL="rag_asr_retrieve"
TRITON_TOP_K=0
MANIFEST_DIR="/ai_sds_wuzz/DATA_ASR/LHOTSE"
SER_MANIFEST_DIR="/ai_sds_wuzz/DATA_SER/LHOTSE"
TSASR_TEST_MANIFEST_DIR="/chenmingjie/mingdong/data/lhotse/ts_hw_test"

usage() { sed -n '2,41p' "$0"; exit "${1:-0}"; }

while getopts ":p:m:f:o:n:j:l:w:R:K:C:W:s:b:E:U:V:Q:M:S:T:h" opt; do
  case "$opt" in
    p) PORT="$OPTARG" ;;
    m) MODEL="$OPTARG" ;;
    f) TEST_PLAN_FILE="$OPTARG" ;;
    o) OUTPUT_DIR="$OPTARG" ;;
    n) RUN_NAME="$OPTARG" ;;
    j) CONCURRENCY="$OPTARG" ;;
    l) DEFAULT_LANG="$OPTARG" ;;
    w) HOTWORDS_PAD_TO="$OPTARG" ;;
    R) HOTWORD_MODE="$OPTARG" ;;
    K) RETRIEVE_TOP_K="$OPTARG" ;;
    C) RETRIEVE_CACHE_DIR="$OPTARG" ;;
    W) RETRIEVE_WORKERS="$OPTARG" ;;
    s) PROMPT_STYLE="$OPTARG"; PROMPT_STYLE_EXPLICIT=1 ;;
    b) BACKEND="$OPTARG" ;;
    E) ENCODER_SOURCE="$OPTARG" ;;
    U) TRITON_URL="$OPTARG" ;;
    V) TRITON_MODEL="$OPTARG" ;;
    Q) TRITON_TOP_K="$OPTARG" ;;
    M) MANIFEST_DIR="$OPTARG" ;;
    S) SER_MANIFEST_DIR="$OPTARG" ;;
    T) TSASR_TEST_MANIFEST_DIR="$OPTARG" ;;
    h) usage 0 ;;
    \?) echo "[eval_vllm] unknown option: -$OPTARG" >&2; usage 1 ;;
    :) echo "[eval_vllm] option -$OPTARG requires an argument" >&2; usage 1 ;;
  esac
done
shift $((OPTIND - 1))
EXTRA_ARGS=("$@")

case "$HOTWORD_MODE" in
  none|random|retrieve|tower_retrieve) ;;
  *) echo "[eval_vllm] invalid -R $HOTWORD_MODE" >&2; usage 1 ;;
esac

case "$BACKEND" in
  auto|chat|transcription) ;;
  *) echo "[eval_vllm] invalid -b $BACKEND" >&2; usage 1 ;;
esac

case "$ENCODER_SOURCE" in
  vllm|triton) ;;
  *) echo "[eval_vllm] invalid -E $ENCODER_SOURCE" >&2; usage 1 ;;
esac

resolve_real_model_name() {
  local port="$1"
  local fallback="$2"
  local served
  served=$(python - "$port" <<'PY' 2>/dev/null || true
import json
import sys
import urllib.request

try:
    with urllib.request.urlopen(f"http://localhost:{sys.argv[1]}/v1/models", timeout=3) as response:
        data = json.load(response).get("data") or []
    for model in data:
        mid = model.get("id")
        if mid:
            print(mid)
except Exception:
    pass
PY
)
  if [[ -z "$served" ]]; then
    echo "$fallback"
    return
  fi
  if grep -Fxq -- "$fallback" <<<"$served"; then
    echo "$fallback"
    return
  fi
  sed -n '1p' <<<"$served"
}

REAL_MODEL="$(resolve_real_model_name "$PORT" "$MODEL")"
if [[ "$REAL_MODEL" != "$MODEL" ]]; then
  echo "[eval_vllm] auto-switch served model: '$MODEL' -> '$REAL_MODEL' (from /v1/models)"
  MODEL="$REAL_MODEL"
fi

if [[ "$PROMPT_STYLE_EXPLICIT" -eq 0 ]]; then
  model_lower="${MODEL,,}"
  if [[ "$model_lower" == *"1.7"* || "$model_lower" == *"qwen3-asr"* || "$model_lower" == *"qwen3_asr"* ]]; then
    PROMPT_STYLE="qwen3_asr"
    echo "[eval_vllm] auto-select prompt-style: qwen3_asr (model=$MODEL)"
  fi
fi

if [[ -z "$RUN_NAME" ]]; then
  CONFIG_BASENAME="$(basename "$TEST_PLAN_FILE")"
  CONFIG_BASENAME="${CONFIG_BASENAME%.yaml}"
  CONFIG_BASENAME="${CONFIG_BASENAME%.yml}"
  SAFE_MODEL="${MODEL//\//_}"; SAFE_MODEL="${SAFE_MODEL// /_}"
  SAFE_CONFIG="${CONFIG_BASENAME//\//_}"; SAFE_CONFIG="${SAFE_CONFIG// /_}"
  RUN_NAME="${SAFE_MODEL}-${SAFE_CONFIG}-$(date +%Y%m%d_%H%M%S)"
fi

CMD=(
  python -m open_audio_llm.integrations.vllm.test_vllm_inference
  --port "$PORT"
  --model "$MODEL"
  --test-plan-file "$TEST_PLAN_FILE"
  --output-dir "$REPO_ROOT/$OUTPUT_DIR"
  --manifest-dir "$MANIFEST_DIR"
  --ser-manifest-dir "$SER_MANIFEST_DIR"
  --tsasr-test-manifest-dir "$TSASR_TEST_MANIFEST_DIR"
  -j "$CONCURRENCY"
  --default-lang "$DEFAULT_LANG"
  --hotwords-pad-to "$HOTWORDS_PAD_TO"
  --hotword-mode "$HOTWORD_MODE"
  --retrieve-top-k "$RETRIEVE_TOP_K"
  --retrieve-cache-dir "$RETRIEVE_CACHE_DIR"
  --retrieve-workers "$RETRIEVE_WORKERS"
  --prompt-style "$PROMPT_STYLE"
  --backend "$BACKEND"
  --encoder-source "$ENCODER_SOURCE"
  --triton-url "$TRITON_URL"
  --triton-model "$TRITON_MODEL"
  --triton-top-k "$TRITON_TOP_K"
  --run-name "$RUN_NAME"
)
[[ ${#EXTRA_ARGS[@]} -gt 0 ]] && CMD+=("${EXTRA_ARGS[@]}")

echo "[eval_vllm] PORT=$PORT MODEL=$MODEL PLAN=$TEST_PLAN_FILE"
echo "[eval_vllm] OUTPUT=$REPO_ROOT/$OUTPUT_DIR RUN_NAME=$RUN_NAME"
echo "[eval_vllm] CONCURRENCY=$CONCURRENCY DEFAULT_LANG=$DEFAULT_LANG HOTWORDS_PAD_TO=$HOTWORDS_PAD_TO PROMPT_STYLE=$PROMPT_STYLE BACKEND=$BACKEND"
echo "[eval_vllm] HOTWORD_MODE=$HOTWORD_MODE RETRIEVE_TOP_K=$RETRIEVE_TOP_K RETRIEVE_WORKERS=$RETRIEVE_WORKERS"
echo "[eval_vllm] RETRIEVE_CACHE_DIR=${RETRIEVE_CACHE_DIR:-<default>}"
echo "[eval_vllm] ENCODER_SOURCE=$ENCODER_SOURCE TRITON_URL=$TRITON_URL TRITON_MODEL=$TRITON_MODEL TRITON_TOP_K=$TRITON_TOP_K"
[[ ${#EXTRA_ARGS[@]} -gt 0 ]] && echo "[eval_vllm] extra=${EXTRA_ARGS[*]}"

PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" exec "${CMD[@]}"
