#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-asr}"
if [[ $# -gt 0 ]]; then shift; fi
case "$MODE" in
    asr) export AMPHION_TSASR_INSERT_SEP=0; DEFAULT_PORT=8000 ;;
    tsasr) export AMPHION_TSASR_INSERT_SEP=1; DEFAULT_PORT=8001 ;;
    *) echo "Usage: bash serve.sh [asr|tsasr] [vLLM options...]" >&2; exit 2 ;;
esac
# TS-ASR 请求开头的注册段长度（秒），服务按这个长度切分注册段与混合段；客户端必须发送同样长度的注册段。
# 默认 5 秒（本包所有目标说话人数字都按 5 秒测）；服务端与客户端的 AMPHION_TSASR_ENROLL_SEC 必须相同。
export AMPHION_TSASR_ENROLL_SEC="${AMPHION_TSASR_ENROLL_SEC:-5}"
export AMPHION_TSASR_ENROLL_NOPAD=0
export COT_AUDIO_CHUNKED_ATTN=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENABLE_V1_MULTIPROCESSING=1
# The backend is registered explicitly; ignore unrelated installed plugins.
export VLLM_PLUGINS=""
export PYTHONPATH="$ROOT/runtime${PYTHONPATH:+:$PYTHONPATH}"
# BF16 下同一请求的冷、热 prefix cache 计算路径可能改变 Chinese/None 的出字判断，所以关闭 prefix cache。
exec "${PYTHON:-python}" "$ROOT/runtime/serve_qwen3.py" "${MODEL:-$ROOT/model}" \
    --host "${HOST:-127.0.0.1}" \
    --port "${PORT:-$DEFAULT_PORT}" \
    --served-model-name "${NAME:-amphionasr-v3-$MODE}" \
    --dtype bfloat16 \
    --tensor-parallel-size 1 \
    --limit-mm-per-prompt '{"audio": 1}' \
    --gpu-memory-utilization "${GPU_UTIL:-0.60}" \
    --max-model-len "${MAX_MODEL_LEN:-8192}" \
    --no-enable-prefix-caching \
    "$@"
