#!/usr/bin/env bash
# Start a swift rollout server for GRPO server-mode validation.
#
# Usage:
#   bash examples/train/rollout/run_rollout_server.sh \
#       -m /path/to/merged_model \
#       -p 8006 \
#       -g 0
#
# Options:
#   -m  merged Open Audio-LLM HF checkpoint (required)
#   -o  rollout model directory (default: <model>_vllm_rollout)
#   -A  vLLM-compatible architecture override (default: TransformersForCausalLM)
#   -p  server port (default: 8006)
#   -g  CUDA_VISIBLE_DEVICES (default: unchanged)
#   -t  tensor parallel size (default: 1)
#   -d  data parallel size (default: 1)
#   -u  vLLM GPU memory utilization (default: 0.9)
#   -l  max model length (default: 1024)
#   -E  enforce eager {true,false} (default: true)
#   -h  print help

set -euo pipefail
umask 0000

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

MODEL="${MODEL:-}"
ROLLOUT_MODEL="${ROLLOUT_MODEL:-}"
VLLM_COMPAT_ARCHITECTURE="${VLLM_COMPAT_ARCHITECTURE:-TransformersForCausalLM}"
PORT="${PORT:-8006}"
GPUS=""
TP_SIZE="${TP_SIZE:-1}"
DP_SIZE="${DP_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
ENFORCE_EAGER="${ENFORCE_EAGER:-true}"

usage() { sed -n '2,22p' "$0"; exit "${1:-0}"; }

while getopts ":m:o:A:p:g:t:d:u:l:E:h" opt; do
  case "$opt" in
    m) MODEL="$OPTARG" ;;
    o) ROLLOUT_MODEL="$OPTARG" ;;
    A) VLLM_COMPAT_ARCHITECTURE="$OPTARG" ;;
    p) PORT="$OPTARG" ;;
    g) GPUS="$OPTARG" ;;
    t) TP_SIZE="$OPTARG" ;;
    d) DP_SIZE="$OPTARG" ;;
    u) GPU_MEMORY_UTILIZATION="$OPTARG" ;;
    l) MAX_MODEL_LEN="$OPTARG" ;;
    E) ENFORCE_EAGER="$OPTARG" ;;
    h) usage 0 ;;
    \?) echo "[rollout] unknown option: -$OPTARG" >&2; usage 1 ;;
    :) echo "[rollout] option -$OPTARG requires an argument" >&2; usage 1 ;;
  esac
done
shift $((OPTIND - 1))

if [[ -z "$MODEL" && $# -gt 0 ]]; then
  MODEL="$1"
  shift
fi
if [[ -z "$MODEL" ]]; then
  echo "[rollout] ERROR: -m is required" >&2
  usage 1
fi
ROLLOUT_MODEL="${ROLLOUT_MODEL:-${MODEL%/}_vllm_rollout}"

if [[ -n "$GPUS" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPUS"
fi

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

python - "$REPO_ROOT" "$MODEL" "$ROLLOUT_MODEL" "$VLLM_COMPAT_ARCHITECTURE" <<'PY'
import json
import shutil
import sys
from pathlib import Path

repo_root = Path(sys.argv[1]).resolve()
source = Path(sys.argv[2]).resolve()
target = Path(sys.argv[3]).resolve()
architecture = sys.argv[4]

if not (source / "config.json").is_file():
    raise SystemExit(f"Missing config.json under MODEL: {source}")

target.mkdir(parents=True, exist_ok=True)

for item in source.iterdir():
    if item.name in {"config.json", "configuration_audio_llm.py", "modeling_audio_llm.py"}:
        continue
    link = target / item.name
    if link.exists() or link.is_symlink():
        continue
    link.symlink_to(item)

config = json.loads((source / "config.json").read_text())
config["architectures"] = [architecture]
(target / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")

for filename in ("configuration_audio_llm.py", "modeling_audio_llm.py"):
    shutil.copy2(repo_root / "src" / "open_audio_llm" / filename, target / filename)

print(f"Prepared vLLM rollout model: {target}")
print(f"vLLM architecture override: {architecture}")
PY

swift rollout \
  --model "$ROLLOUT_MODEL" \
  --external_plugins \
    "$REPO_ROOT/src/open_audio_llm/integrations/ms_swift/register_audio_llm.py" \
    "$REPO_ROOT/src/open_audio_llm/integrations/vllm/patch_registry_subprocess.py" \
  --vllm_tensor_parallel_size "$TP_SIZE" \
  --vllm_data_parallel_size "$DP_SIZE" \
  --vllm_gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
  --vllm_max_model_len "$MAX_MODEL_LEN" \
  --vllm_enforce_eager "$ENFORCE_EAGER" \
  --port "$PORT"
