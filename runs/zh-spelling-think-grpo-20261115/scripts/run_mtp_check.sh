#!/usr/bin/env bash
# 04-mtp-borrow: same as run_eval.sh, but the suites run one after another and both servers' /metrics are saved
# before and after each suite, so the spec-decode counters of every suite can be told apart (show_mtp.py).
# usage: run_mtp_check.sh <configs/eval-name> <wandb proxy url> <ae executable> <suite> [suite ...]
set -u
DIR=$(cd "$1" && pwd); PROXY=$2; AE=$3; shift 3
EXP=$(cd "$DIR/../.." && pwd)
LOG=$DIR/logs; mkdir -p "$LOG"
PY=/workspace/workspace/mingdong/miniconda3/envs/oal-train/bin/python
source /workspace/workspace/mingdong/.config/open-audio-llm/wandb-cluster.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 AMPHION_EVAL_HOME=/workspace/data/ae-state \
  AUDIO_DATA_CATALOG=/workspace/data/registry/catalog AUDIO_DATA_ROOTS_FILE=/workspace/data/registry/roots.json \
  WANDB_HTTP_PROXY=$PROXY WANDB_HTTPS_PROXY=$PROXY
port() { $PY -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['task']['arguments']['--port'])" "$1"; }
declare -A PID PORT
for m in asr tsasr; do
  PORT[$m]=$(port "$DIR/serve-$m.yaml")
  curl -s --noproxy '*' -m 3 "http://127.0.0.1:${PORT[$m]}/v1/models" >/dev/null && { echo "port ${PORT[$m]} busy" >> "$LOG/status.txt"; exit 1; }
  env PYTHONPATH="$EXP/shared/code-v2/src" $PY -m open_audio_llm.cli serve --config "$DIR/serve-$m.yaml" > "$LOG/serve-$m.log" 2>&1 &
  PID[$m]=$!
done
stop() { for m in "${!PID[@]}"; do kill -TERM "${PID[$m]}" 2>/dev/null; done; wait; }
for m in "${!PID[@]}"; do
  until curl -s --noproxy '*' -m 5 "http://127.0.0.1:${PORT[$m]}/v1/models" >/dev/null 2>&1; do
    kill -0 "${PID[$m]}" 2>/dev/null || { echo "$m server exited $(date -u +%FT%TZ)" >> "$LOG/status.txt"; stop; exit 1; }
    sleep 15
  done
  echo "$m server on ${PORT[$m]} ready $(date -u +%FT%TZ)" >> "$LOG/status.txt"
done
scrape() { for m in asr tsasr; do curl -s --noproxy '*' -m 30 "http://127.0.0.1:${PORT[$m]}/metrics" | grep '^vllm:spec_decode' > "$LOG/metrics-$1-$m.txt"; done; }
status=0
for s in "$@"; do
  scrape "before-$s"
  "$AE" open-audio-llm after-training --handoff "$DIR/handoff.json" --config "$DIR/after-training-$s.yaml" > "$LOG/eval-$s.log" 2>&1
  rc=$?; echo "$s exit=$rc $(date -u +%FT%TZ)" >> "$LOG/status.txt"
  scrape "after-$s"
  [ "$rc" = 0 ] || status=$rc
done
stop
echo "done $(date -u +%FT%TZ)" >> "$LOG/status.txt"
exit $status
