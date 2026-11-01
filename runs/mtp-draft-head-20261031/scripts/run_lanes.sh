#!/usr/bin/env bash
# Run one bench command per lane at once on the ACP dev node (GPUs 4-7), W&B through the dev-pod forwarder.
# Usage: run_lanes.sh ATTEMPT_DIR spec CONFIG LANE... | run_lanes.sh ATTEMPT_DIR stream CONFIG...
set -uo pipefail
umask 000
export CUDA_VISIBLE_DEVICES=4,5,6,7
EXP=/workspace/workspace/mingdong/repo/oal-mtp/runs/mtp-draft-head-20261031
PY=/workspace/workspace/mingdong/miniconda3/envs/oal-train/bin/python
ATTEMPT=$1; KIND=$2; shift 2
source /workspace/workspace/mingdong/.config/open-audio-llm/wandb-cluster.sh >/dev/null
export WANDB_HTTP_PROXY=http://10.119.16.21:17920 WANDB_HTTPS_PROXY=http://10.119.16.21:17920
unset no_proxy NO_PROXY
export PYTHONPATH=/workspace/workspace/mingdong/repo/oal-mtp/src
cd "$EXP"
mkdir -p "$ATTEMPT/logs"
pids=()
if [[ $KIND == spec ]]; then
  CONFIG=$1; shift
  cp "$CONFIG" "$ATTEMPT/effective.yaml"
  for lane in "$@"; do
    "$PY" scripts/spec_check.py --config "$CONFIG" --lane "$lane" > "$ATTEMPT/logs/lane$lane.log" 2>&1 &
    pids+=($!)
  done
else
  i=0
  for config in "$@"; do
    cp "$config" "$ATTEMPT/effective-lane$i.yaml"
    "$PY" scripts/stream_bench.py --config "$config" > "$ATTEMPT/logs/lane$i.log" 2>&1 &
    pids+=($!); i=$((i + 1))
  done
fi
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit $status
