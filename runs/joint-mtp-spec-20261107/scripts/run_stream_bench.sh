#!/usr/bin/env bash
# 02-stream-bench on the ACP GRPO dev node: TS streaming replay, one lane per GPU (base, MTP k2, MTP k3) at once.
# Needs shared/draft-v16avg3 from run_spec_bench.sh. Usage: run_stream_bench.sh ATTEMPT_DIR
set -uo pipefail
umask 000
R=/workspace/workspace/mingdong/repo/oal-joint-mtp/runs/joint-mtp-spec-20261107; A=$1; cd $R; mkdir -p $A/logs
PY=/workspace/workspace/mingdong/miniconda3/envs/oal-train/bin/python
[ -f shared/draft-v16avg3/model.safetensors ] || exit 3
echo "start $(date -u +%FT%TZ)" > $A/status.txt
source /workspace/workspace/mingdong/.config/open-audio-llm/wandb-cluster.sh >/dev/null
export WANDB_HTTP_PROXY=http://10.119.17.252:17920 WANDB_HTTPS_PROXY=http://10.119.17.252:17920
unset no_proxy NO_PROXY
export PYTHONPATH=/workspace/workspace/mingdong/repo/oal-joint-mtp/src
pids=(); for i in 0 1 2; do cp configs/stream-lane$i.yaml $A/effective-lane$i.yaml; $PY scripts/stream_bench.py --config configs/stream-lane$i.yaml > $A/logs/lane$i.log 2>&1 & pids+=($!); done
rc=0; for x in "${pids[@]}"; do wait $x || rc=1; done
echo "done rc=$rc $(date -u +%FT%TZ)" >> $A/status.txt; exit $rc
