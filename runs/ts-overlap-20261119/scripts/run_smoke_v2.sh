#!/usr/bin/env bash
# tasks/07 + 08 smoke: 40 steps of each arm side by side (GPUs 0-3 clean feature loss, 4-7 frozen-LLM control).
# W&B goes through the forwarder on 10.119.17.252:17922 (started for this node's IP).
set -u; umask 000
PY=/workspace/workspace/mingdong/miniconda3/envs/oal-train/bin/python
W=/workspace/workspace/mingdong/repo/oal-clean-teacher; X=runs/ts-overlap-20261119
LOGS=$1
echo "node ip $(hostname -I)"
for k in $(seq 1 80); do curl -s -m 10 -o /dev/null -x http://10.119.17.252:17922 https://api.wandb.ai && break; [ $k = 80 ] && { echo "W&B proxy unreachable"; exit 5; }; sleep 15; done
echo "W&B proxy ok $(date -u +%FT%TZ)"
cd $W
pids=()
for arm in feat frozenllm; do
  env -u PYTHONPATH /workspace/workspace/mingdong/repo/open-audio-llm/third_party/amphionkeys/load.sh sensecore -- \
    env PYTHONPATH=$X/shared/code-v2/src $PY -m open_audio_llm.cli train --config $X/configs/smoke-train-$arm.yaml \
    > $LOGS/smoke-$arm.cli.log 2>&1 & pids+=($!)
  sleep 20
done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done; echo "done rc=$rc $(date -u +%FT%TZ)"; exit $rc
