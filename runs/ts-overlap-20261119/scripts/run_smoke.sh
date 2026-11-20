#!/usr/bin/env bash
# tasks/03 + 04 smoke: 40 steps of each arm side by side (GPUs 0-3 clean KL, 4-7 control), after the enrollment evals free the GPUs.
set -u; umask 000
PY=/workspace/workspace/mingdong/miniconda3/envs/oal-train/bin/python
W=/workspace/workspace/mingdong/repo/oal-clean-teacher; X=runs/ts-overlap-20261119
until [ -f /workspace/workspace/mingdong/scratch/acp-tsov/done/02-enroll-length-1117-v12.exit ]; do sleep 20; done
echo "GPUs free $(date -u +%FT%TZ)"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '; echo
cd $W
pids=()
for arm in cleankl control; do
  env -u PYTHONPATH /workspace/workspace/mingdong/repo/open-audio-llm/third_party/amphionkeys/load.sh sensecore -- \
    env PYTHONPATH=$X/shared/code-v1/src $PY -m open_audio_llm.cli train --config $X/configs/smoke-train-$arm.yaml \
    > /workspace/workspace/mingdong/scratch/acp-tsov/logs/smoke2-$arm.cli.log 2>&1 & pids+=($!)
  sleep 20
done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done; echo "done rc=$rc $(date -u +%FT%TZ)"; exit $rc
