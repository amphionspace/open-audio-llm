#!/usr/bin/env bash
# tasks/10 on local GPUs: one 11.17 TS server (pack-mtp, the 11.17 gate stack) per GPU, probe_ts_sampling.py split into
# one shard per server, 16 samples at temperature 1 for every v12 target-present zh sample with overlap >= 0.9.
# Usage: run_grpo_headroom.sh ATTEMPT_DIR GPU [GPU ...]
set -u
A=$1; shift; G=("$@"); K=${#G[@]}
E=/workspace/workspace/mingdong/repo/oal-clean-teacher/runs/ts-overlap-20261119
ENVS=/workspace/workspace/mingdong/miniconda3/envs
PACK=/workspace/workspace/mingdong/repo/oal-joint-mtp/runs/joint-mtp-spec-20261107/shared/pack-mtp
MODEL=/workspace/data/ae-state/model-cache/entries/model-afea989da22b77adc0ce9365
RUN=/workspace/data/ae-runs/whitebox-merge-v18a-es116inc/runs/run-20261117T032925Z-961b82b3
export no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p $A/logs $A/artifacts
pids=()
for i in $(seq 0 $((K - 1))); do
  CUDA_VISIBLE_DEVICES=${G[$i]} MODEL=$MODEL PYTHON=$ENVS/vllm-serving/bin/python PORT=$((19961 + i)) NAME=m1117 \
    GPU_UTIL=0.4 AMPHION_MTP_TOKENS=0 setsid bash $PACK/serve.sh tsasr --max-num-seqs 64 > $A/logs/serve-$i.log 2>&1 &
  pids+=($!)
done
trap 'for p in "${pids[@]}"; do kill -TERM -$p 2>/dev/null; done' EXIT
for i in $(seq 0 $((K - 1))); do
  until curl -s -m 5 http://127.0.0.1:$((19961 + i))/v1/models >/dev/null; do
    kill -0 ${pids[$i]} 2>/dev/null || { echo "server $i died"; exit 1; }; sleep 10
  done
done
echo "servers ready $(date -u +%FT%TZ)"
cp=(); rc=0
for i in $(seq 0 $((K - 1))); do
  env -u PYTHONPATH AUDIO_DATA_CATALOG=/workspace/data/registry/catalog AUDIO_DATA_ROOTS_FILE=/workspace/data/registry/roots.json \
    $ENVS/amphion-eval/bin/python $E/scripts/probe_ts_sampling.py $RUN http://127.0.0.1:$((19961 + i)) m1117 \
    $A/artifacts/shard-$i.jsonl 16 0.9 $i $K > $A/logs/probe-$i.log 2>&1 & cp+=($!)
done
for p in "${cp[@]}"; do wait $p || rc=1; done
echo "exit $rc $(date -u +%FT%TZ)"; exit $rc
