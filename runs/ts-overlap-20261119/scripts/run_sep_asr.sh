#!/usr/bin/env bash
# Two 11.17 TS servers (pack-mtp, as the gate) on the given GPUs, wait for the separation shards, run asr_probe.py.
set -u; S=$(cd $(dirname $0) && pwd); cd $S
PACK=/workspace/workspace/mingdong/repo/oal-joint-mtp/runs/joint-mtp-spec-20261107/shared/pack-mtp
MODEL=/workspace/data/ae-state/model-cache/entries/model-afea989da22b77adc0ce9365
export no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
pids=(); urls=(); k=0
for g in "$@"; do port=$((19981 + k))
  CUDA_VISIBLE_DEVICES=$g MODEL=$MODEL PYTHON=/workspace/workspace/mingdong/miniconda3/envs/vllm-serving/bin/python PORT=$port \
    NAME=m1117 GPU_UTIL=0.35 AMPHION_MTP_TOKENS=0 setsid bash $PACK/serve.sh tsasr --max-num-seqs 64 > logs/serve-$g.log 2>&1 &
  pids+=($!); urls+=(http://127.0.0.1:$port); k=$((k + 1))
done
trap 'for p in "${pids[@]}"; do kill -TERM -$p 2>/dev/null; done' EXIT
for i in "${!urls[@]}"; do until curl -s -m 5 ${urls[$i]}/v1/models >/dev/null; do kill -0 ${pids[$i]} 2>/dev/null || { echo "server $i died"; exit 1; }; sleep 10; done; done
echo "servers ready $(date -u +%FT%TZ)"
until [ $(grep -l "done" logs/sep-*.log 2>/dev/null | wc -l) = 4 ]; do sleep 20; done
echo "separation done $(date -u +%FT%TZ)"
env -u PYTHONPATH /workspace/workspace/mingdong/miniconda3/envs/amphion-eval/bin/python asr_probe.py subset asr.jsonl $(IFS=,; echo "${urls[*]}")
echo "exit $? $(date -u +%FT%TZ)"
