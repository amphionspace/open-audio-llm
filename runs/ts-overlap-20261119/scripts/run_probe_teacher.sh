#!/usr/bin/env bash
# 11.17 TS on the clean version of every generated scene: 8 pack-mtp servers (one per GPU), 8 client shards.
set -u; P=/workspace/workspace/mingdong/.claude/tmp/claude-1001/-workspace-workspace-mingdong-repo-open-audio-llm/ca9aa6d0-4ecc-4cfc-b526-5322bf6c9de5/scratchpad/ovdiag/probe; F=/workspace/data/ts-overlap-20261119/task17-probe; D=$F/scenes100k
PACK=/workspace/workspace/mingdong/repo/oal-joint-mtp/runs/joint-mtp-spec-20261107/shared/pack-mtp
MODEL=/workspace/data/ae-state/model-cache/entries/model-afea989da22b77adc0ce9365
export no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
pids=(); urls=()
for g in 0 1 2 3 4 5 6 7; do port=$((19900 + g))
  CUDA_VISIBLE_DEVICES=$g MODEL=$MODEL PYTHON=/workspace/workspace/mingdong/miniconda3/envs/vllm-serving/bin/python PORT=$port NAME=m1117 GPU_UTIL=0.6 AMPHION_MTP_TOKENS=0 setsid bash $PACK/serve.sh tsasr --max-num-seqs 96 > $F/logs/teacher-serve-$g.log 2>&1 &
  pids+=($!); urls+=(http://127.0.0.1:$port)
done
trap 'for p in "${pids[@]}"; do kill -TERM -$p 2>/dev/null; done' EXIT
for i in "${!urls[@]}"; do until curl -s -m 5 ${urls[$i]}/v1/models >/dev/null; do kill -0 ${pids[$i]} 2>/dev/null || { echo "server $i died"; exit 1; }; sleep 10; done; done
echo "servers ready $(date -u +%FT%TZ)"
cp=(); for s in 0 1 2 3 4 5 6 7; do
  env -u PYTHONPATH /workspace/workspace/mingdong/miniconda3/envs/amphion-eval/bin/python $P/teacher_clean.py $D/manifest.jsonl $D/wav $F/teacher-$s.jsonl ${urls[$s]} $s 8 > $F/logs/teacher-$s.log 2>&1 & cp+=($!)
done
rc=0; for p in "${cp[@]}"; do wait $p || rc=1; done; echo "teacher rc=$rc $(date -u +%FT%TZ)"; exit $rc
