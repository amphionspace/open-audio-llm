#!/usr/bin/env bash
# tasks/19 evaluation: apply the best mask estimator to the heavy-overlap set (task 18 stems) and the side-effect set,
# then 11.17 TS (pack-mtp, as the gate) on [enrollment | extracted] and [enrollment | mixture] in one pass, 8 servers.
set -u; T=/workspace/workspace/mingdong/.claude/tmp/claude-1001/-workspace-workspace-mingdong-repo-open-audio-llm/ca9aa6d0-4ecc-4cfc-b526-5322bf6c9de5/scratchpad/ovdiag/tse; S=/workspace/workspace/mingdong/.claude/tmp/claude-1001/-workspace-workspace-mingdong-repo-open-audio-llm/ca9aa6d0-4ecc-4cfc-b526-5322bf6c9de5/scratchpad/ovdiag/tse/..; W=/workspace/data/ts-overlap-20261119/task19-tse; CK=${1:-$W/ckpt/best.pt}; TAG=${2:-best}; GPUS=(${GPUS:-0 1 2 3 4 5 6 7}); UTIL=${UTIL:-0.6}
PY=/workspace/workspace/mingdong/miniconda3/envs/oal-train/bin/python; AE=/workspace/workspace/mingdong/miniconda3/envs/amphion-eval/bin/python
PACK=/workspace/workspace/mingdong/repo/oal-joint-mtp/runs/joint-mtp-spec-20261107/shared/pack-mtp
MODEL=/workspace/data/ae-state/model-cache/entries/model-afea989da22b77adc0ce9365
export OMP_NUM_THREADS=4 no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
rm -rf $S/subset/tse-$TAG $W/safety/tse-$TAG
CUDA_VISIBLE_DEVICES=${GPUS[0]} $PY $T/tse_apply.py $CK $S/subset $S/subset/tse-$TAG $S/subset/stems > $W/logs/apply-heavy-$TAG.log 2>&1 &
CUDA_VISIBLE_DEVICES=${GPUS[-1]} $PY $T/tse_apply.py $CK $W/safety $W/safety/tse-$TAG > $W/logs/apply-safety-$TAG.log 2>&1 &
wait || exit 1
pids=(); urls=()
for g in "${GPUS[@]}"; do port=$((19900 + g))
  CUDA_VISIBLE_DEVICES=$g MODEL=$MODEL PYTHON=/workspace/workspace/mingdong/miniconda3/envs/vllm-serving/bin/python PORT=$port NAME=m1117 GPU_UTIL=$UTIL AMPHION_MTP_TOKENS=0 setsid bash $PACK/serve.sh tsasr --max-num-seqs 96 > $W/logs/serve-$TAG-$g.log 2>&1 &
  pids+=($!); urls+=(http://127.0.0.1:$port)
done
trap 'for p in "${pids[@]}"; do kill -TERM -$p 2>/dev/null; done' EXIT
for i in "${!urls[@]}"; do until curl -s -m 5 ${urls[$i]}/v1/models >/dev/null; do kill -0 ${pids[$i]} 2>/dev/null || { echo "server $i died"; exit 1; }; sleep 10; done; done
U=$(IFS=,; echo "${urls[*]}")
env -u PYTHONPATH $AE $S/asr_probe.py $S/subset $W/asr-heavy-$TAG.jsonl $U tse-$TAG > $W/logs/asr-heavy-$TAG.log 2>&1 & a1=$!
env -u PYTHONPATH $AE $S/asr_probe.py $W/safety $W/asr-safety-$TAG.jsonl $U tse-$TAG > $W/logs/asr-safety-$TAG.log 2>&1 & a2=$!
wait $a1 && wait $a2 || { echo "asr failed"; exit 1; }
env -u PYTHONPATH $AE $T/tse_report.py $S/subset $W/asr-heavy-$TAG.jsonl $S/asr-oracle.jsonl $W/safety $W/asr-safety-$TAG.jsonl $S/subset/tse-$TAG/sisdr.json $W/report-$TAG.json > $W/logs/report-$TAG.log 2>&1
echo "tse eval rc=$? $(date -u +%FT%TZ)"; cat $W/logs/report-$TAG.log
