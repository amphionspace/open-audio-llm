#!/usr/bin/env bash
# tasks/11 serving check on four local GPUs: 11.17 and the deep debug model on the unmodified pack-mtp, the identity
# loop-init and the true-loop debug model on pack-loop; then serve_equivalence.py over every STEP-th v12 TS request.
# Pass: loop-init == 11.17 and trueloop == deep on every request (and trueloop != 11.17 somewhere: the loop runs).
# Usage: run_loop_serve_check.sh ATTEMPT_DIR GPU0 GPU1 GPU2 GPU3 [STEP]
set -u
A=$1; G=("$2" "$3" "$4" "$5"); STEP=${6:-50}
E=/workspace/workspace/mingdong/repo/oal-clean-teacher/runs/ts-overlap-20261119
ENVS=/workspace/workspace/mingdong/miniconda3/envs
OLD=/workspace/workspace/mingdong/repo/oal-joint-mtp/runs/joint-mtp-spec-20261107/shared/pack-mtp
NEW=$E/shared/pack-loop
D=/workspace/data/ts-overlap-20261119/models
RUN=/workspace/data/ae-runs/whitebox-merge-v18a-es116inc/runs/run-20261117T032925Z-961b82b3
export no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
names=(m1117 loopinit deep trueloop)
models=(/workspace/data/ae-state/model-cache/entries/model-afea989da22b77adc0ce9365 $D/1117-loop16-23-init $D/loop-equivalence-debug/deep $D/loop-equivalence-debug/trueloop)
packs=($OLD $NEW $OLD $NEW)
pids=()
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=${G[$i]} MODEL=${models[$i]} PYTHON=$ENVS/vllm-serving/bin/python PORT=$((19951 + i)) \
    NAME=${names[$i]} GPU_UTIL=0.25 AMPHION_MTP_TOKENS=0 setsid bash ${packs[$i]}/serve.sh tsasr --max-num-seqs 8 \
    > $A/logs/serve-${names[$i]}.log 2>&1 &
  pids+=($!)
done
trap 'for p in "${pids[@]}"; do kill -TERM -$p 2>/dev/null; done' EXIT
for i in 0 1 2 3; do
  until curl -s -m 5 http://127.0.0.1:$((19951 + i))/v1/models >/dev/null; do
    kill -0 ${pids[$i]} 2>/dev/null || { echo "server ${names[$i]} died"; exit 1; }; sleep 10
  done
done
echo "servers ready $(date -u +%FT%TZ)"
specs=(); for i in 0 1 2 3; do specs+=("${names[$i]}=http://127.0.0.1:$((19951 + i))=${names[$i]}"); done
env -u PYTHONPATH AUDIO_DATA_CATALOG=/workspace/data/registry/catalog AUDIO_DATA_ROOTS_FILE=/workspace/data/registry/roots.json \
  $ENVS/amphion-eval/bin/python $E/scripts/serve_equivalence.py $RUN $STEP $A/artifacts/serve-equivalence.jsonl "${specs[@]}"
echo "exit $? $(date -u +%FT%TZ)"
