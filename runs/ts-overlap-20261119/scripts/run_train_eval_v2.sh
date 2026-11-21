#!/usr/bin/env bash
# tasks/07 + 08: both arms 6000 steps side by side (GPUs 0-3 clean feature loss, 4-7 frozen-LLM control), then avg3 of
# 5000/5500/6000 into the
# ae model cache, then each arm on 11.17's suites (two lanes per arm): Chinese TS v12 + English TS, Chinese plain ASR + customer TS.
set -u; umask 000
PY=/workspace/workspace/mingdong/miniconda3/envs/oal-train/bin/python
W=/workspace/workspace/mingdong/repo/oal-clean-teacher; X=runs/ts-overlap-20261119; E=$W/$X
LOGS=$1
export WB_PROXY=http://10.119.17.252:17922
echo "node ip $(hostname -I)"
for k in $(seq 1 80); do curl -s -m 10 -o /dev/null -x $WB_PROXY https://api.wandb.ai && break; [ $k = 80 ] && { echo "W&B proxy unreachable"; exit 5; }; sleep 15; done
echo "W&B proxy ok $(date -u +%FT%TZ)"
cd $W
declare -A before
for t in 07-clean-feature 08-frozen-llm; do before[$t]=$(ls $E/tasks/$t/attempts | sort | tail -1); done
pids=()
for arm in feat frozenllm; do
  env -u PYTHONPATH /workspace/workspace/mingdong/repo/open-audio-llm/third_party/amphionkeys/load.sh sensecore -- \
    env PYTHONPATH=$X/shared/code-v2/src $PY -m open_audio_llm.cli train --config $X/configs/train-$arm.yaml \
    > $LOGS/train-$arm.cli.log 2>&1 & pids+=($!)
  sleep 30
done
for p in "${pids[@]}"; do wait $p; done; echo "training done $(date -u +%FT%TZ)"
declare -A model
for x in 07-clean-feature:feat 08-frozen-llm:frozenllm; do t=${x%%:*}; arm=${x#*:}
  a=$(ls $E/tasks/$t/attempts | sort | awk -v b=${before[$t]} '$1>b' | head -1)
  state=$(python3 -c "import json;print(json.load(open('$E/tasks/$t/attempts/$a/status.json'))['execution'])"); echo "$t $a $state"
  [ "$state" = completed ] || { echo "$t did not complete"; continue; }
  N=ts-overlap-$arm-$a-avg3
  F=$(bash $E/scripts/avg3_cache.sh $t $a $N | tail -1) || { echo "avg3 $N failed"; continue; }
  echo "avg3 $N $F $(date -u +%FT%TZ)"; model[$arm]=$N
done
pids=(); gb=0; pb=19900
for arm in feat frozenllm; do
  [ -n "${model[$arm]:-}" ] || continue
  bash $E/scripts/eval_lanes.sh ${model[$arm]} $arm $gb $pb \
    "eval-sumfull-es116-core/v12 eval-sumfull-es116-core/tsen" \
    "eval-v18avg3-ckpt400-asrfull/asrfull eval-sumfull-es116-core/custts" > $LOGS/eval-$arm.log 2>&1 & pids+=($!)
  gb=$((gb+4)); pb=$((pb+100)); sleep 60
done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
[ ${#model[@]} = 2 ] || rc=1
echo "done rc=$rc $(date -u +%FT%TZ)"; exit $rc
