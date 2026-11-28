#!/usr/bin/env bash
# tasks/11 on one ACP node: train the looped encoder 6000 steps on 8 GPUs (configs/train-loop.yaml), avg3 of
# 5000/5500/6000 into the ae model cache, then the four 11.17 suites on four lanes served by shared/pack-loop.
# v12 reference scoring follows as its own runner step once the v12 run id is known.
# Usage: run_loop_train_eval.sh LOGS_DIR
set -u; umask 000
PY=/workspace/workspace/mingdong/miniconda3/envs/oal-train/bin/python
W=/workspace/workspace/mingdong/repo/oal-clean-teacher; X=runs/ts-overlap-20261119; E=$W/$X
LOGS=$1
export WB_PROXY=http://10.119.17.252:17922
echo "node ip $(hostname -I)"
for k in $(seq 1 80); do curl -s -m 10 -o /dev/null -x $WB_PROXY https://api.wandb.ai && break; [ $k = 80 ] && { echo "W&B proxy unreachable"; exit 5; }; sleep 15; done
echo "W&B proxy ok $(date -u +%FT%TZ)"
cd $W
before=$(ls $E/tasks/11-audio-loop/attempts | sort | tail -1)
env -u PYTHONPATH /workspace/workspace/mingdong/repo/open-audio-llm/third_party/amphionkeys/load.sh sensecore -- \
  env PYTHONPATH=$X/shared/code-v3/src $PY -m open_audio_llm.cli train --config $X/configs/train-loop.yaml \
  > $LOGS/train-loop.cli.log 2>&1
echo "training exit $? $(date -u +%FT%TZ)"
a=$(ls $E/tasks/11-audio-loop/attempts | sort | awk -v b=$before '$1>b' | head -1)
state=$(python3 -c "import json;print(json.load(open('$E/tasks/11-audio-loop/attempts/$a/status.json'))['execution'])")
echo "11-audio-loop $a $state"
[ "$state" = completed ] || { echo "training did not complete"; exit 3; }
N=ts-overlap-loop-$a-avg3
F=$(bash $E/scripts/avg3_cache.sh 11-audio-loop $a $N | tail -1) || { echo "avg3 $N failed"; exit 4; }
echo "avg3 $N $F $(date -u +%FT%TZ)"
PACK=$E/shared/pack-loop bash $E/scripts/eval_lanes.sh $N loop 0 19900 \
  "eval-sumfull-es116-core/v12" "eval-sumfull-es116-core/tsen" \
  "eval-v18avg3-ckpt400-asrfull/asrfull" "eval-sumfull-es116-core/custts" > $LOGS/eval-loop.log 2>&1
rc=$?; echo "evals exit $rc $(date -u +%FT%TZ)"
exit $rc
