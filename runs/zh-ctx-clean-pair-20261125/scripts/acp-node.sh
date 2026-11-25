#!/usr/bin/env bash
# The whole experiment on one ACP node (README):
#   train  train-noclean (configs/train-noclean.yaml, 8 GPUs, about 1.3 h; the checkpoints go to whai);
#   merge  fetch its checkpoint-3000 into the ae model cache, link it with 11.17's mtp/ for serving (noclean-3000),
#          then pm-clean = 11.17 + (control - noclean) (runs/zh-ctx-grpo-20261120/scripts/merge_increment.py);
#   eval   at the same time: pm-clean on the 27 delivery suites (GPUs 0-3, zh-ctx-grpo-20261120 acp-delivery-one.sh) and
#          context-correction v2 think off of pm-clean (4/5) and noclean1106 (6/7) (zh-ctx-grpo-k-20261122 configs).
# Fetch and staging as runs/zh-ctx-grpo-k-20261122/scripts/eval_after_training.sh.
# usage: acp-node.sh [stage ...]   (stages: train merge eval; default all three, in that order)
set -u; umask 002
R=/workspace/workspace/mingdong/repo/oal-ctx-grpo; X=$R/runs/zh-ctx-clean-pair-20261125
K=$R/runs/zh-ctx-grpo-k-20261122; G=$R/runs/zh-ctx-grpo-20261120
M=/workspace/data/ae-runs/ctx-grpo-k-20261122-pairmerge
BASE=/workspace/data/packages/amphionasr-1.7b-v3-v18aes-mtp-candidate/model
CONTROL=$M/control-3000; NOCLEAN=$M/noclean-3000; OUT=$M/1117-plus-control-minus-noclean
CODE=/workspace/workspace/mingdong/repo/oal-think-exp/runs/zh-spelling-think-20261113/shared/code-v1
PY=/workspace/workspace/mingdong/miniconda3/envs/oal-train/bin/python
MODELS=/workspace/workspace/mingdong/scratch/acp-dev/ae-models.sh
AE=/workspace/workspace/mingdong/.local-tools/amphion-eval-0.14.0-eedcf15/bin/ae
PROXY=http://10.119.17.252:17920
now() { date -u +%FT%TZ; }
for i in $(seq 60); do  # the forward only lets registered node IPs through; training and every eval log to W&B
  code=$(curl -s -m 10 -o /dev/null -w '%{http_code}' -x $PROXY https://api.wandb.ai/healthz)
  [ "$code" = 200 ] && break
  echo "W&B through $PROXY: $code, waiting for the node IP to be let through $(now)"; sleep 30
done
[ "$code" = 200 ] || exit 2

train() {
  cd $R
  env -u PYTHONPATH /workspace/workspace/mingdong/repo/open-audio-llm/third_party/amphionkeys/load.sh sensecore -- \
    env PYTHONPATH=$CODE/src:$CODE/audio-data-contract-src $PY \
    -m open_audio_llm.cli experiment run --config runs/zh-ctx-clean-pair-20261125/experiment.yaml --task train-noclean
}

merge() {
  local A s REM C f
  A=$(ls -d $X/tasks/01-train-noclean/attempts/[0-9][0-9][0-9] | sort | tail -n 1)
  s=$($PY -c "import json;print(json.load(open('$A/status.json'))['execution'])" 2>/dev/null)
  echo "train-noclean $(basename $A) $s $(now)"; [ "$s" = completed ] || return 3
  REM=whai:open-audio-llm/runs/zh-ctx-clean-pair-20261125/tasks/01-train-noclean/attempts/$(basename $A)/artifacts/training/checkpoint-3000/
  C=$(bash $MODELS fetch $REM) || return 5
  mkdir -p $NOCLEAN; for f in $C/*; do ln -sfn $f $NOCLEAN/$(basename $f); done; ln -sfn $BASE/mtp $NOCLEAN/mtp
  printf '%s\n' "source: $REM (ae cache $C), files symlinked." \
    "mtp/: symlink to $BASE/mtp (11.17 package), as control-3000: the arm trained without joint MTP; greedy verification." \
    > $NOCLEAN/SOURCE.txt
  [ -f $OUT/model.safetensors ] && { echo "merge exists: $OUT"; return 0; }
  $PY $G/scripts/merge_increment.py $BASE $NOCLEAN $CONTROL $OUT || return 6
  echo "merged $OUT $(now)"
}

evaluate() {
  [ -f $OUT/model.safetensors ] && [ -f $NOCLEAN/model.safetensors ] || { echo "missing $OUT or $NOCLEAN"; return 3; }
  # the generator checks mtp/ is in place, so the ctx v2 configs are made here, after the merge
  (cd $K && $PY scripts/make_eval_configs.py pm-clean $OUT 4 5 18966 \
         && $PY scripts/make_eval_configs.py noclean1106 $NOCLEAN 6 7 18968) || return 6
  bash $G/scripts/acp-delivery-one.sh pm-clean $OUT 0 1 18940 2 3 18942 & local a=$!
  sleep 60
  ctx() { (cd $K && bash scripts/run_eval.sh configs/eval-$1 $PROXY $AE context-correction-v2); local r=$?
          echo "ctx v2 $1 exit $r $(now)"; return $r; }
  ctx pm-clean & local b=$!
  sleep 30  # stagger the vLLM starts
  ctx noclean1106 & local c=$!
  wait $b; local rb=$?; wait $c; local rc=$?; wait $a; local ra=$?
  echo "exit delivery-pm-clean=$ra ctx-v2-pm-clean=$rb ctx-v2-noclean1106=$rc $(now)"
  [ $ra = 0 ] && [ $rb = 0 ] && [ $rc = 0 ]
}

STAGES=("$@"); [ ${#STAGES[@]} -gt 0 ] || STAGES=(train merge eval)
for stage in "${STAGES[@]}"; do
  echo "$stage $(now)"
  case $stage in
    train) train || exit $?;;
    merge) merge || exit $?;;
    eval) evaluate || exit $?;;
    *) echo "unknown stage $stage"; exit 2;;
  esac
done
echo "done $(now)"
