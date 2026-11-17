#!/usr/bin/env bash
# zh-ctx-clue-target-20261117 on one ACP node (inbox runner, as root):
#   1. serving smoke test of the eval stack (vLLM 0.32, think template, joint-MTP draft) on the 11.17 package, GPUs 0/1;
#      a failure is logged and does not stop training (the eval stage can be fixed and rerun alone);
#   2. the two arms one after the other, each on GPUs 0-7 (experiment run skips an arm that already completed);
#   3. both checkpoint-3000s through the ae model cache, evaluated at the same time on the context-correction eval set
#      v2, think off and on: template on GPUs 0/1 (ports 18710/18711), clue-target on GPUs 2/3 (18720/18721).
# usage: node_chain.sh [smoke|train|eval ...]   (default: all three)
set -u; umask 002
R=/workspace/workspace/mingdong/repo/oal-clue-target; E=$R/runs/zh-ctx-clue-target-20261117; S=$E/shared/code-v1
PY=/workspace/workspace/mingdong/miniconda3/envs/oal-train/bin/python
KEYS=/workspace/workspace/mingdong/repo/open-audio-llm/third_party/amphionkeys/load.sh
MODELS=/workspace/workspace/mingdong/scratch/acp-dev/ae-models.sh
AE=/workspace/workspace/mingdong/.local-tools/amphion-eval-0.14.0-eedcf15/bin/ae
PROXY=http://10.119.17.252:17920
PACKAGE=/workspace/data/packages/amphionasr-1.7b-v3-v18aes-mtp-candidate/model
STAGES=("${@:-smoke train eval}"); STAGES=(${STAGES[*]})
now() { date -u +%FT%TZ; }
cd $E

smoke() {
  $PY scripts/make_eval_configs.py smoke "$PACKAGE" 0 1 18700 || return 1
  sed -i 's/samples_per_source: 0$/samples_per_source: 2/' configs/eval-smoke/after-training-context-correction-v2*.yaml
  bash scripts/run_eval.sh configs/eval-smoke $PROXY $AE context-correction-v2-think
}

wandb_reachable() {  # the forward only lets registered node IPs through; training refuses to start without W&B
  for i in $(seq 60); do
    code=$(curl -s -m 10 -o /dev/null -w '%{http_code}' -x $PROXY https://api.wandb.ai/healthz)
    [ "$code" = 200 ] && return 0
    echo "W&B through $PROXY: $code, waiting for the node IP to be let through $(now)"; sleep 30
  done
  return 1
}

train() {
  wandb_reachable || return 2
  for arm in template clue-target; do
    echo "start train-$arm $(now)"
    env -u PYTHONPATH $KEYS sensecore -- env PYTHONPATH=$S/src:$S/audio-data-contract-src \
      $PY -m open_audio_llm.cli experiment run --config runs/zh-ctx-clue-target-20261117/experiment.yaml --task train-$arm \
      </dev/null || { cd $E; echo "train-$arm failed $(now)"; return 3; }
    echo "train-$arm done $(now)"
  done
}

evaluate() {  # arm asr-gpu ts-gpu port
  local f
  f=$(bash $MODELS fetch "whai:open-audio-llm/runs/zh-ctx-clue-target-20261117/tasks/01-train-$1/attempts/001/artifacts/training/checkpoint-3000/") || return 5
  echo "$1 checkpoint $f $(now)"
  $PY scripts/make_eval_configs.py "$1" "$f" "$2" "$3" "$4" || return 6
  bash scripts/run_eval.sh "configs/eval-$1" $PROXY $AE context-correction-v2 context-correction-v2-think
}

for stage in "${STAGES[@]}"; do
  echo "== $stage $(now)"
  case $stage in
    smoke) smoke; echo "smoke exit $? $(now)";;
    train) train || exit $?;;
    eval)
      evaluate template 0 1 18710 & a=$!
      sleep 60  # staggered vLLM starts (torch.distributed port race, 2026-11-12)
      evaluate clue-target 2 3 18720 & b=$!
      wait $a; ra=$?; wait $b; rb=$?
      echo "eval exit template=$ra clue-target=$rb $(now)"
      [ "$ra" = 0 ] && [ "$rb" = 0 ] || exit 4;;
    *) echo "unknown stage $stage"; exit 2;;
  esac
done
echo "chain done $(now)"
