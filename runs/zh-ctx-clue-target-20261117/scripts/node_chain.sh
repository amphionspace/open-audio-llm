#!/usr/bin/env bash
# zh-ctx-clue-target-20261117 on one ACP node (inbox runner, as root). Stages, in the order given:
#   smoke     serving smoke test of the eval stack (vLLM 0.32, think template, joint-MTP draft) on the 11.17 package,
#             GPUs 0/1; a failure is logged and does not stop the chain (eval can be fixed and rerun alone)
#   train     the arms in $ARMS one after the other, each on GPUs 0-7 (experiment run skips an arm already completed)
#   eval      each arm's checkpoint-3000 through the ae model cache, all at the same time on the context-correction
#             eval set v2, think off and on: the i-th arm of $ARMS on GPUs 2i/2i+1, ports 18710+10i/+1
#   implicit  the implicit data's TTS (4 shards) and QC on GPUs 4-7, then QC scoring, build, think (clue + target)
#             and the no-explanation think variant (amphiondata run ctx-implicit-zh-20261117, full-run.sh)
#   configs   make_configs.py --implicit: the two implicit arms' configs and tasks
# usage: [ARMS="a b"] node_chain.sh stage ...   (default ARMS: template clue-target; default stages: smoke train eval)
set -u; umask 002
R=/workspace/workspace/mingdong/repo/oal-clue-target; E=$R/runs/zh-ctx-clue-target-20261117; S=$E/shared/code-v1
PY=/workspace/workspace/mingdong/miniconda3/envs/oal-train/bin/python
KEYS=/workspace/workspace/mingdong/repo/open-audio-llm/third_party/amphionkeys/load.sh
MODELS=/workspace/workspace/mingdong/scratch/acp-dev/ae-models.sh
AE=/workspace/workspace/mingdong/.local-tools/amphion-eval-0.14.0-eedcf15/bin/ae
PROXY=http://10.119.17.252:17920
PACKAGE=/workspace/data/packages/amphionasr-1.7b-v3-v18aes-mtp-candidate/model
DATA=/workspace/workspace/mingdong/data/amphiondata/ctx-implicit-zh-20261117
IMPLICIT=/workspace/data/datasets/context_correction_synth_zh/versions/implicit-v1-20261117
ARMS=(${ARMS:-template clue-target})
STAGES=("$@"); [ ${#STAGES[@]} -gt 0 ] || STAGES=(smoke train eval)
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
  for arm in "${ARMS[@]}"; do
    echo "start train-$arm $(now)"
    cd $R  # the config path is relative to the repository root
    env -u PYTHONPATH $KEYS sensecore -- env PYTHONPATH=$S/src:$S/audio-data-contract-src \
      $PY -m open_audio_llm.cli experiment run --config runs/zh-ctx-clue-target-20261117/experiment.yaml --task train-$arm \
      </dev/null || { cd $E; echo "train-$arm failed $(now)"; return 3; }
    cd $E; echo "train-$arm done $(now)"
  done
}

evaluate() {  # arm asr-gpu ts-gpu port
  local f
  f=$(bash $MODELS fetch "whai:open-audio-llm/runs/zh-ctx-clue-target-20261117/tasks/01-train-$1/attempts/001/artifacts/training/checkpoint-3000/") || return 5
  echo "$1 checkpoint $f $(now)"
  $PY scripts/make_eval_configs.py "$1" "$f" "$2" "$3" "$4" || return 6
  bash scripts/run_eval.sh "configs/eval-$1" $PROXY $AE context-correction-v2 context-correction-v2-think
}

eval_all() {
  local i pids=() rc=0
  for i in "${!ARMS[@]}"; do
    evaluate "${ARMS[$i]}" $((2 * i)) $((2 * i + 1)) $((18710 + 10 * i)) & pids+=($!)
    sleep 60  # staggered vLLM starts (torch.distributed port race, 2026-11-12)
  done
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}"; r=$?; echo "eval ${ARMS[$i]} exit $r $(now)"; [ "$r" = 0 ] || rc=4
  done
  return $rc
}

implicit() {
  local k r pids=()
  [ -s $DATA/full/items.jsonl ] || { echo "no implicit items yet"; return 7; }
  for k in 0 1 2 3; do
    (for attempt in 1 2; do GPU=$((4 + k)) bash $DATA/full-run.sh tts $k/4 >/dev/null && exit 0; done; exit 1) & pids+=($!)
  done
  for k in "${!pids[@]}"; do wait "${pids[$k]}" || { echo "tts shard $k failed $(now)"; return 8; }; done
  bash $DATA/full-run.sh tts-merge || return 8
  GPU=4 bash $DATA/full-run.sh qc-asr sensevoice >/dev/null & a=$!
  GPU=5 bash $DATA/full-run.sh qc-asr qwen3asr >/dev/null & b=$!
  wait $a && wait $b || { echo "qc asr failed $(now)"; return 9; }
  bash $DATA/full-run.sh qc-score && bash $DATA/full-run.sh build && bash $DATA/full-run.sh think || return 10
  $PY scripts/make_implicit_noexplain_think.py || return 10
  # files written here as root go back to the owner of the run and version directories
  chown -R "$(stat -c %u:%g $DATA)" $DATA "$IMPLICIT"
}

for stage in "${STAGES[@]}"; do
  echo "== $stage $(now)"
  case $stage in
    smoke) smoke; echo "smoke exit $? $(now)";;
    train) train || exit $?;;
    eval) eval_all || exit $?;;
    implicit) implicit || exit $?; echo "implicit data done $(now)";;
    configs) $PY scripts/make_configs.py --implicit || exit 11;;
    *) echo "unknown stage $stage"; exit 2;;
  esac
done
echo "chain done $(now)"
