#!/usr/bin/env bash
# The whole paired baseline on one ACP node (README "执行"): sft (K0's SFT through the experiment launcher) -> configs
# (fetch K0's checkpoint-3000 into the ae model cache, scripts/make_configs.py grpo) -> mining (scripts/mining.sh) ->
# grpo (scripts/train.sh) -> eval (scripts/eval_after_training.sh). Each stage refuses to start unless the previous
# one finished; stages can be given explicitly to resume.
# Arm k0s2 (second seed, make_configs.py ARMS) when the first argument is k0s2.
# usage: acp-node.sh [k0s2] [stage ...]   (stages: sft configs mining grpo eval; default all, in that order)
set -u; umask 002
ARM=k0; [ "${1:-}" = k0s2 ] && { ARM=k0s2; shift; }
SFT=$([ $ARM = k0 ] && echo 01-sft-k0 || echo 05-sft-k0s2)
R=/workspace/workspace/mingdong/repo/oal-ctx-grpo; E=$R/runs/zh-ctx-mainline-pair-20261127; cd $E
PY=/workspace/workspace/mingdong/miniconda3/envs/oal-train/bin/python
CODE1=/workspace/workspace/mingdong/repo/oal-clue-target/runs/zh-ctx-clue-target-20261117/shared/code-v1  # K's SFT code
PROXY=http://10.119.18.89:17920
now() { date -u +%FT%TZ; }
for i in $(seq 60); do  # the forward only lets registered node IPs through; every stage logs to W&B
  code=$(curl -s -m 10 -o /dev/null -w '%{http_code}' -x $PROXY https://api.wandb.ai/healthz)
  [ "$code" = 200 ] && break
  echo "W&B through $PROXY: $code, waiting for the node IP to be let through $(now)"; sleep 30
done
[ "$code" = 200 ] || exit 2
sft_done() {
  for d in $E/tasks/$SFT/attempts/*/; do grep -q '"execution": "completed"' $d/status.json 2>/dev/null && basename $d; done | tail -n 1
}
sft() {
  (cd $R && env -u PYTHONPATH /workspace/workspace/mingdong/repo/open-audio-llm/third_party/amphionkeys/load.sh sensecore -- \
     env PYTHONPATH=$CODE1/src:$CODE1/audio-data-contract-src $PY \
     -m open_audio_llm.cli experiment run --config runs/zh-ctx-mainline-pair-20261127/experiment.yaml --task sft-$ARM)
  local rc=$?; echo "sft exit $rc $(now)"; [ -n "$(sft_done)" ]
}
configs() {
  local a start; a=$(sft_done); [ -n "$a" ] || { echo "no completed $SFT attempt"; return 5; }
  start=$(bash /workspace/workspace/mingdong/scratch/acp-dev/ae-models.sh fetch \
    whai:open-audio-llm/runs/zh-ctx-mainline-pair-20261127/tasks/$SFT/attempts/$a/artifacts/training/checkpoint-3000/) || return 5
  echo "$ARM start $start $(now)"
  $PY scripts/make_configs.py grpo "$start" $ARM
}
STAGES=("$@"); [ ${#STAGES[@]} -gt 0 ] || STAGES=(sft configs mining grpo eval)
for stage in "${STAGES[@]}"; do
  echo "$stage $(now)"
  case $stage in
    sft) sft || exit $?;;
    configs) configs || exit $?;;
    mining) bash scripts/mining.sh $ARM || exit $?;;
    grpo) bash scripts/train.sh $ARM || exit $?;;
    eval) bash scripts/eval_after_training.sh $ARM || exit $?;;
    *) echo "unknown stage $stage"; exit 2;;
  esac
done
echo "done $(now)"
