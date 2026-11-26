#!/usr/bin/env bash
# The whole experiment on one ACP node (README "执行"):
#   train  both arms through the experiment launcher: train-real on GPUs 0-3 first; train-real-synth on GPUs 4-7 starts once
#          train-real's data preflight of the shared replay config has finished (the two preflights would build the
#          same index cache at once); waits for both, requires execution "completed" for both;
#   eval   the last checkpoint of each arm (klreal, klsynth): context-correction v2 think off of both first (GPUs 4/5, 6/7,
#          runs/zh-ctx-grpo-k-20261122 configs and run_eval.sh) while klreal runs the 27 delivery suites (GPUs 0-3,
#          runs/zh-ctx-grpo-20261120 acp-delivery-one.sh); then klsynth's 27 suites on GPUs 4-7.
# Outputs: /workspace/data/ae-runs/ctx-kl-anchor-20261126/<arm>/checkpoint-<step> (models),
#          /workspace/data/ae-runs/ctx-grpo-20261120-delivery-<name> and ctx-grpo-k-20261122-<name> (evaluations).
# usage: acp-node.sh [stage ...]   (stages: train eval; default both, in that order)
set -u; umask 002
R=/workspace/workspace/mingdong/repo/oal-ctx-grpo; X=$R/runs/zh-ctx-kl-anchor-20261126
K=$R/runs/zh-ctx-grpo-k-20261122; G=$R/runs/zh-ctx-grpo-20261120
CODE=$G/shared/code-v3
M=/workspace/data/ae-runs/ctx-kl-anchor-20261126
PY=/workspace/workspace/mingdong/miniconda3/envs/oal-train/bin/python
AE=/workspace/workspace/mingdong/.local-tools/amphion-eval-0.14.0-eedcf15/bin/ae
PROXY=http://10.119.18.89:17920  # dev container IP since 2026-11-26
now() { date -u +%FT%TZ; }
for i in $(seq 60); do  # the forward only lets registered node IPs through; training and every eval log to W&B
  code=$(curl -s -m 10 -o /dev/null -w '%{http_code}' -x $PROXY https://api.wandb.ai/healthz)
  [ "$code" = 200 ] && break
  echo "W&B through $PROXY: $code, waiting for the node IP to be let through $(now)"; sleep 30
done
[ "$code" = 200 ] || exit 2

launch() {  # task id -> experiment launcher in the background
  (cd $R && env -u PYTHONPATH /workspace/workspace/mingdong/repo/open-audio-llm/third_party/amphionkeys/load.sh sensecore -- \
     env PYTHONPATH=$CODE/src:$CODE/audio-data-contract-src $PY \
     -m open_audio_llm.cli experiment run --config runs/zh-ctx-kl-anchor-20261126/experiment.yaml --task "$1")
}

last_attempt() { ls -d $X/tasks/$1/attempts/[0-9][0-9][0-9] 2>/dev/null | sort | tail -n 1; }

completed() {
  local A s; A=$(last_attempt "$1")
  s=$($PY -c "import json;print(json.load(open('$A/status.json'))['execution'])" 2>/dev/null)
  echo "$1 $(basename "$A") $s $(now)"; [ "$s" = completed ]
}

train() {
  local before a b ra rb A
  before=$(last_attempt 01-train-real)
  launch train-real & a=$!
  echo "train-real started $(now)"
  for i in $(seq 720); do  # up to 6 h for the preflight; it took about 55 min for a new WenetSpeech index on 11-25
    A=$(last_attempt 01-train-real)
    [ -n "$A" ] && [ "$A" != "$before" ] && [ -f "$A/artifacts/data-preflight-replay.json" ] && break
    kill -0 $a 2>/dev/null || break
    sleep 30
  done
  echo "train-real preflight done (or exited) $(now)"
  launch train-real-synth & b=$!
  echo "train-real-synth started $(now)"
  wait $a; ra=$?; wait $b; rb=$?
  echo "launcher exit train-real=$ra train-real-synth=$rb $(now)"
  completed 01-train-real && completed 02-train-real-synth
}

final_ckpt() { ls -d $M/$1/checkpoint-[0-9]* 2>/dev/null | sort -t- -k2 -n | tail -n 1; }

ctx() { (cd $K && bash scripts/run_eval.sh configs/eval-$1 $PROXY $AE context-correction-v2); local r=$?
        echo "ctx v2 $1 exit $r $(now)"; return $r; }

evaluate() {
  local A B a b c ra rb rc rd
  A=$(final_ckpt real); B=$(final_ckpt real-synth)
  [ -f "$A/model.safetensors" ] && [ -f "$B/model.safetensors" ] || { echo "missing checkpoints: '$A' '$B'"; return 3; }
  chmod -R a+rX $M
  echo "klreal=$A klsynth=$B $(now)"
  (cd $K && $PY scripts/make_eval_configs.py klreal $A 4 5 18966 && $PY scripts/make_eval_configs.py klsynth $B 6 7 18968) || return 6
  bash $G/scripts/acp-delivery-one.sh klreal $A 0 1 18940 2 3 18942 & a=$!
  sleep 60
  ctx klreal & b=$!
  sleep 30  # stagger the vLLM starts
  ctx klsynth & c=$!
  wait $b; rb=$?; wait $c; rc=$?
  bash $G/scripts/acp-delivery-one.sh klsynth $B 4 5 18950 6 7 18952; rd=$?
  wait $a; ra=$?
  echo "exit delivery-klreal=$ra ctx-v2-klreal=$rb ctx-v2-klsynth=$rc delivery-klsynth=$rd $(now)"
  [ $ra = 0 ] && [ $rb = 0 ] && [ $rc = 0 ] && [ $rd = 0 ]
}

STAGES=("$@"); [ ${#STAGES[@]} -gt 0 ] || STAGES=(train eval)
for stage in "${STAGES[@]}"; do
  echo "$stage $(now)"
  case $stage in
    train) train || exit $?;;
    eval) evaluate || exit $?;;
    *) echo "unknown stage $stage"; exit 2;;
  esac
done
echo "done $(now)"
