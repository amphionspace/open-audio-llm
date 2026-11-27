#!/usr/bin/env bash
# The whole experiment on one ACP node (README "执行"):
#   train  both arms of the set through the experiment launcher: arm A (real) on GPUs 0-3 first; arm B (real-synth) on
#          GPUs 4-7 starts once arm A's data preflight of the shared replay config has finished (the two preflights would
#          build the same index cache at once); waits for both, requires execution "completed" for both;
#   eval   the last checkpoint of each arm (base: klreal, klsynth; textce: kltreal, kltsynth; deckl: kldreal, kldsynth;
#          decbatch: kldbreal, kldbsynth):
#          context-correction v2
#          think off of both first (GPUs 4/5, 6/7, runs/zh-ctx-grpo-k-20261122 configs and run_eval.sh) while arm A runs
#          the 27 delivery suites (GPUs 0-3, runs/zh-ctx-grpo-20261120 acp-delivery-one.sh); then arm B's on GPUs 4-7.
# Outputs: /workspace/data/ae-runs/ctx-kl-anchor-20261126/<arm dir>/checkpoint-<step> (models),
#          /workspace/data/ae-runs/ctx-grpo-20261120-delivery-<name> and ctx-grpo-k-20261122-<name> (evaluations).
# usage: acp-node.sh SET [stage ...]   (SET: base = the 11-26 arms, textce = README "后续：CE 只训正文",
#        deckl = README "后续 2：回放 KL 加在场判断项", decbatch = README "后续 3：判断项用大批 TS 样本";
#        stages: train eval; default both, in that order)
set -u; umask 002
R=/workspace/workspace/mingdong/repo/oal-ctx-grpo; X=$R/runs/zh-ctx-kl-anchor-20261126
K=$R/runs/zh-ctx-grpo-k-20261122; G=$R/runs/zh-ctx-grpo-20261120
CODE=$G/shared/code-v3
M=/workspace/data/ae-runs/ctx-kl-anchor-20261126
PY=/workspace/workspace/mingdong/miniconda3/envs/oal-train/bin/python
AE=/workspace/workspace/mingdong/.local-tools/amphion-eval-0.14.0-eedcf15/bin/ae
PROXY=http://10.119.18.89:17920  # dev container IP since 2026-11-26
now() { date -u +%FT%TZ; }
# per set: task ids, task dirs, checkpoint dirs and model names of arm A, then arm B
case ${1:-} in
  base) IA=train-real IB=train-real-synth TA=01-train-real TB=02-train-real-synth DA=real DB=real-synth NA=klreal NB=klsynth;;
  textce) IA=train-real-textce IB=train-real-synth-textce TA=04-train-real-textce TB=05-train-real-synth-textce
          DA=real-textce DB=real-synth-textce NA=kltreal NB=kltsynth;;
  deckl) IA=train-real-deckl IB=train-real-synth-deckl TA=07-train-real-deckl TB=08-train-real-synth-deckl
         DA=real-deckl DB=real-synth-deckl NA=kldreal NB=kldsynth;;
  decbatch) IA=train-real-decbatch IB=train-real-synth-decbatch TA=10-train-real-decbatch TB=11-train-real-synth-decbatch
            DA=real-decbatch DB=real-synth-decbatch NA=kldbreal NB=kldbsynth;;
  *) echo "usage: acp-node.sh base|textce|deckl|decbatch [stage ...]"; exit 2;;
esac
shift
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
  before=$(last_attempt $TA)
  launch $IA & a=$!
  echo "$IA started $(now)"
  for i in $(seq 720); do  # up to 6 h for the preflight; it took about 55 min for a new WenetSpeech index on 11-25
    A=$(last_attempt $TA)
    [ -n "$A" ] && [ "$A" != "$before" ] && [ -f "$A/artifacts/data-preflight-replay.json" ] && break
    kill -0 $a 2>/dev/null || break
    sleep 30
  done
  echo "$IA preflight done (or exited) $(now)"
  launch $IB & b=$!
  echo "$IB started $(now)"
  wait $a; ra=$?; wait $b; rb=$?
  echo "launcher exit $IA=$ra $IB=$rb $(now)"
  completed $TA && completed $TB
}

# The arm's last checkpoint, only if its SOURCE.json says it is the final step (2026-11-26: `sort -t- -k2 -n` keyed on a
# '-' of the path itself and picked checkpoint-778 over checkpoint-1557).
final_ckpt() {
  local c; c=$(ls -d $M/$1/checkpoint-[0-9]* 2>/dev/null | sort -V | tail -n 1)
  [ -n "$c" ] && $PY -c "import json,sys;d=json.load(open(sys.argv[1]+'/SOURCE.json'));sys.exit(d['step']!=d['steps'])" "$c" \
    && echo "$c"
}

ctx() { (cd $K && bash scripts/run_eval.sh configs/eval-$1 $PROXY $AE context-correction-v2); local r=$?
        echo "ctx v2 $1 exit $r $(now)"; return $r; }

evaluate() {
  local A B a b c ra rb rc rd
  A=$(final_ckpt $DA); B=$(final_ckpt $DB)
  [ -f "$A/model.safetensors" ] && [ -f "$B/model.safetensors" ] || { echo "missing checkpoints: '$A' '$B'"; return 3; }
  chmod -R a+rX $M
  echo "$NA=$A $NB=$B $(now)"
  (cd $K && $PY scripts/make_eval_configs.py $NA $A 4 5 18966 && $PY scripts/make_eval_configs.py $NB $B 6 7 18968) || return 6
  bash $G/scripts/acp-delivery-one.sh $NA $A 0 1 18940 2 3 18942 & a=$!
  sleep 60
  ctx $NA & b=$!
  sleep 30  # stagger the vLLM starts
  ctx $NB & c=$!
  wait $b; rb=$?; wait $c; rc=$?
  bash $G/scripts/acp-delivery-one.sh $NB $B 4 5 18950 6 7 18952; rd=$?
  wait $a; ra=$?
  echo "exit delivery-$NA=$ra ctx-v2-$NA=$rb ctx-v2-$NB=$rc delivery-$NB=$rd $(now)"
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
