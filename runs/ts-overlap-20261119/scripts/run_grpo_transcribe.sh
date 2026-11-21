#!/usr/bin/env bash
# tasks/12 on one ACP node: mining on 11.17 (round 26's lanes and candidates, 16 samples per prompt, final text) ->
# six plans (grpo_mining_plans.py) -> GRPO 400 steps (configs/train-grpo-transcribe.yaml) with the checkpoint collector
# -> avg3 of checkpoints 200/300/400 and its 50% interpolation with 11.17 (as round 26) -> the four 11.17 suites for both.
# Mining follows runs/zh-ctx-grpo-20261120/scripts/mining.sh without its ctx lanes; W&B through this experiment's
# forward (10.119.17.252:17922, the node IP let through).
# usage: run_grpo_transcribe.sh [stage ...]   (stages: mining train evals; default all three, in that order)
set -u; umask 000
E=/workspace/workspace/mingdong/repo/oal-clean-teacher/runs/ts-overlap-20261119; cd $E
ENVS=/workspace/workspace/mingdong/miniconda3/envs; C=configs/grpo-mining; T=tasks/12-grpo-transcribe; O=$T/mining
M=/workspace/workspace/mingdong/scratch/acp-dev/ae-models.sh
PROXY=http://10.119.17.252:17922
START=/workspace/data/ae-state/model-cache/entries/model-afea989da22b77adc0ce9365
now() { date -u +%FT%TZ; }
for i in $(seq 60); do
  code=$(curl -s -m 10 -o /dev/null -w '%{http_code}' -x $PROXY https://api.wandb.ai/healthz)
  [ "$code" = 200 ] && break; echo "W&B through $PROXY: $code $(now)"; sleep 30
done
[ "$code" = 200 ] || exit 2
wait_gpus() { for i in $(seq 360); do [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>2000' | wc -l) -eq 0 ] && break; sleep 10; done
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader; }

mining() {
  mkdir -p $O; wait_gpus
  source /workspace/workspace/mingdong/.config/open-audio-llm/wandb-cluster.sh
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 NUMPY_MADVISE_HUGEPAGE=0 \
    WANDB_ENTITY=1016097967-amphion WANDB_PROJECT=open-audio-llm WANDB_DIR=$E/$O WANDB_HTTP_PROXY=$PROXY WANDB_HTTPS_PROXY=$PROXY
  local MINE="env PYTHONPATH=shared/code-v4/src:shared/code-v4/audio-data-contract-src $ENVS/oal-train-vllm/bin/python -m open_audio_llm.scripts.grpo_mining --data_config configs/train-data-grpo-transcribe.yaml --workers 6 --concurrency 48 --samples 16"
  sources() { local k; for k in $(cat $C/sources-$1.txt); do printf -- '--source %s ' "$k"; done; }
  start_servers() {
    local kind=$1; pids=(); ports=()
    for n in 0 1 2 3 4 5 6; do ports+=($(( $2 + n )))
      $ENVS/oal-train/bin/open-audio-llm serve --config $C/$kind-$n.yaml > $O/$kind-$n.log 2>&1 & pids+=($!); done
    for n in 0 1 2 3 4 5 6; do
      until curl -s --noproxy '*' -m 5 http://127.0.0.1:${ports[$n]}/v1/models >/dev/null 2>&1; do
        kill -0 ${pids[$n]} 2>/dev/null || { echo "$kind $n exited $(now)"; kill -TERM "${pids[@]}" 2>/dev/null; return 1; }
        sleep 15
      done
    done; echo "$kind servers ready $(now)"
  }
  stop_servers() { kill -TERM "${pids[@]}" 2>/dev/null; wait "${pids[@]}" 2>/dev/null; sleep 30; }
  wait_all() { local i x rc=0; for i in "${!mpids[@]}"; do wait ${mpids[$i]}; x=$?; echo "${names[$i]} exit $x $(now)"; [ $x -ne 0 ] && rc=$x; done; return $rc; }
  local lanes=() rc
  start_servers serve-asr 19400 || return 1
  mpids=(); names=()
  for n in 0 1 2 3 4 5 6; do url=http://127.0.0.1:${ports[$n]}
    L=$O/plain-lane-$n; mkdir -p $L; lanes+=($L)
    $MINE $(sources plain) --candidates 900 --epoch $((20 + n)) --server $url --model mine-1117-asr --output $L/kept.jsonl > $L/run.log 2>&1 &
    mpids+=($!); names+=(plain-$n)
    L=$O/miss-lane-$n; mkdir -p $L; lanes+=($L)
    $MINE $(sources miss) --candidates 400 --epoch $((40 + n)) --server $url --model mine-1117-asr --output $L/kept.jsonl > $L/run.log 2>&1 &
    mpids+=($!); names+=(miss-$n)
  done
  wait_all; rc=$?; stop_servers; [ $rc -eq 0 ] || return $rc
  start_servers serve-tsasr 19410 || return 1
  mpids=(); names=()
  for n in 0 1 2 3 4 5 6; do L=$O/ts-lane-$n; mkdir -p $L; lanes+=($L); url=http://127.0.0.1:${ports[$n]}
    $MINE $(sources ts) --candidates 75 --epoch $((30 + n)) --server $url --model mine-1117-tsasr \
      --ts_server $url --ts_model mine-1117-tsasr --output $L/kept.jsonl > $L/run.log 2>&1 &
    mpids+=($!); names+=(ts-$n)
  done
  wait_all; rc=$?; stop_servers; [ $rc -eq 0 ] || return $rc
  env PYTHONPATH=shared/code-v4/src $ENVS/oal-train/bin/python scripts/grpo_mining_plans.py \
    $O AmphionASR-ts-overlap-grpo-mining-1117-n16-v26lanes-ts-transcribe "${lanes[@]}" > $O/plans.log 2>&1; rc=$?
  echo "plans exit $rc $(now)"; head -1 $O/plans.log; return $rc
}

train() {
  local p
  for p in hotword nonspeech miss ts-absent ts-present ts-transcribe; do [ -s $O/plan-$p.jsonl ] || { echo "plan $p empty"; return 2; }; done
  wc -l $O/plan-*.jsonl; wait_gpus
  local before=$(ls $T/attempts 2>/dev/null | sort | tail -1)
  cd /workspace/workspace/mingdong/repo/oal-clean-teacher
  env -u PYTHONPATH /workspace/workspace/mingdong/repo/open-audio-llm/third_party/amphionkeys/load.sh sensecore -- \
    env PYTHONPATH=runs/ts-overlap-20261119/shared/code-v4/src $ENVS/oal-train/bin/python -m open_audio_llm.cli train \
    --config runs/ts-overlap-20261119/configs/train-grpo-transcribe.yaml > $E/$T/train.cli.log 2>&1 &
  local tp=$!; cd $E
  local a=; until [ -n "$a" ]; do kill -0 $tp 2>/dev/null || break; a=$(ls $T/attempts 2>/dev/null | sort | awk -v b="$before" '$1>b' | head -1); sleep 10; done
  [ -n "$a" ] && { echo "attempt $a"; bash scripts/grpo_collect_checkpoints.sh $a 12-grpo-transcribe transcribe-$a > $T/attempts/$a/collect.log 2>&1 & }
  wait $tp; local rc=$?; echo "training exit $rc $(now)"; wait
  echo "$a" > $T/last-attempt; return $rc
}

evals() {
  local a=$(cat $T/last-attempt) AVG=/workspace/workspace/mingdong/scratch/ts033/avg_ckpt.py PY=$ENVS/oal-train/bin/python
  local c=() s f
  for s in 200 300 400; do f=$(bash $M fetch ts-overlap-grpo-transcribe-$a-ckpt$s 2>/dev/null) || { echo "fetch ckpt$s failed"; return 4; }
    c+=("$(echo "$f" | tail -1)"); done
  local W=$(bash $M stage); $PY $AVG $W "${c[@]}" | tail -1; rm -f $W/.complete
  local A=$(bash $M put $W --name ts-overlap-grpo-transcribe-$a-avg3 | tail -1); echo "avg3 $A $(now)"
  local W2=$(bash $M stage); $PY $AVG $W2 $START $A | tail -1; rm -f $W2/.complete
  local I=$(bash $M put $W2 --name ts-overlap-grpo-transcribe-$a-interp050 | tail -1); echo "interp050 $I $(now)"
  ls $A/mtp $I/mtp || return 4
  local rc=0 x
  for x in avg3 interp050; do
    bash scripts/eval_lanes.sh ts-overlap-grpo-transcribe-$a-$x grpo-$x 0 19900 \
      "eval-sumfull-es116-core/v12" "eval-sumfull-es116-core/tsen" \
      "eval-v18avg3-ckpt400-asrfull/asrfull" "eval-sumfull-es116-core/custts" > $T/eval-$x.log 2>&1 || rc=1
    echo "evals $x done rc=$rc $(now)"
  done
  return $rc
}

STAGES=("$@"); [ ${#STAGES[@]} -gt 0 ] || STAGES=(mining train evals)
for stage in "${STAGES[@]}"; do
  echo "$stage $(now)"
  $stage || { x=$?; echo "$stage failed $x $(now)"; exit $x; }
done
echo "done $(now)"
