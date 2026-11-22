#!/usr/bin/env bash
# tasks/13 on one ACP node: seven extra TS mining lanes on 11.17 (epochs 70-76, 150 candidates per source, tasks/12's
# servers and sources) -> ts-hard plan from tasks/12's and these TS lanes (grpo_hard_plan.py) -> GRPO 400 steps
# (configs/train-grpo-hard.yaml) with the checkpoint collector -> avg3 and its 50% interpolation with 11.17 -> the four
# 11.17 suites for both, side by side. Same flow as run_grpo_transcribe.sh (tasks/12).
# usage: run_grpo_hard.sh [stage ...]   (stages: mining train evals; default all three, in that order)
set -u; umask 000
E=/workspace/workspace/mingdong/repo/oal-clean-teacher/runs/ts-overlap-20261119; cd $E
ENVS=/workspace/workspace/mingdong/miniconda3/envs; C=configs/grpo-mining-hard; T=tasks/13-grpo-hard; O=$T/mining
M=/workspace/workspace/mingdong/scratch/acp-dev/ae-models.sh
PROXY=http://10.119.17.252:17922
export WB_PROXY=$PROXY  # eval_lanes.sh's W&B forward (its default 17920 admits another session's node only)
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
  local MINE="env PYTHONPATH=shared/code-v4/src:shared/code-v4/audio-data-contract-src $ENVS/oal-train-vllm/bin/python -m open_audio_llm.scripts.grpo_mining --data_config configs/train-data-grpo-hard.yaml --workers 6 --concurrency 48 --samples 16"
  local k src=() pids=() ports=() mpids=() names=() n x rc=0
  for k in $(cat $C/sources-ts.txt); do src+=(--source "$k"); done
  for n in 0 1 2 3 4 5 6; do ports+=($((19410 + n)))
    $ENVS/oal-train/bin/open-audio-llm serve --config $C/serve-tsasr-$n.yaml > $O/serve-tsasr-$n.log 2>&1 & pids+=($!); done
  for n in 0 1 2 3 4 5 6; do
    until curl -s --noproxy '*' -m 5 http://127.0.0.1:${ports[$n]}/v1/models >/dev/null 2>&1; do
      kill -0 ${pids[$n]} 2>/dev/null || { echo "tsasr $n exited $(now)"; kill -TERM "${pids[@]}" 2>/dev/null; return 1; }
      sleep 15
    done
  done; echo "serve-tsasr servers ready $(now)"
  for n in 0 1 2 3 4 5 6; do L=$O/ts-lane-$n; mkdir -p $L; url=http://127.0.0.1:${ports[$n]}
    $MINE "${src[@]}" --candidates 150 --epoch $((70 + n)) --server $url --model mine-1117-tsasr \
      --ts_server $url --ts_model mine-1117-tsasr --output $L/kept.jsonl > $L/run.log 2>&1 & mpids+=($!); names+=(ts-$n)
  done
  for n in "${!mpids[@]}"; do wait ${mpids[$n]}; x=$?; echo "${names[$n]} exit $x $(now)"; [ $x -ne 0 ] && rc=$x; done
  kill -TERM "${pids[@]}" 2>/dev/null; wait "${pids[@]}" 2>/dev/null; sleep 30
  [ $rc -eq 0 ] || return $rc
  env PYTHONPATH=shared/code-v4/src $ENVS/oal-train/bin/python scripts/grpo_hard_plan.py $O \
    AmphionASR-ts-overlap-grpo-mining-1117-n16-ts-hard-spread0.3 tasks/12-grpo-transcribe/mining/ts-lane-* $O/ts-lane-* \
    > $O/plans.log 2>&1; rc=$?
  echo "plans exit $rc $(now)"; head -1 $O/plans.log; return $rc
}

train() {
  local p
  for p in hotword nonspeech miss ts-absent ts-present; do [ -s tasks/12-grpo-transcribe/mining/plan-$p.jsonl ] || { echo "plan $p empty"; return 2; }; done
  [ -s $O/plan-ts-hard.jsonl ] || { echo "plan ts-hard empty"; return 2; }
  wc -l $O/plan-ts-hard.jsonl; wait_gpus
  local before=$(ls $T/attempts 2>/dev/null | sort | tail -1)
  cd /workspace/workspace/mingdong/repo/oal-clean-teacher
  env -u PYTHONPATH /workspace/workspace/mingdong/repo/open-audio-llm/third_party/amphionkeys/load.sh sensecore -- \
    env PYTHONPATH=runs/ts-overlap-20261119/shared/code-v4/src $ENVS/oal-train/bin/python -m open_audio_llm.cli train \
    --config runs/ts-overlap-20261119/configs/train-grpo-hard.yaml > $E/$T/train.cli.log 2>&1 &
  local tp=$!; cd $E
  local a=; until [ -n "$a" ]; do kill -0 $tp 2>/dev/null || break; a=$(ls $T/attempts 2>/dev/null | sort | awk -v b="$before" '$1>b' | head -1); sleep 10; done
  [ -n "$a" ] && { echo "attempt $a"; bash scripts/grpo_collect_checkpoints.sh $a 13-grpo-hard hard-$a > $T/attempts/$a/collect.log 2>&1 & }
  wait $tp; local rc=$?; echo "training exit $rc $(now)"; wait
  echo "$a" > $T/last-attempt; return $rc
}

evals() {
  local a=$(cat $T/last-attempt) AVG=/workspace/workspace/mingdong/scratch/ts033/avg_ckpt.py PY=$ENVS/oal-train/bin/python
  local c=() s f A I N=ts-overlap-grpo-hard-$a
  if A=$(bash $M fetch $N-avg3 2>/dev/null); then A=$(echo "$A" | tail -1); else
    for s in 200 300 400; do f=$(bash $M fetch $N-ckpt$s 2>/dev/null) || { echo "fetch ckpt$s failed"; return 4; }
      c+=("$(echo "$f" | tail -1)"); done
    local W=$(bash $M stage); $PY $AVG $W "${c[@]}" | tail -1; rm -f $W/.complete
    A=$(bash $M put $W --name $N-avg3 | tail -1)
  fi; echo "avg3 $A $(now)"
  if I=$(bash $M fetch $N-interp050 2>/dev/null); then I=$(echo "$I" | tail -1); else
    local W2=$(bash $M stage); $PY $AVG $W2 $START $A | tail -1; rm -f $W2/.complete
    I=$(bash $M put $W2 --name $N-interp050 | tail -1)
  fi; echo "interp050 $I $(now)"
  [ -d $A/mtp ] || { echo "avg3 has no mtp/"; return 4; }  # interp050 has none: 11.17's cache entry carries no mtp/
  local rc=0 x pids=() gb=0 pb=19900
  for x in avg3 interp050; do
    bash scripts/eval_lanes.sh $N-$x hard-$x $gb $pb \
      "eval-sumfull-es116-core/v12 eval-sumfull-es116-core/tsen" \
      "eval-v18avg3-ckpt400-asrfull/asrfull eval-sumfull-es116-core/custts" > $T/eval-$x.log 2>&1 & pids+=($!)
    gb=$((gb + 4)); pb=$((pb + 100)); sleep 60
  done
  for x in 0 1; do wait ${pids[$x]} || rc=1; done
  echo "evals done rc=$rc $(now)"; return $rc
}

STAGES=("$@"); [ ${#STAGES[@]} -gt 0 ] || STAGES=(mining train evals)
for stage in "${STAGES[@]}"; do
  echo "$stage $(now)"
  $stage || { x=$?; echo "$stage failed $x $(now)"; exit $x; }
done
echo "done $(now)"
