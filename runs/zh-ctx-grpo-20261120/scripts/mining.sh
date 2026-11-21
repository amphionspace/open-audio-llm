#!/usr/bin/env bash
# 01-mining on one ACP node: 08's mining (the release round's lanes, candidates and epochs; think off, 16 samples per
# prompt) on the start arm's checkpoint-3000, plus two context-correction lanes per phase on servers 0 and 1, every
# candidate of the ctx sources once per lane. Source keys come from configs/mining/sources-*.txt (make_configs.py,
# final versions). scripts/mining_plans.py then splits the kept prompts into six plans -> tasks/01-mining/plan-*.jsonl.
# usage: mining.sh ARM TARGETS   (ARM: the SFT arm of runs/zh-ctx-clue-target-20261117; TARGETS: finalized targets)
set -u
umask 002
ARM=$1; TARGETS=$2
E=/workspace/workspace/mingdong/repo/oal-ctx-grpo/runs/zh-ctx-grpo-20261120; cd $E
SFT=/workspace/workspace/mingdong/repo/oal-clue-target/runs/zh-ctx-clue-target-20261117
ENVS=/workspace/workspace/mingdong/miniconda3/envs; C=configs/mining; O=tasks/01-mining; mkdir -p $O
a=; for d in $SFT/tasks/01-train-$ARM/attempts/*/; do grep -q '"execution": "completed"' $d/status.json 2>/dev/null && a=$(basename $d); done
[ -n "$a" ] || { echo "no completed training attempt for $ARM"; exit 5; }
START=$(bash /workspace/workspace/mingdong/scratch/acp-dev/ae-models.sh fetch \
  whai:open-audio-llm/runs/zh-ctx-clue-target-20261117/tasks/01-train-$ARM/attempts/$a/artifacts/training/checkpoint-3000/) || exit 5
grep -q "path: $START\$" $C/serve-asr-0.yaml || { echo "start checkpoint is $START; rerun scripts/make_configs.py"; exit 5; }
grep -q "$TARGETS" configs/train-data-grpo.yaml || { echo "data config does not name $TARGETS; rerun scripts/make_configs.py"; exit 5; }
echo "start $START $(date -u +%FT%TZ)"
for i in $(seq 360); do [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>2000' | wc -l) -eq 0 ] && break; sleep 10; done
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
source /workspace/workspace/mingdong/.config/open-audio-llm/wandb-cluster.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 NUMPY_MADVISE_HUGEPAGE=0 \
  WANDB_ENTITY=1016097967-amphion WANDB_PROJECT=open-audio-llm WANDB_DIR=$E/$O \
  WANDB_HTTP_PROXY=http://10.119.17.252:17920 WANDB_HTTPS_PROXY=http://10.119.17.252:17920
MINE="env PYTHONPATH=shared/code-v1/src:shared/code-v1/audio-data-contract-src $ENVS/oal-train-vllm/bin/python -m open_audio_llm.scripts.grpo_mining --data_config configs/train-data-grpo.yaml --workers 6 --concurrency 48 --samples 16"
sources() { local k; for k in $(cat $C/sources-$1.txt); do printf -- '--source %s ' "$k"; done; }
start_servers() {  # $1 = serve-asr | serve-tsasr, $2 = first port
  local kind=$1; pids=(); ports=()
  for n in 0 1 2 3 4 5 6; do ports+=($(( $2 + n )))
    $ENVS/oal-train/bin/open-audio-llm serve --config $C/$kind-$n.yaml > $O/$kind-$n.log 2>&1 & pids+=($!); done
  for n in 0 1 2 3 4 5 6; do
    until curl -s --noproxy '*' -m 5 http://127.0.0.1:${ports[$n]}/v1/models >/dev/null 2>&1; do
      kill -0 ${pids[$n]} 2>/dev/null || { echo "$kind $n exited $(date -u +%FT%TZ)"; kill -TERM "${pids[@]}" 2>/dev/null; exit 1; }
      sleep 15
    done
  done; echo "$kind servers ready $(date -u +%FT%TZ)"
}
stop_servers() { kill -TERM "${pids[@]}" 2>/dev/null; wait "${pids[@]}" 2>/dev/null; sleep 30; }
wait_all() { local i x rc=0; for i in "${!mpids[@]}"; do wait ${mpids[$i]}; x=$?; echo "${names[$i]} exit $x $(date -u +%FT%TZ)"; [ $x -ne 0 ] && rc=$x; done; return $rc; }
lanes=()
start_servers serve-asr 19400
mpids=(); names=()
for n in 0 1 2 3 4 5 6; do url=http://127.0.0.1:${ports[$n]}
  L=$O/plain-lane-$n; mkdir -p $L; lanes+=($L)
  $MINE $(sources plain) --candidates 900 --epoch $((20 + n)) --server $url --model mine-ctx-asr --output $L/kept.jsonl > $L/run.log 2>&1 &
  mpids+=($!); names+=(plain-$n)
  L=$O/miss-lane-$n; mkdir -p $L; lanes+=($L)
  $MINE $(sources miss) --candidates 400 --epoch $((40 + n)) --server $url --model mine-ctx-asr --output $L/kept.jsonl > $L/run.log 2>&1 &
  mpids+=($!); names+=(miss-$n)
done
for n in 0 1; do L=$O/ctx-asr-lane-$n; mkdir -p $L; lanes+=($L)
  $MINE $(sources ctx-asr) --candidates 100000 --epoch $((50 + n)) --server http://127.0.0.1:${ports[$n]} --model mine-ctx-asr \
    --output $L/kept.jsonl > $L/run.log 2>&1 &
  mpids+=($!); names+=(ctx-asr-$n)
done
wait_all; rc=$?; stop_servers; [ $rc -eq 0 ] || exit $rc
start_servers serve-tsasr 19410
mpids=(); names=()
for n in 0 1 2 3 4 5 6; do L=$O/ts-lane-$n; mkdir -p $L; lanes+=($L); url=http://127.0.0.1:${ports[$n]}
  $MINE $(sources ts) --candidates 120 --epoch $((30 + n)) --server $url --model mine-ctx-tsasr \
    --ts_server $url --ts_model mine-ctx-tsasr --output $L/kept.jsonl > $L/run.log 2>&1 &
  mpids+=($!); names+=(ts-$n)
done
for n in 0 1; do L=$O/ctx-ts-lane-$n; mkdir -p $L; lanes+=($L); url=http://127.0.0.1:${ports[$n]}
  $MINE $(sources ctx-ts) --candidates 100000 --epoch $((60 + n)) --server $url --model mine-ctx-tsasr \
    --ts_server $url --ts_model mine-ctx-tsasr --output $L/kept.jsonl > $L/run.log 2>&1 &
  mpids+=($!); names+=(ctx-ts-$n)
done
wait_all; rc=$?; stop_servers; [ $rc -eq 0 ] || exit $rc
env PYTHONPATH=shared/code-v1/src:shared/code-v1/audio-data-contract-src $ENVS/oal-train/bin/python scripts/mining_plans.py \
  $O AmphionASR-ctx-grpo-mining-$ARM-3000-n16-ctxplan "$TARGETS" "${lanes[@]}" > $O/plans.log 2>&1; x=$?
echo "plans exit $x $(date -u +%FT%TZ)"; head -1 $O/plans.log; exit $x
