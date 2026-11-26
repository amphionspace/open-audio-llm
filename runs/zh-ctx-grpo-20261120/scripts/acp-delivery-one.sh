#!/usr/bin/env bash
# One model on the 27 delivery suites (README "交付评估"), two server pairs at the same time; outputs go to
# /workspace/data/ae-runs/ctx-grpo-20261120-delivery-<NAME>, compared afterwards with compare_delivery.py r1117 NAME.
# Same steps as acp-delivery-k.sh (scene-asr first), with the name, GPUs and ports as arguments; first used for K's GRPO
# round (runs/zh-ctx-grpo-k-20261122).
# usage: acp-delivery-one.sh NAME CHECKPOINT A_ASR_GPU A_TS_GPU A_PORT B_ASR_GPU B_TS_GPU B_PORT
set -u; umask 002
E=/workspace/workspace/mingdong/repo/oal-ctx-grpo/runs/zh-ctx-grpo-20261120; cd $E
PY=/workspace/workspace/mingdong/miniconda3/envs/oal-train/bin/python
AE=/workspace/workspace/mingdong/.local-tools/amphion-eval-0.14.0-eedcf15/bin/ae
PROXY=http://10.119.18.89:17920  # dev container IP since 2026-11-26 (was 10.119.17.252)
now() { date -u +%FT%TZ; }
N=$1
$PY scripts/make_delivery_configs.py "$N" "$2" "$3" "$4" "$5" "$6" "$7" "$8" || exit 3
suites() { ls configs/eval-$1/ | sed -n 's/^after-training-\(delivery-.*\)\.yaml$/\1/p' | grep -v '^delivery-scene-asr$' | tr '\n' ' '; }
a="delivery-scene-asr $(suites delivery-$N-a)"; b=$(suites delivery-$N-b)
echo "a: $a $(now)"; bash scripts/run_eval.sh configs/eval-delivery-$N-a $PROXY $AE $a & pa=$!
sleep 60
echo "b: $b $(now)"; bash scripts/run_eval.sh configs/eval-delivery-$N-b $PROXY $AE $b & pb=$!
wait $pa; ra=$?; echo "a exit $ra $(now)"; wait $pb; rb=$?; echo "b exit $rb $(now)"
[ $ra = 0 ] && [ $rb = 0 ]
