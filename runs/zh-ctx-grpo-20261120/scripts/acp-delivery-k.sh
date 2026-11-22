#!/usr/bin/env bash
# K (the context-correction SFT of question 9, runs/zh-ctx-clue-target-20261117, trained from 11.17) on the 27
# delivery suites (README "交付评估"), run by that experiment's node script once its checkpoint-3000 is fetched.
# scene-asr runs first (the long-dialogue drops K is to fix). Pairs a (GPUs 2,3) and b (4,5), beside the ctx eval v2
# on GPUs 0/1.
# usage: acp-delivery-k.sh CHECKPOINT
set -u; umask 002
E=/workspace/workspace/mingdong/repo/oal-ctx-grpo/runs/zh-ctx-grpo-20261120; cd $E
PY=/workspace/workspace/mingdong/miniconda3/envs/oal-train/bin/python
AE=/workspace/workspace/mingdong/.local-tools/amphion-eval-0.14.0-eedcf15/bin/ae
PROXY=http://10.119.17.252:17920
now() { date -u +%FT%TZ; }
$PY scripts/make_delivery_configs.py k "$1" 2 3 18930 4 5 18932 || exit 3
suites() { ls configs/eval-$1/ | sed -n 's/^after-training-\(delivery-.*\)\.yaml$/\1/p' | grep -v '^delivery-scene-asr$' | tr '\n' ' '; }
a="delivery-scene-asr $(suites delivery-k-a)"; b=$(suites delivery-k-b)
echo "a: $a $(now)"; bash scripts/run_eval.sh configs/eval-delivery-k-a $PROXY $AE $a & pa=$!
sleep 60
echo "b: $b $(now)"; bash scripts/run_eval.sh configs/eval-delivery-k-b $PROXY $AE $b & pb=$!
wait $pa; ra=$?; echo "a exit $ra $(now)"; wait $pb; rb=$?; echo "b exit $rb $(now)"
[ $ra = 0 ] && [ $rb = 0 ]
