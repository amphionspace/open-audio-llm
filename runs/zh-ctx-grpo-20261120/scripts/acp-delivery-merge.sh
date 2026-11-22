#!/usr/bin/env bash
# 11.17 + (avg3 - 11.06) on the 27 delivery suites (README "合并诊断"), pairs a (GPUs 4,5) and b (6,7) from
# make_delivery_configs.py merge, beside J's delivery run on GPUs 0-3.
set -u; umask 002
E=/workspace/workspace/mingdong/repo/oal-ctx-grpo/runs/zh-ctx-grpo-20261120; cd $E
AE=/workspace/workspace/mingdong/.local-tools/amphion-eval-0.14.0-eedcf15/bin/ae
PROXY=http://10.119.17.252:17920
now() { date -u +%FT%TZ; }
suites() { ls configs/eval-$1/ | sed -n 's/^after-training-\(delivery-.*\)\.yaml$/\1/p' | tr '\n' ' '; }
a=$(suites delivery-merge-a); b=$(suites delivery-merge-b)
echo "a: $a $(now)"; bash scripts/run_eval.sh configs/eval-delivery-merge-a $PROXY $AE $a & pa=$!
sleep 60
echo "b: $b $(now)"; bash scripts/run_eval.sh configs/eval-delivery-merge-b $PROXY $AE $b & pb=$!
wait $pa; ra=$?; echo "a exit $ra $(now)"; wait $pb; rb=$?; echo "b exit $rb $(now)"
[ $ra = 0 ] && [ $rb = 0 ]
