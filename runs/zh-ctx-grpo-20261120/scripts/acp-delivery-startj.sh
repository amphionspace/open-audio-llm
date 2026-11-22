#!/usr/bin/env bash
# J (GRPO start) on the 27 delivery suites (README "交付评估"), to tell which differences of avg3 from 11.17 J already
# has and which GRPO brought. scene-asr runs first: avg3 drops content on its 30 s+ recordings. Pairs a (GPUs 0,1) and b
# (2,3) from make_delivery_configs.py startj.
set -u; umask 002
E=/workspace/workspace/mingdong/repo/oal-ctx-grpo/runs/zh-ctx-grpo-20261120; cd $E
AE=/workspace/workspace/mingdong/.local-tools/amphion-eval-0.14.0-eedcf15/bin/ae
PROXY=http://10.119.17.252:17920
now() { date -u +%FT%TZ; }
for i in $(seq 60); do
  code=$(curl -s -m 10 -o /dev/null -w '%{http_code}' -x $PROXY https://api.wandb.ai/healthz)
  [ "$code" = 200 ] && break
  echo "W&B through $PROXY: $code, waiting for the node IP to be let through $(now)"; sleep 30
done
[ "$code" = 200 ] || exit 2
suites() { ls configs/eval-$1/ | sed -n 's/^after-training-\(delivery-.*\)\.yaml$/\1/p' | grep -v '^delivery-scene-asr$' | tr '\n' ' '; }
a="delivery-scene-asr $(suites delivery-startj-a)"; b=$(suites delivery-startj-b)
echo "a: $a $(now)"; bash scripts/run_eval.sh configs/eval-delivery-startj-a $PROXY $AE $a & pa=$!
sleep 60
echo "b: $b $(now)"; bash scripts/run_eval.sh configs/eval-delivery-startj-b $PROXY $AE $b & pb=$!
wait $pa; ra=$?; echo "a exit $ra $(now)"; wait $pb; rb=$?; echo "b exit $rb $(now)"
[ $ra = 0 ] && [ $rb = 0 ]
