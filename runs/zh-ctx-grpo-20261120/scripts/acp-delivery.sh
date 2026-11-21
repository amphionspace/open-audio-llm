#!/usr/bin/env bash
# Delivery evaluation (README "交付评估") on one ACP node: avg3 (GPUs 0-3) and 11.17 (GPUs 4-7), each on two server
# pairs (configs/eval-delivery-<name>-{a,b}/, scripts/make_delivery_configs.py), all four at once, vLLM starts staggered.
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
pids=(); names=()
for d in delivery-avg3-a delivery-r1117-a delivery-avg3-b delivery-r1117-b; do
  suites=$(ls configs/eval-$d/ | sed -n 's/^after-training-\(delivery-.*\)\.yaml$/\1/p' | tr '\n' ' ')
  echo "$d: $suites $(now)"
  bash scripts/run_eval.sh configs/eval-$d $PROXY $AE $suites & pids+=($!); names+=($d)
  sleep 60
done
rc=0; for i in "${!pids[@]}"; do wait ${pids[$i]}; x=$?; echo "${names[$i]} exit $x $(now)"; [ $x -ne 0 ] && rc=$x; done
echo "done $(now)"; exit $rc
