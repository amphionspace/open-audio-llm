#!/usr/bin/env bash
# 差值合并诊断 (README) on one ACP node, eval only. Models (built on the dev machine before submitting):
#   pairmerge  11.17 + (J - control), runs/zh-ctx-grpo-20261120/scripts/merge_increment.py   GPUs 0,1 / 2,3
#   control    the control arm of runs/zh-spelling-think-20261113 (SFT from 11.06, no synthetic
#              context-correction data), so J's delivery run (startj, same code-v3 and client) has a same-start pair
#                                                                                           GPUs 4,5 / 6,7
# Delivery runs: runs/zh-ctx-grpo-20261120 scripts/acp-delivery-one.sh, model names pairmerge / control1106.
# usage: acp-pairmerge.sh PAIRMERGE_DIR CONTROL_DIR
set -u; umask 002
G=/workspace/workspace/mingdong/repo/oal-ctx-grpo/runs/zh-ctx-grpo-20261120
PROXY=http://10.119.17.252:17920
now() { date -u +%FT%TZ; }
[ -f "$1/model.safetensors" ] && [ -f "$2/model.safetensors" ] || { echo "missing model: $1 $2"; exit 3; }
for i in $(seq 60); do  # the forward only lets registered node IPs through; every eval logs to W&B
  code=$(curl -s -m 10 -o /dev/null -w '%{http_code}' -x $PROXY https://api.wandb.ai/healthz)
  [ "$code" = 200 ] && break
  echo "W&B through $PROXY: $code, waiting for the node IP to be let through $(now)"; sleep 30
done
[ "$code" = 200 ] || exit 2
bash $G/scripts/acp-delivery-one.sh pairmerge "$1" 0 1 18940 2 3 18942 & m=$!
sleep 60  # stagger the vLLM starts
bash $G/scripts/acp-delivery-one.sh control1106 "$2" 4 5 18944 6 7 18946 & c=$!
wait $m; rm=$?; wait $c; rc=$?
echo "exit delivery-pairmerge=$rm delivery-control1106=$rc $(now)"
[ "$rm" = 0 ] && [ "$rc" = 0 ]
