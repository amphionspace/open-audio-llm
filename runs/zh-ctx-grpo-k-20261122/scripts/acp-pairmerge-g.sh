#!/usr/bin/env bash
# 数据增量诊断 (README) on one ACP node, eval only; same steps as acp-pairmerge.sh. Models:
#   pairmerge-g  11.17 + (J - G), runs/zh-ctx-grpo-20261120/scripts/merge_increment.py, built on the dev machine   GPUs 0,1 / 2,3
#   g1106        G = template-no-synth of runs/zh-ctx-clue-target-20261117 (J's round-2 recipe, MTP and code without the
#                synthetic spelling data and twins; ae cache checkpoint-3000-90c9d6f4fa35), for G - control / J - G   GPUs 4,5 / 6,7
# Delivery runs: runs/zh-ctx-grpo-20261120 scripts/acp-delivery-one.sh.
# usage: acp-pairmerge-g.sh PAIRMERGE_G_DIR G_DIR
set -u; umask 002
G=/workspace/workspace/mingdong/repo/oal-ctx-grpo/runs/zh-ctx-grpo-20261120
PROXY=http://10.119.17.252:17920
now() { date -u +%FT%TZ; }
for d in "$1" "$2"; do [ -f "$d/model.safetensors" ] && [ -f "$d/mtp/model.safetensors" ] || { echo "missing model or mtp/: $d"; exit 3; }; done
for i in $(seq 60); do  # the forward only lets registered node IPs through; every eval logs to W&B
  code=$(curl -s -m 10 -o /dev/null -w '%{http_code}' -x $PROXY https://api.wandb.ai/healthz)
  [ "$code" = 200 ] && break
  echo "W&B through $PROXY: $code, waiting for the node IP to be let through $(now)"; sleep 30
done
[ "$code" = 200 ] || exit 2
bash $G/scripts/acp-delivery-one.sh pairmerge-g "$1" 0 1 18940 2 3 18942 & m=$!
sleep 60  # stagger the vLLM starts
bash $G/scripts/acp-delivery-one.sh g1106 "$2" 4 5 18944 6 7 18946 & c=$!
wait $m; rm=$?; wait $c; rc=$?
echo "exit delivery-pairmerge-g=$rm delivery-g1106=$rc $(now)"
[ "$rm" = 0 ] && [ "$rc" = 0 ]
