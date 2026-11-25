#!/usr/bin/env bash
# 不含合成数据的差值合并 (README) on one ACP node, eval only. Model (runs/zh-ctx-grpo-20261120/scripts/merge_increment.py,
# built on the dev machine under /workspace/data/ae-runs/ctx-grpo-k-20261122-pairmerge/):
#   pm-real      11.17 + (G - control)    1117-plus-g-minus-control
# 1. GPUs 0-3: pm-real on the 27 delivery suites (runs/zh-ctx-grpo-20261120 scripts/acp-delivery-one.sh);
# 2. GPUs 4-7 meanwhile: context-correction v2 think off (configs/eval-<name>, make_eval_configs.py): pm-real (4/5) and
#    G (g1106, 6/7) at once, then the control (control1106, 4/5). 11.17's run is the one of acp-datasplit.sh.
# usage: acp-pmreal.sh
set -u; umask 002
E=/workspace/workspace/mingdong/repo/oal-ctx-grpo/runs/zh-ctx-grpo-k-20261122; cd $E
G=/workspace/workspace/mingdong/repo/oal-ctx-grpo/runs/zh-ctx-grpo-20261120
M=/workspace/data/ae-runs/ctx-grpo-k-20261122-pairmerge
AE=/workspace/workspace/mingdong/.local-tools/amphion-eval-0.14.0-eedcf15/bin/ae
PROXY=http://10.119.17.252:17920
now() { date -u +%FT%TZ; }
d=$M/1117-plus-g-minus-control
[ -f "$d/model.safetensors" ] && [ -f "$d/mtp/model.safetensors" ] || { echo "missing model or mtp/: $d"; exit 3; }
for n in pm-real g1106 control1106; do [ -f configs/eval-$n/serve-asr.yaml ] || { echo "missing configs/eval-$n"; exit 3; }; done
for i in $(seq 60); do  # the forward only lets registered node IPs through; every eval logs to W&B
  code=$(curl -s -m 10 -o /dev/null -w '%{http_code}' -x $PROXY https://api.wandb.ai/healthz)
  [ "$code" = 200 ] && break
  echo "W&B through $PROXY: $code, waiting for the node IP to be let through $(now)"; sleep 30
done
[ "$code" = 200 ] || exit 2
bash $G/scripts/acp-delivery-one.sh pm-real $d 0 1 18940 2 3 18942 & a=$!
sleep 60
ctx() { bash scripts/run_eval.sh configs/eval-$1 $PROXY $AE context-correction-v2; r=$?; echo "ctx v2 $1 exit $r $(now)"; return $r; }
(ctx pm-real; r1=$?; ctx control1106; r2=$?; [ $r1 = 0 ] && [ $r2 = 0 ]) & b=$!
sleep 30  # stagger the vLLM starts
ctx g1106 & c=$!
wait $b; rb=$?; wait $c; rc=$?; wait $a; ra=$?
echo "exit delivery-pm-real=$ra ctx-v2-pm-real+control=$rb ctx-v2-g1106=$rc $(now)"
[ $ra = 0 ] && [ $rb = 0 ] && [ $rc = 0 ]
