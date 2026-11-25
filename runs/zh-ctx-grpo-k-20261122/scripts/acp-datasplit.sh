#!/usr/bin/env bash
# 合成数据拆分诊断 (README) on one ACP node, eval only. Models (runs/zh-ctx-grpo-20261120/scripts/merge_increment.py,
# built on the dev machine under /workspace/data/ae-runs/ctx-grpo-k-20261122-pairmerge/):
#   pm-train     11.17 + (H - G)          1117-plus-h-minus-g
#   pm-train-ts  11.17 + (template - G)   1117-plus-template-minus-g
#   pairmerge-g  11.17 + (J - G)          1117-plus-j-minus-g   (delivery runs already done)
#   r1117        the 11.17 package        (delivery runs already done)
# 1. context-correction v2 think off, all four at once (configs/eval-<name>, make_eval_configs.py, GPUs 0/1 2/3 4/5 6/7);
# 2. the 27 delivery suites (runs/zh-ctx-grpo-20261120 scripts/acp-delivery-one.sh): pm-train GPUs 0-3, pm-train-ts 4-7.
# usage: acp-datasplit.sh
set -u; umask 002
E=/workspace/workspace/mingdong/repo/oal-ctx-grpo/runs/zh-ctx-grpo-k-20261122; cd $E
G=/workspace/workspace/mingdong/repo/oal-ctx-grpo/runs/zh-ctx-grpo-20261120
M=/workspace/data/ae-runs/ctx-grpo-k-20261122-pairmerge
AE=/workspace/workspace/mingdong/.local-tools/amphion-eval-0.14.0-eedcf15/bin/ae
PROXY=http://10.119.17.252:17920
now() { date -u +%FT%TZ; }
for d in $M/1117-plus-h-minus-g $M/1117-plus-template-minus-g $M/1117-plus-j-minus-g; do
  [ -f "$d/model.safetensors" ] && [ -f "$d/mtp/model.safetensors" ] || { echo "missing model or mtp/: $d"; exit 3; }
done
for n in pm-train pm-train-ts pairmerge-g r1117; do [ -f configs/eval-$n/serve-asr.yaml ] || { echo "missing configs/eval-$n"; exit 3; }; done
for i in $(seq 60); do  # the forward only lets registered node IPs through; every eval logs to W&B
  code=$(curl -s -m 10 -o /dev/null -w '%{http_code}' -x $PROXY https://api.wandb.ai/healthz)
  [ "$code" = 200 ] && break
  echo "W&B through $PROXY: $code, waiting for the node IP to be let through $(now)"; sleep 30
done
[ "$code" = 200 ] || exit 2
declare -A P
for n in pm-train pm-train-ts pairmerge-g r1117; do
  bash scripts/run_eval.sh configs/eval-$n $PROXY $AE context-correction-v2 & P[$n]=$!
  sleep 30  # stagger the vLLM starts
done
rc=0
for n in "${!P[@]}"; do wait ${P[$n]}; r=$?; echo "ctx v2 $n exit $r $(now)"; [ $r = 0 ] || rc=1; done
bash $G/scripts/acp-delivery-one.sh pm-train $M/1117-plus-h-minus-g 0 1 18940 2 3 18942 & a=$!
sleep 60
bash $G/scripts/acp-delivery-one.sh pm-train-ts $M/1117-plus-template-minus-g 4 5 18944 6 7 18946 & b=$!
wait $a; ra=$?; wait $b; rb=$?
echo "exit ctx-v2=$rc delivery-pm-train=$ra delivery-pm-train-ts=$rb $(now)"
[ $rc = 0 ] && [ $ra = 0 ] && [ $rb = 0 ]
