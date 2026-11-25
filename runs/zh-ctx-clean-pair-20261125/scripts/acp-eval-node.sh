#!/usr/bin/env bash
# README "后续：说字清洗在 11.06 上的 TS 代价": the 27 delivery suites of noclean (GPUs 0-3) and of 11.06 (GPUs 4-7) at the
# same time on one ACP node, with runs/zh-ctx-grpo-20261120 acp-delivery-one.sh (the control's delivery runs,
# ctx-grpo-20261120-delivery-control1106, use the same configs). Compared afterwards with compare_delivery.py
# control1106 noclean1106 and base1106 noclean1106. Both model dirs carry 11.17's mtp/ (greedy verification).
# usage: acp-eval-node.sh
set -u; umask 002
G=/workspace/workspace/mingdong/repo/oal-ctx-grpo/runs/zh-ctx-grpo-20261120
M=/workspace/data/ae-runs/ctx-grpo-k-20261122-pairmerge
MODELS=/workspace/workspace/mingdong/scratch/acp-dev/ae-models.sh
REM=whai:open-audio-llm/runs/zh-ctx-clean-pair-20261125/tasks/01-train-noclean/attempts/001/artifacts/training/checkpoint-3000/
PROXY=http://10.119.17.252:17920
now() { date -u +%FT%TZ; }
for i in $(seq 60); do  # the forward only lets registered node IPs through; every eval logs to W&B
  code=$(curl -s -m 10 -o /dev/null -w '%{http_code}' -x $PROXY https://api.wandb.ai/healthz)
  [ "$code" = 200 ] && break
  echo "W&B through $PROXY: $code, waiting for the node IP to be let through $(now)"; sleep 30
done
[ "$code" = 200 ] || exit 2
# noclean-3000 links into the ae model cache; fetch again in case the entry was evicted (a no-op when present)
C=$(bash $MODELS fetch $REM) || exit 5
[ "$(readlink -f $M/noclean-3000/model.safetensors)" = "$(readlink -f $C/model.safetensors)" ] || { echo "noclean link != $C"; exit 5; }
[ -f $M/base-1106/model.safetensors ] && [ -d $M/base-1106/mtp ] || { echo "missing $M/base-1106"; exit 3; }
bash $G/scripts/acp-delivery-one.sh noclean1106 $M/noclean-3000 0 1 18940 2 3 18942 & a=$!
sleep 90  # stagger the vLLM starts
bash $G/scripts/acp-delivery-one.sh base1106 $M/base-1106 4 5 18950 6 7 18952 & b=$!
wait $a; ra=$?; wait $b; rb=$?
echo "exit delivery-noclean1106=$ra delivery-base1106=$rb $(now)"
[ $ra = 0 ] && [ $rb = 0 ]
