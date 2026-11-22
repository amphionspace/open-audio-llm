#!/usr/bin/env bash
# 拆分诊断 (README) on one ACP node, eval only. Models (scripts/swap_component.py, built on the dev machine):
#   kgrpo-llm    avg3's LLM + 11.17's audio tower   ctx eval v2 think off/on (GPUs 0/1), then the 27 delivery suites
#                                                   (GPUs 0,1 / 2,3)
#   kgrpo-audio  11.17's LLM + avg3's audio tower   the 27 delivery suites (GPUs 4,5 / 6,7), from the start
# Delivery runs: runs/zh-ctx-grpo-20261120 scripts/acp-delivery-one.sh, model names kgrpollm / kgrpoaudio.
# usage: acp-swap.sh
set -u; umask 002
E=/workspace/workspace/mingdong/repo/oal-ctx-grpo/runs/zh-ctx-grpo-k-20261122; cd $E
G=/workspace/workspace/mingdong/repo/oal-ctx-grpo/runs/zh-ctx-grpo-20261120
O=/workspace/data/ae-runs/ctx-grpo-k-20261122-swap
PY=/workspace/workspace/mingdong/miniconda3/envs/oal-train/bin/python
AE=/workspace/workspace/mingdong/.local-tools/amphion-eval-0.14.0-eedcf15/bin/ae
PROXY=http://10.119.17.252:17920
now() { date -u +%FT%TZ; }
for i in $(seq 60); do  # the forward only lets registered node IPs through; every eval logs to W&B
  code=$(curl -s -m 10 -o /dev/null -w '%{http_code}' -x $PROXY https://api.wandb.ai/healthz)
  [ "$code" = 200 ] && break
  echo "W&B through $PROXY: $code, waiting for the node IP to be let through $(now)"; sleep 30
done
[ "$code" = 200 ] || exit 2
$PY scripts/make_eval_configs.py kgrpo-llm $O/kgrpo-llm 0 1 18966 || exit 6
bash scripts/run_eval.sh configs/eval-kgrpo-llm $PROXY $AE context-correction-v2 context-correction-v2-think & c=$!
sleep 60  # stagger the vLLM starts
bash $G/scripts/acp-delivery-one.sh kgrpoaudio $O/kgrpo-audio 4 5 18944 6 7 18946 & u=$!
wait $c; rc=$?; echo "ctx eval kgrpo-llm exit $rc $(now)"
bash $G/scripts/acp-delivery-one.sh kgrpollm $O/kgrpo-llm 0 1 18940 2 3 18942 & l=$!
wait $l; rl=$?; wait $u; ru=$?
echo "exit ctx=$rc delivery-kgrpollm=$rl delivery-kgrpoaudio=$ru $(now)"
[ "$rc" = 0 ] && [ "$rl" = 0 ] && [ "$ru" = 0 ]
