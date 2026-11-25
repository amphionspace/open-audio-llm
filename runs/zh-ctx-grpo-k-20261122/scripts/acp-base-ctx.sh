#!/usr/bin/env bash
# 续训本身的纠偏收益 (README) on the pm-real node's free GPUs 4-7 (acp-pmreal.sh uses 0-3 by then), eval only:
# context-correction v2 think off of 10.28 (base1028, GPUs 4/5) and 11.06 (base1106, 6/7) at once. Configs from
# make_eval_configs.py; models are the symlink dirs /workspace/data/ae-runs/ctx-grpo-k-20261122-pairmerge/base-{1028,1106}.
# usage: acp-base-ctx.sh
set -u; umask 002
E=/workspace/workspace/mingdong/repo/oal-ctx-grpo/runs/zh-ctx-grpo-k-20261122; cd $E
AE=/workspace/workspace/mingdong/.local-tools/amphion-eval-0.14.0-eedcf15/bin/ae
PROXY=http://10.119.17.252:17920
now() { date -u +%FT%TZ; }
for n in base1028 base1106; do [ -f configs/eval-$n/serve-asr.yaml ] || { echo "missing configs/eval-$n"; exit 3; }; done
ctx() { bash scripts/run_eval.sh configs/eval-$1 $PROXY $AE context-correction-v2; r=$?; echo "ctx v2 $1 exit $r $(now)"; return $r; }
ctx base1028 & a=$!
sleep 30  # stagger the vLLM starts
ctx base1106 & b=$!
wait $a; ra=$?; wait $b; rb=$?
echo "exit ctx-v2-base1028=$ra ctx-v2-base1106=$rb $(now)"
[ $ra = 0 ] && [ $rb = 0 ]
