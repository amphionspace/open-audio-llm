#!/usr/bin/env bash
# Once grpo (attempt 001) completes: avg3 of checkpoints 200/300/400 and its 50% interpolation with the start (round-2 think
# checkpoint-3000), the release round's way of building a candidate; both go into the ae model cache. Then both are
# evaluated at the same time on this node with round 2's suites (context correction think off/on, 8 gates think off,
# 3 gates think on):
#   grpo-avg3       GPUs 0/1, ports 18966/18967
#   grpo-interp050  GPUs 2/3, ports 18968/18969
set -u; umask 002
E=/workspace/workspace/mingdong/repo/oal-think-exp/runs/zh-spelling-think-grpo-20261115
A=$E/tasks/02-grpo/attempts/001
REM=whai:open-audio-llm/runs/zh-spelling-think-grpo-20261115/tasks/02-grpo/attempts/001/artifacts/training
START_REM=whai:open-audio-llm/runs/zh-spelling-think-r2-20261115/tasks/01-train-think/attempts/001/artifacts/training/checkpoint-3000/
PY=/workspace/workspace/mingdong/miniconda3/envs/oal-train/bin/python
MODELS=/workspace/workspace/mingdong/scratch/acp-dev/ae-models.sh
AVG=/workspace/workspace/mingdong/scratch/ts033/avg_ckpt.py
AE=/workspace/workspace/mingdong/.local-tools/amphion-eval-0.14.0-eedcf15/bin/ae
PROXY=http://10.119.17.252:17920
while :; do
  s=$($PY -c "import json;print(json.load(open('$A/status.json'))['execution'])" 2>/dev/null)
  case "$s" in starting|running|"") sleep 60;; *) break;; esac
done
echo "grpo $s $(date -u +%FT%TZ)"; [ "$s" = completed ] || exit 3
C=(); for step in 200 300 400; do c=$(bash $MODELS fetch $REM/checkpoint-$step/) || exit 5; C+=("$c"); done
echo "checkpoints ${C[*]} $(date -u +%FT%TZ)"
W=$(bash $MODELS stage) || exit 5
$PY $AVG "$W" "${C[@]}" || exit 6
F=$(bash $MODELS put "$W" --name spelling-think-grpo-avg3) || exit 5
START=$(bash $MODELS fetch $START_REM) || exit 5
W=$(bash $MODELS stage) || exit 5
$PY $AVG "$W" "$START" "$F" || exit 6
I=$(bash $MODELS put "$W" --name spelling-think-grpo-interp050) || exit 5
echo "avg3 $F interpolation $I $(date -u +%FT%TZ)"
cd $E
SUITES=(context-correction context-correction-think gate-v12 gate-customer-ts gate-asr-benchmark gate-hotwords gate-police
        gate-customer-asr gate-customer-hotwords gate-chime4 gate-v12-think gate-customer-ts-think gate-asr-benchmark-think)
evaluate() {  # name checkpoint asr-gpu ts-gpu port
  $PY scripts/make_eval_configs.py "$1" "$2" "$3" "$4" "$5" && $PY scripts/make_gate_configs.py "$1" v12 customer-ts asr-benchmark || return 6
  bash scripts/run_eval.sh "configs/eval-$1" $PROXY $AE "${SUITES[@]}"
}
evaluate grpo-avg3 "$F" 0 1 18966 & P1=$!
evaluate grpo-interp050 "$I" 2 3 18968 & P2=$!
wait $P1; r1=$?; wait $P2; r2=$?
echo "eval exit grpo-avg3=$r1 grpo-interp050=$r2 $(date -u +%FT%TZ)"
[ "$r1" = 0 ] && [ "$r2" = 0 ]
