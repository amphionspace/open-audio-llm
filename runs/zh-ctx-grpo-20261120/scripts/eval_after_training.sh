#!/usr/bin/env bash
# 03-eval, once grpo completes (its latest attempt, which must be completed): avg3 of checkpoints 200/300/400 and its 50% interpolation with the start
# (J, template-synth-ts-twins checkpoint-3000), as 08 (scripts/eval_after_training.sh there); both go into the ae
# model cache. Then the two candidates and the start are evaluated at the same time with the same code and client
# (README "判定"): context correction v2 think off/on and the gates of make_gate_configs.py, all think off.
#   grpo-avg3       GPUs 0/1, ports 18966/18967
#   grpo-interp050  GPUs 2/3, ports 18968/18969
#   start-j         GPUs 4/5, ports 18970/18971
set -u; umask 002
E=/workspace/workspace/mingdong/repo/oal-ctx-grpo/runs/zh-ctx-grpo-20261120
A=$(ls -d $E/tasks/02-grpo/attempts/[0-9][0-9][0-9] | sort | tail -n 1)
REM=whai:open-audio-llm/runs/zh-ctx-grpo-20261120/tasks/02-grpo/attempts/$(basename $A)/artifacts/training
START_REM=whai:open-audio-llm/runs/zh-ctx-clue-target-20261117/tasks/01-train-template-synth-ts-twins/attempts/001/artifacts/training/checkpoint-3000/
PY=/workspace/workspace/mingdong/miniconda3/envs/oal-train/bin/python
MODELS=/workspace/workspace/mingdong/scratch/acp-dev/ae-models.sh
AVG=/workspace/workspace/mingdong/scratch/ts033/avg_ckpt.py
AE=/workspace/workspace/mingdong/.local-tools/amphion-eval-0.14.0-eedcf15/bin/ae
PROXY=http://10.119.17.252:17920
s=$($PY -c "import json;print(json.load(open('$A/status.json'))['execution'])" 2>/dev/null)
echo "grpo $(basename $A) $s $(date -u +%FT%TZ)"; [ "$s" = completed ] || exit 3
C=(); for step in 200 300 400; do c=$(bash $MODELS fetch $REM/checkpoint-$step/) || exit 5; C+=("$c"); done
echo "checkpoints ${C[*]} $(date -u +%FT%TZ)"
W=$(bash $MODELS stage) || exit 5
$PY $AVG "$W" "${C[@]}" || exit 6
F=$(bash $MODELS put "$W" --name ctx-grpo-avg3) || exit 5
START=$(bash $MODELS fetch $START_REM) || exit 5
W=$(bash $MODELS stage) || exit 5
$PY $AVG "$W" "$START" "$F" || exit 6
I=$(bash $MODELS put "$W" --name ctx-grpo-interp050) || exit 5
echo "avg3 $F interpolation $I start $START $(date -u +%FT%TZ)"
cd $E
SUITES=(context-correction-v2 context-correction-v2-think gate-v12 gate-customer-ts gate-first-other gate-scene-ts
        gate-asr-benchmark gate-hotwords gate-police gate-customer-asr gate-customer-hotwords gate-chime4)
evaluate() {  # name checkpoint asr-gpu ts-gpu port
  $PY scripts/make_eval_configs.py "$1" "$2" "$3" "$4" "$5" && $PY scripts/make_gate_configs.py "$1" || return 6
  bash scripts/run_eval.sh "configs/eval-$1" $PROXY $AE "${SUITES[@]}"
}
evaluate grpo-avg3 "$F" 0 1 18966 & P1=$!
sleep 60  # stagger the vLLM starts
evaluate grpo-interp050 "$I" 2 3 18968 & P2=$!
sleep 60
evaluate start-j "$START" 4 5 18970 & P3=$!
wait $P1; r1=$?; wait $P2; r2=$?; wait $P3; r3=$?
echo "eval exit grpo-avg3=$r1 grpo-interp050=$r2 start-j=$r3 $(date -u +%FT%TZ)"
[ "$r1" = 0 ] && [ "$r2" = 0 ] && [ "$r3" = 0 ]
