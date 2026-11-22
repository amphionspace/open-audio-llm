#!/usr/bin/env bash
# 03-eval, once grpo completes (its latest attempt, which must be completed): avg3 of checkpoints 200/300/400 into the
# ae model cache (as J's round, runs/zh-ctx-grpo-20261120/scripts/eval_after_training.sh; no interpolation here, the
# candidate is avg3). Then at the same time, with the same code and client:
#   kgrpo-avg3  context correction v2 think off/on   GPUs 0/1, ports 18966/18967
#   start-k     the same for the start K            GPUs 2/3, ports 18968/18969
#   kgrpo-avg3  the 27 delivery suites (GRPO experiment zh-ctx-grpo-20261120, scripts/acp-delivery-one.sh)
#               pairs a GPUs 4/5 port 18934, b GPUs 6/7 port 18936
# usage: eval_after_training.sh
set -u; umask 002
E=/workspace/workspace/mingdong/repo/oal-ctx-grpo/runs/zh-ctx-grpo-k-20261122
G=/workspace/workspace/mingdong/repo/oal-ctx-grpo/runs/zh-ctx-grpo-20261120
A=$(ls -d $E/tasks/02-grpo/attempts/[0-9][0-9][0-9] | sort | tail -n 1)
REM=whai:open-audio-llm/runs/zh-ctx-grpo-k-20261122/tasks/02-grpo/attempts/$(basename $A)/artifacts/training
START_REM=whai:open-audio-llm/runs/zh-ctx-clue-target-20261117/tasks/01-train-template-twins-dialog-final-1117/attempts/001/artifacts/training/checkpoint-3000/
PY=/workspace/workspace/mingdong/miniconda3/envs/oal-train/bin/python
MODELS=/workspace/workspace/mingdong/scratch/acp-dev/ae-models.sh
AVG=/workspace/workspace/mingdong/scratch/ts033/avg_ckpt.py
AE=/workspace/workspace/mingdong/.local-tools/amphion-eval-0.14.0-eedcf15/bin/ae
PROXY=http://10.119.17.252:17920
now() { date -u +%FT%TZ; }
s=$($PY -c "import json;print(json.load(open('$A/status.json'))['execution'])" 2>/dev/null)
echo "grpo $(basename $A) $s $(now)"; [ "$s" = completed ] || exit 3
C=(); for step in 200 300 400; do c=$(bash $MODELS fetch $REM/checkpoint-$step/) || exit 5; C+=("$c"); done
echo "checkpoints ${C[*]} $(now)"
W=$(bash $MODELS stage) || exit 5
$PY $AVG "$W" "${C[@]}" || exit 6
F=$(bash $MODELS put "$W" --name ctx-grpo-k-avg3) || exit 5
START=$(bash $MODELS fetch $START_REM) || exit 5
echo "avg3 $F start $START $(now)"
cd $E
evaluate() {  # name checkpoint asr-gpu ts-gpu port
  $PY scripts/make_eval_configs.py "$1" "$2" "$3" "$4" "$5" || return 6
  bash scripts/run_eval.sh "configs/eval-$1" $PROXY $AE context-correction-v2 context-correction-v2-think
}
evaluate kgrpo-avg3 "$F" 0 1 18966 & P1=$!
sleep 60  # stagger the vLLM starts
evaluate start-k "$START" 2 3 18968 & P2=$!
sleep 60
bash $G/scripts/acp-delivery-one.sh kgrpo "$F" 4 5 18934 6 7 18936 & P3=$!
wait $P1; r1=$?; wait $P2; r2=$?; wait $P3; r3=$?
echo "eval exit kgrpo-avg3=$r1 start-k=$r2 delivery-kgrpo=$r3 $(now)"
[ "$r1" = 0 ] && [ "$r2" = 0 ] && [ "$r3" = 0 ]
