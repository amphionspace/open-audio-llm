#!/usr/bin/env bash
# Once train-think (attempt 001) completes: fetch checkpoint-3000 through the ae model cache, build its 50% interpolation
# with the 11.06 release (put into the cache too), and evaluate both at the same time on this node, each with the
# context-correction set (think off and on), the 8 gates (think off) and 3 gates with think on:
#   think-3000           GPUs 0/1, ports 18966/18967
#   think-3000-interp050 GPUs 2/3, ports 18968/18969
set -u; umask 002
E=/workspace/workspace/mingdong/repo/oal-think-exp/runs/zh-spelling-think-r2-20261115
A=$E/tasks/01-train-think/attempts/001
PY=/workspace/workspace/mingdong/miniconda3/envs/oal-train/bin/python
MODELS=/workspace/workspace/mingdong/scratch/acp-dev/ae-models.sh
AE=/workspace/workspace/mingdong/.local-tools/amphion-eval-0.14.0-eedcf15/bin/ae
PROXY=http://10.119.17.252:17920
while :; do
  s=$($PY -c "import json;print(json.load(open('$A/status.json'))['execution'])" 2>/dev/null)
  case "$s" in starting|running|"") sleep 60;; *) break;; esac
done
echo "train-think $s $(date -u +%FT%TZ)"; [ "$s" = completed ] || exit 3
F=$(bash $MODELS fetch whai:open-audio-llm/runs/zh-spelling-think-r2-20261115/tasks/01-train-think/attempts/001/artifacts/training/checkpoint-3000/) || exit 5
echo "checkpoint $F $(date -u +%FT%TZ)"
STAGE=$(bash $MODELS stage) || exit 5
$PY $E/scripts/interp_release.py "$STAGE" "$F" 0.5 || exit 5
bash $MODELS put "$STAGE" --name spelling-think-r2-think-3000-interp050 || exit 5
I=$(bash $MODELS fetch spelling-think-r2-think-3000-interp050) || exit 5
echo "interpolation $I $(date -u +%FT%TZ)"
cd $E
SUITES=(context-correction context-correction-think gate-v12 gate-customer-ts gate-asr-benchmark gate-hotwords gate-police
        gate-customer-asr gate-customer-hotwords gate-chime4 gate-v12-think gate-customer-ts-think gate-asr-benchmark-think)
evaluate() {  # name checkpoint asr-gpu ts-gpu port
  $PY scripts/make_eval_configs.py "$1" "$2" "$3" "$4" "$5" && $PY scripts/make_gate_configs.py "$1" v12 customer-ts asr-benchmark || return 6
  bash scripts/run_eval.sh "configs/eval-$1" $PROXY $AE "${SUITES[@]}"
}
evaluate think-3000 "$F" 0 1 18966 & P1=$!
evaluate think-3000-interp050 "$I" 2 3 18968 & P2=$!
wait $P1; r1=$?; wait $P2; r2=$?
echo "eval exit think-3000=$r1 think-3000-interp050=$r2 $(date -u +%FT%TZ)"
[ "$r1" = 0 ] && [ "$r2" = 0 ]
