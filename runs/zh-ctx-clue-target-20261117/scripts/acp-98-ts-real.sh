#!/usr/bin/env bash
# zh-ctx-clue-target-20261117, question 8 (does J miss real target speech?), eval only, on one ACP node: template and
# J (template-synth-ts-twins), each with one TS server, run gate-v12 then gate-first-other (configs/eval-tsreal-<name>/,
# scripts/make_ts_real_configs.py, generated on the dev machine). Both checkpoint-3000s are already in the ae model
# cache. Template on GPU 0, J on GPU 2; vLLM starts staggered.
set -u; umask 002
E=/workspace/workspace/mingdong/repo/oal-clue-target/runs/zh-ctx-clue-target-20261117; cd $E
AE=/workspace/workspace/mingdong/.local-tools/amphion-eval-0.14.0-eedcf15/bin/ae
PROXY=http://10.119.17.252:17920
now() { date -u +%FT%TZ; }
bash scripts/run_eval.sh configs/eval-tsreal-template $PROXY $AE gate-v12 gate-first-other & t=$!
sleep 120
bash scripts/run_eval.sh configs/eval-tsreal-template-synth-ts-twins $PROXY $AE gate-v12 gate-first-other & j=$!
wait $t; rt=$?; wait $j; rj=$?
echo "ts-real template exit $rt, J exit $rj $(now)"
[ "$rt" = 0 ] && [ "$rj" = 0 ]
