#!/usr/bin/env bash
# zh-ctx-clue-target-20261117, question 7 (TTS voices as somebody else: the synthetic data's own cloned voices in
# target-speaker scenes, 30% twins), on one ACP node: train template-synth-ts-twins (J) on GPUs 0-7, fetch its
# checkpoint-3000, then at the same time its context-correction eval v2 (node_chain.sh eval, GPUs 0/1) and scene TS
# (GPU 4), vLLM starts staggered.
set -u; umask 002
E=/workspace/workspace/mingdong/repo/oal-clue-target/runs/zh-ctx-clue-target-20261117; C=$E/scripts/node_chain.sh
PY=/workspace/workspace/mingdong/miniconda3/envs/oal-train/bin/python
MODELS=/workspace/workspace/mingdong/scratch/acp-dev/ae-models.sh
AE=/workspace/workspace/mingdong/.local-tools/amphion-eval-0.14.0-eedcf15/bin/ae
PROXY=http://10.119.17.252:17920
now() { date -u +%FT%TZ; }
ARMS="template-synth-ts-twins" bash $C train || exit $?
cd $E
a=; for d in tasks/01-train-template-synth-ts-twins/attempts/*/; do grep -q '"execution": "completed"' $d/status.json 2>/dev/null && a=$(basename $d); done
[ -n "$a" ] || { echo "no completed training attempt"; exit 5; }
J=$(bash $MODELS fetch "whai:open-audio-llm/runs/zh-ctx-clue-target-20261117/tasks/01-train-template-synth-ts-twins/attempts/$a/artifacts/training/checkpoint-3000/") || exit 5
echo "template-synth-ts-twins checkpoint $J $(now)"
ARMS="template-synth-ts-twins" bash $C eval & c=$!
sleep 120  # after the context-correction servers have started
{ $PY scripts/make_scene_ts_configs.py template-synth-ts-twins "$J" 4 18791 &&
  bash scripts/run_eval.sh configs/eval-scene-template-synth-ts-twins $PROXY $AE gate-scene-ts; } & s=$!
wait $c; rc=$?; wait $s; rs=$?
echo "ctx eval J exit $rc, scene J exit $rs $(now)"
[ "$rc" = 0 ] && [ "$rs" = 0 ]
