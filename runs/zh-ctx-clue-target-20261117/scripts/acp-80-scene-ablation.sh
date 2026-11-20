#!/usr/bin/env bash
# zh-ctx-clue-target-20261117, question 4 (scene-TS regression: synthetic spelling data or #158/#159's think
# allocation), on one ACP node: train template-no-empty-think (F) and template-no-synth (G) one after the other on
# GPUs 0-7, fetch both checkpoint-3000s (one at a time: the ae model cache is not fetched twice at once), then at the
# same time F's context-correction eval v2 (node_chain.sh eval, GPUs 0/1) and scene TS for F (GPU 2) and G (GPU 3),
# vLLM starts staggered by 60 s.
set -u; umask 002
E=/workspace/workspace/mingdong/repo/oal-clue-target/runs/zh-ctx-clue-target-20261117; C=$E/scripts/node_chain.sh
PY=/workspace/workspace/mingdong/miniconda3/envs/oal-train/bin/python
MODELS=/workspace/workspace/mingdong/scratch/acp-dev/ae-models.sh
AE=/workspace/workspace/mingdong/.local-tools/amphion-eval-0.14.0-eedcf15/bin/ae
PROXY=http://10.119.17.252:17920
now() { date -u +%FT%TZ; }
ARMS="template-no-empty-think template-no-synth" bash $C train || exit $?
cd $E
fetch() {  # arm -> checkpoint path in the ae model cache (latest completed attempt)
  local a d
  for d in tasks/01-train-$1/attempts/*/; do grep -q '"execution": "completed"' $d/status.json 2>/dev/null && a=$(basename $d); done
  [ -n "${a:-}" ] || { echo "no completed training attempt for $1" >&2; return 5; }
  bash $MODELS fetch "whai:open-audio-llm/runs/zh-ctx-clue-target-20261117/tasks/01-train-$1/attempts/$a/artifacts/training/checkpoint-3000/"
}
F=$(fetch template-no-empty-think) || exit 5; echo "template-no-empty-think checkpoint $F $(now)"
G=$(fetch template-no-synth) || exit 5; echo "template-no-synth checkpoint $G $(now)"
scene() {  # name checkpoint gpu port
  $PY scripts/make_scene_ts_configs.py "$1" "$2" "$3" "$4" || return 6
  bash scripts/run_eval.sh "configs/eval-scene-$1" $PROXY $AE gate-scene-ts
}
ARMS="template-no-empty-think" bash $C eval & c=$!
sleep 60
scene template-no-empty-think "$F" 2 18751 & f=$!
sleep 60
scene template-no-synth "$G" 3 18761 & g=$!
wait $c; rc=$?; wait $f; rf=$?; wait $g; rg=$?
echo "ctx eval F exit $rc, scene F exit $rf, scene G exit $rg $(now)"
[ "$rc" = 0 ] && [ "$rf" = 0 ] && [ "$rg" = 0 ]
