#!/usr/bin/env bash
# zh-ctx-clue-target-20261117, question 6 (keep the synthetic spelling data's Emilia gain without the scene-TS
# regression), on one ACP node: train template-no-synth-ts (H) and template-synth-ts-mix (I) one after the other on
# GPUs 0-7, fetch both checkpoint-3000s (one at a time: the ae model cache is not fetched twice at once), then at the
# same time both arms' context-correction eval v2 (node_chain.sh eval, GPUs 0/1 and 2/3) and scene TS for H (GPU 4)
# and I (GPU 5), vLLM starts staggered by 60 s.
set -u; umask 002
E=/workspace/workspace/mingdong/repo/oal-clue-target/runs/zh-ctx-clue-target-20261117; C=$E/scripts/node_chain.sh
PY=/workspace/workspace/mingdong/miniconda3/envs/oal-train/bin/python
MODELS=/workspace/workspace/mingdong/scratch/acp-dev/ae-models.sh
AE=/workspace/workspace/mingdong/.local-tools/amphion-eval-0.14.0-eedcf15/bin/ae
PROXY=http://10.119.17.252:17920
now() { date -u +%FT%TZ; }
ARMS="template-no-synth-ts template-synth-ts-mix" bash $C train || exit $?
cd $E
fetch() {  # arm -> checkpoint path in the ae model cache (latest completed attempt)
  local a d
  for d in tasks/01-train-$1/attempts/*/; do grep -q '"execution": "completed"' $d/status.json 2>/dev/null && a=$(basename $d); done
  [ -n "${a:-}" ] || { echo "no completed training attempt for $1" >&2; return 5; }
  bash $MODELS fetch "whai:open-audio-llm/runs/zh-ctx-clue-target-20261117/tasks/01-train-$1/attempts/$a/artifacts/training/checkpoint-3000/"
}
H=$(fetch template-no-synth-ts) || exit 5; echo "template-no-synth-ts checkpoint $H $(now)"
I=$(fetch template-synth-ts-mix) || exit 5; echo "template-synth-ts-mix checkpoint $I $(now)"
scene() {  # name checkpoint gpu port
  $PY scripts/make_scene_ts_configs.py "$1" "$2" "$3" "$4" || return 6
  bash scripts/run_eval.sh "configs/eval-scene-$1" $PROXY $AE gate-scene-ts
}
ARMS="template-no-synth-ts template-synth-ts-mix" bash $C eval & c=$!
sleep 120  # after both context-correction servers have started (node_chain staggers them by 60 s)
scene template-no-synth-ts "$H" 4 18771 & h=$!
sleep 60
scene template-synth-ts-mix "$I" 5 18781 & i=$!
wait $c; rc=$?; wait $h; rh=$?; wait $i; ri=$?
echo "ctx eval H+I exit $rc, scene H exit $rh, scene I exit $ri $(now)"
[ "$rc" = 0 ] && [ "$rh" = 0 ] && [ "$ri" = 0 ]
