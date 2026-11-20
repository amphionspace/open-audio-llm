#!/usr/bin/env bash
# zh-ctx-clue-target-20261117, question 3: scene TS (tdtech/scene-ts@v1-20261028, the context-correction line's
# delivery blocker, see scripts/make_scene_ts_configs.py) on template's and template-implicit-noexplain's
# checkpoint-3000, GPUs 2 and 3, alongside 50's context-correction eval on GPUs 0/1. Starts once 50's training is done
# and its two eval servers are up (staggered vLLM starts), fetches template-implicit-noexplain only after 50 has.
set -u; umask 002
E=/workspace/workspace/mingdong/repo/oal-clue-target/runs/zh-ctx-clue-target-20261117
R=/workspace/workspace/mingdong/scratch/acp-ctx-template-implicit; L=$R/logs/50-template-implicit.log
PY=/workspace/workspace/mingdong/miniconda3/envs/oal-train/bin/python
MODELS=/workspace/workspace/mingdong/scratch/acp-dev/ae-models.sh
AE=/workspace/workspace/mingdong/.local-tools/amphion-eval-0.14.0-eedcf15/bin/ae
PROXY=http://10.119.17.252:17920
now() { date -u +%FT%TZ; }
wait_for() {  # pattern file: until the pattern shows up, or 50 has exited without it
  until grep -q "$1" "$2" 2>/dev/null; do
    [ -e $R/done/50-template-implicit.exit ] && { echo "50 exited before: $1 $(now)"; exit 3; }
    sleep 30
  done
}
wait_for "train-template-implicit-noexplain done" $L
echo "training done $(now)"
S=$E/configs/eval-template-implicit-noexplain/logs/status.txt
for k in $(seq 40); do [ "$(grep -c 'server on .* ready' $S 2>/dev/null)" -ge 2 ] && break; sleep 30; done
echo "50's eval servers: $(grep -c 'server on .* ready' $S 2>/dev/null) ready $(now)"
cd $E
one() {  # arm gpu port
  local f
  f=$(bash $MODELS fetch "whai:open-audio-llm/runs/zh-ctx-clue-target-20261117/tasks/01-train-$1/attempts/001/artifacts/training/checkpoint-3000/") || return 5
  echo "$1 checkpoint $f $(now)"
  $PY scripts/make_scene_ts_configs.py "$1" "$f" "$2" "$3" || return 6
  bash scripts/run_eval.sh "configs/eval-scene-$1" $PROXY $AE gate-scene-ts
}
one template 2 18731 & a=$!
wait_for "template-implicit-noexplain checkpoint " $L
sleep 60
one template-implicit-noexplain 3 18741 & b=$!
wait $a; ra=$?; wait $b; rb=$?
echo "scene template exit $ra, template-implicit-noexplain exit $rb $(now)"
[ "$ra" = 0 ] && [ "$rb" = 0 ]
