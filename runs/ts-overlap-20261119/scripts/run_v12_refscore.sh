#!/usr/bin/env bash
# tasks/06: the white-box v12 reference scoring (runs/whitebox-20261119-overlap/scripts/ref_score.py, PR #167) on the
# task-04 and task-03 avg3 models, same code (code-v37) and settings as the 11.17 run of 2026-11-20
# (whai:open-audio-llm/runs/whitebox-20261119-overlap/tasks/03-ref-score/). Each model scores its own v12 eval run.
# Usage: run_v12_refscore.sh ATTEMPT
set -u; umask 000
attempt=$1
PY=/workspace/workspace/mingdong/miniconda3/envs/oal-train/bin/python
SRC=/workspace/workspace/mingdong/repo/open-audio-llm-ts-robust/runs/ts-robust-acp-20261015/shared/code-v37/src
SCRIPT=/workspace/workspace/mingdong/repo/oal-retro-overlap/runs/whitebox-20261119-overlap/scripts/ref_score.py
M=/workspace/workspace/mingdong/scratch/acp-dev/ae-models.sh
E=/workspace/workspace/mingdong/repo/oal-clean-teacher/runs/ts-overlap-20261119
A=$E/tasks/06-v12-refscore/attempts/$attempt
B=/workspace/data/ae-runs
mkdir -p $A/logs $A/artifacts
start=$(date -u +%FT%TZ); echo "node $(hostname -I) start $start"
rc=0
for spec in control:ts-overlap-control-004-avg3:run-20261120T123607Z-ac4867f3 cleankl:ts-overlap-cleankl-004-avg3:run-20261120T123537Z-44834ae9; do
  IFS=: read name cache run <<< "$spec"
  ckpt=$(bash $M fetch $cache | tail -1) || { echo "fetch $cache failed"; rc=1; continue; }
  echo "$name $ckpt $run $(date -u +%FT%TZ)"; mkdir -p $A/artifacts/$name
  pids=()
  for g in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$g PYTHONPATH=$SRC $PY $SCRIPT --checkpoint $ckpt --run-dir $B/ts-overlap-$name-004-avg3/runs/$run \
      --out $A/artifacts/$name/shard-$g.jsonl --shard $g --num-shards 8 --workers 6 > $A/logs/$name-shard-$g.log 2>&1 & pids+=($!)
  done
  for g in 0 1 2 3 4 5 6 7; do wait ${pids[$g]} || { echo "$name shard $g failed"; rc=1; }; done
  echo "$name done $(date -u +%FT%TZ) $(cat $A/artifacts/$name/shard-*.jsonl | wc -l) samples"
done
python3 -c "import json,sys; json.dump({'execution': 'completed' if $rc == 0 else 'failed', 'quality': '未核实', 'started': '$start', 'finished': sys.argv[1], 'exit_code': $rc, 'script': 'scripts/run_v12_refscore.sh'}, open('$A/status.json', 'w'), ensure_ascii=False, indent=2)" "$(date -u +%FT%TZ)"
echo "done rc=$rc $(date -u +%FT%TZ)"; exit $rc
