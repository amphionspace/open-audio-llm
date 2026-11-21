#!/usr/bin/env bash
# tasks/05: teacher_signal.py on one node, 8 shards (one per GPU) per model, models one after another:
# 11.17 (the task-03/04 start: the teacher the KL began with), then the task-03 and task-04 avg3 models.
# Usage: run_teacher_signal.sh ATTEMPT SAMPLES   e.g. run_teacher_signal.sh 001 2400
set -u; umask 000
attempt=$1; samples=$2
PY=/workspace/workspace/mingdong/miniconda3/envs/oal-train/bin/python
M=/workspace/workspace/mingdong/scratch/acp-dev/ae-models.sh
E=/workspace/workspace/mingdong/repo/oal-clean-teacher/runs/ts-overlap-20261119
A=$E/tasks/05-teacher-signal/attempts/$attempt
mkdir -p $A/logs $A/artifacts
start=$(date -u +%FT%TZ); echo "node $(hostname -I) start $start"
rc=0
for spec in 1117:whitebox-merge-v18a-es116inc cleankl:ts-overlap-cleankl-004-avg3 control:ts-overlap-control-004-avg3; do
  name=${spec%%:*}; cache=${spec#*:}
  ckpt=$(bash $M fetch $cache | tail -1) || { echo "fetch $cache failed"; rc=1; continue; }
  echo "$name $cache $ckpt $(date -u +%FT%TZ)"
  mkdir -p $A/artifacts/$name
  pids=()
  for g in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$g PYTHONPATH=$E/shared/code-v1/src:$E/shared/code-v1/audio-data-contract-src \
      $PY $E/scripts/teacher_signal.py --checkpoint $ckpt --data-config $E/configs/teacher-signal-data.yaml \
      --out $A/artifacts/$name/shard-$g.jsonl --samples $samples --shard $g --num-shards 8 --workers 8 \
      > $A/logs/$name-shard-$g.log 2>&1 & pids+=($!)
  done
  for g in 0 1 2 3 4 5 6 7; do wait ${pids[$g]} || { echo "$name shard $g failed"; rc=1; }; done
  echo "$name done $(date -u +%FT%TZ) $(cat $A/artifacts/$name/shard-*.jsonl | wc -l) samples"
done
python3 -c "import json,sys; json.dump({'execution': 'completed' if $rc == 0 else 'failed', 'quality': '未核实', 'started': '$start', 'finished': sys.argv[1], 'exit_code': $rc, 'samples_per_model': $samples, 'script': 'scripts/run_teacher_signal.sh'}, open('$A/status.json', 'w'), ensure_ascii=False, indent=2)" "$(date -u +%FT%TZ)"
echo "done rc=$rc $(date -u +%FT%TZ)"
exit $rc
