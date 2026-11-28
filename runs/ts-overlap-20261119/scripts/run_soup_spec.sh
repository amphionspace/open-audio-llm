#!/usr/bin/env bash
# tasks/16 MTP acceptance of soup: spec_check.py (runs/joint-mtp-spec-20261107, the protocol of 11.17's package numbers)
# with configs/spec-check-soup-{zhen-ts,esasr}.yaml, four lanes at once (base / MTP k=2 for each), then summaries.
# usage: run_soup_spec.sh
set -uo pipefail; umask 000
E=/workspace/workspace/mingdong/repo/oal-clean-teacher/runs/ts-overlap-20261119
J=/workspace/workspace/mingdong/repo/oal-joint-mtp/runs/joint-mtp-spec-20261107
PY=/workspace/workspace/mingdong/miniconda3/envs/oal-train/bin/python
source /workspace/workspace/mingdong/.config/open-audio-llm/wandb-cluster.sh >/dev/null
export WANDB_HTTP_PROXY=http://10.119.17.252:17922 WANDB_HTTPS_PROXY=http://10.119.17.252:17922
unset no_proxy NO_PROXY
export PYTHONPATH=/workspace/workspace/mingdong/repo/oal-joint-mtp/src
cd $J
pids=()
for x in zhen-ts:002 esasr:003; do n=${x%%:*}; a=${x##*:}; A=$E/tasks/16-candidate-soup/attempts/$a; mkdir -p $A/logs
  cp $E/configs/spec-check-soup-$n.yaml $A/effective.yaml; echo "start $(date -u +%FT%TZ)" > $A/status.txt
  for lane in 0 1; do $PY scripts/spec_check.py --config $E/configs/spec-check-soup-$n.yaml --lane $lane > $A/logs/lane$lane.log 2>&1 & pids+=($!); done
done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
for a in 002 003; do A=$E/tasks/16-candidate-soup/attempts/$a
  $PY scripts/show_spec.py $A > $A/summary.txt 2>&1; echo "done rc=$rc $(date -u +%FT%TZ)" >> $A/status.txt; done
echo "spec done rc=$rc $(date -u +%FT%TZ)"; exit $rc
