#!/usr/bin/env bash
# tasks/30 ACP 1-node job: one cached model on the seven Arabic eval sets (amphion-eval MR !174, @v1-20261127), four lanes
# on the 8 GPUs, same serving stack as the delivery gate (eval_lanes.sh: pack-mtp serve.sh, no MTP). The shared
# amphion-eval env (0.13.3) has neither these eval sets nor the Arabic scoring (78c1583), so the evaluator runs from the
# venv AE below: the shared env's packages plus the pinned amphion-eval ca7d87f (native scoring v6).
# W&B goes through the dev-machine (10.119.18.89) forwarder on 17922, which admits this node after its IP is added (wandb-ready).
# usage: job_ar_baseline.sh NAME TAG   (outputs under /workspace/data/ts-overlap-20261119/task30-ar-TAG)
set -u
NAME=$1; TAG=$2
O=/workspace/data/ts-overlap-20261119/task30-ar-$TAG; E=/workspace/workspace/mingdong/repo/oal-clean-teacher/runs/ts-overlap-20261119
AE=/workspace/data/ts-overlap-20261119/ar-baseline/ae-ca7d87f-venv
echo "node ip $(hostname -I) $(date -u +%FT%TZ)" > $O/logs/node.txt
for _ in $(seq 120); do [ -f $O/wandb-ready ] && break; sleep 20; done
[ -f $O/wandb-ready ] || { echo "no wandb-ready flag $(date -u +%FT%TZ)" >> $O/logs/node.txt; exit 3; }
cd $E; AE_ENV=$AE TEMPLATES=$E/configs/ar-templates WB_PROXY=http://10.119.18.89:17922 bash scripts/eval_lanes.sh $NAME $TAG 0 19900 \
  "ar/arasr ar/arbab" "ar/tsarv1 ar/xlar" "ar/tsarv2" "ar/arhw ar/tsarhot" > $O/logs/eval.log 2>&1
echo "ar rc=$? $(date -u +%FT%TZ)" >> $O/logs/node.txt
