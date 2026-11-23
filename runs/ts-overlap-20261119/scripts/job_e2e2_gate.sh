#!/usr/bin/env bash
# tasks/23 ACP 1-node job: the delivery gate suites for an exported model (cache NAME, default ts-overlap-e2e2-final001-s30000),
# the 11.17 gate stack (pack-mtp serve.sh without MTP), four lanes on the 8 GPUs: v12 | English TS | Chinese ASR full | customer TS.
# W&B goes through the dev-machine forwarder on 17922, which admits this node only after its IP is added (the wandb-ready flag).
# usage: job_e2e2_gate.sh [NAME TAG]   (outputs under /workspace/data/ts-overlap-20261119/task23-gate[-TAG])
set -u
NAME=${1:-ts-overlap-e2e2-final001-s30000}; TAG=${2:-e2e2-s30000}
O=/workspace/data/ts-overlap-20261119/task23-gate${1:+-$TAG}; E=/workspace/workspace/mingdong/repo/oal-clean-teacher/runs/ts-overlap-20261119
echo "node ip $(hostname -I) $(date -u +%FT%TZ)" > $O/logs/node.txt
for _ in $(seq 120); do [ -f $O/wandb-ready ] && break; sleep 20; done
[ -f $O/wandb-ready ] || { echo "no wandb-ready flag $(date -u +%FT%TZ)" >> $O/logs/node.txt; exit 3; }
cd $E; WB_PROXY=http://10.119.17.252:17922 bash scripts/eval_lanes.sh $NAME $TAG 0 19900 \
  "eval-sumfull-es116-core/v12" "eval-sumfull-es116-core/tsen" "eval-v18avg3-ckpt400-asrfull/asrfull" "eval-sumfull-es116-core/custts" > $O/logs/eval.log 2>&1
echo "gate rc=$? $(date -u +%FT%TZ)" >> $O/logs/node.txt
