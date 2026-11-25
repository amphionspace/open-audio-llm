#!/usr/bin/env bash
# ACP 1-node job: the delivery gate suites (job_e2e2_gate.sh, outputs task23-gate-TAG) and then the whole-meeting TS
# benchmark (job_meeting_bench.sh, outputs task27-bench-TAG-meeting) for one cached model, one after the other on the same
# node. Each part waits for the wandb-ready flag in its own output dir (admit_node.sh raises both).
# usage: job_gate_bench.sh NAME TAG [CATALOG]
set -u
NAME=$1; TAG=$2; CATALOG=${3:-/workspace/data/registry/catalog}
E=/workspace/workspace/mingdong/repo/oal-clean-teacher/runs/ts-overlap-20261119
bash $E/scripts/job_e2e2_gate.sh $NAME $TAG
bash $E/scripts/job_meeting_bench.sh $NAME $TAG-meeting 0 $CATALOG
