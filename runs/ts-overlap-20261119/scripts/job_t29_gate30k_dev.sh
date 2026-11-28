#!/usr/bin/env bash
# task 29 step-30k final gate + meeting bench on the dev machine's GPUs 4-7 (ACP queue was full; ACP job pt-4a2u4nlc
# stopped). Same suites, templates, serving stack and output dirs as job_gate_bench.sh; only the lane layout differs:
# GPUs 4-5 run asrfull (the 66-min long pole), GPUs 6-7 run v12/tsen/custts and then the four meeting suites.
# W&B goes straight through the dev proxy.
set -u; umask 000
E=/workspace/workspace/mingdong/repo/oal-clean-teacher/runs/ts-overlap-20261119; D=/workspace/data/ts-overlap-20261119
NAME=ts-overlap-t29simnegasr-s30000; TAG=t29simnegasr-s30000
G=$D/task23-gate-$TAG; B=$D/task27-bench-$TAG-meeting
unset no_proxy NO_PROXY; export WB_PROXY=http://127.0.0.1:7890
cd $E
echo "node ip $(hostname -I) $(date -u +%FT%TZ) dev GPUs 4-7" > $G/logs/node.txt
bash scripts/eval_lanes.sh $NAME $TAG 4 19970 "eval-v18avg3-ckpt400-asrfull/asrfull" > $G/logs/eval-asrfull.log 2>&1 &
a=$!
( bash scripts/eval_lanes.sh $NAME $TAG 6 19980 "eval-sumfull-es116-core/v12 eval-sumfull-es116-core/tsen eval-sumfull-es116-core/custts" > $G/logs/eval.log 2>&1
  echo "core rc=$? $(date -u +%FT%TZ)" >> $G/logs/node.txt
  echo "node ip $(hostname -I) $(date -u +%FT%TZ) version $(cat $E/configs/meeting-bench/VERSION) catalog /workspace/data/registry/catalog dev GPUs 6-7" > $B/logs/node.txt
  TEMPLATES=$E/configs/meeting-bench CATALOG=/workspace/data/registry/catalog bash scripts/eval_lanes.sh $NAME $TAG-meeting 6 19990 \
    "meeting/a4ramc meeting/alibase meeting/alinear meeting/alisnr" > $B/logs/eval.log 2>&1
  echo "bench rc=$? $(date -u +%FT%TZ)" >> $B/logs/node.txt ) &
b=$!
wait $a; echo "asrfull rc=$? $(date -u +%FT%TZ)" >> $G/logs/node.txt
wait $b; echo "all done $(date -u +%FT%TZ)"
