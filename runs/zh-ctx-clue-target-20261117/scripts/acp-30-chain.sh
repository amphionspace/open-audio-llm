#!/usr/bin/env bash
# zh-ctx-clue-target-20261117, rest of the run after 10-chain's training failed on a config path (2026-11-18 06:35).
# 20-implicit had already started building the implicit data on GPUs 4-7; its parent is stopped here so the build
# finishes alone, then the first two arms train and evaluate, then the two implicit arms.
set -u
R=/workspace/workspace/mingdong/scratch/acp-clue-target
C=/workspace/workspace/mingdong/repo/oal-clue-target/runs/zh-ctx-clue-target-20261117/scripts/node_chain.sh
pkill -TERM -f "running/20-implicit.sh" && echo "stopped the 20-implicit parent $(date -u +%FT%TZ)"
while pgrep -f "node_chain.sh implicit" >/dev/null; do sleep 60; done
echo "implicit build ended $(date -u +%FT%TZ): $(grep -c 'implicit data done' $R/logs/20-implicit.log) done line(s)"
bash $C train eval; rc=$?
echo "first two arms exit $rc $(date -u +%FT%TZ)"
case $rc in 2|3) exit $rc;; esac  # W&B unreachable or a training failure: the implicit arms would fail the same way
grep -q "implicit data done" $R/logs/20-implicit.log || { echo "implicit data not built; implicit arms not run"; exit 7; }
ARMS="clue-target-implicit clue-target-implicit-noexplain" exec bash $C configs train eval
