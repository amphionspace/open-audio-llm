#!/usr/bin/env bash
# Probe features on 8 GPUs: test list (v12 subset, 4 conditions) and train list (generated scenes, mix/clean).
set -u; P=/workspace/workspace/mingdong/.claude/tmp/claude-1001/-workspace-workspace-mingdong-repo-open-audio-llm/ca9aa6d0-4ecc-4cfc-b526-5322bf6c9de5/scratchpad/ovdiag/probe; F=/workspace/data/ts-overlap-20261119/task17-probe
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=/workspace/workspace/mingdong/repo/oal-clean-teacher/runs/ts-overlap-20261119/shared/code-v4/src
PY=/workspace/workspace/mingdong/miniconda3/envs/oal-train/bin/python
mkdir -p $F/logs; pids=()
for g in 0 1 2 3 4 5 6 7; do
  ( CUDA_VISIBLE_DEVICES=$g $PY $P/extract_feats.py $P/test-list.jsonl $F/test $g 8 > $F/logs/test-$g.log 2>&1 &&
    CUDA_VISIBLE_DEVICES=$g $PY $P/extract_feats.py $P/train-list.jsonl $F/train $g 8 > $F/logs/train-$g.log 2>&1 ) & pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done; echo "feats rc=$rc $(date -u +%FT%TZ)"; exit $rc
