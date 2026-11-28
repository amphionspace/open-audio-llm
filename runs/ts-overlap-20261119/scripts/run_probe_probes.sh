#!/usr/bin/env bash
# Four CTC probes (tower layer 0 input, layer 12, layer 24, tower output) in parallel, one GPU each.
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
set -u; P=/workspace/workspace/mingdong/.claude/tmp/claude-1001/-workspace-workspace-mingdong-repo-open-audio-llm/ca9aa6d0-4ecc-4cfc-b526-5322bf6c9de5/scratchpad/ovdiag/probe; F=/workspace/data/ts-overlap-20261119/task17-probe; PY=/workspace/workspace/mingdong/miniconda3/envs/oal-train/bin/python
pids=(); g=0
for L in l0 l12 l24 out; do
  CUDA_VISIBLE_DEVICES=$g $PY $P/train_probe.py $L $F/train $F/test $P/tokens.json $F/probe-$L.json 8000 > $F/logs/probe-$L.log 2>&1 & pids+=($!); g=$((g + 1))
done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done; echo "probes rc=$rc $(date -u +%FT%TZ)"; exit $rc
