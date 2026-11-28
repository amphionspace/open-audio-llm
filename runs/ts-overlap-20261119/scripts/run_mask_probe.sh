#!/usr/bin/env bash
# tasks/22: layer-12 mask probes for B (aux, 4 GPUs), A (ce, 2 GPUs) and 11.17 (2 GPUs), 2000 head-training steps each.
set -u; mkdir -p /workspace/data/ts-overlap-20261119/task22-mask-probe/logs
export OMP_NUM_THREADS=4 AUDIO_DATA_ROOTS_FILE=/workspace/data/registry/roots.json HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/workspace/workspace/mingdong/repo/oal-clean-teacher/runs/ts-overlap-20261119/shared/code-v4/src
PY=/workspace/workspace/mingdong/miniconda3/envs/oal-train/bin/python; pids=(); g=0
for spec in "aux:/workspace/data/ts-overlap-20261119/task21-e2e/aux/step8000.pt:4" "ce:/workspace/data/ts-overlap-20261119/task21-e2e/ce/step8000.pt:2" "base:base:2"; do
  IFS=: read name state n <<< "$spec"
  for s in $(seq 0 $((n - 1))); do
    CUDA_VISIBLE_DEVICES=$g $PY /workspace/workspace/mingdong/.claude/tmp/claude-1001/-workspace-workspace-mingdong-repo-open-audio-llm/ca9aa6d0-4ecc-4cfc-b526-5322bf6c9de5/scratchpad/ovdiag/probe/mask_probe.py $state /workspace/data/ts-overlap-20261119/task22-mask-probe/$name 2000 $s $n > /workspace/data/ts-overlap-20261119/task22-mask-probe/logs/$name-$s.log 2>&1 & pids+=($!); g=$((g + 1))
  done
done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done; chmod -R a+rX /workspace/data/ts-overlap-20261119/task22-mask-probe; echo "mask probe rc=$rc $(date -u +%FT%TZ)"
