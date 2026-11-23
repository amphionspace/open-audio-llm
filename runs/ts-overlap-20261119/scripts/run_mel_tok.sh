#!/usr/bin/env bash
# tasks/22 control: token-rate ideal ratio mask on 11.17's mel input, 615 samples, 4 shards on GPUs 0-3.
set -u; mkdir -p /workspace/data/ts-overlap-20261119/task22-mask-probe/logs
export OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=/workspace/workspace/mingdong/repo/oal-clean-teacher/runs/ts-overlap-20261119/shared/code-v4/src
pids=(); for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g /workspace/workspace/mingdong/miniconda3/envs/oal-train/bin/python /workspace/workspace/mingdong/.claude/tmp/claude-1001/-workspace-workspace-mingdong-repo-open-audio-llm/ca9aa6d0-4ecc-4cfc-b526-5322bf6c9de5/scratchpad/ovdiag/probe/mel_oracle_decode.py /workspace/workspace/mingdong/.claude/tmp/claude-1001/-workspace-workspace-mingdong-repo-open-audio-llm/ca9aa6d0-4ecc-4cfc-b526-5322bf6c9de5/scratchpad/ovdiag/probe/../subset /workspace/data/ts-overlap-20261119/task22-mask-probe/meltok-$g.jsonl melirm_tok $g 4 > /workspace/data/ts-overlap-20261119/task22-mask-probe/logs/meltok-$g.log 2>&1 & pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done; chmod -R a+rX /workspace/data/ts-overlap-20261119/task22-mask-probe; echo "mel tok rc=$rc $(date -u +%FT%TZ)"
