#!/usr/bin/env bash
# tasks/21: end-to-end arms on one node (4 GPUs each), 8000 steps, then HF decoding of the heavy-overlap set (orig + clean)
# and the side-effect set for both arms (2 GPUs each) plus 11.17 on the side-effect set.
set -u; mkdir -p /workspace/data/ts-overlap-20261119/task21-e2e/logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 AUDIO_DATA_ROOTS_FILE=/workspace/data/registry/roots.json HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/workspace/workspace/mingdong/repo/oal-clean-teacher/runs/ts-overlap-20261119/shared/code-v4/src
source /workspace/workspace/mingdong/.config/open-audio-llm/wandb-cluster.sh >/dev/null 2>&1
export WANDB_HTTP_PROXY=http://10.119.17.252:17922 WANDB_HTTPS_PROXY=http://10.119.17.252:17922 WANDB_DIR=/workspace/data/ts-overlap-20261119/task21-e2e
TR=/workspace/workspace/mingdong/miniconda3/envs/oal-train/bin/torchrun; PY=/workspace/workspace/mingdong/miniconda3/envs/oal-train/bin/python
CUDA_VISIBLE_DEVICES=0,1,2,3 $TR --nproc_per_node 4 --master_port 29901 /workspace/workspace/mingdong/.claude/tmp/claude-1001/-workspace-workspace-mingdong-repo-open-audio-llm/ca9aa6d0-4ecc-4cfc-b526-5322bf6c9de5/scratchpad/ovdiag/probe/ft_e2e.py ce /workspace/data/ts-overlap-20261119/task21-e2e/ce 8000 8 10 2.0 001 4000 > /workspace/data/ts-overlap-20261119/task21-e2e/logs/train-ce.log 2>&1 & a=$!
CUDA_VISIBLE_DEVICES=4,5,6,7 $TR --nproc_per_node 4 --master_port 29902 /workspace/workspace/mingdong/.claude/tmp/claude-1001/-workspace-workspace-mingdong-repo-open-audio-llm/ca9aa6d0-4ecc-4cfc-b526-5322bf6c9de5/scratchpad/ovdiag/probe/ft_e2e.py aux /workspace/data/ts-overlap-20261119/task21-e2e/aux 8000 8 10 2.0 001 4000 > /workspace/data/ts-overlap-20261119/task21-e2e/logs/train-aux.log 2>&1 & b=$!
wait $a; ra=$?; wait $b; rb=$?; echo "train ce rc=$ra aux rc=$rb $(date -u +%FT%TZ)"
chmod -R a+rX /workspace/data/ts-overlap-20261119/task21-e2e
export OMP_NUM_THREADS=8
pids=(); g=0
for A in ce aux; do
  [ -f /workspace/data/ts-overlap-20261119/task21-e2e/$A/step8000.pt ] || continue
  for s in 0 1; do
    CUDA_VISIBLE_DEVICES=$g $PY /workspace/workspace/mingdong/.claude/tmp/claude-1001/-workspace-workspace-mingdong-repo-open-audio-llm/ca9aa6d0-4ecc-4cfc-b526-5322bf6c9de5/scratchpad/ovdiag/probe/decode_adapter.py ft /workspace/data/ts-overlap-20261119/task21-e2e/$A/step8000.pt /workspace/workspace/mingdong/.claude/tmp/claude-1001/-workspace-workspace-mingdong-repo-open-audio-llm/ca9aa6d0-4ecc-4cfc-b526-5322bf6c9de5/scratchpad/ovdiag/probe/test-list.jsonl /workspace/workspace/mingdong/.claude/tmp/claude-1001/-workspace-workspace-mingdong-repo-open-audio-llm/ca9aa6d0-4ecc-4cfc-b526-5322bf6c9de5/scratchpad/ovdiag/probe/../subset/manifest.jsonl /workspace/data/ts-overlap-20261119/task21-e2e/decode-$A-heavy-$s.jsonl orig,clean $s 2 > /workspace/data/ts-overlap-20261119/task21-e2e/logs/decode-$A-heavy-$s.log 2>&1 &
    pids+=($!); g=$((g + 1))
  done
  CUDA_VISIBLE_DEVICES=$g $PY /workspace/workspace/mingdong/.claude/tmp/claude-1001/-workspace-workspace-mingdong-repo-open-audio-llm/ca9aa6d0-4ecc-4cfc-b526-5322bf6c9de5/scratchpad/ovdiag/probe/decode_adapter.py ft /workspace/data/ts-overlap-20261119/task21-e2e/$A/step8000.pt /workspace/workspace/mingdong/.claude/tmp/claude-1001/-workspace-workspace-mingdong-repo-open-audio-llm/ca9aa6d0-4ecc-4cfc-b526-5322bf6c9de5/scratchpad/ovdiag/probe/safety-list.jsonl /workspace/data/ts-overlap-20261119/task19-tse/safety/manifest.jsonl /workspace/data/ts-overlap-20261119/task21-e2e/decode-$A-safety.jsonl mix 0 1 > /workspace/data/ts-overlap-20261119/task21-e2e/logs/decode-$A-safety.log 2>&1 &
  pids+=($!); g=$((g + 1))
done
CUDA_VISIBLE_DEVICES=6 $PY /workspace/workspace/mingdong/.claude/tmp/claude-1001/-workspace-workspace-mingdong-repo-open-audio-llm/ca9aa6d0-4ecc-4cfc-b526-5322bf6c9de5/scratchpad/ovdiag/probe/decode_adapter.py out none /workspace/workspace/mingdong/.claude/tmp/claude-1001/-workspace-workspace-mingdong-repo-open-audio-llm/ca9aa6d0-4ecc-4cfc-b526-5322bf6c9de5/scratchpad/ovdiag/probe/safety-list.jsonl /workspace/data/ts-overlap-20261119/task19-tse/safety/manifest.jsonl /workspace/data/ts-overlap-20261119/task21-e2e/decode-1117-safety.jsonl mix 0 1 > /workspace/data/ts-overlap-20261119/task21-e2e/logs/decode-1117-safety.log 2>&1 & pids+=($!)
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done; chmod -R a+rX /workspace/data/ts-overlap-20261119/task21-e2e; echo "decode rc=$rc $(date -u +%FT%TZ)"
