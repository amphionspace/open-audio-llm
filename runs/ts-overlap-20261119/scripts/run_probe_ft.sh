#!/usr/bin/env bash
# Component capacity test: labels with 11.17 punctuation, then enc / dec / both fine-tunes (2 GPUs each, 3000 steps of
# 16 samples), then HF decoding of the v12 test subset (orig + clean) per arm; GPU 6 decodes 11.17 itself on clean.
set -u; P=/workspace/workspace/mingdong/.claude/tmp/claude-1001/-workspace-workspace-mingdong-repo-open-audio-llm/ca9aa6d0-4ecc-4cfc-b526-5322bf6c9de5/scratchpad/ovdiag/probe; F=/workspace/data/ts-overlap-20261119/task17-probe; D=$F/scenes100k; PY=/workspace/workspace/mingdong/miniconda3/envs/oal-train/bin/python
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=/workspace/workspace/mingdong/repo/oal-clean-teacher/runs/ts-overlap-20261119/shared/code-v4/src
source /workspace/workspace/mingdong/.config/open-audio-llm/wandb-cluster.sh >/dev/null 2>&1
export WANDB_HTTP_PROXY=http://10.119.17.252:17922 WANDB_HTTPS_PROXY=http://10.119.17.252:17922 WANDB_DIR=$F
python3 $P/punct_labels.py $D/manifest.jsonl "$F/teacher-*.jsonl" $D/manifest-labels.jsonl || exit 2
CUDA_VISIBLE_DEVICES=6 $PY $P/decode_adapter.py out none $P/test-list.jsonl /workspace/workspace/mingdong/.claude/tmp/claude-1001/-workspace-workspace-mingdong-repo-open-audio-llm/ca9aa6d0-4ecc-4cfc-b526-5322bf6c9de5/scratchpad/ovdiag/subset/manifest.jsonl $F/decode-none-clean.jsonl clean 0 1 > $F/logs/decode-none-clean.log 2>&1 &
nc=$!
pids=(); i=0
for A in enc dec both; do
  CUDA_VISIBLE_DEVICES=$((2 * i)),$((2 * i + 1)) /workspace/workspace/mingdong/miniconda3/envs/oal-train/bin/torchrun --nproc_per_node 2 --master_port $((29700 + i))     $P/ft_component.py $A $D/manifest-labels.jsonl $D/wav $F/ft-$A.pt 3000 8 1 > $F/logs/ft-$A.log 2>&1 &
  pids+=($!); i=$((i + 1))
done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done; echo "ft rc=$rc $(date -u +%FT%TZ)"; [ $rc = 0 ] || exit 1
pids=(); g=0
for A in enc dec both; do for s in 0 1; do
  CUDA_VISIBLE_DEVICES=$g $PY $P/decode_adapter.py ft $F/ft-$A.pt $P/test-list.jsonl /workspace/workspace/mingdong/.claude/tmp/claude-1001/-workspace-workspace-mingdong-repo-open-audio-llm/ca9aa6d0-4ecc-4cfc-b526-5322bf6c9de5/scratchpad/ovdiag/subset/manifest.jsonl $F/decode-ft-$A-$s.jsonl orig,clean $s 2 > $F/logs/decode-ft-$A-$s.log 2>&1 & pids+=($!); g=$((g + 1))
done; done
wait $nc || rc=1; for p in "${pids[@]}"; do wait $p || rc=1; done; echo "decode rc=$rc $(date -u +%FT%TZ)"; exit $rc
