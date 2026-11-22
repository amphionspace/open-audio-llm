#!/usr/bin/env bash
# De-interference maps at tower layer 12, layer 24 and the tower output (GPUs 4-6), then decode the v12 test subset
# (orig mixtures) with each map, two shards per map, plus a no-map HF reproduction (GPU 7).
set -u; P=/workspace/workspace/mingdong/.claude/tmp/claude-1001/-workspace-workspace-mingdong-repo-open-audio-llm/ca9aa6d0-4ecc-4cfc-b526-5322bf6c9de5/scratchpad/ovdiag/probe; F=/workspace/data/ts-overlap-20261119/task17-probe; PY=/workspace/workspace/mingdong/miniconda3/envs/oal-train/bin/python
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=/workspace/workspace/mingdong/repo/oal-clean-teacher/runs/ts-overlap-20261119/shared/code-v4/src
pids=(); g=4
for L in l12 l24 out; do
  CUDA_VISIBLE_DEVICES=$g $PY $P/train_adapter.py $L $F/train $F/adapter-$L.pt 6000 > $F/logs/adapter-$L.log 2>&1 & pids+=($!); g=$((g + 1))
done
CUDA_VISIBLE_DEVICES=7 $PY $P/decode_adapter.py out none $P/test-list.jsonl /workspace/workspace/mingdong/.claude/tmp/claude-1001/-workspace-workspace-mingdong-repo-open-audio-llm/ca9aa6d0-4ecc-4cfc-b526-5322bf6c9de5/scratchpad/ovdiag/subset/manifest.jsonl $F/decode-none-0.jsonl orig 0 1 > $F/logs/decode-none.log 2>&1 & pids+=($!)
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done; echo "adapters rc=$rc $(date -u +%FT%TZ)"; [ $rc = 0 ] || exit 1
pids=(); g=0
for L in l12 l24 out; do for s in 0 1; do
  CUDA_VISIBLE_DEVICES=$g $PY $P/decode_adapter.py $L $F/adapter-$L.pt $P/test-list.jsonl /workspace/workspace/mingdong/.claude/tmp/claude-1001/-workspace-workspace-mingdong-repo-open-audio-llm/ca9aa6d0-4ecc-4cfc-b526-5322bf6c9de5/scratchpad/ovdiag/subset/manifest.jsonl $F/decode-$L-$s.jsonl orig $s 2 > $F/logs/decode-$L-$s.log 2>&1 & pids+=($!); g=$((g + 1))
done; done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done; echo "decode rc=$rc $(date -u +%FT%TZ)"; exit $rc
