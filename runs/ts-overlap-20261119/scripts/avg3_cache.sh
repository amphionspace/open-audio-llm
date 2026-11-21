#!/usr/bin/env bash
# Average checkpoints 5000/5500/6000 of one attempt into the ae model cache and print the cache path (same steps as
# scratch/acp-dev/avg3-cache.sh, which is fixed to ts-robust-acp-20261015 attempts).
# Usage: avg3_cache.sh TASK ATTEMPT NAME   e.g. avg3_cache.sh 03-clean-kl 004 ts-overlap-cleankl-004-avg3
set -u; umask 000
task=$1; a=$2; N=$3; S=/workspace/workspace/mingdong/scratch/acp-dev; M=$S/ae-models.sh
PY=/workspace/workspace/mingdong/miniconda3/envs/oal-train/bin/python
REM=whai:open-audio-llm/runs/ts-overlap-20261119/tasks/$task/attempts/$a/artifacts/training
bash $M fetch $N 2>/dev/null && exit 0
B=(); T=()
for s in 5000 5500 6000; do
  C=$(bash $M fetch "$REM/checkpoint-$s/") || exit 5
  if [ -f "$C/model.safetensors" ]; then B+=("$C"); continue; fi
  W=$(bash $M stage) || exit 5; T+=("$W")
  $PY $S/fp32_shards_to_bf16_into.py "$C" "$W" >&2 || exit 11
  B+=("$W")
done
W=$(bash $M stage) || exit 5
$PY /workspace/workspace/mingdong/scratch/ts033/avg_ckpt.py "$W" "${B[@]}" >&2 || exit 6
$PY $S/add_tied_lm_head.py "$W" >&2 || exit 12
rm -rf "${T[@]}"
bash $M put "$W" --name $N
