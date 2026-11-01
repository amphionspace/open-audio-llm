#!/usr/bin/env bash
# Short draft-head training on the ACP dev node (GPUs 4-7, the GRPO session holds 0-3), no W&B
# (user: validation runs use --report_to none; the formal run goes through configs/train-draft.yaml).
# Usage: train_draft_acp.sh OUTPUT_DIR MAX_STEPS SAVE_STEPS WORKERS_PER_RANK
set -euo pipefail
umask 000
export CUDA_VISIBLE_DEVICES=4,5,6,7
OUT=$1
MAX_STEPS=$2
SAVE_STEPS=$3
WORKERS=$4
REPO=/workspace/workspace/mingdong/repo/oal-mtp
EXP=$REPO/runs/mtp-draft-head-20261031
PY=/workspace/workspace/mingdong/miniconda3/envs/oal-train/bin/python
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export PATH=/workspace/workspace/mingdong/.local-tools/ffmpeg-7.0.2-static/bin:/workspace/workspace/mingdong/miniconda3/envs/oal-train/bin:$PATH
export PYTHONPATH=$REPO/src
mkdir -p "$OUT"
cd "$REPO"
git -c safe.directory="*" rev-parse HEAD > "$OUT/code-commit.txt" || true
git -c safe.directory="*" diff HEAD > "$OUT/code-diff.patch" || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv > "$OUT/gpus-before.csv"
exec "$PY" -m torch.distributed.run --nproc_per_node 4 --master_port 29871 \
  --module open_audio_llm.integrations.ms_swift.train sft \
  --data_config "$EXP/configs/train-data.yaml" \
  --model /workspace/data/packages/amphionasr-1.7b-v3-tsrobust-054avg3/model \
  --model_type amphion_asr_1.7b \
  --external_plugins "$REPO/src/open_audio_llm/integrations/ms_swift/register_qwen3_asr.py" \
  --tuner_type full \
  --torch_dtype bfloat16 --attn_impl sdpa --max_length 2048 \
  --per_device_train_batch_size 12 --per_device_eval_batch_size 4 --gradient_accumulation_steps 2 \
  --draft_head_steps 3 --draft_head_step_decay 0.8 \
  --learning_rate 1e-4 --weight_decay 0.0 --max_steps "$MAX_STEPS" \
  --lr_scheduler_type cosine_with_min_lr --lr_scheduler_kwargs '{"min_lr": 1e-05}' \
  --warmup_ratio 0.0 --warmup_steps 100 \
  --gradient_checkpointing false --vit_gradient_checkpointing false --audio_encoder_batching true \
  --ddp_find_unused_parameters false --dataloader_num_workers "$WORKERS" --performance_logging true \
  --save_steps "$SAVE_STEPS" --eval_strategy no --logging_steps 10 --report_to none --seed 42 \
  --check_model false --load_args false --add_version false \
  --output_dir "$OUT" --ddp_timeout 1800 --max_grad_norm 1.0
