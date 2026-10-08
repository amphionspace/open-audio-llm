#!/usr/bin/env bash
# 全参数 target SOT。dialog 和 long 用 Emilia2 20261004，会议、合成、short 仍用 20260930。
# 权重来自 speaker-events checkpoint-21000，不恢复它的 optimizer。
# 该 checkpoint 的音频 encoder 是窗口注意力：n_window=50，n_window_infer=800，约 8 秒。
# 卷积仍按 100 帧切，权重不改；启动后把 n_window_infer 设成 10^9，整段音频落在同一个注意力块里。
# 动态 batch 由数据配置的 batching.max_duration 决定，每条样本按 metadata.recipe.k 取注册。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
export MODEL="${MODEL:-/222042021/mingdong/workspace/open-audio-llm/runs/qwen3-asr-sot-speaker-events-20260922/training/checkpoint-21000}"
export DATA_CONFIG="${DATA_CONFIG:-$REPO_ROOT/examples/configs/data/target_sot_recipe_20261004.yaml}"
export AUDIO_DATA_CONTRACT_ROOT="${AUDIO_DATA_CONTRACT_ROOT:-$(dirname "$REPO_ROOT")/audio-data-contract}"
export TRAIN_CATALOG="${TRAIN_CATALOG:-/ai_sds_wuzz/DATA_ASR/Derived/target-sot-recipe-20260930/catalog.jsonl}"
export EMILIA_CATALOG="${EMILIA_CATALOG:-/ai_sds_wuzz/DATA_ASR/Derived/target-sot-emilia-20261004/catalog.jsonl}"
export EVAL_CATALOG="${EVAL_CATALOG:-/ai_sds_wuzz/DATA_ASR/Derived/target-sot-supervision-20261001/catalog.jsonl}"
export EMILIA_EVAL_CATALOG="${EMILIA_EVAL_CATALOG:-/ai_sds_wuzz/DATA_ASR/Derived/target-sot-supervision-20261004/catalog.jsonl}"
if [[ -z "${AUDIO_DATA_CATALOG:-}" ]]; then
  export AUDIO_DATA_CATALOG="/ai_sds_wuzz/DATA_ASR/Derived/target-sot-emilia-20261004/catalog-with-train.jsonl"
  cat "$TRAIN_CATALOG" "$EMILIA_CATALOG" "$EVAL_CATALOG" "$EMILIA_EVAL_CATALOG" > "$AUDIO_DATA_CATALOG"
fi
export AUDIO_DATA_ROOTS_FILE="${AUDIO_DATA_ROOTS_FILE:-/222042021/mingdong/data/sot-multispeaker/synthetic-v2-20260915/roots.json}"
export AUDIO_DATA_METADATA_CACHE="${AUDIO_DATA_METADATA_CACHE:-$REPO_ROOT/runs/target-sot-recipe-20261004-metadata}"
export OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/runs/target-sot-recipe-20261004}"
# 6 卡、梯度累积 2。一轮 608868 个 microbatch，每卡 101478 个，正好 50739 步。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-6}"
export MAX_STEPS="${MAX_STEPS:-50739}"
export LLM_LR="${LLM_LR:-1e-5}"
export ENCODER_LR="${ENCODER_LR:-1e-5}"
export ALIGNER_LR="${ALIGNER_LR:-1e-5}"
export MIN_LR="${MIN_LR:-1e-6}"
_GPU_COUNT="$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")"
if [[ "$_GPU_COUNT" -ne "$NPROC_PER_NODE" ]]; then
  echo "CUDA_VISIBLE_DEVICES 有 ${_GPU_COUNT} 张卡，NPROC_PER_NODE=${NPROC_PER_NODE}" >&2
  exit 1
fi
if [[ -e "$OUTPUT_DIR/args.json" ]] || compgen -G "$OUTPUT_DIR/checkpoint-*" >/dev/null; then
  echo "OUTPUT_DIR 里已经有一次训练: $OUTPUT_DIR" >&2
  exit 1
fi
export PYTHONPATH="$REPO_ROOT/src:$AUDIO_DATA_CONTRACT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MASTER_PORT="${MASTER_PORT:-29631}"

"${PYTHON:-python}" - "$MODEL" <<'PY'
import json
import sys
from pathlib import Path

audio = json.loads((Path(sys.argv[1]) / "config.json").read_text())["thinker_config"]["audio_config"]
n_window, infer = int(audio["n_window"]), int(audio["n_window_infer"])
chunk = n_window * 2
print(
    f"encoder n_window={n_window} n_window_infer={infer} "
    f"conv_chunk_frames={chunk} attention_window_seconds={infer / 100:.1f}",
    flush=True,
)
if chunk != 100:
    raise SystemExit("卷积块不是 100 帧，不能只改注意力窗")
if infer < 60000:
    print("当前是窗口注意力。训练进程会把 n_window_infer 设成 10^9，不改权重。", flush=True)
else:
    print("当前注意力窗已经覆盖 600 秒以上。训练进程仍会把 n_window_infer 设成 10^9。", flush=True)
PY

"${PYTHON:-python}" -m open_audio_llm.data.catalog_cache \
  --data_config "$DATA_CONFIG" --workers "${METADATA_WORKERS:-8}" \
  --preflight-report "$OUTPUT_DIR/data-preflight.json" \
  --batch-size 1 --world-size "$NPROC_PER_NODE"

# 现有 10 分钟样本整段最长约 20441 token。16384 会丢掉这些样本。
exec "${PYTHON:-python}" -m torch.distributed.run \
  --nproc_per_node "$NPROC_PER_NODE" --master_port "$MASTER_PORT" \
  --module open_audio_llm.integrations.ms_swift.train sft \
  --model "$MODEL" \
  --model_type amphion_asr_1.7b \
  --external_plugins "$REPO_ROOT/src/open_audio_llm/integrations/ms_swift/register_qwen3_asr.py" \
  --data_config "$DATA_CONFIG" \
  --tuner_type full \
  --freeze_vit false --freeze_aligner false --freeze_llm false \
  --learning_rate "$LLM_LR" --vit_lr "$ENCODER_LR" --aligner_lr "$ALIGNER_LR" \
  --audio_encoder_parallel false --audio_encoder_batching false --audio_encoder_global true \
  --torch_dtype bfloat16 --bf16 true --fp16 false --attn_impl sdpa \
  --max_length 32768 --truncation_strategy delete \
  --per_device_train_batch_size 1 --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps "${GRAD_ACCUM:-2}" \
  --max_steps "$MAX_STEPS" \
  --lr_scheduler_type cosine_with_min_lr --lr_scheduler_kwargs "{\"min_lr\": ${MIN_LR}}" \
  --warmup_ratio 0.03 \
  --gradient_checkpointing true --vit_gradient_checkpointing true \
  --ddp_find_unused_parameters false --ddp_timeout 7200 \
  --dataloader_num_workers "${DATALOADER_WORKERS:-4}" --dataloader_persistent_workers true \
  --eval_strategy steps --eval_steps 1000 \
  --save_steps 1000 --save_total_limit 3 --save_only_model false \
  --logging_steps 10 --report_to none --seed 42 \
  --performance_logging true \
  --output_dir "$OUTPUT_DIR" \
  "$@"
