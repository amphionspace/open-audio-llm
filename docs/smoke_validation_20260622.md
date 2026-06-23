# Smoke Validation 2026-06-22

## Problem Restatement

Validate that AmphionASR integration training assets can run through the new
Open Audio-LLM reproduction path as minimal smoke tests.

## Environment

- Conda env: `amphionft`
- Python: 3.11.15
- torch: 2.10.0
- transformers: 4.57.6
- ms-swift: 4.1.0
- deepspeed: 0.18.9
- vLLM: 0.18.0
- GPU used: `CUDA_VISIBLE_DEVICES=2`

## Assets

- Legacy HF checkpoint:
  `/chenmingjie/mingdong/workspace/AmphionASR/exp/qwen3asr_aut_qwen3_4b_continue_sft_ts_hw_v2/v1-20260424-061154/checkpoint-15180_merged`
- Source smoke data:
  `/chenmingjie/mingdong/workspace/AmphionASR/exp/test/v2-20260430-103019/val_dataset.jsonl`
- Generated smoke data:
  `runs/smoke_data/sft_smoke.jsonl`
  `runs/smoke_data/grpo_smoke.jsonl`

## Results

| Path | Result | Artifact |
| --- | --- | --- |
| Legacy HF conversion | PASS | `runs/converted_open_audio_llm_smoke` |
| Model load | PASS | `AutoModelForCausalLM` loaded `AudioLLMForConditionalGeneration` |
| SFT | PASS | `runs/open_audio_llm_sft_smoke/v0-20260622-070959/checkpoint-1` |
| LoRA merge | PASS | `runs/open_audio_llm_sft_smoke/v0-20260622-070959/checkpoint-1_merged` |
| GRPO | PASS | `runs/open_audio_llm_grpo_smoke/v0-20260622-071613/checkpoint-1` |
| Rollout server | PASS | `examples/train/rollout/run_rollout_server.sh` prepared `checkpoint-1_merged_vllm_rollout`; `/health/` returned `{"status":"ok"}` |

## Key Logs

- SFT completed one step with `train_loss=5.26626873` and saved `checkpoint-1`.
- GRPO completed one step and logged:
  - `rewards/ASRFormatReward/mean`
  - `rewards/ASRAccuracyReward/mean`
  - `rewards/HotwordReward/mean`
- LoRA merge reported `Successfully merged LoRA`.
- Rollout server reached `Application startup complete`, `/health/` returned
  `{"status":"ok"}`, and `/get_engine_type/` returned `LLMEngine`.

## Rollout Compatibility Notes

The root cause was in the vLLM runtime layer. vLLM 0.18 rejected
`AudioLLMForConditionalGeneration`, and `python -m
vllm.model_executor.models.registry` segfaulted during model inspection. The
rollout recipe now avoids custom registry registration and prepares a lightweight
vLLM rollout directory whose `architectures` is `TransformersForCausalLM`.

Two model-side fallbacks are required for this generic Transformers path:

- Text-only `forward`/`generate` delegates to the nested language model when no
  `<speech>` token is present.
- vLLM wrapper calls return hidden states from the nested base language model
  instead of CausalLM logits.

This unblocks GRPO rollout server startup and weight-sync smoke validation. A
true vLLM PagedAttention executor for real audio multimodal requests remains an
optimization task.
