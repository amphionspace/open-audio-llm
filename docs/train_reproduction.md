# Training Reproduction

## Problem Restatement

Open Audio-LLM must reproduce the AmphionASR training workflows formerly under
AmphionASR `src/integrations/scripts/train` while keeping runnable recipes
outside the installable Python package.

## Legacy Mapping

| AmphionASR entry | Open Audio-LLM entry | Purpose |
| --- | --- | --- |
| `/chenmingjie/mingdong/workspace/AmphionASR/src/integrations/scripts/train/sft_swift.sh` | `examples/train/sft/train.sh` | LoRA SFT smoke or short run |
| `/chenmingjie/lx/AmphionASR/src/integrations/scripts/train/sft_swift.sh` | `examples/train/sft/production_swift.sh` | Production-style LoRA SFT |
| `/chenmingjie/lx/AmphionASR/src/integrations/scripts/train/sft_stage1_encoder_aligner.sh` | `examples/train/sft/stage1_encoder_aligner.sh` | Stage 1 encoder + aligner LoRA |
| `/chenmingjie/lx/AmphionASR/src/integrations/scripts/train/sft_stage2_llm.sh` | `examples/train/sft/stage2_llm.sh` | Stage 2 LLM LoRA |
| `/chenmingjie/lx/AmphionASR/src/integrations/scripts/train/sft_stage3_joint.sh` | `examples/train/sft/stage3_joint.sh` | Stage 3 joint LoRA |
| `/chenmingjie/mingdong/workspace/AmphionASR/src/integrations/scripts/train/grpo_swift.sh` | `examples/train/grpo/train.sh` | GRPO smoke or short run |
| `/chenmingjie/lx/AmphionASR/src/integrations/scripts/train/grpo_swift.sh` | `examples/train/grpo/production_swift.sh` | Production-style GRPO |
| `/chenmingjie/mingdong/workspace/AmphionASR/src/integrations/scripts/train/run_rollout_server.sh` | `examples/train/rollout/run_rollout_server.sh` | `swift rollout` server for GRPO server mode |
| `/chenmingjie/mingdong/workspace/AmphionASR/src/integrations/scripts/model/convert_to_hf.sh` | `examples/model/convert_legacy_checkpoint.sh` | Legacy checkpoint to Open Audio-LLM HF directory |
| `/chenmingjie/lx/AmphionASR/src/integrations/scripts/model/convert_to_hf.sh` | `examples/model/convert_amphionasr_checkpoint.sh` | Legacy AmphionASR converter wrapper |
| `/chenmingjie/mingdong/workspace/AmphionASR/src/integrations/scripts/model/merge_lora.sh` | `examples/model/merge_lora.sh` | Merge ms-swift LoRA adapter |
| `/chenmingjie/lx/AmphionASR/src/integrations/scripts/deploy/serve_vllm.sh` | `examples/serve/vllm/serve.sh` | vLLM serve with nested config and Kimi-Audio fix |
| `/chenmingjie/lx/AmphionASR/src/integrations/scripts/eval/eval_vllm.sh` | `examples/eval/vllm/eval.sh` | vLLM batch evaluation |
| `/chenmingjie/lx/AmphionASR/src/integrations/scripts/data/convert_sharegpt.sh` | `examples/data/convert_sharegpt.sh` | Lhotse to ShareGPT conversion |
| `/chenmingjie/lx/AmphionASR/src/integrations/scripts/data/convert_vitw.py` | `examples/data/convert_vitw.py` | Voices-in-the-Wild parquet conversion |

## Smoke-All Procedure

Use AmphionASR workspace assets for the first validation pass. The smoke target
is at least one successful training step per path, not a full training budget.

1. Install the training extras:

   ```bash
   pip install -e ".[hf,swift,data,vllm,dev]"
   ```

2. Convert or point to an Open Audio-LLM HF checkpoint:

   ```bash
   bash examples/model/convert_legacy_checkpoint.sh \
     -c /path/to/legacy.pt \
     -w /path/to/encoder.pth \
     -C /path/to/encoder_config.json \
     -l /path/to/base_llm \
     -o runs/converted_open_audio_llm
   ```

3. Prepare tiny SFT and GRPO JSONL files from existing AmphionASR data. For
   smoke validation, each file can contain a single short audio sample.

4. Run SFT smoke:

   ```bash
   set -a
   source configs/train/sft_smoke.env
   set +a
   bash examples/train/sft/train.sh
   ```

5. Merge the SFT adapter:

   ```bash
   bash examples/model/merge_lora.sh runs/open_audio_llm_sft_smoke/checkpoint-1
   ```

6. Optionally start rollout server for GRPO server mode:

   ```bash
   bash examples/train/rollout/run_rollout_server.sh \
     -m runs/open_audio_llm_sft_smoke/checkpoint-1_merged \
     -p 8006
   ```

   The rollout script prepares a lightweight `${MODEL}_vllm_rollout` directory
   by linking checkpoint files and overriding `architectures` to
   `TransformersForCausalLM` for vLLM 0.18. The original merged checkpoint is
   left unchanged.

7. Run GRPO smoke:

   ```bash
   set -a
   source configs/train/grpo_smoke.env
   set +a
   bash examples/train/grpo/train.sh
   ```

## Pass Criteria

- Checkpoint conversion writes `config.json`, `model.safetensors`, tokenizer
  files, feature extractor files, and an `open_audio_llm/` source copy.
- `config.json` contains `auto_map` entries pointing to `open_audio_llm`.
- SFT reaches at least one training step and writes an adapter checkpoint.
- LoRA merge writes a merged model directory.
- Rollout server starts and reaches a healthy listening state. For ms-swift
  rollout, `GET /health/` should return `{"status":"ok"}` and
  `POST /get_engine_type/` should return `LLMEngine`.
- GRPO reaches at least one training step and logs all three rewards:
  `asr_format_reward`, `asr_accuracy_reward`, and `hotword_reward`.

## Known Limits

- Zipformer/k2 is not a mainline training path for new Open Audio-LLM recipes.
- Static ShareGPT conversion is the fastest compatibility route; dynamic
  `sample_index` plus `LhotseSwiftDataset` remains the longer-term data path.
- Full parity with legacy `/chenmingjie/mingdong/workspace/AmphionASR/src/train.py`
  is out of scope for this reproduction pass.
- The current rollout recipe uses vLLM's generic Transformers backend for
  server startup and GRPO weight-sync smoke tests. Real audio requests still
  need a dedicated vLLM multimodal executor.
