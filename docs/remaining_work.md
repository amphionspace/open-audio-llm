# Remaining Work

## Problem Restatement

Open Audio-LLM establishes the migration skeleton and core contracts. The
remaining work is to turn that skeleton into a production-grade replacement for
AmphionASR's integration path and, later, selected parts of `src/train.py`.

## Assumptions

- The next phase should prioritize measurable ASR SFT/GRPO and serving value
  before attempting full multi-task replacement.
- `src/train.py` remains the reference implementation until data and metric
  parity are demonstrated.
- New code should keep default imports light and delay optional heavy
  dependencies such as ms-swift, vLLM, qwen_asr, lhotse, and torchaudio.

## Done But Needs Hardening

### Model Layer

- `AudioLLMConfig` exists, but config round-trip should be tested with real
  `AutoConfig.from_pretrained(..., trust_remote_code=True)`.
- `AudioLLMForConditionalGeneration` exists, but it needs runtime tests against
  real Qwen/Qwen-Omni text configs.
- `SlotMerger` supports multiple `<speech>` slots, but needs tests for labels,
  left padding, empty enrollment slots, and mixed ASR/TS-ASR batches.
- Qwen3-ASR and Qwen3-Omni adapters exist as optional adapters, but have not
  been validated in an environment with their actual dependencies.
- `SwooshR` is pure PyTorch, but numerical parity with legacy/k2 behavior has
  not been benchmarked.

### Framework Integrations

- HF conversion entry now includes historical `.pt` key remapping, but still
  needs a real checkpoint dry run.
- ms-swift registration and `AudioLLMTemplate` exist, but a real `swift sft`
  run has not been executed.
- ms-swift audio feature extraction and collator integration exist for smoke
  paths, but need validation against real Qwen/Omni assets.
- ms-swift rollout starts through vLLM's generic Transformers backend for smoke
  validation. The dedicated AudioLLM vLLM executor is still an optimization
  task for real audio multimodal serving.

### Data Layer

- `SampleIndexRecord` exists, but schema should be checked against all active
  datasets in the source project.
- Lhotse recordings/supervisions builder exists, but cuts-file ingestion is
  incomplete.
- Dynamic dataset renders prompts and hotwords, but does not yet load or
  transform waveform tensors.
- Duration-aware batching utilities exist, but need distributed training seed
  and rank behavior.
- ESC foreground mix currently records sampled policy metadata. Actual audio
  rendering needs a Lhotse or torchaudio backend.

### Evaluation Layer

- Lightweight ASR, hotword, and multi-task metrics exist.
- Full compatibility with legacy `src/decode.py`, `src/eval_plan.py`, and
  `src/eval_io.py` remains incomplete.
- SER/SEC/SEPC/SEI/ESC metric definitions need to be aligned with legacy
  reports before replacing source-project evaluation.

## Next Doable Work

### Phase A: Make Core Runtime Testable

1. Install target development extras in a clean environment:
   `pip install -e .[dev,hf]`.
2. Run `pytest tests`.
3. Add a tiny HF save/load test using `AutoModelForCausalLM.from_pretrained`
   with `trust_remote_code=True`.
4. Add label-aware `SlotMerger` tests.
5. Add a fixture for single-audio ASR and dual-audio TS-ASR prompt layouts.

Completion signal:

- Core tests pass without qwen_asr, ms-swift, vLLM, lhotse, or k2.

### Phase B: Finish Legacy Checkpoint Conversion

1. Harden the key mapping now available in
   `src/open_audio_llm/integrations/hf/convert_amphion_to_hf.py`.
2. Map legacy prefixes into:
   `audio_tower`, `connector`, `language_model`, and `slot_merger`.
3. Expand conversion tests beyond prefix remapping to synthetic state dicts.
4. Add a real checkpoint dry run if a small checkpoint is available.

Completion signal:

- A legacy Qwen/Omni checkpoint inherited from AmphionASR can be converted and
  loaded as an `AudioLLMForConditionalGeneration`.

### Phase C: Make ms-swift SFT Real

1. Validate `AudioLLMTemplate` against the installed ms-swift version.
2. Convert sample index records to the exact columns expected by swift.
3. Run a one-batch LoRA SFT smoke test.
4. Preserve `solution`, `task`, and `candidate_hotwords` columns for GRPO.

Completion signal:

- `swift sft` completes a tiny run on a converted AudioLLM checkpoint.

### Phase D: Make GRPO Real

1. Validate reward plugin registration with ms-swift.
2. Ensure group-level hotword sampling is deterministic inside each GRPO group.
3. Add task-aware reward dispatch for ASR, hotword, and format rewards.
4. Run a tiny `swift rlhf --rlhf_type grpo` smoke test.

Completion signal:

- GRPO can score structured ASR completions with hotword-aware rewards.

### Phase E: Replace Static ShareGPT With Dynamic Data

1. Add cuts-file ingestion to `lhotse_index_builder.py`.
2. Implement waveform segment loading from `audio_path`, `start`, and
   `duration`.
3. Add Lhotse or torchaudio-backed hooks for speed perturbation, RIR, MUSAN,
   and SpecAugment.
4. Implement real ESC foreground mix rendering for training.
5. Add distributed sampler behavior with epoch/rank-aware seeds.

Completion signal:

- One sample index can produce different epoch-level augmentations while
  validation/test remains fixed.

### Phase F: Restore Multi-Task Parity

1. Verify task prompts against legacy `TASK_PROMPTS` in `src/train.py`.
2. Add sample-index fields for SER, SEC, SEPC, SEI, ESC, and TS-ASR.
3. Validate multi-audio ordering for TS-ASR:
   `[enrollment, mixed]`.
4. Add task-aware eval plan files.
5. Compare small-run metrics with the source project.

Completion signal:

- ASR, hotword ASR, TS-ASR, SER, SEC, SEPC, SEI, and ESC each have a tiny
  train/eval smoke path.

### Phase G: Optimize vLLM

1. Replace the generic Transformers rollout fallback with a true vLLM model
   executor.
2. Load the language model through vLLM's registered model path.
3. Run audio tower and connector outside the KV-cache path.
4. Support multiple audio slots and `--limit-mm-per-prompt`.
5. Add a single-request audio serving smoke test.

Completion signal:

- vLLM serves an AudioLLM checkpoint with real audio inputs and no source-project
  imports, beyond the current text-only rollout startup fallback.

### Phase H: Decommission Legacy Dependencies

1. Confirm all target runtime paths work without `k2`.
2. Keep Zipformer behind an optional legacy adapter only if historical models
   still need it.
3. Remove any accidental imports from `src/train.py`, `src/zipformer`, or
   `src/asr_datamodule.py`.
4. Record final migration status in `docs/migration_from_amphionasr.md`.

Completion signal:

- A fresh target-project environment can train, evaluate, and serve the main
  Qwen/Omni AudioLLM path without installing k2.

## Known Non-Goals For Now

- Do not make the new project depend on `src/train.py`.
- Do not reintroduce k2 as a default dependency for consistency.
- Do not make Zipformer a first-class new-experiment path.
- Do not bake online augmentation randomness into static JSONL.
- Do not claim replacement parity until metrics are compared against legacy
  evaluation.
