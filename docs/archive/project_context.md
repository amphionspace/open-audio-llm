# Project Context

## Problem Restatement

Open Audio-LLM exists to move the legacy source stack into an open-ecosystem
Audio-LLM package while preserving the core research invariant: any compatible
audio encoder can be connected to any compatible Hugging Face causal LM through
an explicit connector and slot-merging layer.

## Source Project Snapshot

The source project is `AmphionASR`.

Important source paths:

- `src/train.py`: legacy multi-task training entry. It owns dataset mux,
  dynamic Lhotse loading, prompt rendering, online hotwords, optimizer,
  checkpointing, validation, and several task-specific training paths.
- `src/model.py`: original composable model. It has the strongest multi-audio
  `<speech>` slot merge implementation and supports TS-ASR-style
  `[enrollment, mixed]` audio ordering.
- `src/asr_datamodule.py`: legacy DynamicBucketing, CutSet transforms, RIR,
  MUSAN, speed perturbation, SpecAugment, and ESC routing.
- `src/esc_mixing.py`: ESC foreground speech mixing policy and metadata
  handling. Training mix is stochastic; validation/test mix should be fixed.
- AmphionASR `src/integrations/huggingface`: HF checkpoint format, config,
  model wrapper, projector, encoder registry, conversion scripts, and smoke
  tests.
- AmphionASR `src/integrations/ms_swift`: ms-swift model registration,
  templates, ShareGPT conversion, hotword utilities, SFT/GRPO scripts, and
  rewards.
- AmphionASR `src/integrations/vllm`: vLLM plugin and serving/test scripts.
- `src/decode.py`, `src/eval_plan.py`, `src/eval_io.py`: legacy evaluation
  and reporting utilities.

## Target Project Snapshot

The target project is `open-audio-llm`, with the Python package
`open_audio_llm`.

Implemented target paths:

- `src/open_audio_llm/configuration_audio_llm.py`: componentized HF config.
- `src/open_audio_llm/modeling_audio_llm.py`: AudioTower + Connector +
  SlotMerger + CausalLM HF model.
- `src/open_audio_llm/audio`: audio tower interfaces and optional Qwen
  adapters.
- `src/open_audio_llm/connectors`: connector interface and pure PyTorch
  `mlp_downsample` connector.
- `src/open_audio_llm/merge`: multi-`<speech>` slot merge.
- `src/open_audio_llm/data`: sample index, dynamic dataset, hotword
  sampling, samplers, augmentation hooks, and ESC foreground policy metadata.
- `src/open_audio_llm/integrations`: HF conversion and legacy compatibility,
  ms-swift shims/rewards/data conversion, and vLLM registration/serving
  helpers.
- `src/open_audio_llm/eval`: lightweight ASR, hotword, and multi-task
  metrics.
- `tests`: dependency-light unit/smoke tests for core contracts.

## Design Invariants

- Default runtime must not require `k2`.
- Zipformer is legacy-only unless explicitly reintroduced through an optional
  adapter.
- Audio encoders must implement the `AudioTower` contract:
  `forward(features, lengths) -> (hidden, hidden_lengths)`.
- vLLM compatibility requires deterministic `get_output_lengths`.
- Connector code must not know task semantics.
- Slot merging must support one or more `<speech>` positions.
- Offline data files store facts; online Dataset/Sampler/Collator own training
  randomization.
- Validation/test random processes must be fixed for reproducibility.

## Current Validation State

Completed checks:

- Python AST syntax check across `src` and `tests`.
- IDE linter check for `open-audio-llm`.

Blocked checks in the current environment:

- `pytest` was unavailable.
- `transformers` was unavailable.
- Runtime HF/ms-swift/vLLM smoke tests were not executed.

## Immediate Risk Layer

The current target project is a scaffold plus contract implementation. It is
not yet a proven replacement for `src/train.py`.

The highest-risk areas are:

- full old `.pt` checkpoint key mapping,
- real Qwen/Omni encoder runtime validation,
- true ms-swift Template integration with audio feature processing,
- optimized vLLM PagedAttention model implementation,
- Lhotse-backed online audio augmentation parity,
- task-level metric parity with legacy `decode.py` and `eval_io.py`.
