# Open Audio-LLM

`open-audio-llm` is a composable framework for connecting arbitrary audio
encoders to Hugging Face causal language models. It standardizes the training
and serving path around Hugging Face, ms-swift, and vLLM while keeping audio
towers, connectors, and slot merging as explicit runtime contracts.

The default path intentionally avoids `k2` and Zipformer. Zipformer support is
treated as a legacy adapter for historical checkpoint conversion or inference.

## Core Shape

```text
AudioProcessor -> AudioTower -> Connector -> SlotMerger -> CausalLM
```

- `AudioTower` converts audio features into `(B, T, D)` hidden states.
- `Connector` maps encoder hidden states to the LLM hidden size.
- `SlotMerger` replaces one or more `<speech>` slots with continuous audio
  embeddings.
- `CausalLM` is any Hugging Face causal LM that supports `inputs_embeds`.

## Status

This directory is the migration landing zone. It contains the new package
skeleton, model composition layer, sample-index data boundary, ms-swift/vLLM
integration shims, and evaluation utilities needed to migrate away from the
legacy `src/train.py` path in phases.

## Provenance

Open Audio-LLM incorporates design and migration work originally developed in
AmphionASR. AmphionASR is referenced only as the upstream source project and as
the legacy checkpoint format, not as the current project identity.

## Documentation

- `docs/architecture.md`: component contracts and model composition.
- `docs/data_boundary.md`: offline sample facts vs online training randomness.
- `docs/migration_from_amphionasr.md`: migration boundary and ownership.
- `docs/project_context.md`: source-project and target-project handoff context.
- `docs/remaining_work.md`: unfinished work and next doable tasks.
- `docs/train_reproduction.md`: smoke training reproduction from AmphionASR.
- `docs/vllm_triton_bypass.md`: vLLM Qwen3-ASR Triton audio embedding bypass.
- `docs/compatibility_matrix.md`: supported defaults and optional paths.
- `docs/legacy_deps.md`: k2 and Zipformer legacy dependency policy.
