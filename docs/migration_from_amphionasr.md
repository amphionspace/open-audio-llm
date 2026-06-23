# Migration From AmphionASR

## Problem Restatement

The migration is not a second top-level `src/integrations/` tree. The goal is
to make Open Audio-LLM an open ecosystem project that preserves AmphionASR's
composable audio-encoder-to-LLM design while moving post-training, deployment,
and data interfaces onto Hugging Face, ms-swift, and vLLM under
`src/open_audio_llm/integrations`.

## Boundary Decisions

- New experiments should use Qwen/Omni or other registered `AudioTower`
  adapters, not Zipformer.
- Zipformer is legacy-only for historical checkpoint conversion or inference.
- `k2` is not a default dependency of this project.
- Legacy `src/train.py` remains the reference for multi-task pretraining until
  the dynamic ms-swift data pipeline reaches parity.
- Static ShareGPT conversion remains a baseline, not the final data pipeline.

## Capability Ownership

| Capability | New Project Owner | Legacy Status |
| --- | --- | --- |
| HF model format | `open_audio_llm.modeling_audio_llm` | replace integration copy |
| ms-swift SFT/GRPO | `open_audio_llm.integrations.ms_swift` | replace AmphionASR `src/integrations` plugin |
| vLLM serving | `integrations.vllm` | replace ASR-specific plugin |
| ASR/hotword rewards | `integrations.ms_swift.rewards` | port existing logic |
| Dynamic data pipeline | `data` package | port from `src/asr_datamodule.py` |
| ESC foreground mix | `data.esc_mix` | port from `src/esc_mixing.py` |
| Full multi-task pretraining | staged migration | legacy `src/train.py` until parity |
| Zipformer training | none | deprecated |

## k2 Exit Criteria

The new project is k2-free when a fresh environment can run:

1. sample index construction,
2. HF model save/load,
3. ms-swift SFT/GRPO smoke tests,
4. vLLM import/serve smoke tests,
5. task evaluation utilities,

without installing `k2` or importing `src/zipformer`.
