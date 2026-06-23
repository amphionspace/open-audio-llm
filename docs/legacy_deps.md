# Legacy Dependencies

Open Audio-LLM does not require `k2` in its default or training dependencies.

## Decisions

- `SwooshR` is implemented in pure PyTorch in
  `open_audio_llm.connectors.mlp_downsample`.
- Zipformer training is not a supported mainline path.
- Historical Zipformer checkpoints should be converted or served through a
  legacy extra only.
- Legacy `src/train.py` remains outside this package and should not be imported
  by new runtime modules.

## Counterfactual

If `k2` stays in the default path, every HF/ms-swift/vLLM user inherits CUDA and
icefall compatibility risk even when they train only Qwen/Omni-based models.
Removing it from the default project reduces the environment failure domain
without blocking explicit legacy recovery work.
