# Data Boundary

The data boundary is split into a deterministic offline layer and a stochastic
online layer.

## Offline Sample Index

The offline layer stores sample facts:

- dataset id and task,
- audio path plus optional `start` and `duration`,
- text, language, and labels,
- real hotwords,
- enrollment audio for target-speaker ASR,
- ESC background metadata,
- validation/test fixed mixes.

It should not permanently bake in training-time random hotword distractors,
RIR/noise choices, SpecAugment masks, or ESC foreground speech choices.

## Online Training Layer

The online layer owns training distribution:

- dataset mux weights and repetitions,
- duration-aware batching,
- hotword prompt sampling and hard negatives,
- speed perturbation, RIR, MUSAN, and SpecAugment,
- ESC foreground mix for training,
- GRPO group-level deterministic sampling.

Validation and test random processes must be fixed offline so metrics remain
comparable.
