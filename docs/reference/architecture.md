# Architecture

The project treats Audio-LLM training as component composition rather than a
single fixed ASR architecture.

```mermaid
flowchart TD
    audioInput[AudioInput] --> audioProcessor[AudioProcessor]
    audioProcessor --> audioTower[AudioTower]
    audioTower --> connector[Connector]
    textInput[TextInput] --> tokenizer[Tokenizer]
    tokenizer --> slotMerger[SlotMerger]
    connector --> slotMerger
    slotMerger --> causalLM[CausalLM]
    causalLM --> output[TextOutput]
```

The Catalog training path is deliberately layered:

```mermaid
flowchart LR
    catalog[DatasetCatalog] --> cutset[Lhotse CutSet]
    cutset --> record[AudioRecord]
    record --> renderer[Versioned Prompt Renderer]
    renderer --> example[AudioExample]
    example --> swift[ms-swift adapter]
```

Lhotse owns audio manifests, segment access, and decoding; the online dataset applies waveform and feature augmentation. `AudioRecord`
owns stable task facts and ordered audio references. `AudioExample` owns the
rendered multimodal message. No layer stores machine-specific absolute paths
as part of its portable contract.

## Component Contracts

- `AudioTower.forward(features, lengths)` returns `(hidden, hidden_lengths)`.
- `AudioTower.get_output_lengths(input_lengths)` must be deterministic so vLLM
  can reserve prompt-replacement space.
- `Connector.forward(hidden, hidden_lengths)` returns LLM-width embeddings and
  updated lengths after any temporal downsampling.
- `SlotMerger` replaces each `<speech>` placeholder with the corresponding
  audio embedding sequence. Multi-audio prompts are first-class.
- `CausalLM` is loaded through `AutoModelForCausalLM` and receives
  `inputs_embeds`.

## Extension Rule

Adding a new audio encoder should require a new `AudioTower` adapter and config
entry only. It should not require editing the core model forward path.
