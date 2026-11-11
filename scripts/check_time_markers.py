"""Check that training and vLLM serving build the same audio span with time markers, for one real clip.

    # 1. training template (oal-train env, ms-swift): replace_tag output tokenized with the checkpoint tokenizer
    OPEN_AUDIO_LLM_AUDIO_TIME_MARKER_SECONDS=5 python check_time_markers.py train CKPT WAV OUT.json
    # 2. vLLM processor (vllm-serving env, plugin registered from this source tree): prompt token ids of the same clip
    OPEN_AUDIO_LLM_AUDIO_TIME_MARKER_SECONDS=5 python check_time_markers.py vllm CKPT WAV OUT.json

Each mode writes the token ids from <|audio_start|> through <|audio_end|>; mode ``vllm`` also compares with
``OUT.json`` of mode ``train`` when it exists next to it (``--against``).
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


def load(wav):
    x, sr = sf.read(wav, dtype="float32", always_2d=True)
    assert sr == 16000, sr
    return x.mean(axis=1)


def train(ckpt, wav, out):
    from types import SimpleNamespace

    from transformers import AutoTokenizer

    from open_audio_llm.integrations.ms_swift import register_qwen3_asr as module

    template = object.__new__(module.Qwen3ASRTemplate)
    template._hop_length = 160
    module.Qwen3ASRTemplate._mix_source = lambda self, inputs: None
    tags = template.replace_tag("audio", 0, SimpleNamespace(audios=[wav], _audio_llm_waveforms=[load(wav)]))
    ids = AutoTokenizer.from_pretrained(ckpt).encode("".join(tags), add_special_tokens=False)
    Path(out).write_text(json.dumps({"ids": ids, "pads": tags.count("<|audio_pad|>")}))
    print("train span", len(ids), "pads", tags.count("<|audio_pad|>"))


def vllm(ckpt, wav, out, against=None):
    from vllm.config import ModelConfig
    from vllm.multimodal import MULTIMODAL_REGISTRY

    from open_audio_llm.integrations.vllm.plugin import register

    register()
    config = ModelConfig(model=ckpt, tokenizer=ckpt, dtype="bfloat16", limit_mm_per_prompt={"audio": 1})
    processor = MULTIMODAL_REGISTRY.create_processor(config)
    print("processor", type(processor).__name__)
    prompt = ("<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|>"
              "<|im_end|>\n<|im_start|>assistant\n")
    from vllm.multimodal.processing import ProcessorInputs
    from vllm.multimodal.processing.context import TimingContext

    items = processor.info.parse_mm_data({"audio": [(load(wav), 16000)]})
    result = processor.apply(ProcessorInputs(prompt, items), TimingContext(enabled=False))
    ids = list(result["prompt_token_ids"])
    tokenizer = processor.info.get_tokenizer()
    start, end = tokenizer.convert_tokens_to_ids("<|audio_start|>"), tokenizer.convert_tokens_to_ids("<|audio_end|>")
    span = ids[ids.index(start): ids.index(end) + 1]
    pad = tokenizer.convert_tokens_to_ids("<|audio_pad|>")
    placeholder = result["mm_placeholders"]["audio"][0]
    embeds = int(np.asarray(placeholder.is_embed).sum()) if placeholder.is_embed is not None else placeholder.length
    Path(out).write_text(json.dumps({"ids": span, "pads": span.count(pad), "embed_positions": embeds}))
    print("vllm span", len(span), "pads", span.count(pad), "embed positions", embeds)
    if against:
        ref = json.loads(Path(against).read_text())
        same = ref["ids"] == span
        print("identical to training span:", same)
        if not same:
            diff = next(i for i, (a, b) in enumerate(zip(ref["ids"], span)) if a != b)
            print("first difference at", diff, ref["ids"][diff - 3: diff + 3], span[diff - 3: diff + 3])
            sys.exit(1)


if __name__ == "__main__":
    mode, ckpt, wav, out, *rest = sys.argv[1:]
    {"train": train, "vllm": vllm}[mode](ckpt, wav, out, *rest)
