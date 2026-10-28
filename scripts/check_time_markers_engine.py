"""Engine-level check (the server path, multimodal processor cache on): generate on one clip with time markers and
compare the prompt's audio span with the training template's (check_time_markers.py train output).

    OPEN_AUDIO_LLM_AUDIO_TIME_MARKER_SECONDS=5 vllm-serving/bin/python check_time_markers_engine.py CKPT WAV TRAIN.json
"""
import json
import sys
from pathlib import Path

import soundfile as sf


def main(ckpt, wav, train):
    from open_audio_llm.integrations.vllm.plugin import register

    register()
    from vllm import LLM, SamplingParams

    x, sr = sf.read(wav, dtype="float32")
    llm = LLM(model=ckpt, tokenizer=ckpt, dtype="bfloat16", gpu_memory_utilization=0.3, max_model_len=8192,
              limit_mm_per_prompt={"audio": 1}, enforce_eager=True)
    prompt = ("<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|>"
              "<|im_end|>\n<|im_start|>assistant\n")
    for attempt in (1, 2):  # second request hits the processor cache
        out = llm.generate({"prompt": prompt, "multi_modal_data": {"audio": (x, sr)}},
                           SamplingParams(temperature=0, max_tokens=200))[0]
        ids = list(out.prompt_token_ids)
        tok = llm.get_tokenizer()
        s, e = ids.index(tok.convert_tokens_to_ids("<|audio_start|>")), ids.index(tok.convert_tokens_to_ids("<|audio_end|>"))
        span = ids[s: e + 1]
        ref = json.loads(Path(train).read_text())["ids"]
        print(f"request {attempt}: span {len(span)} identical to training: {span == ref}; text: {out.outputs[0].text[:160]!r}")


if __name__ == "__main__":  # vLLM spawns the engine process, which re-imports this file
    main(*sys.argv[1:4])
