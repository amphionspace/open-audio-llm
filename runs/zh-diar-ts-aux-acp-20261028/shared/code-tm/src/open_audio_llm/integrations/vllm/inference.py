#!/usr/bin/env python3
"""AmphionASR vLLM inference script.

Prerequisites:
    pip install vllm
    cd src/open_audio_llm/integrations/vllm/plugin && pip install -e .

Usage:
    python src/open_audio_llm/integrations/vllm/inference.py --audio /path/to/audio.wav
    python src/open_audio_llm/integrations/vllm/inference.py --audio /path/to/audio.wav --task asr_zh
"""

import argparse
import time

import soundfile as sf
from loguru import logger
from vllm import LLM, SamplingParams

MODEL_DIR = "models/Qwen3-ASR-AuT-Qwen3-4B-Instruct-hf"

DEFAULT_SPEECH_TOKEN = "<speech>"
START_SPEECH_TOKEN = "<start_speech>"
END_SPEECH_TOKEN = "<end_speech>"

TASK_PROMPTS = {
    "asr": "Transcribe the following audio:",
    "asr_en": "Transcribe the following English audio:",
    "asr_zh": "Transcribe the following Chinese audio:",
}

def build_prompt(task: str = "asr", custom_prompt: str = None) -> str:
    """Build a prompt string with audio placeholder tokens.

    For offline inference the special tokens are embedded directly in the
    string.  The tokenizer's chat template handles them transparently.
    """
    speech_placeholder = (
        f"{START_SPEECH_TOKEN}{DEFAULT_SPEECH_TOKEN}{END_SPEECH_TOKEN}"
    )
    base = custom_prompt or TASK_PROMPTS.get(task, TASK_PROMPTS["asr"])
    return f"{base}{speech_placeholder}"


def main():
    parser = argparse.ArgumentParser(description="AmphionASR vLLM inference")
    parser.add_argument(
        "--audio", type=str, required=True, help="Audio file path",
    )
    parser.add_argument(
        "--model-dir", type=str, default=MODEL_DIR, help="HF model dir",
    )
    parser.add_argument(
        "--task", type=str, default="asr",
        choices=list(TASK_PROMPTS.keys()),
    )
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--max-tokens", type=int, default=200)
    args = parser.parse_args()

    # 1. Load audio
    logger.info("=" * 60)
    logger.info("STEP 1: Loading audio")
    logger.info("=" * 60)
    audio, sr = sf.read(args.audio)
    logger.info(f"  File:        {args.audio}")
    logger.info(f"  Sample rate: {sr} Hz")
    logger.info(f"  Duration:    {len(audio) / sr:.2f} s")

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000
        logger.info("  Resampled to 16 kHz")

    # 2. Build prompt
    user_content = build_prompt(task=args.task, custom_prompt=args.prompt)
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": ""},
    ]

    # 3. Initialize vLLM
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Loading model with vLLM")
    logger.info("=" * 60)
    t0 = time.time()

    llm = LLM(
        model=args.model_dir,
        trust_remote_code=True,
        limit_mm_per_prompt={"audio": 1},
    )
    logger.info(f"  Load time: {time.time() - t0:.1f}s")

    # 4. Build prompt text using tokenizer (uses the built-in chat template)
    tokenizer = llm.get_tokenizer()
    text_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    logger.info(f"  Prompt: {repr(text_prompt[:100])}...")

    # 5. Run inference
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Running inference")
    logger.info("=" * 60)

    sampling_params = SamplingParams(
        temperature=0,
        top_p=1.0,
        max_tokens=args.max_tokens,
    )

    t0 = time.time()
    outputs = llm.generate(
        [
            {
                "prompt": text_prompt,
                "multi_modal_data": {
                    "audio": (audio, sr),
                },
            }
        ],
        sampling_params=sampling_params,
    )
    gen_time = time.time() - t0

    output_text = outputs[0].outputs[0].text
    num_tokens = len(outputs[0].outputs[0].token_ids)

    logger.info(f"  Generated tokens: {num_tokens}")
    logger.info(f"  Inference time:   {gen_time:.2f}s")

    # 6. Result
    logger.info("\n" + "=" * 60)
    logger.info("Result")
    logger.info("=" * 60)
    logger.info(f"\n  {output_text}\n")


if __name__ == "__main__":
    main()
