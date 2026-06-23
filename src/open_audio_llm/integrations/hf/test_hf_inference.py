#!/usr/bin/env python3
"""
AmphionASR HuggingFace 推理测试脚本

单条音频模式:
  python test_hf_inference.py --audio /path/to/audio.wav
  python test_hf_inference.py --audio /path/to/audio.wav --task asr_zh

Lhotse 批量模式:
  python test_hf_inference.py --recordings recs.jsonl.gz --supervisions sups.jsonl.gz
  python test_hf_inference.py --recordings recs.jsonl.gz --supervisions sups.jsonl.gz -n 50 --output results.jsonl
"""

import argparse
import gzip
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, WhisperFeatureExtractor

try:
    from .constants import (
        DEFAULT_SPEECH_TOKEN,
        END_SPEECH_TOKEN,
        START_SPEECH_TOKEN,
    )
    from .processing_amphion_asr import kaldi_fbank_extract
except ImportError:
    from constants import (
        DEFAULT_SPEECH_TOKEN,
        END_SPEECH_TOKEN,
        START_SPEECH_TOKEN,
    )
    from processing_amphion_asr import kaldi_fbank_extract

MODEL_DIR = "models/Zipformer-noncausal-Qwen2.5-1.5B-Instruct-hf"

CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\\n' + message['content']}}"
    "{% if loop.last %}{{''}}{% else %}{{ '<|im_end|>\\n' }}"
    "{% endif %}"
    "{% endfor %}"
)

TASK_PROMPTS = {
    "asr":    "Transcribe the following audio:",
    "asr_en": "Transcribe the following English audio:",
    "asr_zh": "Transcribe the following Chinese audio:",
}


# ======================================================================
# Helpers
# ======================================================================

def build_prompt(task: str = "asr", custom_prompt: str = None) -> str:
    speech_placeholder = f"{START_SPEECH_TOKEN}{DEFAULT_SPEECH_TOKEN}{END_SPEECH_TOKEN}"
    base = custom_prompt or TASK_PROMPTS.get(task, TASK_PROMPTS["asr"])
    return f"{base}{speech_placeholder}"


def load_audio(path: str, offset: float = 0.0, duration: float = None) -> np.ndarray:
    """Load audio file (or segment), convert to mono 16 kHz."""
    info = sf.info(path)
    sr = info.samplerate
    start_sample = int(offset * sr)
    stop_sample = int((offset + duration) * sr) if duration else None
    audio, sr = sf.read(path, start=start_sample, stop=stop_sample)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    return audio


def read_jsonl_gz(path: str) -> list[dict]:
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def infer_recordings_path(supervisions_path: str) -> str:
    """Infer recordings manifest path from supervisions path.

    aishell_supervisions_test.jsonl.gz      -> aishell_recordings_test.jsonl.gz
    aishell_supervisions_test_punc.jsonl.gz -> aishell_recordings_test_punc.jsonl.gz
                                            -> aishell_recordings_test.jsonl.gz  (fallback)
    """
    p = Path(supervisions_path)
    name = p.name.replace("supervisions", "recordings")
    candidate = p.parent / name
    if candidate.exists():
        return str(candidate)

    # fallback: strip _punc / _raw etc. suffix before .jsonl(.gz)
    stem = p.name
    for ext in (".gz", ".jsonl"):
        stem = stem.removesuffix(ext)
    base = stem.replace("supervisions", "recordings")
    for suffix in ("_punc", "_raw", "_norm"):
        fallback = p.parent / (base.removesuffix(suffix) + "".join(p.suffixes))
        if fallback.exists():
            return str(fallback)

    raise FileNotFoundError(
        f"无法自动推断 recordings 文件，已尝试: {candidate}\n"
        f"请通过 --recordings 手动指定"
    )


def load_lhotse_pairs(recordings_path: str, supervisions_path: str):
    """Return list of dicts from lhotse manifests (supervision-centric).

    Each dict: {id, audio, ref, start, duration}. Supports long recordings
    where each supervision is a short segment within the recording.
    """
    recs = read_jsonl_gz(recordings_path)
    sups = read_jsonl_gz(supervisions_path)

    rec_by_id = {}
    for r in recs:
        rec_by_id[r["id"]] = r["sources"][0]["source"]

    pairs = []
    for s in sups:
        rid = s.get("recording_id", s["id"])
        audio_path = rec_by_id.get(rid)
        if audio_path is None:
            continue
        pairs.append({
            "id": s["id"],
            "audio": audio_path,
            "ref": s.get("text", ""),
            "start": s.get("start", 0.0),
            "duration": s.get("duration", None),
        })
    return pairs


# ======================================================================
# Inference engine
# ======================================================================

class ASREngine:
    def __init__(self, model_dir: str, device: str, dtype: torch.dtype):
        self.device = device
        self.dtype = dtype

        config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        self.feat_type = getattr(config, "feature_extractor_type", "whisper")
        logger.info(f"  特征提取: {self.feat_type}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        if self.feat_type == "whisper":
            self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=True, dtype=dtype, device_map=device,
        )
        self.model.eval()

    def _extract_features(self, audio: np.ndarray):
        """Return (input_features, feature_lens) as (1, T, F) and (1,)."""
        if self.feat_type == "kaldi_fbank":
            feats = kaldi_fbank_extract(audio)  # (T, F)
            return feats.unsqueeze(0), torch.tensor([feats.shape[0]], dtype=torch.long)
        feat_out = self.feature_extractor(
            audio, sampling_rate=16000, return_tensors="pt",
            padding=True, return_attention_mask=True,
        )
        input_features = feat_out["input_features"].transpose(1, 2)
        feature_lens = feat_out["attention_mask"].sum(dim=-1).long()
        return input_features, feature_lens

    @torch.no_grad()
    def transcribe(self, audio: np.ndarray, task: str = "asr",
                   custom_prompt: str = None, max_new_tokens: int = 200) -> str:
        input_features, feature_lens = self._extract_features(audio)

        user_content = build_prompt(task=task, custom_prompt=custom_prompt)
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": ""},
        ]
        text_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
            chat_template=CHAT_TEMPLATE,
        )
        tok_out = self.tokenizer(text_prompt, return_tensors="pt", padding=True, truncation=True)

        dev = self.model.device
        generated_ids = self.model.generate(
            input_ids=tok_out["input_ids"].to(dev),
            attention_mask=tok_out["attention_mask"].to(dev),
            input_features=input_features.to(device=dev, dtype=self.dtype),
            feature_lens=feature_lens.to(dev),
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)


# ======================================================================
# Single-audio mode
# ======================================================================

def run_single(args):
    logger.info("=" * 60)
    logger.info("单条音频推理")
    logger.info("=" * 60)

    audio = load_audio(args.audio)
    logger.info(f"  文件:   {args.audio}")
    logger.info(f"  时长:   {len(audio) / 16000:.2f}s")

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    logger.info(f"\n加载模型: {args.model_dir}")
    engine = ASREngine(args.model_dir, device, dtype)

    t0 = time.time()
    result = engine.transcribe(audio, task=args.task, custom_prompt=args.prompt,
                               max_new_tokens=args.max_new_tokens)
    elapsed = time.time() - t0

    logger.info(f"\n  推理耗时: {elapsed:.2f}s")
    logger.info(f"  结果:     {result}")


# ======================================================================
# Lhotse batch mode
# ======================================================================

def run_lhotse(args):
    pairs = load_lhotse_pairs(args.recordings, args.supervisions)
    total = len(pairs)
    if args.num_samples and args.num_samples < total:
        pairs = pairs[:args.num_samples]
    n = len(pairs)

    logger.info("=" * 60)
    logger.info(f"Lhotse 批量推理  ({n}/{total} 条)")
    logger.info("=" * 60)

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    logger.info(f"加载模型: {args.model_dir}")
    engine = ASREngine(args.model_dir, device, dtype)
    logger.info("")

    results = []
    total_time = 0.0
    total_duration = 0.0

    for i, item in enumerate(pairs):
        utt_id = item["id"]
        ref_text = item["ref"]
        try:
            audio = load_audio(item["audio"], item["start"], item["duration"])
        except Exception as e:
            logger.error(f"[{i+1}/{n}] {utt_id}  *** 加载失败: {e}")
            results.append({"id": utt_id, "audio": item["audio"], "ref": ref_text,
                            "hyp": "", "error": str(e)})
            continue

        dur = len(audio) / 16000
        total_duration += dur

        t0 = time.time()
        hyp = engine.transcribe(audio, task=args.task, custom_prompt=args.prompt,
                                max_new_tokens=args.max_new_tokens)
        elapsed = time.time() - t0
        total_time += elapsed

        match = "✓" if hyp.strip() == ref_text.strip() else "✗"
        logger.info(f"[{i+1}/{n}] {utt_id}  ({dur:.1f}s, {elapsed:.2f}s)  {match}")
        logger.info(f"  REF: {ref_text}")
        logger.info(f"  HYP: {hyp}")

        results.append({"id": utt_id, "audio": item["audio"], "ref": ref_text, "hyp": hyp})

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("汇总")
    logger.info("=" * 60)
    correct = sum(1 for r in results if r.get("hyp", "").strip() == r["ref"].strip() and "error" not in r)
    errors = sum(1 for r in results if "error" in r)
    logger.info(f"  总数:       {n}")
    logger.info(f"  完全匹配:   {correct}/{n - errors}  ({correct / max(n - errors, 1) * 100:.1f}%)")
    logger.info(f"  加载失败:   {errors}")
    logger.info(f"  总音频时长: {total_duration:.1f}s")
    logger.info(f"  总推理耗时: {total_time:.1f}s")
    if total_duration > 0:
        logger.info(f"  RTF:        {total_time / total_duration:.3f}")

    if args.output:
        out_path = Path(args.output)
        with open(out_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info(f"\n结果已保存至: {out_path}")


# ======================================================================
# Entry
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="AmphionASR HF 推理测试 (单条音频 / Lhotse 批量)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input source
    grp = parser.add_argument_group("输入源 (二选一)")
    grp.add_argument("--audio", type=str, help="单条音频文件路径 (wav/flac/mp3)")
    grp.add_argument("--supervisions", type=str,
                     help="Lhotse supervisions manifest (jsonl / jsonl.gz)，自动推断 recordings")
    parser.add_argument("--recordings", type=str, default=None,
                        help="Lhotse recordings manifest (可选，默认从 supervisions 路径推断)")

    # Common
    parser.add_argument("--model-dir", type=str, default=MODEL_DIR, help="HF 模型目录")
    parser.add_argument("--task", type=str, default="asr", choices=list(TASK_PROMPTS.keys()))
    parser.add_argument("--prompt", type=str, default=None, help="自定义 prompt (覆盖 --task)")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "bfloat16", "float32"])

    # Lhotse-specific
    parser.add_argument("-n", "--num-samples", type=int, default=None,
                        help="仅处理前 N 条 (Lhotse 模式)")
    parser.add_argument("--output", type=str, default=None,
                        help="结果输出文件路径 (JSONL)")

    args = parser.parse_args()

    if args.supervisions:
        if not args.recordings:
            args.recordings = infer_recordings_path(args.supervisions)
            logger.info(f"自动推断 recordings: {args.recordings}")
        run_lhotse(args)
    elif args.audio:
        run_single(args)
    else:
        parser.error("请指定 --audio 或 --supervisions")


if __name__ == "__main__":
    main()
