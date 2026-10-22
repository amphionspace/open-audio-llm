"""HF Qwen3-ASR TS-ASR inputs, batched generation and completion log-probs.

Inputs follow the SFT template (``register_qwen3_asr`` TS path): enrollment and
mixture are padded to whole Mel hops and get their own Mel, the features are
concatenated with ``enroll_n_frames``, and the prompt holds conv(enroll) + 1
SEP + conv(mix) audio pads. The audio tower uses one attention window, as the
TS-ASR vLLM service does (``n_window_infer`` override).
"""

from __future__ import annotations

import numpy as np
import torch

from open_audio_llm.tsasr.global_audio_attn import GLOBAL_N_WINDOW_INFER
from open_audio_llm.tsasr.sep_token import audio_token_count, extract_enroll_mix_mel, load_qwen3_asr
from open_audio_llm.tsasr.ts_prompt import TS_CONCAT_SYSTEM

SAMPLE_RATE = 16000
MEL_HOP = 160
EOS_IDS = (151643, 151645)  # <|endoftext|>, <|im_end|> (generation_config.json)
PAD_ID = 151643
AUDIO_START, AUDIO_PAD, AUDIO_END = "<|audio_start|>", "<|audio_pad|>", "<|audio_end|>"
DEFAULT_SYSTEM = TS_CONCAT_SYSTEM


def load(model_dir: str, dtype: torch.dtype, device):
    """Model with the trained SEP attached, and its processor.

    Load each copy separately: ``attach_sep_token`` installs closures bound to
    the loaded thinker, so a ``deepcopy`` would still route through the
    original and lose the enrollment/mixture split.
    """
    import qwen_asr  # noqa: F401  registers the Qwen3-ASR classes
    from transformers import AutoProcessor

    model = load_qwen3_asr(model_dir, torch_dtype=dtype).to(device)
    model.thinker.audio_tower.n_window_infer = GLOBAL_N_WINDOW_INFER
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    return model, processor


def _hop_pad(wav) -> np.ndarray:
    wav = np.ascontiguousarray(wav, dtype=np.float32).reshape(-1)
    return np.pad(wav, (0, -len(wav) % MEL_HOP))


def build_inputs(processor, enroll, mix, system: str = DEFAULT_SYSTEM) -> dict:
    feat, mask, enroll_n = extract_enroll_mix_mel(
        processor.feature_extractor, _hop_pad(enroll), _hop_pad(mix), SAMPLE_RATE)
    n_tokens = audio_token_count(int(feat.shape[-1]), True, int(enroll_n))
    text = (f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n"
            f"{AUDIO_START}{AUDIO_PAD * n_tokens}{AUDIO_END}<|im_end|>\n<|im_start|>assistant\n")
    ids = processor.tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids
    return {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "input_features": feat,
        "feature_attention_mask": mask,
        "enroll_n_frames": int(enroll_n),
    }


def _to_device(inputs: dict, device, dtype) -> dict:
    out = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            value = value.to(device, dtype) if key == "input_features" else value.to(device)
        out[key] = value
    return out


@torch.no_grad()
def generate(model, prompts: list[dict], n: int, temperature: float, max_new_tokens: int,
             device, dtype, seed: int | None = None) -> list[list[list[int]]]:
    """``n`` completions for each prompt in one ``generate`` call.

    Prompts are left-padded and Mel features right-padded with their masks;
    every repeated item carries its own ``enroll_n_frames``. Completions keep
    the EOS token when one was emitted.
    """
    items = [p for p in prompts for _ in range(n)]
    b = len(items)
    lmax = max(p["input_ids"].shape[1] for p in items)
    tmax = max(p["input_features"].shape[-1] for p in items)
    ids = torch.full((b, lmax), PAD_ID, dtype=torch.long)
    attn = torch.zeros((b, lmax), dtype=torch.long)
    feats = torch.zeros((b, items[0]["input_features"].shape[1], tmax))
    fmask = torch.zeros((b, tmax), dtype=torch.long)
    for i, p in enumerate(items):
        length, frames = p["input_ids"].shape[1], p["input_features"].shape[-1]
        ids[i, lmax - length:] = p["input_ids"][0]
        attn[i, lmax - length:] = 1
        feats[i, :, :frames] = p["input_features"][0]
        fmask[i, :frames] = p["feature_attention_mask"][0]
    if seed is not None:
        torch.manual_seed(seed)
    kwargs = dict(do_sample=temperature > 0, max_new_tokens=max_new_tokens, eos_token_id=list(EOS_IDS),
                  pad_token_id=PAD_ID,
                  # The shipped config.json has text_config.use_cache=false; without the
                  # cache every decode step re-runs the whole prompt (about 20x slower).
                  use_cache=True)
    if temperature > 0:
        kwargs.update(temperature=temperature, top_p=1.0, top_k=0)
    out = model.thinker.generate(
        input_ids=ids.to(device), attention_mask=attn.to(device),
        input_features=feats.to(device, dtype), feature_attention_mask=fmask.to(device),
        enroll_n_frames=[p["enroll_n_frames"] for p in items], **kwargs)
    completions = []
    for seq in out[:, lmax:].tolist():
        cut = []
        for token in seq:
            cut.append(token)
            if token in EOS_IDS:
                break
        completions.append(cut)
    return [completions[i * n:(i + 1) * n] for i in range(len(prompts))]


def completion_logprobs(model, prompt: dict, completions: list[list[int]], device, dtype):
    """Per-sequence sum of completion-token log-probs (with grad) and token counts.

    All completions share one prompt, so the prompt needs no padding;
    completions are right-padded and masked.
    """
    inputs = _to_device(prompt, device, dtype)
    n = len(completions)
    lp = inputs["input_ids"].shape[1]
    lmax = max(len(c) for c in completions)
    comp = torch.full((n, lmax), PAD_ID, dtype=torch.long, device=device)
    cmask = torch.zeros((n, lmax), dtype=torch.long, device=device)
    for i, c in enumerate(completions):
        comp[i, :len(c)] = torch.tensor(c, device=device)
        cmask[i, :len(c)] = 1
    out = model.thinker(
        input_ids=torch.cat([inputs["input_ids"].expand(n, -1), comp], dim=1),
        attention_mask=torch.cat([inputs["attention_mask"].expand(n, -1), cmask], dim=1),
        input_features=inputs["input_features"].expand(n, -1, -1),
        feature_attention_mask=inputs["feature_attention_mask"].expand(n, -1),
        enroll_n_frames=[inputs["enroll_n_frames"]] * n,
        use_cache=False,
    )
    logits = out.logits[:, lp - 1: lp - 1 + lmax].float()
    logp = torch.log_softmax(logits, dim=-1).gather(-1, comp.unsqueeze(-1)).squeeze(-1)
    return (logp * cmask).sum(-1), cmask.sum(-1)
