"""Variable-length enrollment encoding, using the v4 separate-conv/SEP layout.

Only explicitly enabled checkpoints acquire the SEP parameter. No global
feature-extractor patches or per-forward mutable enrollment state are used.
"""

import json
from importlib.metadata import version
from pathlib import Path
from types import MethodType

import torch
from torch import nn


def feature_tokens(frames):
    return frames // 100 * 13 + (frames % 100 + 7) // 8


def segment_tokens(lengths):
    return sum(feature_tokens(n) for n in lengths) + len(lengths) - 1


def extract_segments(extractor, waveforms):
    """Independent Mel extraction, including full mixture and final partial hop."""
    features, lengths = [], []
    hop = extractor.hop_length
    for wav in waveforms:
        if not len(wav):
            raise ValueError("Empty enrollment or mixture")
        out = extractor(wav, sampling_rate=extractor.sampling_rate,
                        return_attention_mask=True, return_tensors="pt",
                        padding="max_length", truncation=False,
                        max_length=max(extractor.n_samples, ((len(wav) + hop - 1) // hop) * hop))
        length = int(out["attention_mask"].sum())
        features.append(out["input_features"][0, :, :length])
        lengths.append(length)
    return torch.cat(features, dim=-1).unsqueeze(0), lengths


def attach_separator(tower):
    if not hasattr(tower, "sep_token"):
        parameter = next(tower.parameters())
        tower.sep_token = nn.Embedding(1, tower.config.d_model,
                                       device=parameter.device, dtype=parameter.dtype)
        nn.init.normal_(tower.sep_token.weight, std=0.02)


def encode_segments(tower, packed, lengths):
    """Encode one example; all its segments attend jointly, never other examples."""
    from open_audio_llm.integrations.ms_swift.audio_batching import conv_segments

    if len(lengths) == 1:
        return tower(packed[:, :lengths[0]], feature_lens=torch.tensor(
            lengths, device=packed.device)).last_hidden_state
    if not 2 <= len(lengths) <= 4 or any(n <= 0 for n in lengths) or sum(lengths) > packed.shape[-1]:
        raise ValueError("Invalid enrollment/mixture frame lengths")
    if not hasattr(tower, "sep_token"):
        raise ValueError("Checkpoint has no enrollment SEP parameter")
    pieces = list(packed[:, :sum(lengths)].split(lengths, dim=-1))
    # conv_segments preserves each segment's native convolution padding shape.
    width = max(lengths)
    padded = torch.stack([torch.nn.functional.pad(p, (0, width - p.shape[-1])) for p in pieces])
    hidden = conv_segments(tower, padded, lengths)
    sep = tower.sep_token.weight.to(hidden[0].dtype)
    joined = []
    for i, part in enumerate(hidden):
        if i:
            joined.append(sep)
        joined.append(part)
    hidden = torch.cat(joined)
    # Qwen3-ASR 0.0.6 SDPA uses full attention within a single example.
    boundaries = torch.tensor([0, len(hidden)], dtype=torch.int32, device=hidden.device)
    for layer in tower.layers:
        hidden = layer(hidden, boundaries)[0]
    hidden = tower.ln_post(hidden)
    return tower.proj2(tower.act(tower.proj1(hidden)))


def checkpoint_tensors(path, prefix):
    """Read only required tensors from local safetensors checkpoints."""
    from safetensors import safe_open

    path = Path(path)
    index = path / "model.safetensors.index.json"
    if index.exists():
        mapping = json.loads(index.read_text())["weight_map"]
        files = sorted({name for key, name in mapping.items() if key.startswith(prefix)})
    else:
        files = ["model.safetensors"]
    for name in files:
        with safe_open(path / name, framework="pt", device="cpu") as reader:
            for key in reader.keys():  # noqa: SIM118 -- safe_open is not a dict or iterable.
                if key.startswith(prefix):
                    yield key[len(prefix):], reader.get_tensor(key)


def enable_target_audio(model, model_dir=None):
    if version("qwen-asr") != "0.0.6":
        raise ValueError("Target audio requires qwen-asr==0.0.6")
    thinker, tower = model.thinker, model.thinker.audio_tower
    if tower.config._attn_implementation != "sdpa" or tower.n_window * 2 != 100:
        raise ValueError("Target audio requires SDPA and native 100-frame conv chunks")
    if getattr(thinker, "_target_audio_enabled", False):
        return
    was_enabled = getattr(model.config, "target_sot_audio", False)
    attach_separator(tower)
    if was_enabled and model_dir:
        weights = dict(checkpoint_tensors(model_dir, "thinker.audio_tower.sep_token."))
        tower.sep_token.load_state_dict(weights, strict=True)
    model.config.target_sot_audio = True
    original = thinker.forward

    def forward(self, input_ids=None, input_features=None, attention_mask=None,
                feature_attention_mask=None, audio_feature_lengths=None,
                inputs_embeds=None, enrollment_lengths=None, **kwargs):
        if enrollment_lengths is not None and input_features is not None:
            lengths = (feature_attention_mask.sum(-1) if feature_attention_mask is not None
                       else audio_feature_lengths).tolist()
            rows = enrollment_lengths.tolist()
            if len(rows) != len(lengths) or len(rows) != len(input_features):
                raise ValueError("Enrollment metadata batch mismatch")
            outputs = []
            for features, total, row in zip(input_features, lengths, rows):
                if len(row) != 3 or any(n < 0 for n in row) or any(
                    row[i] > 0 and row[i - 1] == 0 for i in (1, 2)
                ):
                    raise ValueError("Enrollment lengths require 1--3 positive entries then zeros")
                enroll = [n for n in row if n > 0]
                if sum(enroll) >= total:
                    raise ValueError("Enrollment consumes the mixture")
                outputs.append(encode_segments(self.audio_tower, features, [*enroll, total - sum(enroll)]))
            audio = torch.cat(outputs)
            if inputs_embeds is None:
                inputs_embeds = self.get_input_embeddings()(input_ids)
            mask = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds)
            if len(mask) != len(outputs) or any(
                int(m.sum()) != out.numel() for m, out in zip(mask, outputs)
            ):
                raise ValueError("Audio placeholder count does not match segment encoding")
            inputs_embeds = inputs_embeds.masked_scatter(mask, audio.to(inputs_embeds))
            input_features, feature_attention_mask, audio_feature_lengths = None, None, None
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        # Ordinary-only batches must participate in DDP reduction for SEP too.
        inputs_embeds = inputs_embeds + self.audio_tower.sep_token.weight.sum().to(inputs_embeds) * 0
        return original(input_ids=input_ids, input_features=input_features,
                        attention_mask=attention_mask, feature_attention_mask=feature_attention_mask,
                        audio_feature_lengths=audio_feature_lengths, inputs_embeds=inputs_embeds, **kwargs)

    thinker.forward = MethodType(forward, thinker)
    thinker._target_audio_enabled = True
