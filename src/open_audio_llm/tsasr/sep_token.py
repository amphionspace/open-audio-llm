"""Learned [SEP] token between enroll and mix, after conv and before transformer.

TS-ASR extracts Mel and runs conv on enroll and mix independently. Packed
conv tokens are concatenated as ``[enroll, SEP, mix]`` only when entering
the audio transformer. Plain ASR / hotword-ASR never split or insert SEP.

``thinker.audio_tower.sep_token`` is trainable. The conv stack
(``conv2d1/2/3``, ``conv_out``) is frozen in the default full-SFT script.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

SEP_TOKENS = 1
SEP_INIT_STD = 0.02
_ENV_INSERT = "AMPHION_TSASR_INSERT_SEP"


def conv_token_count(n_frames: int) -> int:
    """Whisper conv + chunked encoder token count (no SEP)."""
    n_frames = int(n_frames)
    leave = n_frames % 100
    feat_lengths = (leave - 1) // 2 + 1
    return int(((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (n_frames // 100) * 13)


def feat_extract_output_lengths(input_lengths: torch.Tensor) -> torch.Tensor:
    """Tensor form of ``conv_token_count`` (same formula as Qwen3-ASR)."""
    input_lengths = input_lengths.to(dtype=torch.long)
    leave = input_lengths % 100
    feat_lengths = (leave - 1) // 2 + 1
    return ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13


def audio_token_count(
    n_frames: int,
    insert_sep: bool,
    enroll_n_frames: int = 0,
) -> int:
    """LLM audio-pad count.

    TS-ASR: ``conv(enroll) + 1 + conv(mix)``. When enroll is a multiple of 100
    frames this equals ``conv(enroll+mix) + 1``.
    """
    n_frames = int(n_frames)
    enroll = int(enroll_n_frames or 0)
    if insert_sep and enroll > 0:
        mix = max(n_frames - enroll, 0)
        return conv_token_count(enroll) + SEP_TOKENS + conv_token_count(mix)
    n = conv_token_count(n_frames)
    return n + SEP_TOKENS if insert_sep else n


def tsasr_sep_enabled() -> bool:
    raw = os.environ.get(_ENV_INSERT, "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def default_enroll_n_frames(
    hop_length: int = 160,
    enroll_sec: float | None = None,
    sr: int = 16000,
) -> int:
    """Mel frames of the enroll prefix. 0 unless TS-eval env is set."""
    if not tsasr_sep_enabled():
        return 0
    if enroll_sec is None:
        from .concat_audio import configured_enroll_sec

        enroll_sec = configured_enroll_sec()
    return int(round(float(enroll_sec) * sr)) // int(hop_length)


def extract_wav_mel(feature_extractor, wav, sampling_rate: int | None = None):
    """One clip -> cropped Mel ``(1, n_mels, T)`` plus mask and T."""
    import numpy as np

    sr = int(sampling_rate or getattr(feature_extractor, "sampling_rate", 16000))
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    hop = int(getattr(feature_extractor, "hop_length", 160))
    if wav.size == 0:
        wav = np.zeros(hop * 8, dtype=np.float32)
    out = feature_extractor(
        [wav],
        sampling_rate=sr,
        return_attention_mask=True,
        return_tensors="pt",
        padding=True,
        truncation=False,
    )
    mask = out["attention_mask"]
    feat_len = int(mask.sum(dim=-1)[0].item())
    feat = out["input_features"][..., :feat_len]
    mask = mask[..., :feat_len]
    return feat, mask, feat_len


def extract_enroll_mix_mel(feature_extractor, enroll_wav, mix_wav, sampling_rate: int | None = None):
    """Independent Mel for enroll and mix, concatenated along time for transport."""
    e_f, e_m, e_len = extract_wav_mel(feature_extractor, enroll_wav, sampling_rate)
    m_f, m_m, _m_len = extract_wav_mel(feature_extractor, mix_wav, sampling_rate)
    return torch.cat([e_f, m_f], dim=-1), torch.cat([e_m, m_m], dim=-1), e_len


def _iter_raw_wavs(raw_speech):
    import numpy as np

    if torch.is_tensor(raw_speech):
        raw_speech = raw_speech.detach().cpu().numpy()
    if isinstance(raw_speech, np.ndarray) and raw_speech.ndim == 1:
        yield np.asarray(raw_speech, dtype=np.float32).reshape(-1)
        return
    if not isinstance(raw_speech, (list, tuple)):
        yield np.asarray(raw_speech, dtype=np.float32).reshape(-1)
        return
    for item in raw_speech:
        wav = item
        if isinstance(item, (list, tuple)) and len(item) == 2:
            wav = item[0]
        if torch.is_tensor(wav):
            wav = wav.detach().cpu().numpy()
        yield np.asarray(wav, dtype=np.float32).reshape(-1)


def _crop_fe_item(out, index: int = 0):
    feat = out["input_features"]
    mask = out["attention_mask"]
    if torch.is_tensor(feat):
        feat_len = int(mask[index].sum().item())
        return feat[index : index + 1, ..., :feat_len], mask[index : index + 1, :feat_len]
    import numpy as np

    feat_len = int(np.asarray(mask[index]).sum())
    return feat[index : index + 1, ..., :feat_len], mask[index : index + 1, :feat_len]


def _pad_stack_mel(feats, masks):
    max_t = max(int(f.shape[-1]) for f in feats)
    padded_f, padded_m = [], []
    for f, m in zip(feats, masks):
        pad = max_t - int(f.shape[-1])
        if pad > 0:
            if torch.is_tensor(f):
                f = F.pad(f, (0, pad))
                m = F.pad(m, (0, pad))
            else:
                import numpy as np

                f = np.pad(f, ((0, 0), (0, 0), (0, pad)))
                m = np.pad(m, ((0, 0), (0, pad)))
        padded_f.append(f)
        padded_m.append(m)
    if torch.is_tensor(padded_f[0]):
        return torch.cat(padded_f, dim=0), torch.cat(padded_m, dim=0)
    import numpy as np

    return np.concatenate(padded_f, axis=0), np.concatenate(padded_m, axis=0)


def _replace_fe_out(first_out, feats, masks):
    data = dict(first_out)
    data["input_features"] = feats
    data["attention_mask"] = masks
    cls = type(first_out)
    try:
        return cls(data)
    except Exception:
        first_out["input_features"] = feats
        first_out["attention_mask"] = masks
        return first_out


def install_independent_mel_fe() -> None:
    """TS-eval: Whisper FE splits concat wav (first 3s / rest) and extracts separately.

    Training does not set ``AMPHION_TSASR_INSERT_SEP``; the template extracts
    two clips directly. Eval concat-wav APIs go through this wrapper.
    """
    from transformers import WhisperFeatureExtractor

    from .concat_audio import split_enroll_mix_wav

    if getattr(WhisperFeatureExtractor.__call__, "_ts_indep_mel", False):
        return
    orig = WhisperFeatureExtractor.__call__

    def wrapped(self, raw_speech, *args, **kwargs):
        if not tsasr_sep_enabled():
            return orig(self, raw_speech, *args, **kwargs)
        wavs = list(_iter_raw_wavs(raw_speech))
        if not wavs:
            return orig(self, raw_speech, *args, **kwargs)
        sr = kwargs.get("sampling_rate") or getattr(self, "sampling_rate", 16000)
        feats, masks = [], []
        first_out = None
        for wav in wavs:
            enroll, mix = split_enroll_mix_wav(wav, sr=int(sr))
            e_out = orig(self, [enroll], *args, **kwargs)
            if first_out is None:
                first_out = e_out
            e_f, e_m = _crop_fe_item(e_out, 0)
            if mix.shape[0] > 0:
                m_out = orig(self, [mix], *args, **kwargs)
                m_f, m_m = _crop_fe_item(m_out, 0)
                if torch.is_tensor(e_f):
                    feats.append(torch.cat([e_f, m_f], dim=-1))
                    masks.append(torch.cat([e_m, m_m], dim=-1))
                else:
                    import numpy as np

                    feats.append(np.concatenate([e_f, m_f], axis=-1))
                    masks.append(np.concatenate([e_m, m_m], axis=-1))
            else:
                feats.append(e_f)
                masks.append(e_m)
        stacked_f, stacked_m = _pad_stack_mel(feats, masks)
        return _replace_fe_out(first_out, stacked_f, stacked_m)

    wrapped._ts_indep_mel = True
    WhisperFeatureExtractor.__call__ = wrapped


class AudioSepToken(nn.Module):
    """Single randomly initialized encoder token. No speech sinusoidal PE."""

    def __init__(self, d_model: int, std: float = SEP_INIT_STD):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(int(d_model)))
        self.std = float(std)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, mean=0.0, std=self.std)

    def forward(self) -> torch.Tensor:
        return self.weight


def rebuild_cu_seqlens(aftercnn_lens, window_aftercnn, device) -> torch.Tensor:
    cu_chunk_lens = [0]
    window_aftercnn = int(window_aftercnn)
    if window_aftercnn <= 0:
        window_aftercnn = 10**9
    lens = aftercnn_lens.tolist() if torch.is_tensor(aftercnn_lens) else list(aftercnn_lens)
    if not isinstance(lens, (list, tuple)):
        lens = [lens]
    for cnn_len in lens:
        cnn_len = int(cnn_len)
        if cnn_len <= 0:
            continue
        n_full = cnn_len // window_aftercnn
        rem = cnn_len % window_aftercnn
        cu_chunk_lens.extend([window_aftercnn] * n_full)
        if rem:
            cu_chunk_lens.append(rem)
    if len(cu_chunk_lens) == 1:
        cu_chunk_lens.append(0)
    return torch.tensor(cu_chunk_lens, device=device, dtype=torch.int32).cumsum(
        -1, dtype=torch.int32
    )


def _enroll_frames_per_utt(enroll_n_frames, n_utt: int) -> list[int]:
    if enroll_n_frames is None:
        return [default_enroll_n_frames()] * n_utt
    if torch.is_tensor(enroll_n_frames):
        vals = [int(x) for x in enroll_n_frames.detach().reshape(-1).tolist()]
    elif isinstance(enroll_n_frames, (int, float)):
        vals = [int(enroll_n_frames)]
    else:
        vals = [int(x) for x in enroll_n_frames]
    if len(vals) == 1 and n_utt != 1:
        vals = vals * n_utt
    if len(vals) != n_utt:
        raise ValueError(f"enroll_n_frames has {len(vals)} values for {n_utt} utterances")
    return vals


def transformer_window_aftercnn(tower: nn.Module) -> int:
    """cu_seqlens window size after conv. Independent of enroll/mix split."""
    chunk_frames = int(tower.n_window) * 2
    factor = int(tower.n_window_infer) // chunk_frames
    return conv_token_count(chunk_frames) * factor


def expand_enroll_mix_lens(feature_lens, enroll_n_frames):
    """Split each utterance into enroll/mix segments for separate conv.

    Returns ``(seg_lens, groups)`` where each group is ``(n_seg, insert_sep)``
    covering consecutive items in ``seg_lens``.
    """
    if torch.is_tensor(feature_lens) and feature_lens.ndim == 0:
        feature_lens = feature_lens.unsqueeze(0)
    lens = [int(x) for x in feature_lens.reshape(-1).tolist()]
    enrolls = _enroll_frames_per_utt(enroll_n_frames, len(lens))
    segs: list[int] = []
    groups: list[tuple[int, bool]] = []
    for t, n_enr in zip(lens, enrolls):
        n_enr = min(max(int(n_enr), 0), t)
        if n_enr > 0 and t > n_enr:
            segs.extend([n_enr, t - n_enr])
            groups.append((2, True))
        elif n_enr > 0:
            segs.append(t)
            groups.append((1, True))
        else:
            segs.append(t)
            groups.append((1, False))
    device = feature_lens.device
    dtype = feature_lens.dtype if feature_lens.dtype in (torch.int32, torch.int64) else torch.long
    return torch.tensor(segs, device=device, dtype=dtype), groups


def stitch_sep_between_segments(
    hidden_states: torch.Tensor,
    segment_cnn_lens,
    groups: list[tuple[int, bool]],
    sep_embed: torch.Tensor,
):
    """Turn separately-convolved enroll/mix packs into ``[enroll, SEP, mix]``."""
    if torch.is_tensor(segment_cnn_lens) and segment_cnn_lens.ndim == 0:
        segment_cnn_lens = segment_cnn_lens.unsqueeze(0)
    lens = [int(x) for x in segment_cnn_lens.tolist()]
    chunks = list(torch.split(hidden_states, lens, dim=0))
    sep = sep_embed.to(device=hidden_states.device, dtype=hidden_states.dtype)
    if sep.ndim == 1:
        sep = sep.unsqueeze(0)
    out = []
    new_lens = []
    idx = 0
    for n_seg, insert_sep in groups:
        parts = chunks[idx : idx + n_seg]
        idx += n_seg
        if insert_sep:
            if n_seg == 2:
                h = torch.cat([parts[0], sep, parts[1]], dim=0)
            else:
                h = torch.cat([parts[0], sep], dim=0)
        else:
            h = parts[0] if n_seg == 1 else torch.cat(parts, dim=0)
        out.append(h)
        new_lens.append(int(h.shape[0]))
    new_hidden = torch.cat(out, dim=0) if out else hidden_states
    new_lens_t = torch.tensor(
        new_lens, device=segment_cnn_lens.device, dtype=segment_cnn_lens.dtype
    )
    return new_hidden, new_lens_t


def conv_pack_features(tower: nn.Module, input_features: torch.Tensor, feature_lens: torch.Tensor):
    """Conv2d + per-chunk sinusoidal PE + pack. No transformer, no SEP.

    ``feature_lens`` may list several segments (enroll and mix as two items).
    Each segment is chunked independently, then all chunks share one conv batch.
    """
    if feature_lens.ndim == 0:
        feature_lens = feature_lens.unsqueeze(0)
    feature_lens = feature_lens.to(device=input_features.device, dtype=torch.long)
    aftercnn_lens = feat_extract_output_lengths(feature_lens)
    chunk_size = int(tower.n_window) * 2
    chunk_num = torch.ceil(feature_lens / chunk_size).long()
    chunk_lengths = torch.tensor(
        [chunk_size] * int(chunk_num.sum().item()),
        dtype=torch.long,
        device=feature_lens.device,
    )
    tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
    chunk_lengths[tail_chunk_index] = feature_lens % chunk_size
    chunk_lengths[chunk_lengths == 0] = chunk_size

    chunk_list = input_features.T.split(chunk_lengths.tolist(), dim=0)
    padded_feature = nn.utils.rnn.pad_sequence(chunk_list, batch_first=True).transpose(1, 2)
    feature_lens_after_cnn = feat_extract_output_lengths(chunk_lengths)
    padded_mask_after_cnn = nn.utils.rnn.pad_sequence(
        [
            torch.ones(int(length), dtype=torch.bool, device=padded_feature.device)
            for length in feature_lens_after_cnn.tolist()
        ],
        batch_first=True,
    )
    padded_feature = padded_feature.unsqueeze(1)
    padded_embeds = []
    conv_chunksize = int(getattr(tower, "conv_chunksize", 256) or 256)
    for chunk in padded_feature.split(conv_chunksize, dim=0):
        padded_embed = F.gelu(tower.conv2d1(chunk))
        padded_embed = F.gelu(tower.conv2d2(padded_embed))
        padded_embed = F.gelu(tower.conv2d3(padded_embed))
        padded_embeds.append(padded_embed)
    padded_embed = torch.cat(padded_embeds, dim=0)
    b, c, f, t = padded_embed.size()
    padded_embed = tower.conv_out(padded_embed.permute(0, 3, 1, 2).contiguous().view(b, t, c * f))
    positional_embedding = (
        tower.positional_embedding.positional_embedding[: padded_embed.shape[1], :]
        .unsqueeze(0)
        .to(padded_embed.dtype)
    )
    padded_embed = padded_embed + positional_embedding
    hidden_states = padded_embed[padded_mask_after_cnn]
    return hidden_states, aftercnn_lens


def insert_sep_tokens(
    hidden_states: torch.Tensor,
    aftercnn_lens,
    sep_embed: torch.Tensor,
    enroll_n_frames,
):
    """Deprecated joint-pack insert. Encoder uses separate conv + stitch instead."""
    if torch.is_tensor(aftercnn_lens) and aftercnn_lens.ndim == 0:
        aftercnn_lens = aftercnn_lens.unsqueeze(0)
    lens = [int(x) for x in aftercnn_lens.tolist()]
    n_utt = len(lens)
    enroll_frames = _enroll_frames_per_utt(enroll_n_frames, n_utt)
    sep = sep_embed.to(device=hidden_states.device, dtype=hidden_states.dtype)
    if sep.ndim == 1:
        sep = sep.unsqueeze(0)
    chunks = torch.split(hidden_states, lens, dim=0)
    out = []
    new_lens = []
    for chunk, n_mel in zip(chunks, enroll_frames):
        if n_mel <= 0 or chunk.shape[0] == 0:
            out.append(chunk)
            new_lens.append(int(chunk.shape[0]))
            continue
        pos = min(conv_token_count(n_mel), int(chunk.shape[0]))
        out.append(torch.cat([chunk[:pos], sep, chunk[pos:]], dim=0))
        new_lens.append(int(chunk.shape[0]) + SEP_TOKENS)
    new_hidden = torch.cat(out, dim=0) if out else hidden_states
    new_lens_t = torch.tensor(new_lens, device=aftercnn_lens.device, dtype=aftercnn_lens.dtype)
    return new_hidden, new_lens_t


def _d_model_of_tower(tower: nn.Module) -> int:
    cfg = getattr(tower, "config", None)
    if cfg is not None and hasattr(cfg, "d_model"):
        return int(cfg.d_model)
    return int(tower.conv_out.out_features)


def import_qwen3_asr() -> None:
    import qwen_asr  # noqa: F401


def load_qwen3_asr(model_dir: str, torch_dtype=torch.bfloat16) -> nn.Module:
    import_qwen3_asr()
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        model_dir,
        dtype=torch_dtype,
        trust_remote_code=True,
    )
    return attach_sep_token(model, model_dir=model_dir)


def attach_sep_token(model: nn.Module, model_dir: str | None = None) -> nn.Module:
    """Add ``audio_tower.sep_token`` and patch encoder / get_audio_features.

    Call this *before* ms-swift ``use_submodel_func`` so the copied ``forward``
    still accepts ``enroll_n_frames``.
    """
    thinker = model.thinker if hasattr(model, "thinker") else model
    tower = thinker.audio_tower
    d_model = _d_model_of_tower(tower)
    if not isinstance(getattr(tower, "sep_token", None), AudioSepToken):
        tower.add_module("sep_token", AudioSepToken(d_model))
        ref = tower.conv_out.weight
        tower.sep_token.to(device=ref.device, dtype=ref.dtype)
    _patch_hf_encoder_forward(tower)
    _patch_get_audio_features(thinker)
    _patch_thinker_forward(thinker)
    _patch_prepare_inputs(thinker)
    _patch_processor_audio_len()
    install_independent_mel_fe()
    if model_dir:
        load_sep_token_weights(tower.sep_token, model_dir)
    return model


def _patch_processor_audio_len() -> None:
    """HF processor pad count +1 only when TS-eval env is set.

    Training pad counts come from the ms-swift template (per-sample), so this
    must not fire during mixed ASR/HW/TS batches.
    """
    import qwen_asr.core.transformers_backend.processing_qwen3_asr as proc_mod

    fn = proc_mod._get_feat_extract_output_lengths
    if getattr(fn, "_sep_token", False):
        return

    def wrapped(input_lengths):
        if not tsasr_sep_enabled():
            return orig(input_lengths)
        enroll = default_enroll_n_frames()
        if torch.is_tensor(input_lengths):
            vals = [audio_token_count(int(x), True, enroll) for x in input_lengths.reshape(-1).tolist()]
            out = torch.tensor(vals, device=input_lengths.device, dtype=torch.long)
            return out.reshape(-1) if input_lengths.ndim == 1 else out.reshape(input_lengths.shape)
        if isinstance(input_lengths, (int, float)):
            return audio_token_count(int(input_lengths), True, enroll)
        return [audio_token_count(int(x), True, enroll) for x in input_lengths]

    orig = fn
    wrapped._sep_token = True
    proc_mod._get_feat_extract_output_lengths = wrapped


def load_sep_token_weights(module: nn.Module, model_dir: str) -> int:
    root = Path(model_dir)
    if not root.is_dir():
        return 0
    loaded: dict[str, torch.Tensor] = {}
    prefixes = (
        "thinker.audio_tower.sep_token.",
        "audio_tower.sep_token.",
        "sep_token.",
    )
    try:
        from safetensors.torch import safe_open
    except ImportError:
        safe_open = None
    if safe_open is not None:
        for p in sorted(root.glob("*.safetensors")):
            with safe_open(str(p), framework="pt") as f:
                for k in f.keys():
                    for pref in prefixes:
                        if k.startswith(pref):
                            loaded[k[len(pref) :]] = f.get_tensor(k)
    bin_path = root / "pytorch_model.bin"
    if not loaded and bin_path.is_file():
        state = torch.load(str(bin_path), map_location="cpu")
        for k, v in state.items():
            for pref in prefixes:
                if k.startswith(pref):
                    loaded[k[len(pref) :]] = v
    if not loaded:
        return 0
    missing, unexpected = module.load_state_dict(loaded, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected sep_token keys: {unexpected[:8]}")
    return sum(1 for k in loaded if k not in missing)


def _patch_hf_encoder_forward(tower: nn.Module) -> None:
    if getattr(tower, "_sep_token_patched", False):
        return

    from qwen_asr.core.transformers_backend.modeling_qwen3_asr import BaseModelOutput

    def forward(self, input_features, feature_lens=None, aftercnn_lens=None, enroll_n_frames=None):
        seg_lens, groups = expand_enroll_mix_lens(feature_lens, enroll_n_frames)
        hidden_states, aftercnn_lens = conv_pack_features(self, input_features, seg_lens)
        sep_mod = getattr(self, "sep_token", None)
        if isinstance(sep_mod, AudioSepToken) and any(use_sep for _, use_sep in groups):
            hidden_states, aftercnn_lens = stitch_sep_between_segments(
                hidden_states, aftercnn_lens, groups, sep_mod.weight
            )
        # Frozen convs produce a hidden state with requires_grad=False on plain ASR.
        # Checkpointed encoder layers then skip backward, so this rank never joins
        # the encoder gradient allreduce while a rank that saw a sep token does.
        if (
            self.training
            and torch.is_grad_enabled()
            and not hidden_states.requires_grad
        ):
            hidden_states = hidden_states.detach().requires_grad_(True)
        window_aftercnn = transformer_window_aftercnn(self)
        cu_seqlens = rebuild_cu_seqlens(aftercnn_lens, window_aftercnn, aftercnn_lens.device)

        for encoder_layer in self.layers:
            layer_outputs = encoder_layer(hidden_states, cu_seqlens)
            hidden_states = layer_outputs[0]

        hidden_states = self.ln_post(hidden_states)
        hidden_states = self.proj1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.proj2(hidden_states)
        return BaseModelOutput(last_hidden_state=hidden_states)

    tower.forward = forward.__get__(tower, type(tower))
    tower._sep_token_patched = True


def _patch_get_audio_features(thinker: nn.Module) -> None:
    if getattr(thinker, "_sep_get_audio_patched", False):
        return

    def get_audio_features(
        self,
        input_features,
        feature_attention_mask=None,
        audio_feature_lengths=None,
        enroll_n_frames=None,
    ):
        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
        feature_lens = audio_feature_lengths if audio_feature_lengths is not None else feature_attention_mask.sum(-1)
        n = int(feature_lens.shape[0])
        frames = _enroll_frames_per_utt(enroll_n_frames, n)
        audio_features = []
        for i, (input_feature, feature_len) in enumerate(zip(input_features, feature_lens)):
            audio_output = self.audio_tower(
                input_feature[:, :feature_len],
                feature_lens=feature_len.unsqueeze(0),
                enroll_n_frames=frames[i],
            )
            audio_features.append(audio_output.last_hidden_state)
        return torch.cat(audio_features, dim=0)

    thinker.get_audio_features = get_audio_features.__get__(thinker, type(thinker))
    thinker._sep_get_audio_patched = True


def _patch_thinker_forward(thinker: nn.Module) -> None:
    if getattr(thinker, "_sep_forward_patched", False):
        return

    orig = thinker.forward
    _get = thinker.get_audio_features

    def get_with_pending(
        self,
        input_features,
        feature_attention_mask=None,
        audio_feature_lengths=None,
        enroll_n_frames=None,
    ):
        if enroll_n_frames is None:
            enroll_n_frames = getattr(self, "_pending_enroll_n_frames", None)
        return _get(
            input_features,
            feature_attention_mask=feature_attention_mask,
            audio_feature_lengths=audio_feature_lengths,
            enroll_n_frames=enroll_n_frames,
        )

    thinker.get_audio_features = get_with_pending.__get__(thinker, type(thinker))

    def wrapped_forward(
        self,
        input_ids=None,
        input_features=None,
        attention_mask=None,
        feature_attention_mask=None,
        audio_feature_lengths=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        rope_deltas=None,
        labels=None,
        use_cache=None,
        cache_position=None,
        enroll_n_frames=None,
        **kwargs,
    ):
        kwargs.pop("enroll_n_frames", None)
        self._pending_enroll_n_frames = enroll_n_frames
        try:
            return orig(
                input_ids=input_ids,
                input_features=input_features,
                attention_mask=attention_mask,
                feature_attention_mask=feature_attention_mask,
                audio_feature_lengths=audio_feature_lengths,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                rope_deltas=rope_deltas,
                labels=labels,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )
        finally:
            self._pending_enroll_n_frames = None

    thinker.forward = wrapped_forward.__get__(thinker, type(thinker))
    thinker._sep_forward_patched = True


def _patch_prepare_inputs(thinker: nn.Module) -> None:
    orig = getattr(thinker, "prepare_inputs_for_generation", None)
    if orig is None or getattr(thinker, "_sep_prepare_patched", False):
        return

    def wrapped(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        input_features=None,
        feature_attention_mask=None,
        enroll_n_frames=None,
        **kwargs,
    ):
        kwargs.pop("enroll_n_frames", None)
        out = orig(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            **kwargs,
        )
        if enroll_n_frames is not None:
            out["enroll_n_frames"] = enroll_n_frames
        return out

    thinker.prepare_inputs_for_generation = wrapped.__get__(thinker, type(thinker))
    thinker._sep_prepare_patched = True
