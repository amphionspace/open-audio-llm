"""Native Qwen3-ASR encoder batching and precision-preserving frozen concurrency."""

from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from types import MethodType

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers.modeling_outputs import BaseModelOutput


def _enroll_list(enroll_n_frames, n_utt):
    if enroll_n_frames is None:
        from open_audio_llm.tsasr.sep_token import default_enroll_n_frames

        # Training passes an explicit vector (0 for plain ASR). Eval concat-wav
        # APIs leave this unset and rely on AMPHION_TSASR_INSERT_SEP.
        return [default_enroll_n_frames()] * n_utt
    if torch.is_tensor(enroll_n_frames):
        values = [int(x) for x in enroll_n_frames.detach().reshape(-1).tolist()]
    elif isinstance(enroll_n_frames, (int, float)):
        values = [int(enroll_n_frames)]
    else:
        values = [int(x) for x in enroll_n_frames]
    if len(values) == 1 and n_utt != 1:
        values = values * n_utt
    if len(values) != n_utt:
        raise ValueError(f"enroll_n_frames has {len(values)} values for {n_utt} utterances")
    return values


def _conv_clips(encoder, clips):
    """Conv each clip independently, using the native 100-frame chunk width.

    ``clips`` are ``(n_mels, T)`` views. Grouping by ``min(T, 100)`` keeps the
    short-clip convolution tensor the same width as the serial encoder.
    """
    lengths = [int(clip.shape[-1]) for clip in clips]
    device = clips[0].device
    groups = {}
    for index, length in enumerate(lengths):
        groups.setdefault(min(length, 100), []).append(index)
    audio_hidden = [None] * len(clips)
    for indexes in groups.values():
        chunks, chunk_lengths, output_lengths = [], [], []
        for index in indexes:
            parts = clips[index].T.split(100)
            chunks.extend(parts)
            chunk_lengths.extend(len(part) for part in parts)
            output_lengths.append(sum((len(part) + 7) // 8 for part in parts))
        padded = pad_sequence(chunks, batch_first=True).transpose(1, 2).unsqueeze(1)
        embedded = []
        for part in padded.split(encoder.conv_chunksize):
            part = F.gelu(encoder.conv2d1(part))
            part = F.gelu(encoder.conv2d2(part))
            embedded.append(F.gelu(encoder.conv2d3(part)))
        embedded = torch.cat(embedded)
        batch, channels, frequency, time = embedded.shape
        embedded = encoder.conv_out(
            embedded.permute(0, 3, 1, 2).contiguous().view(batch, time, channels * frequency)
        )
        embedded = embedded + encoder.positional_embedding.positional_embedding[:time].to(embedded.dtype)
        valid = torch.arange(time, device=device)[None, :] < torch.tensor(
            [(length + 7) // 8 for length in chunk_lengths], device=device,
        )[:, None]
        for index, hidden in zip(indexes, embedded[valid].split(output_lengths)):
            audio_hidden[index] = hidden
    return audio_hidden


def _stitch_sep(encoder, hidden_rows, groups):
    if not any(insert for _, insert in groups):
        return hidden_rows
    sep_mod = getattr(encoder, "sep_token", None)
    if sep_mod is None or not hasattr(sep_mod, "weight"):
        raise ValueError("Batched TS-ASR requires thinker.audio_tower.sep_token")
    sep = sep_mod.weight
    if sep.ndim == 1:
        sep = sep.unsqueeze(0)
    stitched = []
    cursor = 0
    for n_seg, insert in groups:
        parts = hidden_rows[cursor:cursor + n_seg]
        cursor += n_seg
        if insert:
            token = sep.to(device=parts[0].device, dtype=parts[0].dtype)
            if n_seg == 2:
                stitched.append(torch.cat([parts[0], token, parts[1]], dim=0))
            else:
                stitched.append(torch.cat([parts[0], token], dim=0))
        else:
            stitched.append(parts[0] if n_seg == 1 else torch.cat(parts, dim=0))
    return stitched


def _encode_batch(encoder, features, lengths, enroll_n_frames=None):
    enrolls = _enroll_list(enroll_n_frames, len(lengths))
    clips, groups = [], []
    for index, length in enumerate(lengths):
        enroll = min(max(enrolls[index], 0), length)
        feat = features[index]
        if enroll > 0 and length > enroll:
            clips.extend((feat[:, :enroll], feat[:, enroll:length]))
            groups.append((2, True))
        elif enroll > 0:
            clips.append(feat[:, :length])
            groups.append((1, True))
        else:
            clips.append(feat[:, :length])
            groups.append((1, False))
    audio_hidden = _stitch_sep(encoder, _conv_clips(encoder, clips), groups)
    hidden = pad_sequence(audio_hidden, batch_first=True)
    valid = torch.arange(hidden.shape[1], device=features.device)[None, :] < torch.tensor(
        [len(row) for row in audio_hidden], device=features.device,
    )[:, None]
    # Frozen convs leave a plain-ASR hidden state without grad. Checkpointed
    # encoder layers would then skip backward and drop out of the allreduce.
    if encoder.training and torch.is_grad_enabled() and not hidden.requires_grad:
        hidden = hidden.detach().requires_grad_(True)
    # SDPA ignores native cu_seqlens. Use a batch dimension to isolate utterances
    # and mask padding, retaining full attention within each enrollment+mixture.
    for layer in encoder.layers:
        hidden = layer(hidden, None, valid[:, None, None, :])[0]
    hidden = encoder.ln_post(hidden[valid])
    return BaseModelOutput(last_hidden_state=encoder.proj2(encoder.act(encoder.proj1(hidden))))


def _batched_attention(attention, hidden_states, attention_mask):
    batch, time, _ = hidden_states.shape

    def project(module):
        return module(hidden_states).view(batch, time, attention.num_heads, -1).transpose(1, 2)

    hidden = F.scaled_dot_product_attention(
        project(attention.q_proj), project(attention.k_proj), project(attention.v_proj),
        attn_mask=attention_mask, dropout_p=0.0, is_causal=False, scale=attention.scaling,
    )
    return attention.out_proj(hidden.transpose(1, 2).reshape(batch, time, -1))


def enable_batched_audio(model):
    """Batch a trainable or frozen tower; keep weights and native layer checkpointing.

    Batching changes BF16 reduction order; it is not bitwise equivalent to the
    serial path. Keep this opt-in and compare recognition before adoption.

    When ``enroll_n_frames`` is set, enroll and mix are convolved as separate
    clips and a learned SEP token is inserted before the transformer, matching
    the v3 TS-ASR protocol. Plain ASR leaves ``enroll_n_frames`` at 0.
    """
    if getattr(model, "_audio_encoder_batching_enabled", False):
        return lambda: None
    if version("qwen-asr") != "0.0.6":
        raise ValueError("Batched audio is verified for qwen-asr==0.0.6")
    thinker = model.thinker
    encoder = thinker.audio_tower
    if encoder.config._attn_implementation != "sdpa" or encoder.n_window * 2 != 100:
        raise ValueError("Batched audio requires SDPA with native 100-frame convolution chunks")
    if any(getattr(encoder.config, name) for name in ("dropout", "attention_dropout", "activation_dropout")):
        raise ValueError("Batched audio requires a dropout-free audio encoder")
    original_features, original_encoder = thinker.get_audio_features, encoder.forward
    original_attention = [(layer.self_attn, layer.self_attn.forward) for layer in encoder.layers]

    def forward(self, input_features, feature_lens=None, aftercnn_lens=None, enroll_n_frames=None):
        if input_features.ndim != 3:
            if enroll_n_frames is None:
                return original_encoder(input_features, feature_lens, aftercnn_lens)
            return original_encoder(
                input_features, feature_lens, aftercnn_lens, enroll_n_frames=enroll_n_frames,
            )
        lengths = feature_lens.tolist() if isinstance(feature_lens, torch.Tensor) else feature_lens
        if lengths is None or len(lengths) != len(input_features) or any(
            length <= 0 or length > input_features.shape[-1] for length in lengths
        ):
            raise ValueError("Batched audio requires one positive, in-range length per utterance")
        return _encode_batch(self, input_features, lengths, enroll_n_frames)

    def get_audio_features(
        self, input_features, feature_attention_mask=None, audio_feature_lengths=None, enroll_n_frames=None,
    ):
        lengths = feature_attention_mask.sum(-1) if feature_attention_mask is not None else audio_feature_lengths
        if enroll_n_frames is None:
            enroll_n_frames = getattr(self, "_pending_enroll_n_frames", None)
        return self.audio_tower(
            input_features, feature_lens=lengths, enroll_n_frames=enroll_n_frames,
        ).last_hidden_state

    def wrap_attention(original):
        def forward(self, hidden_states, cu_seqlens=None, attention_mask=None, **kwargs):
            if hidden_states.ndim == 3:
                return _batched_attention(self, hidden_states, attention_mask)
            return original(hidden_states, cu_seqlens, attention_mask, **kwargs)
        return forward

    encoder.forward = MethodType(forward, encoder)
    thinker.get_audio_features = MethodType(get_audio_features, thinker)
    model._audio_encoder_batching_enabled = True
    for attention, original in original_attention:
        attention.forward = MethodType(wrap_attention(original), attention)

    def restore():
        thinker.get_audio_features = original_features
        encoder.forward = original_encoder
        for attention, original in original_attention:
            attention.forward = original
        model._audio_encoder_batching_enabled = False

    return restore

# Two streams improved measured encoder latency; four increased host contention.
_ENCODER_STREAMS = 2


def enable_parallel_audio(model):
    """Keep native per-audio shapes and precision; return a cleanup/restore function."""
    if version("qwen-asr") != "0.0.6":
        raise ValueError("Parallel audio is verified for qwen-asr==0.0.6")
    thinker = model.thinker
    encoder = thinker.audio_tower
    if encoder.config._attn_implementation != "sdpa":
        raise ValueError("Parallel audio requires the native SDPA backend")
    if any(p.requires_grad for p in encoder.parameters()):
        raise ValueError("Parallel audio requires a frozen audio encoder")
    if any(
        getattr(encoder.config, name)
        for name in ("dropout", "attention_dropout", "activation_dropout")
    ):
        raise ValueError("Parallel audio requires a dropout-free audio encoder")
    device = next(encoder.parameters()).device
    streams = [torch.cuda.Stream(device=device) for _ in range(_ENCODER_STREAMS)]
    pool = ThreadPoolExecutor(_ENCODER_STREAMS, thread_name_prefix="audio-encoder")
    original = thinker.get_audio_features

    def get_audio_features(
        self, input_features, feature_attention_mask=None, audio_feature_lengths=None
    ):
        lengths = (
            feature_attention_mask.sum(-1)
            if feature_attention_mask is not None
            else audio_feature_lengths
        )
        if lengths is None:
            raise ValueError("Audio feature lengths are required")
        # SDPA ignores cu_seqlens. CPU lengths avoid repeated scalar GPU synchronizations
        # while preserving the original encoder's convolution and attention shapes.
        lengths = lengths.tolist()
        current = torch.cuda.current_stream(device)
        for stream in streams:
            stream.wait_stream(current)
            input_features.record_stream(stream)
        grad_enabled = torch.is_grad_enabled()
        inference = torch.is_inference_mode_enabled()
        autocast = torch.is_autocast_enabled("cuda")
        dtype = torch.get_autocast_dtype("cuda")

        def encode(part):
            with (
                torch.cuda.stream(streams[part]),
                torch.inference_mode(inference),
                torch.set_grad_enabled(grad_enabled),
                torch.autocast("cuda", enabled=autocast, dtype=dtype),
            ):
                return [
                    (
                        i,
                        encoder(
                            input_features[i, :, : lengths[i]],
                            feature_lens=torch.tensor([lengths[i]], device="cpu"),
                        ).last_hidden_state,
                    )
                    for i in range(part, len(lengths), _ENCODER_STREAMS)
                ]

        pairs = [
            pair for rows in pool.map(encode, range(_ENCODER_STREAMS)) for pair in rows
        ]
        for stream in streams:
            current.wait_stream(stream)
        outputs = []
        for _, output in sorted(pairs):
            # Outputs were allocated on worker streams and are consumed on the caller's.
            output.record_stream(current)
            outputs.append(output)
        return torch.cat(outputs)

    thinker.get_audio_features = MethodType(get_audio_features, thinker)

    def restore():
        pool.shutdown(wait=True)
        for stream in streams:
            stream.synchronize()
        thinker.get_audio_features = original

    return restore
