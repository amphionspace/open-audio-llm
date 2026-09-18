"""Native Qwen3-ASR encoder batching and precision-preserving frozen concurrency."""

from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from types import MethodType

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers.modeling_outputs import BaseModelOutput


def _encode_batch(encoder, features, lengths):
    # Native short utterances use a narrower convolution tensor. Padding those
    # to 100 frames changes boundary activations after the strided convolutions.
    groups = {}
    for index, length in enumerate(lengths):
        groups.setdefault(min(length, 100), []).append(index)
    audio_hidden = [None] * len(lengths)
    for indexes in groups.values():
        chunks, chunk_lengths, output_lengths = [], [], []
        for index in indexes:
            parts = features[index, :, :lengths[index]].T.split(100)
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
        valid = torch.arange(time, device=features.device)[None, :] < torch.tensor(
            [(length + 7) // 8 for length in chunk_lengths], device=features.device,
        )[:, None]
        for index, hidden in zip(indexes, embedded[valid].split(output_lengths)):
            audio_hidden[index] = hidden
    hidden = pad_sequence(audio_hidden, batch_first=True)
    valid = torch.arange(hidden.shape[1], device=features.device)[None, :] < torch.tensor(
        [len(row) for row in audio_hidden], device=features.device,
    )[:, None]
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
    """
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

    def forward(self, input_features, feature_lens=None, aftercnn_lens=None):
        if input_features.ndim != 3:
            return original_encoder(input_features, feature_lens, aftercnn_lens)
        lengths = feature_lens.tolist() if isinstance(feature_lens, torch.Tensor) else feature_lens
        if lengths is None or len(lengths) != len(input_features) or any(
            length <= 0 or length > input_features.shape[-1] for length in lengths
        ):
            raise ValueError("Batched audio requires one positive, in-range length per utterance")
        return _encode_batch(self, input_features, lengths)

    def get_audio_features(self, input_features, feature_attention_mask=None, audio_feature_lengths=None):
        lengths = feature_attention_mask.sum(-1) if feature_attention_mask is not None else audio_feature_lengths
        return self.audio_tower(input_features, feature_lens=lengths).last_hidden_state

    def wrap_attention(original):
        def forward(self, hidden_states, cu_seqlens=None, attention_mask=None, **kwargs):
            if hidden_states.ndim == 3:
                return _batched_attention(self, hidden_states, attention_mask)
            return original(hidden_states, cu_seqlens, attention_mask, **kwargs)
        return forward

    encoder.forward = MethodType(forward, encoder)
    thinker.get_audio_features = MethodType(get_audio_features, thinker)
    for attention, original in original_attention:
        attention.forward = MethodType(wrap_attention(original), attention)

    def restore():
        thinker.get_audio_features = original_features
        encoder.forward = original_encoder
        for attention, original in original_attention:
            attention.forward = original

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
