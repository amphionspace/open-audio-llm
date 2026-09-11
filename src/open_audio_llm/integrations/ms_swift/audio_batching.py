"""Run frozen native audio encoders concurrently without changing BF16 kernels."""

from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from types import MethodType

import torch

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
