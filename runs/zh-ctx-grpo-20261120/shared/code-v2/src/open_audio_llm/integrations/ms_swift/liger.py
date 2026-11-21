"""Liger Triton kernels for the Qwen3-ASR thinker's text decoder."""

from __future__ import annotations

from collections import Counter


def enable_liger_kernels(model):
    """Run RMSNorm, SwiGLU and RoPE of ``model.thinker.model`` on Liger kernels.

    transformers applies ``--use_liger_kernel`` by ``config.model_type``. Liger has no
    entry for ``qwen3_asr``, so the Trainer only logs that and the model runs unchanged.
    Norm and MLP forwards change on this instance only. RoPE is a module-level function
    in qwen_asr, so it changes every Qwen3-ASR model in the process until the returned
    callable restores it.
    """
    from liger_kernel.transformers import LigerSwiGLUMLP, liger_rotary_pos_emb
    from liger_kernel.transformers.monkey_patch import _patch_rms_norm_module, _patch_swiglu_module
    from qwen_asr.core.transformers_backend import modeling_qwen3_asr as qwen3_asr

    text = model.thinker.model
    if text.config.hidden_act not in {"silu", "swish"}:
        raise ValueError(f"Liger SwiGLU computes SiLU, not {text.config.hidden_act}")
    norms = (qwen3_asr.Qwen3ASRTextRMSNorm, qwen3_asr.Qwen3ASRThinkerTextRMSNorm)
    mlps = (qwen3_asr.Qwen3ASRTextMLP, qwen3_asr.Qwen3ASRThinkerTextMLP)
    patched = Counter()
    for module in text.modules():
        if isinstance(module, norms):
            # Qwen3-ASR casts the normalized states back before the weight: Liger's "llama" mode.
            _patch_rms_norm_module(module)
            patched["rms_norm"] += 1
        elif isinstance(module, mlps):
            _patch_swiglu_module(module, LigerSwiGLUMLP)
            patched["swiglu"] += 1
    if not patched["rms_norm"] or not patched["swiglu"]:
        raise ValueError(f"No Qwen3-ASR text norms or MLPs to patch: {dict(patched)}")
    rope = qwen3_asr.apply_rotary_pos_emb
    qwen3_asr.apply_rotary_pos_emb = liger_rotary_pos_emb

    def restore():
        qwen3_asr.apply_rotary_pos_emb = rope

    return dict(patched), restore
