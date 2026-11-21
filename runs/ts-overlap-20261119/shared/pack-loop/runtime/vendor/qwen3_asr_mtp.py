"""vLLM 0.18 speculative-decoding draft for the jointly trained MTP module (``joint_mtp.py``).

The module is served through vLLM's EAGLE proposer (``method: "eagle"``): vLLM's own MTP path
only knows the model types in ``MTPModelTypes`` and Qwen3-Next's gated, MoE layer, while ours is
one dense Qwen3 decoder layer. The EAGLE proposer feeds the draft exactly what the module was
trained on:

- the target's final-norm hidden state at ``i`` and the embedding of token ``i + 1`` (text
  embedding only; audio positions use the placeholder token's embedding, as in training);
- 1-D positions (the target's MRoPE positions are equal across sections for Qwen3-ASR);
- for a further step, the draft's own returned hidden state one position later, through the
  same layer (``joint_mtp.MTPModule.unrolled``).

The draft returns ``(h', h')``: the post-norm output both feeds ``lm_head`` and is the next
step's hidden state. ``embed_tokens`` and ``lm_head`` are not in the checkpoint, so the proposer
shares the target's, as in training.

``mtp/`` is the draft directory: ``write_vllm_config`` adds the Hugging Face Qwen3 fields vLLM
needs to ``mtp/config.json``; serve with
``--speculative-config '{"method": "eagle", "model": "<checkpoint>/mtp", "num_speculative_tokens": k}'``.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Iterable
from pathlib import Path

# vLLM validates the draft directory's architecture against the registry before EAGLEConfig
# renames it ("X" -> "EagleX", unless it already starts with "Eagle"), so config.json names the
# registered class itself.
ARCHITECTURE = "EagleQwen3ASRMTP"


def vllm_config_fields(mtp_config: dict, *, max_position_embeddings: int) -> dict:
    """Hugging Face ``Qwen3Config`` fields for one MTP layer, from ``mtp/config.json``."""
    return {
        "architectures": [ARCHITECTURE],
        "model_type": "qwen3",
        "num_hidden_layers": 1,
        "hidden_act": "silu",
        "attention_bias": False,
        "tie_word_embeddings": False,
        "max_position_embeddings": max_position_embeddings,
        "torch_dtype": "bfloat16",
        **{k: mtp_config[k] for k in (
            "hidden_size", "intermediate_size", "num_attention_heads", "num_key_value_heads",
            "head_dim", "rms_norm_eps", "rope_theta", "vocab_size")},
    }


def write_vllm_config(directory, *, max_position_embeddings: int) -> None:
    path = Path(directory) / "config.json"
    config = json.loads(path.read_text())
    config.update(vllm_config_fields(config, max_position_embeddings=max_position_embeddings))
    path.write_text(json.dumps(config, indent=1) + "\n")


@functools.cache
def _model_classes():
    import torch
    from torch import nn
    from vllm.compilation.decorators import support_torch_compile
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.model_executor.layers.linear import ReplicatedLinear
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader
    from vllm.model_executor.models.qwen3 import Qwen3DecoderLayer, Qwen3ForCausalLM
    from vllm.model_executor.models.utils import (
        AutoWeightsLoader,
        get_draft_quant_config,
        maybe_prefix,
        process_eagle_weight,
    )

    # Explicit: with postponed annotations vLLM cannot infer the tensor arguments from the hints.
    @support_torch_compile(dynamic_arg_dims={"input_ids": 0, "positions": -1, "hidden_states": 0})
    class Qwen3ASRMTPModel(nn.Module):
        def __init__(self, *, vllm_config: VllmConfig, prefix: str = "", start_layer_id: int = 0):
            super().__init__()
            config = vllm_config.speculative_config.draft_model_config.hf_config
            self.config = config
            quant_config = get_draft_quant_config(vllm_config)
            hidden = config.hidden_size
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size, hidden, prefix=maybe_prefix(prefix, "embed_tokens"))
            self.pre_fc_norm_embedding = RMSNorm(hidden, eps=config.rms_norm_eps)
            self.pre_fc_norm_hidden = RMSNorm(hidden, eps=config.rms_norm_eps)
            self.fc = ReplicatedLinear(
                input_size=2 * hidden, output_size=hidden, bias=False,
                params_dtype=vllm_config.model_config.dtype, quant_config=quant_config,
                prefix=maybe_prefix(prefix, "fc"), return_bias=False)
            # The attention layer name must not collide with the target's layers.
            self.layers = nn.ModuleList([Qwen3DecoderLayer(
                config, cache_config=vllm_config.cache_config, quant_config=quant_config,
                prefix=maybe_prefix(prefix, f"layers.{start_layer_id}"))])
            self.norm = RMSNorm(hidden, eps=config.rms_norm_eps)

        def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
            return self.embed_tokens(input_ids)

        def forward(self, input_ids: torch.Tensor, positions: torch.Tensor,
                    hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            embeds = self.pre_fc_norm_embedding(self.embed_tokens(input_ids))
            x = self.fc(torch.cat((embeds, self.pre_fc_norm_hidden(hidden_states)), dim=-1))
            x, residual = self.layers[0](positions, x, None)
            x, _ = self.norm(x, residual)
            return x, x

        def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
            stacked = [(".qkv_proj", ".q_proj", "q"), (".qkv_proj", ".k_proj", "k"),
                       (".qkv_proj", ".v_proj", "v"), (".gate_up_proj", ".gate_proj", 0),
                       (".gate_up_proj", ".up_proj", 1)]
            params = dict(self.named_parameters())
            loaded = set()
            for name, weight in weights:
                for param_name, shard_name, shard_id in stacked:
                    if shard_name in name:
                        name = name.replace(shard_name, param_name)
                        params[name].weight_loader(params[name], weight, shard_id)
                        break
                else:
                    param = params[name]
                    getattr(param, "weight_loader", default_weight_loader)(param, weight)
                loaded.add(name)
            return loaded

    class EagleQwen3ASRMTP(Qwen3ForCausalLM):
        def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
            nn.Module.__init__(self)
            self.config = vllm_config.speculative_config.draft_model_config.hf_config
            if getattr(self.config, "draft_vocab_size", None) is None:
                self.config.draft_vocab_size = self.config.vocab_size
            target_layers = vllm_config.model_config.get_num_layers(vllm_config.parallel_config)
            self.model = Qwen3ASRMTPModel(
                vllm_config=vllm_config, prefix="model", start_layer_id=target_layers)
            self.logits_processor = LogitsProcessor(self.config.vocab_size)

        def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
            # No multimodal_embeddings argument: the proposer's probe then falls back to
            # text-only drafting, which is what the module was trained on.
            return self.model.embed_input_ids(input_ids)

        def forward(self, input_ids: torch.Tensor, positions: torch.Tensor,
                    hidden_states: torch.Tensor, inputs_embeds: torch.Tensor | None = None):
            if inputs_embeds is not None:
                raise NotImplementedError("the MTP draft is text-only")
            return self.model(input_ids, positions, hidden_states)

        def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
            def prefixed(item):
                name, weight = item
                name = name if name.startswith("lm_head") else "model." + name
                process_eagle_weight(self, name)
                return name, weight

            AutoWeightsLoader(self, skip_prefixes=None).load_weights(map(prefixed, weights))

    return EagleQwen3ASRMTP


def register() -> None:
    from vllm import ModelRegistry

    ModelRegistry.register_model(ARCHITECTURE, "open_audio_llm.tsasr.vllm_mtp:EagleQwen3ASRMTP")


def __getattr__(name):
    if name == "EagleQwen3ASRMTP":
        return _model_classes()
    raise AttributeError(name)
