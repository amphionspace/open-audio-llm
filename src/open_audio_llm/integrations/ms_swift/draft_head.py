"""EAGLE draft head distilled from a frozen Qwen3-ASR model, served by vLLM speculative decoding.

The head is exactly what vLLM 0.18 runs for ``--speculative-config '{"method": "eagle", ...}'``
with a draft checkpoint whose config names ``LlamaForCausalLM`` (``EagleLlamaForCausalLM`` in
``vllm/model_executor/models/llama_eagle.py``, driven by ``vllm/v1/spec_decode/eagle.py``):

- Draft row ``i`` of the first pass takes the target's final-norm output at position ``i`` and
  the embedding of token ``i + 1``, sits at RoPE position ``i`` and predicts token ``i + 2``.
  The draft class has no multimodal ``embed_input_ids``, so vLLM drafts text-only: audio
  placeholders keep the embedding of their placeholder token id, as ``embed_tokens(input_ids)``
  gives here.
- Each further draft step feeds the head's own output feature back with the drafted token, one
  position later. It attends to the first-pass KV of the verified prefix and to the KV that the
  earlier steps of the same proposal wrote.
- Block: ``fc(cat(embed, hidden))`` -> Llama decoder layer without ``input_layernorm`` ->
  ``hidden + residual`` -> the target's ``lm_head`` (tied to ``embed_tokens``). There is no
  final norm and no q/k norm. Embeddings and ``lm_head`` are shared with the target, so the
  exported checkpoint holds neither.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import MethodType

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd.function import once_differentiable
from transformers.modeling_outputs import CausalLMOutputWithPast

DRAFT = "mtp_draft"
# One FP32 block of the 151,936-token vocabulary is about 150 MiB at this row count.
_LOSS_CHUNK_ROWS = 256


class RMSNorm(nn.Module):
    def __init__(self, size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x):
        normed = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * normed.to(x.dtype)


class _Attention(nn.Module):
    def __init__(self, hidden, heads, kv_heads, head_dim):
        super().__init__()
        self.q_proj = nn.Linear(hidden, heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(heads * head_dim, hidden, bias=False)


class _MLP(nn.Module):
    def __init__(self, hidden, intermediate):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _Layer(nn.Module):
    """Llama decoder layer whose input_layernorm vLLM replaces by Identity (EAGLE-1)."""

    def __init__(self, hidden, intermediate, heads, kv_heads, head_dim, eps):
        super().__init__()
        self.self_attn = _Attention(hidden, heads, kv_heads, head_dim)
        self.post_attention_layernorm = RMSNorm(hidden, eps)
        self.mlp = _MLP(hidden, intermediate)


def _rotate_half(x):
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class DraftHead(nn.Module):
    """Parameter names equal the vLLM EAGLE checkpoint names (``fc``, ``layers.0.*``)."""

    def __init__(self, *, hidden_size, intermediate_size, num_attention_heads, num_key_value_heads,
                 head_dim, rms_norm_eps, rope_theta):
        super().__init__()
        if num_attention_heads % num_key_value_heads:
            raise ValueError("attention heads must be a multiple of the KV heads")
        self.config = {
            "hidden_size": hidden_size, "intermediate_size": intermediate_size,
            "num_attention_heads": num_attention_heads, "num_key_value_heads": num_key_value_heads,
            "head_dim": head_dim, "rms_norm_eps": rms_norm_eps, "rope_theta": rope_theta,
        }
        self.fc = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.layers = nn.ModuleList([_Layer(hidden_size, intermediate_size, num_attention_heads,
                                            num_key_value_heads, head_dim, rms_norm_eps)])

    @classmethod
    def from_thinker(cls, thinker):
        """A float32 head shaped like the text decoder, initialised from its last layer."""
        text = thinker.config.text_config
        head = cls(
            hidden_size=text.hidden_size, intermediate_size=text.intermediate_size,
            num_attention_heads=text.num_attention_heads, num_key_value_heads=text.num_key_value_heads,
            head_dim=getattr(text, "head_dim", text.hidden_size // text.num_attention_heads),
            rms_norm_eps=text.rms_norm_eps, rope_theta=float(text.rope_theta),
        )
        last = thinker.model.layers[-1]
        layer = head.layers[0]
        with torch.no_grad():
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                getattr(layer.self_attn, name).weight.copy_(getattr(last.self_attn, name).weight)
            for name in ("gate_proj", "up_proj", "down_proj"):
                getattr(layer.mlp, name).weight.copy_(getattr(last.mlp, name).weight)
            layer.post_attention_layernorm.weight.copy_(last.post_attention_layernorm.weight)
        return head.to(device=last.mlp.down_proj.weight.device, dtype=torch.float32)

    def _rope(self, length, device):
        dim = self.config["head_dim"]
        inv_freq = 1.0 / (self.config["rope_theta"] ** (
            torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
        angles = torch.outer(torch.arange(length, device=device, dtype=torch.float32), inv_freq)
        angles = torch.cat((angles, angles), dim=-1)
        return angles.cos(), angles.sin()

    def unrolled(self, embeds, hidden, steps):
        """Features of ``steps`` draft steps for every position of right-padded rows.

        ``embeds[b, q]`` embeds token ``q + 1``, ``hidden[b, q]`` is the target's final-norm
        output at ``q``. Step ``k`` (1-based) at position ``q`` is the k-th draft of the proposal
        whose first draft row is ``a = q - k + 1``: its input feature is step ``k - 1`` at
        ``q - 1`` and it attends to first-pass keys at ``<= a`` plus step ``m``'s key at
        ``a + m - 1`` for ``m = 2..k``. Under teacher forcing the drafted tokens are the
        sequence's own tokens, so every step reuses ``embeds``.
        """
        batch, length, _ = hidden.shape
        heads, kv_heads, head_dim = (self.config[k] for k in
                                     ("num_attention_heads", "num_key_value_heads", "head_dim"))
        cos, sin = self._rope(length, hidden.device)
        layer = self.layers[0]
        attention = layer.self_attn
        keys, values, features = [], [], []
        previous = hidden
        for step in range(1, steps + 1):
            if step > 1:
                previous = F.pad(features[-1][:, :-1], (0, 0, 1, 0))
            x = self.fc(torch.cat((embeds.to(previous.dtype), previous), dim=-1))
            q = attention.q_proj(x).view(batch, length, heads, head_dim).transpose(1, 2)
            k = attention.k_proj(x).view(batch, length, kv_heads, head_dim).transpose(1, 2)
            v = attention.v_proj(x).view(batch, length, kv_heads, head_dim).transpose(1, 2)
            q = q * cos.to(q.dtype) + _rotate_half(q) * sin.to(q.dtype)
            k = k * cos.to(k.dtype) + _rotate_half(k) * sin.to(k.dtype)
            keys.append(k.repeat_interleave(heads // kv_heads, dim=1))
            values.append(v.repeat_interleave(heads // kv_heads, dim=1))
            mask = unrolled_mask(length, step, hidden.device)
            out = F.scaled_dot_product_attention(
                q, torch.cat(keys, dim=2), torch.cat(values, dim=2),
                attn_mask=None if step == 1 else mask, is_causal=step == 1,
            )
            h = x + attention.o_proj(out.transpose(1, 2).reshape(batch, length, heads * head_dim))
            features.append(h + layer.mlp(layer.post_attention_layernorm(h)))
        return features


def unrolled_mask(length, step, device=None):
    """Which concatenated keys ``[K_1 | ... | K_step]`` the step-``step`` query at ``q`` sees."""
    q = torch.arange(length, device=device)[:, None]
    p = torch.arange(length, device=device)[None, :]
    blocks = [p <= q - step + 1]
    blocks += [p == q - step + m for m in range(2, step + 1)]
    return torch.cat(blocks, dim=1)


def _shift(mask, offset):
    """``result[:, q] = mask[:, q + offset]``, False outside the row."""
    result = torch.zeros_like(mask)
    length = mask.shape[1]
    if offset >= 0:
        result[:, :length - offset] = mask[:, offset:]
    else:
        result[:, -offset:] = mask[:, :length + offset]
    return result


def draft_rows(labels, steps):
    """Per step, the positions whose input and predicted tokens are both generated.

    Step ``k`` at ``q`` drafts token ``q + 2`` in a proposal that began right after the target
    emitted token ``q - k + 2``; vLLM only drafts inside the assistant turn, so both tokens
    must be supervised (the supervised span is contiguous).
    """
    supervised = labels != -100
    return [_shift(supervised, 2 - step) & _shift(supervised, 2) for step in range(1, steps + 1)]


class _ChunkedHeadKL(torch.autograd.Function):
    """KL(target || draft) through the shared ``lm_head`` without keeping vocabulary-sized logits."""

    @staticmethod
    def forward(ctx, student, teacher, weight):
        ctx.save_for_backward(student, teacher, weight)
        kl = student.new_empty(len(student), dtype=torch.float32)
        agree = torch.empty(len(student), dtype=torch.bool, device=student.device)
        for start in range(0, len(student), _LOSS_CHUNK_ROWS):
            end = start + _LOSS_CHUNK_ROWS
            draft = (student[start:end].to(weight.dtype) @ weight.t()).float().log_softmax(-1)
            target = (teacher[start:end].to(weight.dtype) @ weight.t()).float().log_softmax(-1)
            kl[start:end] = (target.exp() * (target - draft)).sum(-1)
            agree[start:end] = draft.argmax(-1) == target.argmax(-1)
        ctx.mark_non_differentiable(agree)
        return kl, agree

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_kl, grad_agree):
        student, teacher, weight = ctx.saved_tensors
        gradient = torch.empty_like(student)
        for start in range(0, len(student), _LOSS_CHUNK_ROWS):
            end = start + _LOSS_CHUNK_ROWS
            draft = (student[start:end].to(weight.dtype) @ weight.t()).float().softmax(-1)
            target = (teacher[start:end].to(weight.dtype) @ weight.t()).float().softmax(-1)
            logits_grad = (draft - target) * grad_kl[start:end, None]
            gradient[start:end] = (logits_grad.to(weight.dtype) @ weight).to(student.dtype)
        return gradient, None, None


def draft_loss(head, weight, embeds, hidden, labels, steps, decay):
    """Weighted mean over steps of the per-row KL to the target's next-next distribution.

    Returns the loss and, per step, the rows' KL and argmax agreement, plus each complete
    proposal's accepted length under teacher forcing (consecutive agreeing steps).
    """
    embeds = F.pad(embeds[:, 1:], (0, 0, 0, 1))
    features = head.unrolled(embeds, hidden, steps)
    # The target's distribution over token q + 2 comes from its output at q + 1.
    teacher = F.pad(hidden[:, 1:], (0, 0, 0, 1)).flatten(0, 1)
    rows = draft_rows(labels, steps)
    loss, total_weight, stats = 0.0, 0.0, []
    agreement = []
    for step, (feature, valid) in enumerate(zip(features, rows), start=1):
        index = valid.flatten().nonzero().squeeze(1)
        kl, agree = _ChunkedHeadKL.apply(feature.flatten(0, 1).index_select(0, index),
                                         teacher.index_select(0, index), weight)
        step_weight = decay ** (step - 1)
        total_weight += step_weight
        if len(index):
            loss = loss + step_weight * kl.mean()
        else:
            # Keep every head parameter in the graph so all DDP ranks reduce gradients.
            loss = loss + feature.sum() * 0
        stats.append((kl.detach(), agree))
        agreement.append(torch.zeros_like(valid).flatten().index_fill(0, index[agree], True).view_as(valid))
    # A proposal is complete when its last step is still inside the assistant turn.
    complete = rows[-1]
    accepted = torch.zeros(labels.shape, dtype=torch.long, device=labels.device)
    running = torch.ones_like(complete)
    for step in range(1, steps + 1):
        running = running & _shift(agreement[step - 1], step - 1)
        accepted += running
    return loss / total_weight, stats, accepted[_shift(complete, steps - 1)]


def export_eagle(head, directory, *, vocab_size, eos_token_id=None, pad_token_id=None,
                 dtype=torch.bfloat16, info=None):
    """Write the head as a vLLM EAGLE draft: Llama config plus fc/layers.0 weights only."""
    from safetensors.torch import save_file

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "num_hidden_layers": len(head.layers),
        "hidden_act": "silu",
        "max_position_embeddings": 65536,
        "rope_scaling": None,
        "attention_bias": False,
        "mlp_bias": False,
        "tie_word_embeddings": True,
        "vocab_size": vocab_size,
        "torch_dtype": str(dtype).removeprefix("torch."),
        "eos_token_id": eos_token_id,
        "pad_token_id": pad_token_id,
        **head.config,
    }
    tensors = {name: value.detach().to(dtype).contiguous().cpu()
               for name, value in head.state_dict().items()}
    save_file(tensors, str(directory / "model.safetensors"), metadata={"format": "pt"})
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    if info is not None:
        (directory / "draft_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n")
    return directory


def enable_draft_head(trainer, steps, decay):
    """Train only an EAGLE draft head on the frozen model; checkpoints hold only the head.

    Each checkpoint keeps the head under ``mtp_draft.*`` in ``model.safetensors`` (for
    resuming) and a vLLM draft directory in ``eagle/``.
    """
    if steps < 1 or not 0 < decay <= 1 or not math.isfinite(decay):
        raise ValueError("draft_head_steps must be >= 1 and draft_head_step_decay in (0, 1]")
    model = trainer.model
    thinker = model.thinker
    head = DraftHead.from_thinker(thinker)
    model.add_module(DRAFT, head)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(f"{DRAFT}."))
    lm_head = thinker.lm_head
    embed_tokens = thinker.get_input_embeddings()
    forward = model.forward
    pending = []

    def draft_forward(self, *args, labels=None, **kwargs):
        if labels is None or args or not self.training:
            return forward(*args, labels=labels, **kwargs)
        captured = {}

        def capture(module, inputs):
            # Keep the final-norm output and project nothing: the head computes its own targets.
            captured["hidden"] = inputs[0]
            return (inputs[0][:, :0],)

        hook = lm_head.register_forward_pre_hook(capture)
        thinker.eval()
        try:
            with torch.no_grad():
                forward(**kwargs, use_cache=False)
                embeds = embed_tokens(kwargs["input_ids"])
        finally:
            hook.remove()
            thinker.train(self.training)
        mask = kwargs.get("attention_mask")
        if mask is not None and (mask[:, 1:] > mask[:, :-1]).any():
            # Positions and the causal key range assume right padding.
            raise ValueError("draft head training needs right-padded batches")
        loss, stats, accepted = draft_loss(head, lm_head.weight, embeds, captured["hidden"],
                                           labels, steps, decay)
        pending.append((stats, accepted))
        return CausalLMOutputWithPast(loss=loss)

    log = trainer.log

    def log_with_draft(self, logs, *args, **kwargs):
        if pending:
            metrics = self.custom_metrics["train"]
            for step in range(steps):
                metrics[f"draft_kl_{step + 1}"].update(torch.cat([s[step][0] for s, _ in pending]))
                metrics[f"draft_acc_{step + 1}"].update(torch.cat([s[step][1] for s, _ in pending]).float())
            metrics[f"draft_accepted_of_{steps}"].update(torch.cat([a for _, a in pending]).float())
            pending.clear()
        return log(logs, *args, **kwargs)

    save = trainer._save_model

    def save_draft(self, output_dir=None, state_dict=None):
        output_dir = output_dir or self.args.output_dir
        if state_dict is None:
            state_dict = self.model.state_dict()
        save(output_dir, {k: v for k, v in state_dict.items() if k.startswith(f"{DRAFT}.")})
        text = thinker.config.text_config
        export_eagle(head, Path(output_dir) / "eagle", vocab_size=text.vocab_size,
                     eos_token_id=getattr(model.config, "eos_token_id", None),
                     pad_token_id=getattr(model.config, "pad_token_id", None),
                     info={"backbone": getattr(model.config, "_name_or_path", None),
                           "global_step": self.state.global_step, "train_steps": steps,
                           "step_decay": decay})

    model.forward = MethodType(draft_forward, model)
    trainer.log = MethodType(log_with_draft, trainer)
    trainer._save_model = MethodType(save_draft, trainer)
    return head
