"""EAGLE draft heads distilled from a frozen Qwen3-ASR model, served by vLLM speculative decoding.

Each head is exactly what vLLM 0.18 runs for its ``--speculative-config`` method, driven by
``vllm/v1/spec_decode/eagle.py`` with the target's embeddings and ``lm_head`` shared:

- ``eagle`` (EAGLE-1, ``EagleLlamaForCausalLM`` in ``llama_eagle.py``): draft row ``i`` of the
  first pass takes the target's final-norm output at ``i`` and the embedding of token
  ``i + 1``; ``fc(cat(embed, hidden))`` -> Llama layer without ``input_layernorm`` ->
  ``hidden + residual`` -> ``lm_head``. No final norm, no q/k norm.
- ``eagle3`` (EAGLE-3, ``Eagle3LlamaForCausalLM`` in ``llama_eagle3.py``): the first pass takes
  ``fc(cat(aux))`` of the target's residual stream entering decoder layers ``aux_layers``
  (``Qwen2Model`` collects ``hidden + residual`` after layer ``j - 1`` for id ``j``), then
  ``cat(input_layernorm(embed), hidden_norm(h))`` -> attention with residual ``h`` -> MLP ->
  final ``norm`` -> ``lm_head``. Later steps feed back the pre-norm output, not ``fc``.

Both: row ``i`` sits at RoPE position ``i`` and predicts token ``i + 2``. Neither draft class
merges multimodal embeddings, so audio placeholders keep the embedding of their placeholder
token id, as ``embed_tokens(input_ids)`` gives here. Each further draft step feeds the head's
own feature back with the drafted token, one position later, attending to the first-pass KV of
the verified prefix and to the KV that the earlier steps of the same proposal wrote.
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
# The serving benchmark's output cap; generation stops earlier at the end token.
GREEDY_MAX_NEW_TOKENS = 256
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
    def __init__(self, hidden, heads, kv_heads, head_dim, input_size):
        super().__init__()
        self.q_proj = nn.Linear(input_size, heads * head_dim, bias=False)
        self.k_proj = nn.Linear(input_size, kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(input_size, kv_heads * head_dim, bias=False)
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
    """EAGLE-1: Llama layer whose input_layernorm vLLM replaces by Identity.

    EAGLE-3: attention reads ``cat(input_layernorm(embed), hidden_norm(hidden))``.
    """

    def __init__(self, hidden, intermediate, heads, kv_heads, head_dim, eps, eagle3):
        super().__init__()
        self.self_attn = _Attention(hidden, heads, kv_heads, head_dim, 2 * hidden if eagle3 else hidden)
        if eagle3:
            self.input_layernorm = RMSNorm(hidden, eps)
            self.hidden_norm = RMSNorm(hidden, eps)
        self.post_attention_layernorm = RMSNorm(hidden, eps)
        self.mlp = _MLP(hidden, intermediate)


def _rotate_half(x):
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class DraftHead(nn.Module):
    """Parameter names equal the vLLM draft checkpoint names (``fc``, ``layers.0.*``, ``norm``).

    ``aux_layers`` empty: EAGLE-1. Otherwise EAGLE-3 over those target layer ids.
    """

    def __init__(self, *, hidden_size, intermediate_size, num_attention_heads, num_key_value_heads,
                 head_dim, rms_norm_eps, rope_theta, aux_layers=()):
        super().__init__()
        if num_attention_heads % num_key_value_heads:
            raise ValueError("attention heads must be a multiple of the KV heads")
        self.config = {
            "hidden_size": hidden_size, "intermediate_size": intermediate_size,
            "num_attention_heads": num_attention_heads, "num_key_value_heads": num_key_value_heads,
            "head_dim": head_dim, "rms_norm_eps": rms_norm_eps, "rope_theta": rope_theta,
        }
        self.aux_layers = tuple(aux_layers)
        if list(self.aux_layers) != sorted(set(self.aux_layers)):
            # vLLM appends aux states while walking the layers, so the fc input is in layer order.
            raise ValueError("aux_layers must be strictly increasing")
        eagle3 = bool(self.aux_layers)
        fc_input = len(self.aux_layers) * hidden_size if eagle3 else 2 * hidden_size
        self.fc = nn.Linear(fc_input, hidden_size, bias=False)
        self.layers = nn.ModuleList([_Layer(hidden_size, intermediate_size, num_attention_heads,
                                            num_key_value_heads, head_dim, rms_norm_eps, eagle3)])
        if eagle3:
            self.norm = RMSNorm(hidden_size, rms_norm_eps)

    @property
    def eagle3(self):
        return bool(self.aux_layers)

    @classmethod
    def from_thinker(cls, thinker, arch="eagle1"):
        """A float32 head shaped like the text decoder, initialised as transformers does.

        Copying the decoder's last layer instead starts far worse: without the final norm its
        MLP output turns into logits with a KL of about 200 nats to the target's distribution.
        EAGLE-3 reads vLLM's default layers for the target, ``(2, L // 2, L - 3)``.
        """
        if arch not in {"eagle1", "eagle3"}:
            raise ValueError("draft head arch must be eagle1 or eagle3")
        text = thinker.config.text_config
        depth = text.num_hidden_layers
        head = cls(
            hidden_size=text.hidden_size, intermediate_size=text.intermediate_size,
            num_attention_heads=text.num_attention_heads, num_key_value_heads=text.num_key_value_heads,
            head_dim=getattr(text, "head_dim", text.hidden_size // text.num_attention_heads),
            rms_norm_eps=text.rms_norm_eps, rope_theta=float(text.rope_theta),
            aux_layers=(2, depth // 2, depth - 3) if arch == "eagle3" else (),
        )
        with torch.no_grad():
            for module in head.modules():
                if isinstance(module, nn.Linear):
                    module.weight.normal_(0.0, text.initializer_range)
        return head.to(device=thinker.lm_head.weight.device, dtype=torch.float32)

    def _rope(self, length, device):
        dim = self.config["head_dim"]
        inv_freq = 1.0 / (self.config["rope_theta"] ** (
            torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
        angles = torch.outer(torch.arange(length, device=device, dtype=torch.float32), inv_freq)
        angles = torch.cat((angles, angles), dim=-1)
        return angles.cos(), angles.sin()

    def unrolled(self, embeds, target, steps):
        """Per step, the features fed to ``lm_head`` for every position of right-padded rows.

        ``embeds[b, q]`` embeds token ``q + 1``. ``target[b, q]`` is the target's final-norm
        output at ``q`` (EAGLE-1) or its aux hidden states there, concatenated in layer order
        (EAGLE-3). Step ``k`` (1-based) at position ``q`` is the k-th draft of the proposal whose
        first draft row is ``a = q - k + 1``: its input feature is step ``k - 1``'s output at
        ``q - 1`` (EAGLE-3: before the final norm) and it attends to first-pass keys at ``<= a``
        plus step ``m``'s key at ``a + m - 1`` for ``m = 2..k``. Under teacher forcing the
        drafted tokens are the sequence's own tokens, so every step reuses ``embeds``.
        """
        batch, length, _ = target.shape
        heads, kv_heads, head_dim = (self.config[k] for k in
                                     ("num_attention_heads", "num_key_value_heads", "head_dim"))
        cos, sin = self._rope(length, target.device)
        layer = self.layers[0]
        attention = layer.self_attn
        keys, values, carried, features = [], [], [], []
        for step in range(1, steps + 1):
            if step == 1:
                previous = self.fc(target) if self.eagle3 else target
            else:
                previous = F.pad(carried[-1][:, :-1], (0, 0, 1, 0))
            if self.eagle3:
                residual = previous
                x = torch.cat((layer.input_layernorm(embeds), layer.hidden_norm(previous)), dim=-1)
            else:
                x = residual = self.fc(torch.cat((embeds.to(previous.dtype), previous), dim=-1))
            q = attention.q_proj(x).view(batch, length, heads, head_dim).transpose(1, 2)
            k = attention.k_proj(x).view(batch, length, kv_heads, head_dim).transpose(1, 2)
            v = attention.v_proj(x).view(batch, length, kv_heads, head_dim).transpose(1, 2)
            q = q * cos.to(q.dtype) + _rotate_half(q) * sin.to(q.dtype)
            k = k * cos.to(k.dtype) + _rotate_half(k) * sin.to(k.dtype)
            keys.append(k.repeat_interleave(heads // kv_heads, dim=1))
            values.append(v.repeat_interleave(heads // kv_heads, dim=1))
            mask = unrolled_mask(length, step, target.device)
            out = F.scaled_dot_product_attention(
                q, torch.cat(keys, dim=2), torch.cat(values, dim=2),
                attn_mask=None if step == 1 else mask, is_causal=step == 1,
            )
            h = residual + attention.o_proj(out.transpose(1, 2).reshape(batch, length, heads * head_dim))
            carried.append(h + layer.mlp(layer.post_attention_layernorm(h)))
            features.append(self.norm(carried[-1]) if self.eagle3 else carried[-1])
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


def draft_loss(head, weight, embeds, hidden, labels, steps, decay, aux=None):
    """Weighted mean over steps of the per-row KL to the target's next-next distribution.

    ``hidden`` is the target's final-norm output (the teacher, and EAGLE-1's input); ``aux``
    its concatenated aux hidden states (EAGLE-3's input). Returns the loss and, per step, the
    rows' KL and argmax agreement, plus each complete proposal's accepted length under teacher
    forcing (consecutive agreeing steps).
    """
    if head.eagle3 == (aux is None):
        raise ValueError("EAGLE-3 heads take the aux hidden states, EAGLE-1 heads only the output")
    embeds = F.pad(embeds[:, 1:], (0, 0, 0, 1))
    features = head.unrolled(embeds, aux if head.eagle3 else hidden, steps)
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
    """Write the head as a vLLM draft: Llama config plus the head's weights only.

    EAGLE-3 drafts name ``LlamaForCausalLMEagle3`` and pin the target layers they read in
    ``eagle_aux_hidden_state_layer_ids`` (vLLM otherwise falls back to the target's default).
    """
    from safetensors.torch import save_file

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    eagle3 = {
        "eagle_aux_hidden_state_layer_ids": list(head.aux_layers),
        "target_hidden_size": head.config["hidden_size"],
    } if head.eagle3 else {}
    config = {
        "architectures": ["LlamaForCausalLMEagle3" if head.eagle3 else "LlamaForCausalLM"],
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
        **eagle3,
    }
    tensors = {name: value.detach().to(dtype).contiguous().cpu()
               for name, value in head.state_dict().items()}
    save_file(tensors, str(directory / "model.safetensors"), metadata={"format": "pt"})
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    if info is not None:
        (directory / "draft_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n")
    return directory


def capture_aux(layers, ids, store):
    """Hooks that store the residual stream entering ``layers[j]`` (after the last one for ``j == len``).

    This is vLLM's aux hidden state ``j``: ``Qwen2Model`` keeps ``hidden + residual`` after
    layer ``j - 1`` (the embeddings for ``j == 0``).
    """
    handles = []
    for j in ids:
        if j < len(layers):
            handles.append(layers[j].register_forward_pre_hook(
                lambda module, args, kwargs, j=j: store.__setitem__(
                    j, args[0] if args else kwargs["hidden_states"]), with_kwargs=True))
        elif j == len(layers):
            handles.append(layers[-1].register_forward_hook(
                lambda module, args, output, j=j: store.__setitem__(
                    j, output[0] if isinstance(output, tuple) else output)))
        else:
            raise ValueError(f"aux layer {j} is past the decoder's {len(layers)} layers")
    return handles


@torch.no_grad()
def greedy_relabel(thinker, kwargs, labels, pad_id, eos_ids, max_new_tokens=GREEDY_MAX_NEW_TOKENS):
    """The batch with every assistant span replaced by the frozen model's greedy continuation.

    Generation starts from each row's prompt (everything before its first supervised token),
    left-padded as batched generation needs; rows are rebuilt right-padded as prompt + greedy
    tokens up to and including the first end token, all supervised. Audio inputs are reused
    unchanged. Returns the new keyword inputs, labels and which rows changed.
    """
    input_ids = kwargs["input_ids"]
    supervised = labels != -100
    if not supervised.any(1).all():
        raise ValueError("every row needs an assistant span to regenerate")
    starts = supervised.int().argmax(1).tolist()
    width = max(starts)
    prompt = input_ids.new_full((len(starts), width), pad_id)
    prompt_mask = torch.zeros_like(prompt)
    for row, start in enumerate(starts):
        prompt[row, width - start:] = input_ids[row, :start]
        prompt_mask[row, width - start:] = 1
    audio = {key: kwargs[key] for key in ("input_features", "feature_attention_mask", "enroll_n_frames",
                                          "enrollment_lengths") if kwargs.get(key) is not None}
    generated = thinker.generate(input_ids=prompt, attention_mask=prompt_mask, max_new_tokens=max_new_tokens,
                                 do_sample=False, eos_token_id=list(eos_ids), pad_token_id=pad_id, **audio)
    rows, spans = [], []
    for row, start in enumerate(starts):
        tokens = generated[row, width:].tolist()
        end = next((i + 1 for i, token in enumerate(tokens) if token in eos_ids), len(tokens))
        spans.append(tokens[:end])
        rows.append(input_ids[row, :start].tolist() + tokens[:end])
    length = max(map(len, rows))
    new_ids = input_ids.new_full((len(rows), length), pad_id)
    new_labels = torch.full_like(new_ids, -100)
    new_mask = torch.zeros_like(new_ids)
    for row, (tokens, start) in enumerate(zip(rows, starts)):
        new_ids[row, :len(tokens)] = torch.tensor(tokens, device=new_ids.device)
        new_labels[row, start:len(tokens)] = new_ids[row, start:len(tokens)]
        new_mask[row, :len(tokens)] = 1
    changed = torch.tensor([span != input_ids[row][supervised[row]].tolist()
                            for row, span in enumerate(spans)], device=new_ids.device)
    return {**kwargs, "input_ids": new_ids, "attention_mask": new_mask}, new_labels, changed


def enable_draft_head(trainer, steps, decay, arch="eagle1", targets="reference"):
    """Train only an EAGLE draft head on the frozen model; checkpoints hold only the head.

    Each checkpoint keeps the head under ``mtp_draft.*`` in ``model.safetensors`` (for
    resuming) and a vLLM draft directory in ``eagle/``.
    """
    if steps < 1 or not 0 < decay <= 1 or not math.isfinite(decay):
        raise ValueError("draft_head_steps must be >= 1 and draft_head_step_decay in (0, 1]")
    if targets not in {"reference", "greedy"}:
        # reference: the data's transcripts; greedy: the frozen model's own greedy outputs.
        raise ValueError("draft head targets must be reference or greedy")
    model = trainer.model
    thinker = model.thinker
    head = DraftHead.from_thinker(thinker, arch)
    model.add_module(DRAFT, head)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(f"{DRAFT}."))
    lm_head = thinker.lm_head
    embed_tokens = thinker.get_input_embeddings()
    forward = model.forward
    pending, pending_changed = [], []
    # The checkpoint's generation_config.json belongs to the top-level model; the thinker's own
    # generation config carries no special tokens.
    generation = model.generation_config
    pad_id = generation.pad_token_id
    eos_ids = set(generation.eos_token_id if isinstance(generation.eos_token_id, list)
                  else [generation.eos_token_id])
    if targets == "greedy" and (pad_id is None or None in eos_ids):
        raise ValueError("greedy draft targets need pad and end tokens in the model's generation_config")

    def draft_forward(self, *args, labels=None, **kwargs):
        if labels is None or args or not self.training:
            return forward(*args, labels=labels, **kwargs)
        captured = {}

        def capture(module, inputs):
            # Keep the final-norm output and project nothing: the head computes its own targets.
            captured["hidden"] = inputs[0]
            return (inputs[0][:, :0],)

        if targets == "greedy":
            thinker.eval()
            try:
                kwargs, labels, changed = greedy_relabel(thinker, kwargs, labels, pad_id, eos_ids)
            finally:
                thinker.train(self.training)
            pending_changed.append(changed)
        aux = {}
        hooks = [lm_head.register_forward_pre_hook(capture),
                 *capture_aux(thinker.model.layers, head.aux_layers, aux)]
        thinker.eval()
        try:
            with torch.no_grad():
                forward(**kwargs, use_cache=False)
                embeds = embed_tokens(kwargs["input_ids"])
        finally:
            for hook in hooks:
                hook.remove()
            thinker.train(self.training)
        mask = kwargs.get("attention_mask")
        if mask is not None and (mask[:, 1:] > mask[:, :-1]).any():
            # Positions and the causal key range assume right padding.
            raise ValueError("draft head training needs right-padded batches")
        loss, stats, accepted = draft_loss(
            head, lm_head.weight, embeds, captured["hidden"], labels, steps, decay,
            aux=torch.cat([aux[j] for j in head.aux_layers], dim=-1) if head.eagle3 else None)
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
        if pending_changed:
            # Share of rows whose greedy output differs from the data's transcript.
            self.custom_metrics["train"]["draft_greedy_changed"].update(torch.cat(pending_changed).float())
            pending_changed.clear()
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
                           "step_decay": decay, "arch": arch, "targets": targets,
                           "method": "eagle3" if head.eagle3 else "eagle"})

    model.forward = MethodType(draft_forward, model)
    trainer.log = MethodType(log_with_draft, trainer)
    trainer._save_model = MethodType(save_draft, trainer)
    return head
