"""DeepSeek-V3 style multi-token prediction (MTP) trained jointly with Qwen3-ASR SFT.

Unlike the frozen-backbone draft heads in ``draft_head.py``, the MTP module is trained together
with the model: the loss is the ordinary next-token cross entropy plus ``weight`` times the MTP
module's cross entropy on the token after next (DeepSeek-V3, section 2.2). The backbone, its
token embedding and its ``lm_head`` all receive the MTP gradient; the embedding and ``lm_head``
are shared with the module, as in DeepSeek-V3.

The module has the layout vLLM 0.18 serves for ``method: "mtp"`` with Qwen3-Next
(``vllm/model_executor/models/qwen3_next_mtp.py``), with a dense Qwen3 decoder layer:

    x = fc(cat(pre_fc_norm_embedding(embed(t[i + 1])), pre_fc_norm_hidden(h[i])))
    h' = norm(Qwen3DecoderLayer(x))          # predicts t[i + 2] through the shared lm_head

where ``h[i]`` is the backbone's final-norm output at ``i``. A further speculative step feeds
``h'`` back one position later with the next drafted token and reuses the same layer, attending
to the first pass and the proposal's earlier steps; ``steps > 1`` trains those steps under
teacher forcing (mask and row selection shared with ``draft_head.py``), weighted ``decay**(k-1)``.

Checkpoints keep the backbone in ``model.safetensors`` exactly as before (so every existing
loader and evaluation reads them unchanged) and the module in ``mtp/model.safetensors`` with
``mtp/config.json``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import MethodType

import torch
import torch.nn.functional as F
from torch import nn

from .draft_head import _MLP, RMSNorm, _rotate_half, draft_rows, unrolled_mask
from .retention import _ChunkedCrossEntropy, supervised_forward

MTP = "mtp"


class _QwenAttention(nn.Module):
    def __init__(self, hidden, heads, kv_heads, head_dim, eps):
        super().__init__()
        self.q_proj = nn.Linear(hidden, heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(heads * head_dim, hidden, bias=False)
        self.q_norm = RMSNorm(head_dim, eps)
        self.k_norm = RMSNorm(head_dim, eps)


class _QwenLayer(nn.Module):
    """Parameter names equal transformers' ``Qwen3DecoderLayer`` and vLLM's."""

    def __init__(self, hidden, intermediate, heads, kv_heads, head_dim, eps):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden, eps)
        self.self_attn = _QwenAttention(hidden, heads, kv_heads, head_dim, eps)
        self.post_attention_layernorm = RMSNorm(hidden, eps)
        self.mlp = _MLP(hidden, intermediate)


class MTPModule(nn.Module):
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
        self.pre_fc_norm_embedding = RMSNorm(hidden_size, rms_norm_eps)
        self.pre_fc_norm_hidden = RMSNorm(hidden_size, rms_norm_eps)
        self.fc = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.layers = nn.ModuleList([_QwenLayer(hidden_size, intermediate_size, num_attention_heads,
                                                num_key_value_heads, head_dim, rms_norm_eps)])
        self.norm = RMSNorm(hidden_size, rms_norm_eps)

    @classmethod
    def from_thinker(cls, thinker):
        """Starts by passing the backbone's output through: ``fc = [0 | I]``, the last decoder
        layer's weights with zero output projections (``o_proj``, ``down_proj``), unit norms.

        The first pass then returns ``h / rms(h)`` for the backbone's final-norm output ``h``:
        the backbone's own next-token logits up to a positive scale, a few nats from the token
        after next. Running the copied layer on that normalised input instead started at about
        14 nats, worse than uniform (11.9). The embedding half of ``fc`` and both output
        projections learn from zero.
        """
        text = thinker.config.text_config
        module = cls(
            hidden_size=text.hidden_size, intermediate_size=text.intermediate_size,
            num_attention_heads=text.num_attention_heads, num_key_value_heads=text.num_key_value_heads,
            head_dim=getattr(text, "head_dim", text.hidden_size // text.num_attention_heads),
            rms_norm_eps=text.rms_norm_eps, rope_theta=float(text.rope_theta),
        )
        reference = thinker.lm_head.weight
        module.to(device=reference.device, dtype=reference.dtype)
        with torch.no_grad():
            module.layers[0].load_state_dict(thinker.model.layers[-1].state_dict())
            module.layers[0].self_attn.o_proj.weight.zero_()
            module.layers[0].mlp.down_proj.weight.zero_()
            hidden = text.hidden_size
            module.fc.weight.zero_()
            module.fc.weight[:, hidden:].copy_(torch.eye(hidden, dtype=module.fc.weight.dtype))
        return module

    def _rope(self, length, device):
        dim = self.config["head_dim"]
        inv_freq = 1.0 / (self.config["rope_theta"] ** (
            torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
        angles = torch.outer(torch.arange(length, device=device, dtype=torch.float32), inv_freq)
        angles = torch.cat((angles, angles), dim=-1)
        return angles.cos(), angles.sin()

    def unrolled(self, embeds, hidden, steps):
        """Per step, the post-norm features fed to ``lm_head`` for every position.

        ``embeds[b, q]`` embeds token ``q + 1``; ``hidden[b, q]`` is the backbone's final-norm
        output at ``q``. Step ``k`` at ``q`` belongs to the proposal whose first row is
        ``q - k + 1`` and takes step ``k - 1``'s (post-norm) output at ``q - 1``, as vLLM feeds
        back what the MTP module returns.
        """
        batch, length, _ = hidden.shape
        heads, kv_heads, head_dim = (self.config[k] for k in
                                     ("num_attention_heads", "num_key_value_heads", "head_dim"))
        cos, sin = self._rope(length, hidden.device)
        layer = self.layers[0]
        attention = layer.self_attn
        embeds = self.pre_fc_norm_embedding(embeds)
        keys, values, features = [], [], []
        for step in range(1, steps + 1):
            previous = hidden if step == 1 else F.pad(features[-1][:, :-1], (0, 0, 1, 0))
            x = self.fc(torch.cat((embeds, self.pre_fc_norm_hidden(previous)), dim=-1))
            normed = layer.input_layernorm(x)
            q = attention.q_norm(attention.q_proj(normed).view(batch, length, heads, head_dim)).transpose(1, 2)
            k = attention.k_norm(attention.k_proj(normed).view(batch, length, kv_heads, head_dim)).transpose(1, 2)
            v = attention.v_proj(normed).view(batch, length, kv_heads, head_dim).transpose(1, 2)
            q = q * cos.to(q.dtype) + _rotate_half(q) * sin.to(q.dtype)
            k = k * cos.to(k.dtype) + _rotate_half(k) * sin.to(k.dtype)
            keys.append(k.repeat_interleave(heads // kv_heads, dim=1))
            values.append(v.repeat_interleave(heads // kv_heads, dim=1))
            out = F.scaled_dot_product_attention(
                q, torch.cat(keys, dim=2), torch.cat(values, dim=2),
                attn_mask=None if step == 1 else unrolled_mask(length, step, hidden.device),
                is_causal=step == 1,
            )
            h = x + attention.o_proj(out.transpose(1, 2).reshape(batch, length, heads * head_dim))
            h = h + layer.mlp(layer.post_attention_layernorm(h))
            features.append(self.norm(h))
        return features


def mtp_loss(module, weight, embeds, hidden, labels, steps, decay):
    """Weighted mean over steps of the token-mean cross entropy on the token after next.

    ``embeds`` embeds the input ids (token ``q`` at ``q``); ``hidden`` is the backbone's
    final-norm output. Returns the loss and, per step, the rows' CE and argmax correctness.
    """
    embeds = F.pad(embeds[:, 1:], (0, 0, 0, 1))
    features = module.unrolled(embeds, hidden, steps)
    targets = F.pad(labels[:, 2:], (0, 2), value=-100).flatten()
    loss, total_weight, stats = 0.0, 0.0, []
    for step, (feature, valid) in enumerate(zip(features, draft_rows(labels, steps)), start=1):
        index = valid.flatten().nonzero().squeeze(1)
        step_weight = decay ** (step - 1)
        total_weight += step_weight
        selected = feature.flatten(0, 1).index_select(0, index)
        target = targets.index_select(0, index)
        logits = selected @ weight.t()
        ce = _ChunkedCrossEntropy.apply(logits, target)
        if len(index):
            loss = loss + step_weight * ce.mean()
        else:
            # Keep every module parameter in the graph so all DDP ranks reduce gradients.
            loss = loss + feature.sum() * 0
        stats.append((ce.detach(), logits.detach().argmax(-1) == target))
    return loss / total_weight, stats


def save_mtp(module, directory, *, vocab_size, max_position_embeddings, info):
    """``directory`` is also the vLLM draft directory (``open_audio_llm.tsasr.vllm_mtp``)."""
    from safetensors.torch import save_file

    from ...tsasr.vllm_mtp import vllm_config_fields

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().contiguous().cpu() for k, v in module.state_dict().items()},
              directory / "model.safetensors")
    config = {**module.config, "vocab_size": vocab_size, "num_nextn_predict_layers": 1, **info}
    config.update(vllm_config_fields(config, max_position_embeddings=max_position_embeddings))
    (directory / "config.json").write_text(json.dumps(config, indent=1) + "\n")


def _own_lr_groups(trainer, module, lr):
    if lr is None:
        return
    if not 0 < lr or not math.isfinite(lr):
        raise ValueError("joint_mtp_lr must be positive")
    # The module starts from the backbone-based init while the backbone fine-tunes at a few
    # 1e-6; its own parameter groups keep the scheduler's shape at this peak instead.
    create_optimizer = trainer.create_optimizer
    own = {id(p) for p in module.parameters()}

    def create_optimizer_with_mtp():
        optimizer = create_optimizer()
        if any(group.get("joint_mtp") for group in optimizer.param_groups):
            return optimizer
        extra = []
        for group in optimizer.param_groups:
            params = [p for p in group["params"] if id(p) in own]
            if params:
                group["params"] = [p for p in group["params"] if id(p) not in own]
                extra.append({**{k: v for k, v in group.items() if k != "params"},
                              "params": params, "lr": lr, "joint_mtp": True})
        optimizer.param_groups[:] = [group for group in optimizer.param_groups if group["params"]]
        for group in extra:
            optimizer.add_param_group(group)
        return optimizer

    trainer.create_optimizer = create_optimizer_with_mtp


def _save_split(trainer, module, model, thinker, weight, steps, decay, init):
    """Backbone into model.safetensors as before, the module into mtp/."""
    save = trainer._save_model

    def save_with_mtp(self, output_dir=None, state_dict=None):
        output_dir = output_dir or self.args.output_dir
        if state_dict is None:
            state_dict = self.model.state_dict()
        save(output_dir, {k: v for k, v in state_dict.items() if not k.startswith(f"{MTP}.")})
        if self.args.should_save:
            save_mtp(module, Path(output_dir) / MTP, vocab_size=thinker.config.text_config.vocab_size,
                     max_position_embeddings=thinker.config.text_config.max_position_embeddings,
                     info={"backbone": getattr(model.config, "_name_or_path", None),
                           "global_step": self.state.global_step, "joint_mtp_weight": weight,
                           "train_steps": steps, "step_decay": decay, "init": init,
                           "layout": "qwen3_next_mtp with a dense Qwen3 decoder layer"})

    trainer._save_model = MethodType(save_with_mtp, trainer)


def enable_joint_mtp(trainer, weight, steps=1, decay=0.8, init=None, lr=None):
    """Add a DeepSeek-V3 style MTP module to supervised-head SFT and train it with the model.

    The training loss becomes ``CE(next) + weight * MTP_CE(after next)``; both are per-microbatch
    token means over the assistant turn, the scale ms-swift's supervised-head path already uses.
    """
    if not 0 < weight or not math.isfinite(weight):
        raise ValueError("joint_mtp_weight must be positive")
    if steps < 1 or not 0 < decay <= 1 or not math.isfinite(decay):
        raise ValueError("joint_mtp_steps must be >= 1 and joint_mtp_step_decay in (0, 1]")
    if trainer.args.acc_strategy != "token":
        raise ValueError("joint MTP runs on the supervised-head path, which reports token accuracy only")
    model = trainer.model
    thinker = model.thinker
    module = MTPModule.from_thinker(thinker)
    if init:
        from safetensors.torch import load_file

        module.load_state_dict(load_file(init))
    model.add_module(MTP, module)
    _own_lr_groups(trainer, module, lr)
    lm_head = thinker.lm_head
    embed_tokens = thinker.get_input_embeddings()
    forward = model.forward
    pending = []

    def joint_forward(self, *args, labels=None, **kwargs):
        if labels is None or args or not self.training:
            return forward(*args, labels=labels, **kwargs)
        mask = kwargs.get("attention_mask")
        if mask is not None and (mask[:, 1:] > mask[:, :-1]).any():
            # RoPE positions and the causal key range of the MTP module assume right padding.
            raise ValueError("joint MTP training needs right-padded batches")
        captured = {}
        # Registered before supervised_forward's own pre-hook, so it sees every position.
        hook = lm_head.register_forward_pre_hook(lambda m, a: captured.__setitem__("hidden", a[0]))
        try:
            outputs, targets, _, _ = supervised_forward(forward, lm_head, kwargs, labels)
        finally:
            hook.remove()
        logits = outputs.logits
        main = F.cross_entropy(logits.float(), targets)
        auxiliary, stats = mtp_loss(module, lm_head.weight, embed_tokens(kwargs["input_ids"]),
                                    captured["hidden"], labels, steps, decay)
        pending.append((logits.detach().argmax(-1) == targets, main.detach(), stats))
        return type(outputs)(loss=main + weight * auxiliary)

    log = trainer.log

    def log_with_mtp(self, logs, *args, **kwargs):
        if pending:
            metrics = self.custom_metrics["train"]
            for correct, _, _ in pending:
                metrics["token_acc"].update(correct)
            metrics["main_loss"].update(torch.stack([m for _, m, _ in pending]))
            for step in range(steps):
                metrics[f"mtp_ce_{step + 1}"].update(torch.cat([s[step][0] for _, _, s in pending]))
                metrics[f"mtp_acc_{step + 1}"].update(torch.cat([s[step][1] for _, _, s in pending]).float())
            pending.clear()
        return log(logs, *args, **kwargs)

    model.forward = MethodType(joint_forward, model)
    trainer.log = MethodType(log_with_mtp, trainer)
    _save_split(trainer, module, model, thinker, weight, steps, decay, init)
    return module


def install_grpo_mtp(trainer, weight, steps=1, decay=0.8, init=None, lr=None):
    """Keep training a joint MTP module during GRPO on the policy's own best completion.

    Each micro-batch holds one prompt's generations. After the GRPO loss, the completion with the
    highest advantage is run once more as an ordinary causal row (prompt + completion, right padded)
    and ``weight`` times the MTP cross entropy on its completion tokens joins the loss, so the module
    tracks the moving policy on the text it actually generates. Rows that are not right padded skip
    the MTP term (counted as ``mtp/skipped``).
    """
    if not 0 < weight or not math.isfinite(weight):
        raise ValueError("joint_mtp_weight must be positive")
    if steps < 1 or not 0 < decay <= 1 or not math.isfinite(decay):
        raise ValueError("joint_mtp_steps must be >= 1 and joint_mtp_step_decay in (0, 1]")
    model = trainer.model
    thinker = model.thinker
    module = MTPModule.from_thinker(thinker)
    if init:
        from safetensors.torch import load_file

        module.load_state_dict(load_file(init))
    model.add_module(MTP, module)
    _own_lr_groups(trainer, module, lr)
    lm_head = thinker.lm_head
    embed_tokens = thinker.get_input_embeddings()
    compute_loss = trainer.compute_loss

    def metric(name, value):
        trainer._metrics["train"][name].append(value)

    def compute_loss_with_mtp(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss = compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)
        if not (torch.is_grad_enabled() and self.model.training):
            return loss
        batch = inputs[0] if isinstance(inputs, list) else inputs
        model_inputs, grpo_batch = batch["model_inputs"], batch["grpo_batch"]
        ids, mask = model_inputs["input_ids"], model_inputs.get("attention_mask")
        completion = grpo_batch.completion_mask.bool()
        advantages = grpo_batch.advantages
        score = advantages if advantages.dim() == 1 else (advantages * completion).sum(1) / completion.sum(1).clamp_min(1)
        row = int(score.argmax())
        if mask is None or mask.dim() != 2 or (mask[row, 1:] > mask[row, :-1]).any() or not completion[row].any():
            metric("mtp/skipped", 1.0)
            return loss
        keep = completion.shape[1]
        labels = torch.full_like(ids[row:row + 1], -100)
        labels[0, -keep:] = torch.where(completion[row], ids[row, -keep:], torch.full_like(ids[row, -keep:], -100))
        single = {k: (v[row:row + 1] if torch.is_tensor(v) and v.dim() and v.shape[0] == len(ids) else v)
                  for k, v in model_inputs.items() if k not in ("use_cache", "logits_to_keep")}
        captured = {}
        hook = lm_head.register_forward_pre_hook(
            lambda m, a: (captured.__setitem__("hidden", a[0]), (a[0][:, :0],))[1])
        try:
            model(**single, use_cache=False)
        finally:
            hook.remove()
        auxiliary, stats = mtp_loss(module, lm_head.weight, embed_tokens(single["input_ids"]),
                                    captured["hidden"], labels, steps, decay)
        metric("mtp/skipped", 0.0)
        metric("mtp/loss", auxiliary.item())
        for step, (_, correct) in enumerate(stats, start=1):
            if len(correct):
                metric(f"mtp/acc_{step}", correct.float().mean().item())
        return loss + weight * auxiliary

    trainer.compute_loss = MethodType(compute_loss_with_mtp, trainer)
    _save_split(trainer, module, model, thinker, weight, steps, decay, init)
    return module
