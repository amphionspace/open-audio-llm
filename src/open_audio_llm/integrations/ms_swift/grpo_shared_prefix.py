"""Score a GRPO micro-batch as one sequence: the shared prompt once, then every completion.

A micro-batch holds the generations of one prompt: the same audio and prompt tokens, then ~18 completion
tokens each. ms-swift scores every row separately, so the audio encoder, the ~200-300 prompt positions and
the LM head run once per generation, in the old-policy, reference and training forwards and in backward.
Here the rows become ``prompt + completion_1 + ... + completion_n`` with a mask that lets each completion
see the prompt and its own earlier tokens only, and position ids that restart after the prompt for every
completion. Each completion then sees exactly what it saw alone, so the log-probabilities are the same
function of the weights (up to kernel rounding), and backward sums the prompt's gradient over completions.

With one optimizer step per generation, the old-policy forward of ms-swift 4.5.3 recomputes what the
training forward already yields (``old = logps.detach()`` when it is missing), so it is skipped.

Rows that do not share a prompt, audio and per-row extras fall back to ms-swift's path.
"""

from __future__ import annotations

import os

import torch
from trl.trainer.utils import selective_log_softmax

# Model inputs that are one per row and must agree across the packed rows.
_PER_ROW = ("input_features", "feature_attention_mask", "enroll_n_frames", "catalog_task")


def pack_shared_prefix(model_inputs: dict, logits_to_keep: int, dtype: torch.dtype):
    """Packed inputs and the logit row that predicts each completion token, or None to fall back."""
    ids, mask = model_inputs["input_ids"], model_inputs.get("attention_mask")
    # ms-swift writes use_cache / logits_to_keep into the shared dict on its own forwards.
    if mask is None or mask.dim() != 2 or set(model_inputs) - {"input_ids", "attention_mask", "use_cache",
                                                               "logits_to_keep", *_PER_ROW}:
        return None
    rows, length = ids.shape
    prompt = length - logits_to_keep
    if rows < 2 or prompt <= 0 or not bool(mask[:, :prompt].all()):
        return None
    if not torch.equal(ids[:, :prompt], ids[:1, :prompt].expand(rows, -1)):
        return None
    tail = mask[:, prompt:]
    lens = tail.sum(1)
    columns = torch.arange(tail.shape[1], device=tail.device)
    if not torch.equal(tail.bool(), columns[None] < lens[:, None]):
        return None  # completions are not right-padded runs
    packed = {}
    for key in _PER_ROW:
        value = model_inputs.get(key)
        if value is None:
            continue
        if value.shape[0] != rows or not torch.equal(value, value[:1].expand_as(value)):
            return None
        packed[key] = value[:1]
    device = ids.device
    lens_list = lens.tolist()
    starts = torch.tensor([0, *lens_list[:-1]], device=device).cumsum(0) + prompt
    tokens = torch.cat([ids[row, prompt:prompt + n] for row, n in enumerate(lens_list)])
    total = prompt + len(tokens)
    segment = torch.cat([torch.zeros(prompt, dtype=torch.long, device=device),
                         torch.repeat_interleave(torch.arange(1, rows + 1, device=device), lens)])
    position = torch.cat([torch.arange(prompt, device=device),
                          *(torch.arange(prompt, prompt + n, device=device) for n in lens_list)])
    order = torch.arange(total, device=device)
    allowed = (order[None, :] <= order[:, None]) & ((segment[None, :] == 0) | (segment[None, :] == segment[:, None]))
    attention = torch.zeros(1, 1, total, total, dtype=dtype, device=device)
    attention.masked_fill_(~allowed, torch.finfo(dtype).min)
    # Token j of a completion is predicted at the prompt's last position (j = 0) or at its own token j - 1.
    within = torch.cat([torch.arange(n, device=device) for n in lens_list])
    start = torch.repeat_interleave(starts, lens)
    predictor = torch.where(within == 0, torch.full_like(within, prompt - 1), start + within - 1)
    packed.update(input_ids=torch.cat([ids[0, :prompt], tokens])[None], attention_mask=attention,
                  position_ids=position[None])
    return packed, predictor, tokens, lens


def shared_prefix_logps(model, packed, predictor, tokens, lens, logits_to_keep, temperature,
                        compute_entropy=False, entropy_fn=None):
    """Per-token log-probabilities [rows, logits_to_keep] (zeros past each completion), as ms-swift returns."""
    logits = model(**packed, use_cache=False).logits[0].index_select(0, predictor)
    logits = logits / temperature
    flat = selective_log_softmax(logits, tokens)
    rows = len(lens)
    out = flat.new_zeros(rows, logits_to_keep)
    column = torch.cat([torch.arange(n, device=flat.device) for n in lens.tolist()])
    row = torch.repeat_interleave(torch.arange(rows, device=flat.device), lens)
    out = out.index_put((row, column), flat)
    entropies = None
    if compute_entropy:
        entropies = flat.new_zeros(rows, logits_to_keep).index_put((row, column), entropy_fn(logits))
    return out, entropies


def install_efficient_logps(trainer) -> None:
    """Shared-prefix scoring and no redundant old-policy forward (env GRPO_EFFICIENT_LOGPS=1).

    GRPO_EFFICIENT_LOGPS_VERIFY=N also runs ms-swift's own path for the first N gradient-free packed
    forwards and logs the difference as ``efficient_logps/max_abs_diff`` and ``mean_abs_diff``.
    """
    original_local = trainer._get_logps_via_local_forward
    original_logps = trainer._get_per_token_logps_and_entropies
    verify = [int(os.environ.get("GRPO_EFFICIENT_LOGPS_VERIFY", "0"))]
    metrics = trainer._metrics["train"]
    try:
        from trl.trainer.utils import entropy_from_logits
    except ImportError:  # pragma: no cover - older trl
        from swift.rlhf_trainers.utils import entropy_from_logits

    def per_token_logps(model, model_inputs, grpo_batch, compute_entropy=False, origin_data=None):
        on_policy = not trainer.old_policy() and not getattr(trainer, "kl_in_reward", False)
        if (on_policy and model is trainer.model and trainer.ref_model is not None
                and not torch.is_grad_enabled() and trainer.model.training):
            metrics["efficient_logps/old_policy_forward_skipped"].append(1.0)
            return None, None
        return original_logps(model, model_inputs, grpo_batch, compute_entropy=compute_entropy,
                              origin_data=origin_data)

    def local_forward(model, model_inputs, logits_to_keep, input_ids, compute_entropy=False):
        dtype = next(model.parameters()).dtype
        packed = pack_shared_prefix(model_inputs, logits_to_keep, dtype)
        if packed is None:
            metrics["efficient_logps/packed"].append(0.0)
            return original_local(model, model_inputs, logits_to_keep, input_ids, compute_entropy=compute_entropy)
        metrics["efficient_logps/packed"].append(1.0)
        logps, entropies = shared_prefix_logps(model, *packed, logits_to_keep, trainer.temperature,
                                               compute_entropy, entropy_from_logits)
        if verify[0] > 0 and not torch.is_grad_enabled():
            verify[0] -= 1
            reference, _ = original_local(model, dict(model_inputs), logits_to_keep, input_ids)
            tail = model_inputs["attention_mask"][:, -logits_to_keep:].bool()
            diff = (logps - reference).abs()[tail]
            metrics["efficient_logps/max_abs_diff"].append(diff.max().item())
            metrics["efficient_logps/mean_abs_diff"].append(diff.mean().item())
        return logps, entropies

    trainer._get_per_token_logps_and_entropies = per_token_logps
    trainer._get_logps_via_local_forward = local_forward
