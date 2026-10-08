"""Full-LLM SFT with token-mean cross-entropy and frozen-base ASR replay KL."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from types import MethodType

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

LOG = logging.getLogger(__name__)
# One FP32 block is about 74 MiB for Qwen3-ASR's 151,936-token vocabulary.
_LOSS_CHUNK_TOKENS = 128


class _ChunkedCrossEntropy(torch.autograd.Function):
    """Recompute softmax by block instead of saving a full FP32 vocabulary tensor."""

    @staticmethod
    def forward(ctx, logits, targets):
        ctx.save_for_backward(logits, targets)
        result = logits.new_empty(len(targets), dtype=torch.float32)
        for start in range(0, len(targets), _LOSS_CHUNK_TOKENS):
            end = start + _LOSS_CHUNK_TOKENS
            result[start:end] = F.cross_entropy(
                logits[start:end].float(), targets[start:end], reduction="none")
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        logits, targets = ctx.saved_tensors
        # Write directly into the result: concatenating per-block gradients would
        # briefly allocate a second full token-by-vocabulary gradient matrix.
        gradient = torch.empty_like(logits)
        for start in range(0, len(targets), _LOSS_CHUNK_TOKENS):
            end = start + _LOSS_CHUNK_TOKENS
            target = targets[start:end]
            valid = target != -100
            probability = logits[start:end].float().softmax(-1)
            probability[torch.arange(len(target), device=target.device), target.clamp_min(0)] -= valid.to(probability.dtype)
            probability.mul_((grad_output[start:end] * valid).unsqueeze(-1))
            gradient[start:end].copy_(probability)
        return gradient, None


def supervised_forward(model, head, inputs, labels):
    """Project only supervised positions; Qwen-ASR 0.0.6 otherwise projects audio too."""
    shifted = F.pad(labels[:, 1:], (0, 1), value=-100)
    mask = shifted != -100
    rows = torch.arange(labels.shape[0], device=labels.device)[:, None].expand_as(mask)[mask]
    hook = head.register_forward_pre_hook(lambda module, args: (args[0][mask],))
    try:
        outputs = model(**inputs, use_cache=False)
    finally:
        hook.remove()
    return outputs, shifted[mask], rows, mask.sum(-1)


class _ChunkedLinearCrossEntropy(torch.autograd.Function):
    """Token cross-entropy for every label token in one matmul.

    Audio positions are omitted before this runs. Logits are not saved; backward
    recomputes the same projection. The whole label span is one matrix so the
    temporary vocabulary tensor is ``label_tokens * vocab``, not ``B * T * vocab``.
    """

    @staticmethod
    def forward(ctx, hidden, weight, targets):
        ctx.save_for_backward(hidden, weight, targets)
        logits = F.linear(hidden, weight).float()
        return F.cross_entropy(logits, targets, reduction="none")

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        hidden, weight, targets = ctx.saved_tensors
        probability = F.linear(hidden, weight).float().softmax(-1)
        valid = targets != -100
        probability[torch.arange(hidden.shape[0], device=targets.device), targets.clamp_min(0)] -= valid.to(probability.dtype)
        probability.mul_((grad_output * valid).unsqueeze(-1))
        grad = probability.to(hidden.dtype)
        return grad @ weight, grad.transpose(0, 1) @ hidden, None


def _supervised_positions(labels):
    shifted = F.pad(labels[:, 1:], (0, 1), value=-100)
    mask = shifted != -100
    if not mask.any():
        raise ValueError("Batch has no supervised tokens")
    return shifted, mask


def chunked_supervised_loss(model, head, inputs, labels):
    """Local token-mean loss, projecting only positions that predict a label.

    ``lm_head`` is bypassed so a long audio prompt never becomes a vocabulary
    matrix. The returned scalar matches ``ForCausalLMLoss`` reduction ``mean``
    on this rank; the caller applies the trainer's global token denominator.
    """
    if head.bias is not None:
        raise ValueError("Chunked supervised loss requires an unbiased lm_head")
    shifted, mask = _supervised_positions(labels)
    inputs = dict(inputs)
    inputs.pop("use_cache", None)
    captured = {}

    def capture(hidden):
        captured["hidden"] = hidden
        return hidden.new_empty(0)

    original = head.forward
    head.forward = capture
    try:
        outputs = model(**inputs, use_cache=False)
    finally:
        head.forward = original
    hidden = captured["hidden"]
    if hidden.shape[:mask.ndim] != mask.shape:
        raise ValueError(
            f"Hidden states {tuple(hidden.shape)} do not match labels {tuple(mask.shape)}")
    selected = hidden[mask]
    targets = shifted[mask]
    token_losses = _ChunkedLinearCrossEntropy.apply(selected, head.weight, targets)
    with torch.no_grad():
        predicted = F.linear(selected, head.weight).argmax(-1)
        token_acc = predicted.eq(targets).float()
    outputs.logits = None
    return outputs, token_losses.sum() / targets.shape[0], token_acc


def install_chunked_supervised_loss(trainer):
    """Replace the dense vocabulary loss on the normal SFT path.

    Retention already projects supervised positions and is left unchanged.
    Evaluation gathers predictions only when ``prediction_loss_only`` is false;
    the dense logits tensor would otherwise be copied back at ``eval_steps``.
    """
    head = trainer.model.thinker.lm_head
    if head.bias is not None:
        raise ValueError("Chunked supervised loss requires an unbiased lm_head")
    if getattr(trainer.args, "enable_dft_loss", False) or getattr(trainer.args, "enable_channel_loss", False):
        raise ValueError("Chunked supervised loss does not implement DFT or channel loss")
    if trainer.label_smoother is not None:
        raise ValueError("Chunked supervised loss does not implement label smoothing")
    trainer.args.prediction_loss_only = True
    LOG.warning(
        "Supervised loss projects every label token in one matmul "
        "and does not allocate [batch, sequence, vocab] logits"
    )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = dict(inputs)
        if inputs.pop("compute_loss_func", None) is not None:
            raise ValueError("A custom loss function cannot be combined with chunked supervised loss")
        if inputs.pop("loss_scale", None) is not None:
            raise ValueError("Token loss_scale cannot be combined with chunked supervised loss")
        inputs.pop("text_position_ids", None)
        inputs.pop("channel", None)
        labels = inputs.pop("labels")
        outputs, loss, token_acc = chunked_supervised_loss(model, head, inputs, labels)
        shift_count = (labels[:, 1:] != -100).sum()
        if num_items_in_batch is not None:
            loss = loss * (shift_count / num_items_in_batch)
        if (getattr(self.args, "average_tokens_across_devices", False)
                and self.model_accepts_loss_kwargs and num_items_in_batch is not None):
            loss = loss * self.accelerator.num_processes
        mode = "train" if model.training else "eval"
        self.custom_metrics[mode]["token_acc"].update(token_acc.detach())
        outputs.loss = loss
        if not compute_loss.logged:
            LOG.warning(
                "Chunked supervised loss tokens: supervised %s / padded sequence %s",
                int(shift_count), int(labels.numel()),
            )
            compute_loss.logged = True
        return (loss, outputs) if return_outputs else loss

    compute_loss.logged = False
    trainer.compute_loss = MethodType(compute_loss, trainer)


def average_answer_parts(values, rows, counts, prefix=None):
    if prefix is None:
        return values.new_zeros(len(counts)).scatter_add(0, rows, values) / counts.clamp_min(1)
    prefix_counts = counts.new_zeros(len(counts)).scatter_add(0, rows, prefix.long())
    header = values.new_zeros(len(counts)).scatter_add(0, rows, values * prefix)
    body = values.new_zeros(len(counts)).scatter_add(0, rows, values * ~prefix)
    return header / prefix_counts.clamp_min(1) + body / (counts - prefix_counts).clamp_min(1)


def token_mean_loss(example_losses, counts, num_items_in_batch=None, world_size=1):
    """Recover the token sum from per-example means and divide by supervised tokens.

    A length-10 transcript therefore contributes ten times the gradient of a
    one-token transcript. ``num_items_in_batch`` is the global token count for
    this optimizer step; multiplying by ``world_size`` cancels DDP's average.
    """
    weighted = example_losses * counts
    if num_items_in_batch is None:
        return weighted.sum() / counts.sum().clamp_min(1)
    return weighted.sum() * world_size / num_items_in_batch


def sample_losses(logits, targets, rows, counts, teacher_logits=None, ordinary=None, kl_weight=1.0, prefix=None):
    """Average tokens within examples, then combine CE and KL on ordinary ASR only."""
    token_ce = _ChunkedCrossEntropy.apply(logits, targets)
    ce = average_answer_parts(token_ce, rows, counts, prefix)
    kl = ce.new_zeros(len(counts))
    if teacher_logits is not None:
        selected = logits[ordinary]
        if selected.shape != teacher_logits.shape:
            raise ValueError("Teacher and student supervised tokens differ")
        # Bound the temporary FP32 vocabulary distributions on long batches.
        pieces = []
        for student, teacher in zip(selected.split(128), teacher_logits.detach().split(128)):
            pieces.append(F.kl_div(student.float().log_softmax(-1), teacher.float().softmax(-1),
                                  reduction="none").sum(-1))
        token_kl = torch.cat(pieces)
        kl = average_answer_parts(token_kl, rows[ordinary], counts,
                                  None if prefix is None else prefix[ordinary])
    return ce + kl_weight * kl, ce, kl


def select_examples(inputs, indexes):
    result = {}
    for key, value in inputs.items():
        if key == "position_ids":
            result[key] = value.index_select(1 if value.ndim == 3 else 0, indexes)
        elif isinstance(value, torch.Tensor):
            result[key] = value.index_select(0, indexes)
        else:
            result[key] = value
    return result


def install_retention_objective(trainer, args):
    from qwen_asr import Qwen3ASRModel

    if args.tuner_type != "full" or not args.retention_teacher:
        raise ValueError("Token-mean replay requires full LLM training and --retention_teacher")
    if args.deepspeed or args.use_logits_to_keep or trainer.optimizer is not None:
        raise ValueError("Replay objective requires DDP, untrimmed labels and installation before optimizer creation")
    coefficient = float(args._catalog_config["objective"]["replay_kl_weight"])
    if args._catalog_config["objective"].get("separate_prefix", False):
        LOG.warning("separate_prefix is ignored; the objective averages supervised tokens")
    if not math.isfinite(coefficient) or coefficient <= 0:
        raise ValueError("replay_kl_weight must be finite and positive")
    student = trainer.model
    thinker = student.thinker
    # Training forwards explicitly disable caching; exported configs should retain
    # normal inference defaults even when gradient checkpointing is enabled.
    thinker.model.config.use_cache = True
    thinker.generation_config.use_cache = True
    if any(p.requires_grad for p in thinker.audio_tower.parameters()) and args.audio_encoder_parallel:
        raise ValueError("Trainable audio encoder requires audio_encoder_parallel=false")
    if not all(p.requires_grad for p in thinker.model.parameters()) or not thinker.lm_head.weight.requires_grad:
        raise ValueError("All LLM parameters, including embeddings and output head, must be trainable")
    # AdamW needs FP32 master weights/moments for small full-model updates.
    # Trainer's BF16 autocast still performs the expensive matrix multiplies in BF16.
    for parameter in student.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    teacher = Qwen3ASRModel.from_pretrained(
        args.retention_teacher, dtype=torch.bfloat16,
        device_map=str(trainer.accelerator.device), attn_implementation="sdpa",
    ).model.eval().requires_grad_(False)
    teacher.thinker.config.use_cache = False
    trainer.retention_teacher = teacher
    trainer.model_accepts_loss_kwargs = True
    trainer.args.average_tokens_across_devices = True

    def count_supervised_tokens(self, batches, device):
        count = torch.tensor(
            sum(int((batch["labels"][:, 1:] != -100).sum()) for batch in batches), device=device)
        return self.accelerator.gather(count).sum()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = dict(inputs)
        tasks = inputs.pop("catalog_task")
        labels = inputs.pop("labels")
        inputs.pop("compute_loss_func", None)
        inputs.pop("text_position_ids", None)
        if inputs.pop("loss_scale", None) is not None:
            raise ValueError("Token loss_scale cannot be combined with token-mean replay")
        outputs, targets, rows, counts = supervised_forward(model, thinker.lm_head, inputs, labels)
        indexes = (tasks == 0).nonzero().flatten()
        teacher_logits = None
        ordinary = tasks[rows] == 0
        if indexes.numel():
            teacher_inputs = select_examples(inputs, indexes)
            teacher_inputs.pop("enrollment_lengths", None)
            # Accelerate wraps only the student forward in autocast.
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                teacher_outputs, teacher_targets, _, _ = supervised_forward(
                    teacher, teacher.thinker.lm_head, teacher_inputs,
                    labels.index_select(0, indexes),
                )
            if not torch.equal(targets[ordinary], teacher_targets):
                raise ValueError("Teacher and student labels differ")
            teacher_logits = teacher_outputs.logits
        losses, ce, kl = sample_losses(outputs.logits, targets, rows, counts,
                                      teacher_logits, ordinary, coefficient)
        mode = "train" if model.training else "eval"
        metrics = self.custom_metrics[mode]
        # Create the same metrics on every rank even when a batch lacks a task.
        for task, name in enumerate(("asr", "ts_positive", "ts_negative", "sot")):
            metric = metrics[f"ce_{name}"]
            selected = ce.detach()[tasks == task]
            if selected.numel():
                metric.update(selected)
        metric = metrics["replay_kl"]
        if indexes.numel():
            metric.update(kl.detach()[indexes])
        # counts restores the token sum; the denominator is every supervised token.
        loss = token_mean_loss(
            losses, counts, num_items_in_batch, self.accelerator.num_processes)
        outputs.loss = loss
        return (loss, outputs) if return_outputs else loss

    trainer._get_num_items_in_batch = MethodType(count_supervised_tokens, trainer)
    trainer.compute_loss = MethodType(compute_loss, trainer)
    summary = {
        "teacher": args.retention_teacher,
        "objective": "sum_tokens(CE + ordinary_asr * replay_kl_weight * KL(base || student)) / supervised_tokens",
        "reduction": "token_mean",
        "separate_prefix": False,
        "replay_kl_weight": coefficient,
        "trainable_parameters": sum(p.numel() for p in student.parameters() if p.requires_grad),
        "frozen_parameters": sum(p.numel() for p in student.parameters() if not p.requires_grad),
        "audio_trainable_parameters": sum(p.numel() for p in thinker.audio_tower.parameters() if p.requires_grad),
        "llm_trainable_parameters": sum(p.numel() for name, p in thinker.named_parameters()
                                        if p.requires_grad and name.startswith(("model.", "lm_head."))),
        "learning_rates": {"llm": args.learning_rate, "encoder": args.vit_lr, "aligner": args.aligner_lr},
        "master_dtype": "float32", "compute_dtype": "bfloat16",
        "teacher_trainable_parameters": sum(p.numel() for p in teacher.parameters() if p.requires_grad),
    }
    LOG.warning("Full LLM replay: %s", json.dumps(summary))
    if trainer.is_world_process_zero():
        path = Path(trainer.args.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "retention-objective.json").write_text(json.dumps(summary, indent=2) + "\n")
