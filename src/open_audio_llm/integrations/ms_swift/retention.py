"""Full-LLM SFT with equal example weight and frozen-base ASR replay KL."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from types import MethodType

import torch
import torch.nn.functional as F

LOG = logging.getLogger(__name__)


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


def average_answer_parts(values, rows, counts, prefix=None):
    if prefix is None:
        return values.new_zeros(len(counts)).scatter_add(0, rows, values) / counts.clamp_min(1)
    prefix_counts = counts.new_zeros(len(counts)).scatter_add(0, rows, prefix.long())
    header = values.new_zeros(len(counts)).scatter_add(0, rows, values * prefix)
    body = values.new_zeros(len(counts)).scatter_add(0, rows, values * ~prefix)
    return header / prefix_counts.clamp_min(1) + body / (counts - prefix_counts).clamp_min(1)


def sample_losses(logits, targets, rows, counts, teacher_logits=None, ordinary=None, kl_weight=1.0, prefix=None):
    """Average tokens within examples, then combine CE and KL on ordinary ASR only."""
    token_ce = F.cross_entropy(logits.float(), targets, reduction="none")
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
        raise ValueError("Sample-mean replay requires full LLM training and --retention_teacher")
    if args.deepspeed or args.use_logits_to_keep or trainer.optimizer is not None:
        raise ValueError("Replay objective requires DDP, untrimmed labels and installation before optimizer creation")
    coefficient = float(args._catalog_config["objective"]["replay_kl_weight"])
    separate_prefix = args._catalog_config["objective"].get("separate_prefix", False)
    asr_text_id = trainer.template.tokenizer.convert_tokens_to_ids("<asr_text>")
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

    def count_examples(self, batches, device):
        count = torch.tensor(sum(batch["labels"].shape[0] for batch in batches), device=device)
        return self.accelerator.gather(count).sum()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = dict(inputs)
        tasks = inputs.pop("catalog_task")
        labels = inputs.pop("labels")
        inputs.pop("compute_loss_func", None)
        inputs.pop("text_position_ids", None)
        if inputs.pop("loss_scale", None) is not None:
            raise ValueError("Token loss_scale cannot be combined with sample-mean replay")
        outputs, targets, rows, counts = supervised_forward(model, thinker.lm_head, inputs, labels)
        prefix = None
        if separate_prefix:
            shifted = labels[:, 1:]
            boundary = shifted == asr_text_id
            prefix = ((boundary.cumsum(-1) == 0) | boundary)[shifted != -100]
        indexes = (tasks == 0).nonzero().flatten()
        teacher_logits = None
        ordinary = tasks[rows] == 0
        if indexes.numel():
            # Accelerate wraps only the student forward in autocast.
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                teacher_outputs, teacher_targets, _, _ = supervised_forward(
                    teacher, teacher.thinker.lm_head, select_examples(inputs, indexes),
                    labels.index_select(0, indexes),
                )
            if not torch.equal(targets[ordinary], teacher_targets):
                raise ValueError("Teacher and student labels differ")
            teacher_logits = teacher_outputs.logits
        losses, ce, kl = sample_losses(outputs.logits, targets, rows, counts,
                                      teacher_logits, ordinary, coefficient, prefix)
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
        if num_items_in_batch is None:
            loss = losses.mean()
        else:
            # Global example count covers all microbatches in this optimizer step.
            # DDP averages rank gradients, so compensate exactly once here.
            loss = losses.sum() * self.accelerator.num_processes / num_items_in_batch
        outputs.loss = loss
        return (loss, outputs) if return_outputs else loss

    trainer._get_num_items_in_batch = MethodType(count_examples, trainer)
    trainer.compute_loss = MethodType(compute_loss, trainer)
    summary = {
        "teacher": args.retention_teacher,
        "objective": "mean_examples(answer_reduction(CE) + ordinary_asr * replay_kl_weight * answer_reduction(KL(base || student)))",
        "separate_prefix": separate_prefix,
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
