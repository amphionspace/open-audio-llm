"""Equal-example SFT with an explicitly optional frozen-base ASR replay KL."""

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
    # The one host sync, before the forward: every later selection reuses these
    # row-major positions, so the host never waits for the forward to finish.
    index = mask.flatten().nonzero().squeeze(1)
    rows = index.div(labels.shape[1], rounding_mode="floor")
    hook = head.register_forward_pre_hook(
        lambda module, args: (args[0].flatten(0, 1).index_select(0, index),))
    try:
        outputs = model(**inputs, use_cache=False)
    finally:
        hook.remove()
    return outputs, shifted.flatten().index_select(0, index), rows, mask.sum(-1)


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


def token_kl(student, teacher):
    """Per-token KL(teacher || student) over the vocabulary, in 128-token FP32 blocks."""
    pieces = []
    for s, t in zip(student.split(128), teacher.detach().split(128)):
        pieces.append(F.kl_div(s.float().log_softmax(-1), t.float().softmax(-1), reduction="none").sum(-1))
    return torch.cat(pieces) if pieces else student.new_zeros(0, dtype=torch.float32)


def clean_teacher_kl(student_model, head, inputs, labels, logits, targets, rows, counts, clean_rows, clean_features):
    """Per-example KL from the model's own prediction on the clean target to its prediction on the mixture.

    Rows in ``clean_rows`` are target-present TS samples mixed online: ``clean_features`` holds their enrollment
    followed by the target alone, on the mixture's timeline, so the teacher has the student's input ids and
    supervised positions. The teacher is the student itself, without gradient: it pulls the mixture's answer
    distribution toward what the model says when nobody talks over the target.
    Returns (per-example KL, zero outside ``clean_rows``; the selected row indexes).
    """
    indexes = clean_rows.nonzero().flatten()
    kl = logits.new_zeros(len(counts), dtype=torch.float32)
    if not indexes.numel():
        return kl, indexes
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        teacher_inputs = select_examples(inputs, indexes)
        teacher_inputs["input_features"] = clean_features.index_select(0, indexes).to(inputs["input_features"].dtype)
        teacher_outputs, teacher_targets, _, _ = supervised_forward(
            student_model, head, teacher_inputs, labels.index_select(0, indexes))
    picked = torch.zeros(len(counts), dtype=torch.bool, device=rows.device)
    picked[indexes] = True
    positions = picked[rows]
    if not torch.equal(targets[positions], teacher_targets):
        raise ValueError("Clean teacher and student labels differ")
    values = token_kl(logits[positions], teacher_outputs.logits)
    kl = average_answer_parts(values, rows[positions], counts)
    return kl, indexes


def frozen_tower_copy(tower):
    """A frozen BF16 copy of the (SEP-patched) audio tower: the clean-target feature teacher.

    ``attach_sep_token`` binds the SEP-aware ``forward`` on the instance; a deep copy would keep that method bound
    to the student, so the copy is taken without it and patched again.
    """
    import copy

    from open_audio_llm.tsasr.sep_token import _patch_hf_encoder_forward

    forward = tower.__dict__.pop("forward", None)
    try:
        teacher = copy.deepcopy(tower)
    finally:
        if forward is not None:
            tower.forward = forward
    teacher._sep_token_patched = False
    _patch_hf_encoder_forward(teacher)
    return teacher.to(torch.bfloat16).eval().requires_grad_(False)


def tower_tokens(tower, features, feature_lens, enroll_frames):
    """Audio tokens (enrollment, SEP, mixture) per row, as ``get_audio_features`` computes them."""
    return [tower(feature[:, :int(length)], feature_lens=length.reshape(1), enroll_n_frames=int(frames)).last_hidden_state
            for feature, length, frames in zip(features, feature_lens, enroll_frames)]


def clean_teacher_features(student_tower, teacher_tower, inputs, clean_rows, clean_features, n_examples):
    """Per-example relative squared error from the student's audio tokens on the mixture to the frozen teacher
    tower's audio tokens on the clean target: sum_tokens ||s - t||^2 / sum_tokens ||t||^2.

    ``clean_features`` hold the enrollment followed by the target alone on the mixture's timeline, so both towers
    emit the same tokens in the same places. The teacher is the starting audio tower: what the LLM receives when
    nobody talks over the target. Gradient reaches the student tower only.
    Returns (per-example error, zero outside ``clean_rows``; the selected row indexes).
    """
    indexes = clean_rows.nonzero().flatten()
    error = torch.zeros(n_examples, dtype=torch.float32, device=clean_rows.device)
    if not indexes.numel():
        return error, indexes
    lengths = inputs["feature_attention_mask"].index_select(0, indexes).sum(-1)
    frames = inputs["enroll_n_frames"].index_select(0, indexes)
    dtype = inputs["input_features"].dtype
    with torch.autocast("cuda", dtype=torch.bfloat16):
        student = tower_tokens(student_tower, inputs["input_features"].index_select(0, indexes), lengths, frames)
        with torch.no_grad():
            teacher = tower_tokens(teacher_tower, clean_features.index_select(0, indexes).to(dtype), lengths, frames)
    values = []
    for s, t in zip(student, teacher):
        if s.shape != t.shape:
            raise ValueError(f"student audio tokens {tuple(s.shape)} != clean teacher tokens {tuple(t.shape)}")
        s, t = s.float(), t.float()
        values.append((s - t).square().sum() / t.square().sum().clamp_min(1e-6))
    return error.index_put((indexes,), torch.stack(values)), indexes


def answer_parts(labels, asr_text_id, think_end_id):
    """(think rows, header mask over supervised tokens) for ``separate_prefix``.

    The header is ``language X<asr_text>``. A think block before it (data/think.py) is answer
    content: it is averaged with the transcript, not with the header.
    """
    shifted = labels[:, 1:]
    closed = shifted == think_end_id
    think_rows = closed.any(-1)
    after_think = (closed.cumsum(-1) > 0) & ~closed
    boundary = shifted == asr_text_id
    header = (~think_rows[:, None] | after_think) & ((boundary.cumsum(-1) == 0) | boundary)
    return think_rows, header[shifted != -100]


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


def has_plain_asr_source(dataset):
    """Whether any training source still yields plain ASR after task conversion.

    Sources that declare ``task`` are decided from the config first. Reading a record opens the
    Lance index in the trainer process, and the train DataLoader then forks its workers from a
    process holding Lance's threads; those workers crash with a segmentation fault.
    """
    candidates = [(source, indices) for source, indices in zip(dataset.sources, dataset.source_ranges)
                  if source.get('replay_kl', True)
                  and not (source.get('single_speaker_format') or source.get('sot_timestamps'))]
    if any(source.get('task') not in (None, 'speaker_attributed_asr') for source, _ in candidates):
        return True
    return any('task' not in source
               and dataset.records[indices.start].record.task != 'speaker_attributed_asr'
               for source, indices in candidates)


def install_retention_objective(trainer, args):
    from qwen_asr import Qwen3ASRModel

    if args.tuner_type not in {"full", "lora"}:
        raise ValueError("Sample-mean SFT supports full or lora training")
    if args.deepspeed or args.use_logits_to_keep or trainer.optimizer is not None:
        raise ValueError("Replay objective requires DDP, untrimmed labels and installation before optimizer creation")
    coefficient = float(args._catalog_config["objective"]["replay_kl_weight"])
    separate_prefix = args._catalog_config["objective"].get("separate_prefix", False)
    reduction = args._catalog_config["objective"].get("reduction", "example_mean")
    if reduction not in {"example_mean", "token_mean"}:
        raise ValueError("objective.reduction must be example_mean or token_mean")
    if reduction == "token_mean" and separate_prefix:
        # Prefix/body averages cannot be recovered as a token sum.
        raise ValueError("separate_prefix requires reduction=example_mean")
    asr_text_id = trainer.template.tokenizer.convert_tokens_to_ids("<asr_text>")
    think_end_id = trainer.template.tokenizer.convert_tokens_to_ids("</think>")
    if not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError("replay_kl_weight must be finite and nonnegative")
    if coefficient and not args.retention_teacher:
        raise ValueError("Positive replay_kl_weight requires retention_teacher")
    clean_weight = float(args._catalog_config["objective"].get("clean_teacher_kl_weight", 0.0))
    if not math.isfinite(clean_weight) or clean_weight < 0:
        raise ValueError("clean_teacher_kl_weight must be finite and nonnegative")
    if clean_weight and separate_prefix:
        raise ValueError("clean_teacher_kl_weight averages KL over the whole answer; use separate_prefix false")
    feature_weight = float(args._catalog_config["objective"].get("clean_teacher_feature_weight", 0.0))
    if not math.isfinite(feature_weight) or feature_weight < 0:
        raise ValueError("clean_teacher_feature_weight must be finite and nonnegative")
    if coefficient and not has_plain_asr_source(trainer.train_dataset):
        raise ValueError("ASR replay KL has no plain-ASR examples after task conversion; "
                         "set replay_kl_weight=0 for supervised diarization replay")
    student = trainer.model
    thinker = student.thinker
    # Training forwards explicitly disable caching; exported configs should retain
    # normal inference defaults even when gradient checkpointing is enabled.
    thinker.model.config.use_cache = True
    thinker.generation_config.use_cache = True
    if any(p.requires_grad for p in thinker.audio_tower.parameters()) and args.audio_encoder_parallel:
        raise ValueError("Trainable audio encoder requires audio_encoder_parallel=false")
    llm = [*thinker.model.parameters(), thinker.lm_head.weight]
    if feature_weight and not any(p.requires_grad for p in thinker.audio_tower.parameters()):
        raise ValueError("clean_teacher_feature_weight trains the audio tower; unfreeze it")
    if args.tuner_type == 'full' and getattr(args, "freeze_llm", False):
        # An encoder-only round: the LLM stays exactly as it starts.
        if any(p.requires_grad for p in llm):
            raise ValueError("freeze_llm left LLM parameters trainable")
    elif args.tuner_type == 'full' and not all(p.requires_grad for p in llm):
        raise ValueError("All LLM parameters, including embeddings and output head, must be trainable")
    # Taken before the FP32 cast below: the starting tower, in BF16.
    feature_teacher = frozen_tower_copy(thinker.audio_tower) if feature_weight else None
    # AdamW needs FP32 master weights/moments for small full-model updates.
    # Trainer's BF16 autocast still performs the expensive matrix multiplies in BF16.
    for parameter in student.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    teacher = None
    if coefficient:
        teacher = Qwen3ASRModel.from_pretrained(
            args.retention_teacher, dtype=torch.bfloat16,
            device_map=str(trainer.accelerator.device), attn_implementation="sdpa",
        ).model.eval().requires_grad_(False)
        teacher.thinker.config.use_cache = False
    trainer.retention_teacher = teacher
    if feature_teacher is not None:
        feature_teacher.to(trainer.accelerator.device)
    trainer.clean_feature_teacher = feature_teacher
    trainer.model_accepts_loss_kwargs = True
    trainer.args.average_tokens_across_devices = True

    def count_examples(self, batches, device):
        count = torch.tensor(sum(batch["labels"].shape[0] for batch in batches), device=device)
        return self.accelerator.gather(count).sum()

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
            raise ValueError("Token loss_scale cannot be combined with the retention objective")
        clean_rows, clean_features = inputs.pop("clean_rows", None), inputs.pop("clean_input_features", None)
        outputs, targets, rows, counts = supervised_forward(model, thinker.lm_head, inputs, labels)
        think_rows, prefix = answer_parts(labels, asr_text_id, think_end_id)
        if not separate_prefix:
            prefix = None
        # The frozen base has never seen a think block: replay KL covers think-off ASR only.
        plain = (tasks == 0) & ~think_rows
        indexes = plain.nonzero().flatten()
        teacher_logits = None
        ordinary = plain[rows]
        if coefficient and indexes.numel():
            # Accelerate wraps only the student forward in autocast.
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                teacher_inputs = select_examples(inputs, indexes)
                # Replay rows are plain ASR; the base teacher has no enrollment input.
                teacher_inputs.pop("enrollment_lengths", None)
                teacher_outputs, teacher_targets, _, _ = supervised_forward(
                    teacher, teacher.thinker.lm_head, teacher_inputs,
                    labels.index_select(0, indexes),
                )
            if not torch.equal(targets[ordinary], teacher_targets):
                raise ValueError("Teacher and student labels differ")
            teacher_logits = teacher_outputs.logits
        losses, ce, kl = sample_losses(outputs.logits, targets, rows, counts,
                                      teacher_logits, ordinary, coefficient, prefix)
        clean_indexes = None
        if (clean_weight or feature_weight) and clean_rows is not None and model.training:
            # Target-present TS rows without a think block (the clean target is only built for those).
            clean_rows = clean_rows.to(tasks.device) & (tasks == 1) & ~think_rows
        if clean_weight and clean_rows is not None and model.training:
            clean_kl, clean_indexes = clean_teacher_kl(
                student, thinker.lm_head, inputs, labels, outputs.logits, targets, rows, counts,
                clean_rows, clean_features)
            losses = losses + clean_weight * clean_kl
        feature_indexes = None
        if feature_weight and clean_rows is not None and model.training:
            # Per example, like CE: token_mean weighs it by the row's supervised tokens.
            clean_feat, feature_indexes = clean_teacher_features(
                thinker.audio_tower, self.clean_feature_teacher, inputs, clean_rows, clean_features, len(counts))
            losses = losses + feature_weight * clean_feat
        mode = "train" if model.training else "eval"
        metrics = self.custom_metrics[mode]
        if clean_weight and mode == "train":
            metric = metrics["clean_kl"]
            if clean_indexes is not None and clean_indexes.numel():
                metric.update(clean_kl.detach()[clean_indexes])
        if feature_weight and mode == "train":
            metric = metrics["clean_feat"]
            if feature_indexes is not None and feature_indexes.numel():
                metric.update(clean_feat.detach()[feature_indexes])
        if (clean_weight or feature_weight) and mode == "train":
            picked = clean_indexes if clean_indexes is not None else feature_indexes
            metrics["clean_rows"].update(torch.tensor(
                [float(picked.numel() if picked is not None else 0)], device=tasks.device))
        # Create the same metrics on every rank even when a batch lacks a task.
        for task, name in enumerate(("asr", "ts_positive", "ts_negative", "sot", "asr_hotwords", "asr_no_kl")):
            metric = metrics[f"ce_{name}"]
            selected = ce.detach()[tasks == task]
            if selected.numel():
                metric.update(selected)
        metric = metrics["ce_think"]
        selected = ce.detach()[think_rows]
        if selected.numel():
            metric.update(selected)
        metric = metrics["replay_kl"]
        if coefficient and indexes.numel():
            metric.update(kl.detach()[indexes])
        if reduction == "token_mean":
            loss = token_mean_loss(losses, counts, num_items_in_batch, self.accelerator.num_processes)
        elif num_items_in_batch is None:
            loss = losses.mean()
        else:
            # Global example count covers all microbatches in this optimizer step.
            # DDP averages rank gradients, so compensate exactly once here.
            loss = losses.sum() * self.accelerator.num_processes / num_items_in_batch
        outputs.loss = loss
        return (loss, outputs) if return_outputs else loss

    trainer._get_num_items_in_batch = MethodType(
        count_supervised_tokens if reduction == "token_mean" else count_examples, trainer)
    trainer.compute_loss = MethodType(compute_loss, trainer)
    summary = {
        "teacher": args.retention_teacher if teacher is not None else None,
        "teacher_enabled": teacher is not None,
        "tuner_type": args.tuner_type,
        "objective": (
            "sum_tokens(CE + ordinary_asr * replay_kl_weight * KL(base || student)"
            " + ts_present_online * clean_teacher_kl_weight * KL(student(clean target) || student)"
            " + ts_present_online * clean_teacher_feature_weight * |tower(mix) - start_tower(clean target)|^2 / |start_tower(clean target)|^2"
            ") / supervised_tokens"
            if reduction == "token_mean" else
            "mean_examples(answer_reduction(CE) + ordinary_asr * replay_kl_weight * answer_reduction(KL(base || student)))"
        ),
        "reduction": reduction,
        "separate_prefix": separate_prefix,
        "replay_kl_weight": coefficient,
        "clean_teacher_kl_weight": clean_weight,
        "clean_teacher_feature_weight": feature_weight,
        "freeze_llm": bool(getattr(args, "freeze_llm", False)),
        "trainable_parameters": sum(p.numel() for p in student.parameters() if p.requires_grad),
        "frozen_parameters": sum(p.numel() for p in student.parameters() if not p.requires_grad),
        "audio_trainable_parameters": sum(p.numel() for p in thinker.audio_tower.parameters() if p.requires_grad),
        "llm_trainable_parameters": sum(p.numel() for name, p in thinker.named_parameters()
                                        if p.requires_grad and name.startswith(("model.", "lm_head."))),
        "learning_rates": {"llm": args.learning_rate, "encoder": args.vit_lr, "aligner": args.aligner_lr},
        "master_dtype": "float32", "compute_dtype": "bfloat16",
        "teacher_trainable_parameters": sum(p.numel() for p in teacher.parameters() if p.requires_grad) if teacher is not None else 0,
    }
    LOG.warning("Sample-mean objective: %s", json.dumps(summary))
    if trainer.is_world_process_zero():
        path = Path(trainer.args.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "retention-objective.json").write_text(json.dumps(summary, indent=2) + "\n")
