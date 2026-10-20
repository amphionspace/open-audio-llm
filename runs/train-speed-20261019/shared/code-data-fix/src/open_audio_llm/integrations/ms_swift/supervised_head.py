"""Score only supervised positions on ms-swift's ordinary token-mean SFT path."""

from __future__ import annotations

from types import MethodType

import torch.nn.functional as F

from .retention import supervised_forward


def enable_supervised_head(trainer):
    """Project only supervised positions; ms-swift's loss scaling stays unchanged.

    Qwen3-ASR 0.0.6 projects every position, audio placeholders included, onto the
    151,936-token vocabulary and upcasts all of it for the loss. The loss returned here
    is the same per-microbatch token mean as transformers' ``ForCausalLMLoss``, which
    ms-swift then scales by the update's supervised-token count. ``token_acc`` stays on
    the GPU until the next log instead of copying every microbatch's logits to the CPU.
    """
    if trainer.args.acc_strategy != "token":
        raise ValueError("supervised_head_loss reports token accuracy only")
    model = trainer.model
    head = model.thinker.lm_head
    forward = model.forward
    log = trainer.log
    pending = []

    def supervised(self, *args, labels=None, **kwargs):
        # Evaluation and label-free calls (custom ms-swift losses pop labels) keep full logits.
        if labels is None or args or not self.training:
            return forward(*args, labels=labels, **kwargs)
        outputs, targets, _, _ = supervised_forward(forward, head, kwargs, labels)
        logits = outputs.logits
        pending.append(logits.detach().argmax(-1) == targets)
        # Without logits ms-swift skips its own accuracy pass.
        return type(outputs)(loss=F.cross_entropy(logits.float(), targets))

    def log_with_accuracy(self, logs, *args, **kwargs):
        if pending:
            metric = self.custom_metrics["train"]["token_acc"]
            for correct in pending:
                metric.update(correct)
            pending.clear()
        return log(logs, *args, **kwargs)

    model.forward = MethodType(supervised, model)
    trainer.log = MethodType(log_with_accuracy, trainer)
