"""Paired look-alike loss for TS-ASR SFT: same mixture, real enrollment speaks, look-alike enrollment says None.

White-box task 12 (``runs/whitebox-20261107/tasks/12-grad-attribution``) found that target-absent training
signal, GRPO or cross-entropy, moves the presence decision of target-present samples as much as that of hard
negatives: it only shifts the threshold. On the very same mixture, the direction
``grad m(twin) - grad m(present)`` instead raises the ``None`` margin of hard negatives and lowers it on
target-present samples (m = log p(None) - logsumexp over language names, at the token after ``language``).

The dataset (``objective.paired_lookalike``) gives some online target-present samples a *twin enrollment*:
the nearest same-corpus look-alike speaker's enrollment, processed like the real one and cut to the same
length, so the twin shares input ids and mixture features and differs only in the enrollment Mel frames.
Here each such row is run a second time with those frames swapped, and per pair

    softplus(margin - m_twin) + softplus(gap - (m_twin - m_present))

summed over pairs and divided by the micro-batch's supervised tokens (the supervised-head token-mean scale).
The second term pushes the pair apart with one weight on both sides, ``grad m_present - grad m_twin``, until the
twin's margin is ``gap`` above the real one; plain cross-entropy on the present row vanishes once it is right,
which leaves only the threshold-shifting half (task 12, attempts 005-006). The bound matters: an unbounded
``w * m_present`` term (first version) drove m_present to -160 nats and let 10% of present rows say None.
"""

from __future__ import annotations

from types import MethodType

import torch
import torch.nn.functional as F

from .retention import supervised_forward

LANGUAGE_TOKENS = (" Chinese", " English")  # the languages of the TS-ASR training data
PAIR_KEYS = ("twin_enroll_features", "twin_rows")


def decision_labels(labels):
    """Labels that keep only the language-name token (the one after the first assistant token ``language``)."""
    first = (labels != -100).int().argmax(-1)
    keep = torch.zeros_like(labels, dtype=torch.bool)
    keep[torch.arange(len(labels), device=labels.device), (first + 1).clamp(max=labels.shape[1] - 1)] = True
    return labels.masked_fill(~keep, -100)


def pair_inputs(kwargs, labels, rows, twin_features):
    """Present copies of ``rows`` followed by their twins: every batch-indexed tensor twice, the twin half with its
    enrollment Mel frames (the first ``enroll_n_frames`` of each row) replaced by ``twin_features``."""
    batch = labels.shape[0]
    inputs = {}
    for key, value in kwargs.items():
        if torch.is_tensor(value) and value.dim() > 0 and value.shape[0] == batch:
            picked = value.index_select(0, rows)
            if key == "input_features":
                twin = picked.clone()
                frames = kwargs["enroll_n_frames"].index_select(0, rows)
                for i, n in enumerate(frames.tolist()):
                    twin[i, :, :n] = twin_features[rows[i], :, :n].to(twin.dtype)
                inputs[key] = torch.cat([picked, twin])
            else:
                inputs[key] = torch.cat([picked, picked])
        else:
            inputs[key] = value
    picked = decision_labels(labels.index_select(0, rows))
    return inputs, torch.cat([picked, picked])


def presence_margin(logits, none_id, language_ids):
    logp = torch.log_softmax(logits.float(), -1)
    return logp[:, none_id] - torch.logsumexp(logp[:, language_ids], -1)


def pair_loss(m_present, m_twin, margin, gap, tokens):
    return (F.softplus(margin - m_twin) + F.softplus(gap - (m_twin - m_present))).sum() / tokens


def enable_paired_lookalike(trainer, weight, margin, gap):
    """Add ``weight`` x the pair loss; install after every other forward wrapper (it takes the twin columns off)."""
    if not weight > 0:
        raise ValueError("paired_lookalike weight must be positive")
    model = trainer.model
    tokenizer = trainer.template.tokenizer
    ids = [tokenizer.encode(w, add_special_tokens=False) for w in (" None", *LANGUAGE_TOKENS)]
    if any(len(i) != 1 for i in ids):
        raise ValueError(f"presence tokens are not single tokens: {ids}")
    none_id, language_ids = ids[0][0], [i[0] for i in ids[1:]]
    head = model.thinker.lm_head
    forward = model.forward
    pending = []

    def paired(self, *args, labels=None, twin_enroll_features=None, twin_rows=None, **kwargs):
        outputs = forward(*args, labels=labels, **kwargs)
        if (labels is None or args or not self.training or twin_rows is None or not bool(twin_rows.any())
                or getattr(outputs, "loss", None) is None):
            return outputs
        rows = twin_rows.nonzero().squeeze(1)
        inputs, pair_labels = pair_inputs(kwargs, labels, rows, twin_enroll_features)
        inputs.pop("ts_activity", None)
        # Label-free through the other wrappers (they pass such calls on), projecting only the decision tokens.
        pair_outputs, _, _, _ = supervised_forward(forward, head, inputs, pair_labels)
        m = presence_margin(pair_outputs.logits, none_id, language_ids)
        m_present, m_twin = m[: len(rows)], m[len(rows):]
        tokens = (labels[:, 1:] != -100).sum().clamp_min(1)
        loss = pair_loss(m_present, m_twin, margin, gap, tokens)
        pending.append(torch.stack([loss.detach(), (m_twin > 0).float().mean(), (m_present < 0).float().mean(),
                                    m_twin.detach().mean(), m_present.detach().mean(),
                                    torch.tensor(float(len(rows)), device=m.device)]))
        outputs.loss = outputs.loss + weight * loss
        return outputs

    log = trainer.log

    def log_with_pairs(self, logs, *args, **kwargs):
        if pending:
            values = torch.stack(pending)
            pending.clear()
            metrics = self.custom_metrics["train"]
            for i, name in enumerate(("pair_loss", "pair_twin_none", "pair_present_speaks", "pair_twin_margin",
                                      "pair_present_margin", "pair_rows")):
                metrics[name].update(values[:, i])
        return log(logs, *args, **kwargs)

    model.forward = MethodType(paired, model)
    trainer.log = MethodType(log_with_pairs, trainer)
