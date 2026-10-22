"""Frame-level target-speaker activity as an auxiliary TS-ASR training loss.

The text loss tells the model who the target is only through which words it writes;
a target-absent sample contributes a single token. This loss supervises every mixture
frame instead: a linear head on the audio encoder output classifies each frame as
silence, target only, others only, or target with others (``ts_online.activity_codes``).

The head exists only during training. It is left out of the exported weights, so
serving and evaluation load the same checkpoint layout as without it; the DeepSpeed
state keeps it, so resuming a run restores it.
"""

from __future__ import annotations

from types import MethodType

import torch
import torch.nn.functional as F

from open_audio_llm.tsasr.sep_token import conv_token_count

from .audio_batching import _enroll_list

# silence, target only, others only, target with others: the activity code itself.
CLASSES = 4
HEAD = "ts_activity_head"
# The Qwen3-ASR encoder convolves the Mel input in chunks of 100 frames; output j of a
# chunk covers its Mel frames 8j..8j+7, 13 outputs for a full chunk (conv_token_count).
CHUNK, CHUNK_OUTPUTS, STRIDE = 100, 13, 8


def frame_labels(codes, mel_frames):
    """Activity code at the centre of each encoder output of a ``mel_frames``-frame clip.

    ``codes`` holds one code per Mel frame (-1: unlabelled); missing frames count as -1.
    """
    index = torch.arange(conv_token_count(mel_frames), device=codes.device)
    chunk, position = index // CHUNK_OUTPUTS, index % CHUNK_OUTPUTS
    length = torch.clamp(mel_frames - chunk * CHUNK, max=CHUNK)
    centre = chunk * CHUNK + torch.minimum(position * STRIDE + STRIDE // 2, length - 1)
    padded = F.pad(codes, (0, max(0, mel_frames - len(codes))), value=-1)
    return padded[centre]


def activity_loss(head, features, codes, lengths, enroll_n_frames):
    """Cross-entropy of the mixture frames, their correct count and their number.

    ``features`` are the encoder outputs of the whole micro-batch in batch order, each
    utterance laid out as the batched encoder builds it: enrollment, SEP, mixture.
    """
    lengths = [int(x) for x in lengths.tolist()]
    enrolls = _enroll_list(enroll_n_frames, len(lengths))
    rows, labels, offset = [], [], 0
    for index, (length, enroll) in enumerate(zip(lengths, enrolls)):
        enroll = min(max(enroll, 0), length)
        if enroll > 0 and length > enroll:
            start = offset + conv_token_count(enroll) + 1  # the SEP frame
            mixture = conv_token_count(length - enroll)
            if codes is not None and index < len(codes):
                target = frame_labels(codes[index], length - enroll)
                keep = (target >= 0).nonzero().squeeze(-1)
                rows.append(start + keep)
                labels.append(target[keep])
            offset = start + mixture
        else:
            offset += conv_token_count(length) + (1 if enroll > 0 else 0)
    if offset != len(features):
        raise ValueError(f"Encoder gave {len(features)} frames, the batch layout {offset}")
    if not rows or sum(len(r) for r in rows) == 0:
        # Keep the head in the graph so every rank reduces its gradient.
        zero = head(features[:1]).float().sum() * 0
        return zero, zero.detach(), zero.detach()
    rows, labels = torch.cat(rows), torch.cat(labels).to(features.device)
    logits = head(features[rows]).float()
    loss = F.cross_entropy(logits, labels)
    correct = (logits.argmax(-1) == labels).sum().to(loss.dtype)
    return loss, correct, torch.tensor(float(len(labels)), device=loss.device)


def enable_ts_activity_loss(trainer, weight):
    """Add ``weight`` x activity cross-entropy to the training loss; install after other forward hooks."""
    model = trainer.model
    if not getattr(model, "_audio_encoder_batching_enabled", False):
        raise ValueError("ts_activity_weight needs audio_encoder_batching: the loss follows its frame layout")
    tower = model.thinker.audio_tower
    reference = tower.proj2.weight
    head = torch.nn.Linear(reference.shape[0], CLASSES).to(device=reference.device, dtype=reference.dtype)
    model.thinker.add_module(HEAD, head)
    captured = {}

    def capture(module, args, output):
        if module.training:
            captured["features"] = output.last_hidden_state

    tower.register_forward_hook(capture)
    forward = model.forward
    pending = []

    def with_activity(self, *args, ts_activity=None, **kwargs):
        captured.clear()
        outputs = forward(*args, **kwargs)
        features = captured.pop("features", None)
        if not self.training or features is None or getattr(outputs, "loss", None) is None:
            return outputs
        loss, correct, frames = activity_loss(
            head, features, ts_activity, kwargs["feature_attention_mask"].sum(-1),
            kwargs.get("enroll_n_frames"),
        )
        pending.append(torch.stack([loss.detach() * frames, correct, frames]))
        outputs.loss = outputs.loss + weight * loss
        return outputs

    log = trainer.log

    def log_with_activity(self, logs, *args, **kwargs):
        if pending:
            loss_sum, correct, frames = torch.stack(pending).sum(0)
            pending.clear()
            if frames > 0:
                metrics = self.custom_metrics["train"]
                metrics["ts_activity_loss"].update(loss_sum / frames)
                metrics["ts_activity_acc"].update(correct / frames)
                metrics["ts_activity_frames"].update(frames)
        return log(logs, *args, **kwargs)

    save = trainer._save_model

    def save_without_head(self, output_dir=None, state_dict=None):
        if state_dict is None:
            state_dict = self.model.state_dict()
        state_dict = {key: value for key, value in state_dict.items() if f"{HEAD}." not in key}
        return save(output_dir, state_dict)

    model.forward = MethodType(with_activity, model)
    trainer.log = MethodType(log_with_activity, trainer)
    trainer._save_model = MethodType(save_without_head, trainer)
    return head
