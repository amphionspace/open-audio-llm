from collections import defaultdict
from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutput, CausalLMOutputWithPast

from open_audio_llm.integrations.ms_swift.ts_activity import (
    HEAD,
    activity_loss,
    enable_ts_activity_loss,
    frame_labels,
)
from open_audio_llm.tsasr.sep_token import conv_token_count


def test_frame_labels_read_the_code_at_each_encoder_output_centre():
    mel = 213  # two full chunks and a 13-frame remainder
    labels = frame_labels(torch.arange(mel), mel)
    centres = [100 * c + min(8 * j + 4, min(100, mel - 100 * c) - 1)
               for c in range(3) for j in range(conv_token_count(min(100, mel - 100 * c)))]
    assert len(labels) == conv_token_count(mel) == 28 and labels.tolist() == centres
    # Outputs centred past the coded Mel frames are unlabelled: centres 4, 12, ..., 44 fall inside 50.
    assert (frame_labels(torch.zeros(50, dtype=torch.long), 120) >= 0).sum() == 6


def _layout(lengths, enrolls):
    """Encoder rows of each utterance and the mixture rows inside them."""
    rows, offset = [], 0
    for length, enroll in zip(lengths, enrolls):
        if enroll and length > enroll:
            start = offset + conv_token_count(enroll) + 1
            rows.append(torch.arange(start, start + conv_token_count(length - enroll)))
            offset = start + conv_token_count(length - enroll)
        else:
            rows.append(None)
            offset += conv_token_count(length) + (1 if enroll else 0)
    return rows, offset


def test_activity_loss_reads_only_the_labelled_mixture_frames():
    torch.manual_seed(0)
    lengths, enrolls = torch.tensor([250, 80, 230]), torch.tensor([100, 0, 100])
    rows, total = _layout(lengths.tolist(), enrolls.tolist())
    features = torch.randn(total, 6)
    head = torch.nn.Linear(6, 4)
    codes = torch.full((3, 150), -1, dtype=torch.long)
    codes[0] = torch.randint(0, 4, (150,))  # the third utterance has no labels
    loss, correct, frames = activity_loss(head, features, codes, lengths, enrolls)
    labels = frame_labels(codes[0], 150)
    logits = head(features[rows[0]])
    torch.testing.assert_close(loss, F.cross_entropy(logits, labels))
    assert frames == len(rows[0]) and correct == (logits.argmax(-1) == labels).sum()
    with pytest.raises(ValueError, match="batch layout"):
        activity_loss(head, torch.randn(total + 1, 6), codes, lengths, enrolls)
    # Nothing labelled: zero loss, yet the head stays in the graph.
    loss, _, frames = activity_loss(head, features, None, lengths, enrolls)
    loss.backward()
    assert loss == 0 and frames == 0 and head.weight.grad is not None


def test_activity_loss_matches_the_batched_encoder_layout():
    pytest.importorskip("qwen_asr")
    from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRAudioEncoderConfig
    from qwen_asr.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRAudioEncoder

    from open_audio_llm.integrations.ms_swift.audio_batching import enable_batched_audio
    from open_audio_llm.tsasr.sep_token import AudioSepToken, _patch_hf_encoder_forward

    torch.manual_seed(5)
    config = Qwen3ASRAudioEncoderConfig(
        d_model=32, encoder_layers=1, encoder_attention_heads=4, encoder_ffn_dim=64, output_dim=16,
        downsample_hidden_size=8, n_window=50, conv_chunksize=3,
    )
    config._attn_implementation = "sdpa"
    encoder = Qwen3ASRAudioEncoder(config).train()
    encoder.add_module("sep_token", AudioSepToken(config.d_model))
    _patch_hf_encoder_forward(encoder)
    thinker = SimpleNamespace(audio_tower=encoder, get_audio_features=None)
    enable_batched_audio(SimpleNamespace(thinker=thinker))
    lengths, enrolls = torch.tensor([80, 120, 250, 100]), torch.tensor([0, 50, 100, 100])
    features = thinker.get_audio_features(
        torch.randn(4, 128, 260), audio_feature_lengths=lengths, enroll_n_frames=enrolls)
    codes = torch.randint(0, 4, (4, 150))
    head = torch.nn.Linear(16, 4)
    _, _, frames = activity_loss(head, features, codes, lengths, enrolls)
    assert frames == conv_token_count(70) + conv_token_count(150)


class Tower(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Parameter(torch.randn(64, 5))
        self.proj2 = torch.nn.Linear(5, 6)

    def forward(self, rows):
        return BaseModelOutput(last_hidden_state=self.proj2(self.base[:rows]))


class TinyTS(torch.nn.Module):
    """Loss inside forward, encoder at ``thinker.audio_tower``, as Qwen3-ASR under ms-swift."""

    def __init__(self):
        super().__init__()
        self.thinker = torch.nn.Module()
        self.thinker.audio_tower = Tower()
        self._audio_encoder_batching_enabled = True

    def forward(self, feature_attention_mask, enroll_n_frames=None, labels=None):
        rows = _layout(feature_attention_mask.sum(-1).tolist(), enroll_n_frames.tolist())[1]
        hidden = self.thinker.audio_tower(rows).last_hidden_state
        return CausalLMOutputWithPast(loss=hidden.pow(2).mean())


class Mean:
    def __init__(self):
        self.values = []

    def update(self, value):
        self.values.append(float(value))


def test_enabled_loss_adds_its_weighted_term_logs_it_and_stays_out_of_saved_weights():
    torch.manual_seed(1)
    model = TinyTS().train()
    saved, logged = {}, []
    trainer = SimpleNamespace(model=model, custom_metrics={"train": defaultdict(Mean)})
    trainer.log = MethodType(lambda self, logs: logged.append(dict(logs)), trainer)
    trainer._save_model = MethodType(lambda self, output_dir=None, state_dict=None: saved.update(state_dict), trainer)
    head = enable_ts_activity_loss(trainer, 0.5)
    assert model.thinker.ts_activity_head is head

    mask = torch.arange(260)[None, :] < torch.tensor([250, 80])[:, None]
    batch = {"feature_attention_mask": mask, "enroll_n_frames": torch.tensor([100, 0])}
    codes = torch.randint(0, 4, (2, 150))
    codes[1] = -1
    base = TinyTS.forward(model, **batch).loss
    features = model.thinker.audio_tower(_layout([250, 80], [100, 0])[1]).last_hidden_state
    expected, _, frames = activity_loss(head, features, codes, mask.sum(-1), batch["enroll_n_frames"])
    outputs = model(**batch, ts_activity=codes)
    torch.testing.assert_close(outputs.loss, base + 0.5 * expected)
    outputs.loss.backward()
    assert torch.count_nonzero(head.weight.grad) > 0 and model.thinker.audio_tower.base.grad is not None

    trainer.log({"loss": 1.0})
    metrics = trainer.custom_metrics["train"]
    assert metrics["ts_activity_loss"].values == pytest.approx([float(expected.detach())])
    assert metrics["ts_activity_frames"].values == [float(frames)]
    assert logged == [{"loss": 1.0}]

    trainer._save_model("out", dict(model.state_dict()))
    assert f"thinker.{HEAD}.weight" in model.state_dict()
    assert not any(HEAD in key for key in saved) and "thinker.audio_tower.proj2.weight" in saved

    model.eval()  # evaluation keeps the plain loss
    torch.testing.assert_close(model(**batch, ts_activity=codes).loss, base)


def test_loss_requires_the_batched_encoder_layout():
    model = TinyTS()
    model._audio_encoder_batching_enabled = False
    with pytest.raises(ValueError, match="audio_encoder_batching"):
        enable_ts_activity_loss(SimpleNamespace(model=model), 0.1)
