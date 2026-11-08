import random
from types import SimpleNamespace

import numpy as np
import torch

from open_audio_llm.data import catalog_dataset
from open_audio_llm.data.augment import AugmentConfig
from open_audio_llm.integrations.ms_swift.paired_lookalike import (
    decision_labels,
    pair_inputs,
    pair_loss,
    presence_margin,
)


def test_decision_labels_keep_only_the_language_name():
    labels = torch.tensor([[-100, -100, 7, 8, 9, 10], [-100, 7, 8, 9, -100, -100]])
    assert decision_labels(labels).tolist() == [[-100, -100, -100, 8, -100, -100], [-100, -100, 8, -100, -100, -100]]


def test_pair_inputs_repeat_rows_and_swap_only_the_twin_enrollment_frames():
    features = torch.arange(3 * 2 * 6, dtype=torch.float32).reshape(3, 2, 6)
    twins = -torch.ones(3, 2, 4)
    kwargs = {"input_ids": torch.tensor([[1, 2], [3, 4], [5, 6]]), "input_features": features,
              "enroll_n_frames": torch.tensor([2, 3, 4]), "use_cache": False}
    labels = torch.tensor([[-100, 7, 8], [-100, 7, 9], [-100, 7, 10]])
    rows = torch.tensor([1, 2])
    inputs, pair_labels = pair_inputs(kwargs, labels, rows, twins)
    assert inputs["input_ids"].tolist() == [[3, 4], [5, 6], [3, 4], [5, 6]]
    assert inputs["use_cache"] is False
    assert torch.equal(inputs["input_features"][:2], features[1:])  # present copies untouched
    assert torch.equal(inputs["input_features"][2, :, :3], -torch.ones(2, 3))  # row 1: 3 enrollment frames
    assert torch.equal(inputs["input_features"][2, :, 3:], features[1, :, 3:])  # mixture frames kept
    assert torch.equal(inputs["input_features"][3, :, :4], -torch.ones(2, 4))
    assert pair_labels.tolist() == [[-100, -100, 9], [-100, -100, 10]] * 2


def test_pair_loss_pushes_the_pair_apart_until_the_gap():
    m_present = torch.tensor([-5.0, -30.0], requires_grad=True)
    m_twin = torch.tensor([-3.0, 4.0], requires_grad=True)
    pair_loss(m_present, m_twin, 2.0, 20.0, torch.tensor(4)).backward()
    apart = torch.sigmoid(20.0 - (m_twin.detach() - m_present.detach())) / 4
    anchor = torch.sigmoid(2.0 + m_present.detach()) / 4
    assert torch.allclose(m_present.grad, apart + anchor)  # descent lowers m_present: speak on the real speaker
    assert torch.allclose(m_twin.grad, -apart - torch.sigmoid(2.0 - m_twin.detach()) / 4)  # and raises m_twin
    assert m_present.grad[1] < 1e-3  # 34 nats apart and far from a tie: the real speaker is not pushed further


def test_a_real_enrollment_drifting_to_none_is_pulled_back_even_when_the_twin_is_far_above():
    m_present = torch.tensor([1.0], requires_grad=True)  # the real speaker now says None
    m_twin = torch.tensor([30.0], requires_grad=True)  # gap 29 > 20: the gap term is saturated
    pair_loss(m_present, m_twin, 2.0, 20.0, torch.tensor(1)).backward()
    assert m_present.grad[0] > 0.9


def test_presence_margin_against_the_language_names():
    logits = torch.log(torch.tensor([[0.5, 0.25, 0.25, 0.0]]) + 1e-12)
    assert torch.allclose(presence_margin(logits, 0, [1, 2]), torch.tensor([0.0]), atol=1e-5)


class _Table:
    pool_ids = np.array(["p0", "p1"])

    def candidates(self, enroll_id):
        return np.array([0, 1]) if enroll_id == "e1" else np.zeros(0, dtype=np.int32)


def _record(**metadata):
    slot = SimpleNamespace(ref=SimpleNamespace(cut_id="e1"))
    return SimpleNamespace(language="zh", metadata={"online_mix": {}, **metadata}, slot=lambda name: slot)


def _dataset():
    return SimpleNamespace(
        paired_lookalike={"prob": 1.0, "top": 4}, sampling_rate=16000,
        lookalike={"zh": [(_Table(), {"p0": "short", "p1": "long"})]},
        _nonspeech_loader=lambda floor=False: (lambda r, sr: np.zeros(sr, np.float32)),
        _rir_loader=lambda: None, noise_groups=[], noise_weights=[])


def test_twin_enrollment_matches_the_length_and_leaves_the_sample_rng_alone(monkeypatch):
    clips = {"short": np.ones(100, np.float32), "long": np.full(5000, 0.1, np.float32)}
    monkeypatch.setattr(catalog_dataset, "load_mono", lambda item, sr: clips[item])
    rng = random.Random(3)
    state = rng.getstate()
    twin = catalog_dataset.CatalogSwiftDataset._twin_enrollment(
        _dataset(), _record(), np.zeros(1600, np.float32), AugmentConfig(), rng)
    assert twin is not None and twin.shape == (1600,) and twin.dtype == np.float32
    assert rng.getstate() == state  # the real sample draws the same mixture with or without the twin


def test_no_twin_for_target_absent_or_scene_samples_or_unknown_enrollments(monkeypatch):
    monkeypatch.setattr(catalog_dataset, "load_mono", lambda item, sr: np.ones(5000, np.float32))
    twin = catalog_dataset.CatalogSwiftDataset._twin_enrollment
    for record in (_record(online_absent=True), _record(online_nonspeech=True), _record(scene={"mode": "full"})):
        assert twin(_dataset(), record, np.zeros(1600, np.float32), AugmentConfig(), random.Random(0)) is None
    other = _record()
    other.slot = lambda name: SimpleNamespace(ref=SimpleNamespace(cut_id="e9"))
    assert twin(_dataset(), other, np.zeros(1600, np.float32), AugmentConfig(), random.Random(0)) is None
