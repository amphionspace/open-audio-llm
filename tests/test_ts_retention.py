from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from audio_data_contract import AudioRecord, AudioRef, AudioSlot

from open_audio_llm.data.catalog_dataset import in_ts_partition
from open_audio_llm.integrations.ms_swift.retention import sample_losses, supervised_forward


def test_short_negative_has_same_example_weight_as_long_transcript():
    logits = torch.zeros(11, 3, requires_grad=True)
    targets = torch.zeros(11, dtype=torch.long)
    rows = torch.tensor([0] * 10 + [1])
    loss, _, _ = sample_losses(logits, targets, rows, torch.tensor([10, 1]))
    loss.mean().backward()
    assert logits.grad[:10].abs().sum() == pytest.approx(logits.grad[10:].abs().sum())


def test_presence_decision_is_not_overweighted_by_short_negative_body():
    # Both answers have 3 header tokens, but positive and negative bodies have 20/1.
    logits = torch.zeros(27, 4, requires_grad=True)
    targets = torch.zeros(27, dtype=torch.long)
    rows = torch.tensor([0] * 23 + [1] * 4)
    prefix = torch.tensor([True] * 3 + [False] * 20 + [True] * 3 + [False])
    losses, _, _ = sample_losses(logits, targets, rows, torch.tensor([23, 4]), prefix=prefix)
    losses.mean().backward()
    torch.testing.assert_close(logits.grad[:3], logits.grad[23:26])
    torch.testing.assert_close(logits.grad[3:23].sum(0), logits.grad[26])


def test_kl_anchors_only_plain_asr_and_does_not_train_teacher():
    student = torch.tensor([[2., 0., -1.], [0., 2., -1.]], requires_grad=True)
    teacher = torch.tensor([[0., 2., -1.]], requires_grad=True)
    labels, rows, counts = torch.tensor([0, 1]), torch.tensor([0, 1]), torch.ones(2)
    losses, ce, kl = sample_losses(student, labels, rows, counts, teacher,
                                   torch.tensor([True, False]), 2.0)
    assert kl[0] > 0 and kl[1] == 0
    assert losses[1] == ce[1]
    losses.sum().backward()
    assert teacher.grad is None
    _, _, matching = sample_losses(student, labels, rows, counts, student[:1].detach(),
                                   torch.tensor([True, False]))
    assert matching.abs().max().item() < 1e-6


def test_global_example_normalization_matches_unequal_rank_microbatches():
    torch.manual_seed(42)
    weight = torch.randn(3, 4, requires_grad=True)
    inputs = torch.randn(7, 3)
    targets = torch.tensor([0, 1, 1, 3, 0, 2, 1])
    # Three examples, lengths 2, 1, 4. Rank 0 has two microbatches; rank 1 has one.
    rows = torch.tensor([0, 0, 1, 2, 2, 2, 2])
    reference, _, _ = sample_losses(inputs @ weight, targets, rows, torch.tensor([2, 1, 4]))
    expected = torch.autograd.grad(reference.mean(), weight)[0]
    gradients = []
    for slices in ((slice(0, 2), slice(2, 3)), (slice(3, 7),)):
        total = 0
        for section in slices:
            size = len(targets[section])
            loss, _, _ = sample_losses(inputs[section] @ weight, targets[section],
                                       torch.zeros(size, dtype=torch.long), torch.tensor([size]))
            total = total + loss.sum() * 2 / 3
        gradients.append(torch.autograd.grad(total, weight)[0])
    torch.testing.assert_close(sum(gradients) / 2, expected)


def test_projecting_supervised_positions_preserves_loss_and_gradients():
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(9, 5)
            self.head = torch.nn.Linear(5, 9)

        def forward(self, input_ids, use_cache=False):
            return SimpleNamespace(logits=self.head(self.embedding(input_ids)))

    torch.manual_seed(3)
    model = TinyModel()
    inputs = {"input_ids": torch.tensor([[1, 2, 3, 4], [2, 4, 5, 0]])}
    labels = torch.tensor([[-100, -100, 3, 4], [-100, 4, 5, -100]])
    full = model(**inputs).logits
    expected = F.cross_entropy(full[:, :-1].reshape(-1, 9), labels[:, 1:].reshape(-1))
    expected_grads = torch.autograd.grad(expected, tuple(model.parameters()))
    output, targets, rows, counts = supervised_forward(model, model.head, inputs, labels)
    actual, _, _ = sample_losses(output.logits, targets, rows, counts)
    assert targets.tolist() == [3, 4, 4, 5]
    assert counts.tolist() == [2, 2]
    torch.testing.assert_close(actual.mean(), expected)
    for actual_grad, expected_grad in zip(torch.autograd.grad(actual.mean(), tuple(model.parameters())), expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)
    assert model(**inputs).logits.shape == (2, 4, 9)  # Hook cannot leak into generation.


@pytest.mark.parametrize("dataset,ids", [
    ("aishellmix", ["C0659_0122_D0017_0083|C0659_0122", "C0659_0122_D0017_0083_D0138_0113|D0017_0083", "neg|C0659_0122_D0017_0083|C9343_0053"]),
    ("librimix", ["8975-270782-0029_229-130880-0066|8975-270782-0029", "8975-270782-0029_2836-5355-0039_6529-62556-0046|2836-5355-0039", "neg|8975-270782-0029_229-130880-0066|233-134440-0025"]),
])
def test_ts_derivatives_and_all_targets_share_one_partition(dataset, ids):
    record = AudioRecord("example", "ts_asr", tuple(AudioSlot(name, AudioRef(dataset, "v1", "train", name))
                         for name in ("enrollment", "mixture")), "hello", metadata={"source_record_id":ids[0]})
    for seed in range(20):
        train = {"part":"train", "modulus":3, "seed":seed}
        dev = {**train, "part":"dev"}
        records = [replace(record, metadata={"source_record_id":value}) for value in ids]
        assert len({in_ts_partition(r, train) for r in records}) == 1
        assert all(in_ts_partition(r, train) != in_ts_partition(r, dev) for r in records)
