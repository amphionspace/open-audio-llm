import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from audio_data_contract import AudioRecord, AudioRef, AudioSlot

from open_audio_llm.data.catalog_dataset import in_ts_partition
from open_audio_llm.integrations.ms_swift.retention import sample_losses, supervised_forward, token_mean_loss


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('separate_prefix', [False, True])
def test_chunked_ce_preserves_dense_objective_and_gradients(dtype, separate_prefix):
    from open_audio_llm.integrations.ms_swift.retention import average_answer_parts

    torch.manual_seed(19)
    student = torch.randn(389, 31, dtype=dtype, requires_grad=True)
    teacher = torch.randn(130, 31, dtype=dtype, requires_grad=True)
    targets = torch.randint(31, (389,))
    rows = torch.tensor([0] * 130 + [1] * 258 + [2])
    counts = torch.tensor([130, 258, 1])
    ordinary = rows == 0
    prefix = torch.arange(389) % 7 == 0 if separate_prefix else None
    ce = average_answer_parts(F.cross_entropy(student.float(), targets, reduction='none'),
                              rows, counts, prefix)
    token_kl = F.kl_div(student[ordinary].float().log_softmax(-1),
                        teacher.detach().float().softmax(-1), reduction='none').sum(-1)
    kl = average_answer_parts(token_kl, rows[ordinary], counts,
                              None if prefix is None else prefix[ordinary])
    expected = ce + 2 * kl
    weight = torch.tensor([0.3, 0.6, 0.1])
    expected_gradient = torch.autograd.grad((expected * weight).sum(), student)[0]
    actual, actual_ce, actual_kl = sample_losses(student, targets, rows, counts,
                                                teacher, ordinary, 2, prefix)
    for value, reference in zip((actual, actual_ce, actual_kl), (expected, ce, kl)):
        torch.testing.assert_close(value, reference)
    (actual * weight).sum().backward()
    torch.testing.assert_close(student.grad, expected_gradient, atol=2e-6, rtol=0.01 if dtype == torch.bfloat16 else 1e-5)
    assert teacher.grad is None


def test_chunked_ce_does_not_retain_full_softmax_and_ignores_masked_targets():
    from open_audio_llm.integrations.ms_swift.retention import _ChunkedCrossEntropy

    logits = torch.randn(259, 17, requires_grad=True)
    targets = torch.randint(17, (259,))
    targets[::13] = -100
    saved = []
    def remember(tensor):
        saved.append(tensor.data_ptr())
        return tensor
    with torch.autograd.graph.saved_tensors_hooks(remember, lambda tensor: tensor):
        actual = _ChunkedCrossEntropy.apply(logits, targets)
    assert saved == [logits.data_ptr(), targets.data_ptr()]
    expected = F.cross_entropy(logits, targets, reduction='none')
    torch.testing.assert_close(actual, expected)
    reference = torch.autograd.grad(expected.sum(), logits)[0]
    actual.sum().backward()
    torch.testing.assert_close(logits.grad, reference)


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


def test_lora_supervised_replay_does_not_load_unused_teacher(tmp_path, monkeypatch):
    from qwen_asr import Qwen3ASRModel
    from open_audio_llm.integrations.ms_swift.retention import install_retention_objective

    monkeypatch.setattr(Qwen3ASRModel, 'from_pretrained',
                        lambda *a, **k: pytest.fail('zero KL must not load a teacher'))
    thinker = torch.nn.Module()
    thinker.model = torch.nn.Linear(3, 3).requires_grad_(False)
    thinker.model.config = SimpleNamespace(use_cache=False)
    thinker.audio_tower = torch.nn.Linear(3, 3).requires_grad_(False)
    thinker.lm_head = torch.nn.Linear(3, 3).requires_grad_(False)
    thinker.lora_adapter = torch.nn.Linear(3, 3)
    thinker.generation_config = SimpleNamespace(use_cache=False)
    student = torch.nn.Module()
    student.thinker = thinker
    trainer = SimpleNamespace(
        model=student, optimizer=None,
        template=SimpleNamespace(tokenizer=SimpleNamespace(convert_tokens_to_ids=lambda x: 1)),
        args=SimpleNamespace(output_dir=str(tmp_path)),
        is_world_process_zero=lambda: True,
    )
    args = SimpleNamespace(tuner_type='lora', retention_teacher=None, deepspeed=None,
                           use_logits_to_keep=False, audio_encoder_parallel=False,
                           learning_rate=2e-5, vit_lr=0, aligner_lr=0,
                           _catalog_config={'objective': {'replay_kl_weight': 0}})
    install_retention_objective(trainer, args)
    assert trainer.retention_teacher is None
    assert not any(p.requires_grad for p in thinker.audio_tower.parameters())
    assert all(p.requires_grad for p in thinker.lora_adapter.parameters())
    assert not json.loads((tmp_path / 'retention-objective.json').read_text())['teacher_enabled']


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


def test_token_mean_long_transcript_contributes_in_proportion_to_its_tokens():
    logits = torch.zeros(11, 3, requires_grad=True)
    targets = torch.zeros(11, dtype=torch.long)
    rows = torch.tensor([0] * 10 + [1])
    counts = torch.tensor([10, 1])
    losses, _, _ = sample_losses(logits, targets, rows, counts)
    token_mean_loss(losses, counts).backward()
    assert logits.grad[:10].abs().sum() == pytest.approx(10 * logits.grad[10:].abs().sum())


def test_token_mean_gives_each_supervised_token_equal_weight():
    logits = torch.zeros(27, 4, requires_grad=True)
    targets = torch.zeros(27, dtype=torch.long)
    rows = torch.tensor([0] * 23 + [1] * 4)
    counts = torch.tensor([23, 4])
    losses, _, _ = sample_losses(logits, targets, rows, counts)
    token_mean_loss(losses, counts).backward()
    torch.testing.assert_close(logits.grad[0], logits.grad[26])
    torch.testing.assert_close(logits.grad[:23].sum(0) / 23, logits.grad[23:].sum(0) / 4)


def test_global_token_normalization_matches_unequal_rank_microbatches():
    torch.manual_seed(42)
    weight = torch.randn(3, 4, requires_grad=True)
    inputs = torch.randn(7, 3)
    targets = torch.tensor([0, 1, 1, 3, 0, 2, 1])
    # Three examples, lengths 2, 1, 4 (7 tokens). Rank 0 has two; rank 1 has one.
    rows = torch.tensor([0, 0, 1, 2, 2, 2, 2])
    counts = torch.tensor([2, 1, 4])
    reference, _, _ = sample_losses(inputs @ weight, targets, rows, counts)
    expected = torch.autograd.grad(token_mean_loss(reference, counts), weight)[0]
    gradients = []
    for slices in ((slice(0, 2), slice(2, 3)), (slice(3, 7),)):
        total = 0
        for section in slices:
            size = len(targets[section])
            loss, _, _ = sample_losses(inputs[section] @ weight, targets[section],
                                       torch.zeros(size, dtype=torch.long), torch.tensor([size]))
            total = total + token_mean_loss(loss, torch.tensor([size]), 7, world_size=2)
        gradients.append(torch.autograd.grad(total, weight)[0])
    torch.testing.assert_close(sum(gradients) / 2, expected)


@pytest.mark.parametrize('objective', [
    {'replay_kl_weight': 0, 'reduction': 'sum'},
    {'replay_kl_weight': 0, 'reduction': 'token_mean', 'separate_prefix': True},
])
def test_invalid_reduction_settings_are_rejected(objective):
    from open_audio_llm.integrations.ms_swift.retention import install_retention_objective

    trainer = SimpleNamespace(optimizer=None, template=SimpleNamespace(
        tokenizer=SimpleNamespace(convert_tokens_to_ids=lambda x: 1)))
    args = SimpleNamespace(tuner_type='lora', retention_teacher=None, deepspeed=None,
                           use_logits_to_keep=False, _catalog_config={'objective': objective})
    with pytest.raises(ValueError, match='reduction'):
        install_retention_objective(trainer, args)
