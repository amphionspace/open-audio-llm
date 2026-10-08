from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from open_audio_llm.integrations.ms_swift.retention import (
    _ChunkedLinearCrossEntropy,
    chunked_supervised_loss,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_chunked_linear_ce_matches_dense_loss_without_saving_vocab_logits(dtype):
    torch.manual_seed(7)
    hidden = torch.randn(300, 8, dtype=dtype, requires_grad=True)
    weight = torch.randn(19, 8, dtype=dtype, requires_grad=True)
    targets = torch.randint(19, (300,))
    targets[::17] = -100
    saved = []

    def remember(tensor):
        saved.append(tuple(tensor.shape))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(remember, lambda tensor: tensor):
        actual = _ChunkedLinearCrossEntropy.apply(hidden, weight, targets)
    assert saved == [(300, 8), (19, 8), (300,)]
    dense = F.cross_entropy(F.linear(hidden, weight).float(), targets, reduction="none")
    torch.testing.assert_close(actual, dense)
    expected_grads = torch.autograd.grad(dense.sum(), (hidden, weight))
    actual.sum().backward()
    tolerance = {"atol": 2e-6, "rtol": 1e-5} if dtype == torch.float32 else {"atol": 1e-2, "rtol": 2e-2}
    torch.testing.assert_close(hidden.grad, expected_grads[0], **tolerance)
    torch.testing.assert_close(weight.grad, expected_grads[1], **tolerance)


def test_chunked_supervised_loss_ignores_audio_positions_and_restores_head():
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(9, 5)
            self.head = torch.nn.Linear(5, 9, bias=False)

        def forward(self, input_ids, use_cache=False):
            return SimpleNamespace(logits=self.head(self.embed(input_ids)), loss=None)

    torch.manual_seed(3)
    model = Tiny()
    inputs = {"input_ids": torch.tensor([[1, 2, 3, 4], [2, 4, 5, 0]])}
    labels = torch.tensor([[-100, -100, 3, 4], [-100, 4, 5, -100]])
    full = model(**inputs).logits
    expected = F.cross_entropy(full[:, :-1].reshape(-1, 9).float(), labels[:, 1:].reshape(-1))
    expected_grads = torch.autograd.grad(expected, tuple(model.parameters()))
    outputs, actual, correct = chunked_supervised_loss(model, model.head, inputs, labels)
    assert outputs.logits is None
    assert correct.shape == (4,)
    torch.testing.assert_close(actual, expected)
    for actual_grad, expected_grad in zip(
        torch.autograd.grad(actual, tuple(model.parameters())), expected_grads
    ):
        torch.testing.assert_close(actual_grad, expected_grad)
    assert model(**inputs).logits.shape == (2, 4, 9)
