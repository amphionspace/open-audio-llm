from types import MethodType, SimpleNamespace

import torch
from transformers.loss.loss_utils import ForCausalLMLoss
from transformers.modeling_outputs import CausalLMOutputWithPast

from open_audio_llm.integrations.ms_swift.supervised_head import enable_supervised_head


class TinyAudioLM(torch.nn.Module):
    """Mirrors the Qwen3-ASR wrapper: loss inside forward, head at ``thinker.lm_head``."""

    def __init__(self):
        super().__init__()
        self.thinker = torch.nn.Module()
        self.thinker.embedding = torch.nn.Embedding(11, 6)
        self.thinker.lm_head = torch.nn.Linear(6, 11, bias=False)

    def forward(self, input_ids, attention_mask=None, labels=None, use_cache=None):
        logits = self.thinker.lm_head(self.thinker.embedding(input_ids))
        loss = None if labels is None else ForCausalLMLoss(logits, labels, vocab_size=11)
        return CausalLMOutputWithPast(loss=loss, logits=logits)


class Accuracy:
    def __init__(self):
        self.values = []

    def update(self, values):
        self.values.extend(values.tolist())


def _trainer(model):
    logged = []
    trainer = SimpleNamespace(args=SimpleNamespace(acc_strategy="token"), model=model,
                              custom_metrics={"train": {"token_acc": Accuracy()}})
    trainer.log = MethodType(lambda self, logs: logged.append(dict(logs)), trainer)
    return trainer, logged


def test_supervised_head_matches_full_vocabulary_loss_gradients_and_accuracy():
    torch.manual_seed(7)
    reference = TinyAudioLM()
    model = TinyAudioLM()
    model.load_state_dict(reference.state_dict())
    inputs = {"input_ids": torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 0]]),
              "attention_mask": torch.ones(2, 5, dtype=torch.long)}
    labels = torch.tensor([[-100, -100, 3, 4, 5], [-100, 7, -100, 9, -100]])
    expected = reference(**inputs, labels=labels)
    expected_grads = torch.autograd.grad(expected.loss, tuple(reference.parameters()))
    shifted = labels[:, 1:]
    expected_accuracy = (expected.logits[:, :-1].argmax(-1) == shifted)[shifted != -100].tolist()

    trainer, logged = _trainer(model)
    enable_supervised_head(trainer)
    actual = model(**inputs, labels=labels)

    assert actual.logits is None  # ms-swift skips its CPU accuracy pass.
    torch.testing.assert_close(actual.loss, expected.loss)
    for actual_grad, expected_grad in zip(torch.autograd.grad(actual.loss, tuple(model.parameters())), expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)
    assert trainer.custom_metrics["train"]["token_acc"].values == []
    trainer.log({"loss": 1.0})
    assert trainer.custom_metrics["train"]["token_acc"].values == expected_accuracy
    assert logged == [{"loss": 1.0}]
    trainer.log({"loss": 2.0})  # Flushed values are not counted twice.
    assert trainer.custom_metrics["train"]["token_acc"].values == expected_accuracy


def test_supervised_head_keeps_full_logits_for_evaluation_and_label_free_calls():
    model = TinyAudioLM()
    trainer, _ = _trainer(model)
    enable_supervised_head(trainer)
    inputs = {"input_ids": torch.tensor([[1, 2, 3]])}
    assert model(**inputs).logits.shape == (1, 3, 11)
    model.eval()
    output = model(**inputs, labels=torch.tensor([[-100, 2, 3]]))
    assert output.logits.shape == (1, 3, 11) and output.loss is not None
    model.train()
    model(**inputs, labels=torch.tensor([[-100, 2, 3]]))
    assert model(**inputs).logits.shape == (1, 3, 11)  # The head hook never outlives a call.
