import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from trl.trainer.utils import selective_log_softmax

from open_audio_llm.integrations.ms_swift.grpo_shared_prefix import pack_shared_prefix, shared_prefix_logps

PAD = 0


def tiny_model():
    torch.manual_seed(0)
    config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=128)
    config._attn_implementation = "sdpa"
    return Qwen3ForCausalLM(config).float().eval()


def batch(prompt, completions):
    """ms-swift's layout: one shared prompt, right-padded completions, logits_to_keep = longest completion."""
    keep = max(len(c) for c in completions)
    ids = torch.full((len(completions), len(prompt) + keep), PAD)
    mask = torch.zeros_like(ids)
    for row, completion in enumerate(completions):
        ids[row, :len(prompt) + len(completion)] = torch.tensor(prompt + completion)
        mask[row, :len(prompt) + len(completion)] = 1
    return {"input_ids": ids, "attention_mask": mask}, keep


def separate_logps(model, inputs, keep, temperature):
    """ms-swift's own path (_get_logps_via_local_forward), one padded batch."""
    logits = model(**inputs).logits[:, -(keep + 1):-1, :] / temperature
    return selective_log_softmax(logits, inputs["input_ids"][:, -keep:])


def test_packed_scores_equal_separate_scores_and_gradients():
    model = tiny_model()
    inputs, keep = batch([5, 6, 7, 8, 9, 10], [[11, 12, 13], [14], [15, 16, 17, 18], [19, 20]])
    mask = inputs["attention_mask"][:, -keep:].bool()
    reference = separate_logps(model, inputs, keep, 0.7)
    (reference * mask).sum().backward()
    reference_grad = {n: p.grad.clone() for n, p in model.named_parameters()}
    model.zero_grad()

    packed = pack_shared_prefix(inputs, keep, torch.float32)
    assert packed is not None
    assert packed[0]["input_ids"].shape == (1, 6 + 3 + 1 + 4 + 2)
    logps, _ = shared_prefix_logps(model, *packed, keep, 0.7)
    assert torch.allclose(logps[mask], reference[mask], atol=1e-5)
    assert (logps[~mask] == 0).all()
    (logps * mask).sum().backward()
    for name, param in model.named_parameters():
        assert torch.allclose(param.grad, reference_grad[name], atol=1e-5), name


def test_rows_that_differ_fall_back():
    inputs, keep = batch([5, 6, 7], [[11, 12], [13]])
    inputs["input_ids"][1, 0] = 9  # another prompt
    assert pack_shared_prefix(inputs, keep, torch.float32) is None

    inputs, keep = batch([5, 6, 7], [[11, 12], [13]])
    inputs["input_features"] = torch.stack([torch.zeros(4, 3), torch.ones(4, 3)])  # another clip
    assert pack_shared_prefix(inputs, keep, torch.float32) is None

    inputs, keep = batch([5, 6, 7], [[11, 12], [13]])
    inputs["pixel_values"] = torch.zeros(2, 3)  # an input whose batching is unknown here
    assert pack_shared_prefix(inputs, keep, torch.float32) is None


def test_per_row_audio_inputs_are_sent_once():
    inputs, keep = batch([5, 6, 7], [[11, 12], [13]])
    inputs["input_features"] = torch.ones(2, 4, 3)
    inputs["feature_attention_mask"] = torch.ones(2, 3, dtype=torch.long)
    inputs["enroll_n_frames"] = torch.tensor([2, 2])
    inputs["use_cache"] = False  # ms-swift writes this into the shared dict
    packed, predictor, tokens, lens = pack_shared_prefix(inputs, keep, torch.float32)
    assert packed["input_features"].shape == (1, 4, 3)
    assert packed["enroll_n_frames"].tolist() == [2]
    assert "use_cache" not in packed
    assert packed["position_ids"].tolist() == [[0, 1, 2, 3, 4, 3]]
    assert predictor.tolist() == [2, 3, 2]
    assert tokens.tolist() == [11, 12, 13] and lens.tolist() == [2, 1]
    allowed = packed["attention_mask"][0, 0] == 0
    assert allowed.tolist() == [
        [True, False, False, False, False, False],
        [True, True, False, False, False, False],
        [True, True, True, False, False, False],
        [True, True, True, True, False, False],
        [True, True, True, True, True, False],
        [True, True, True, False, False, True],  # the second completion does not see the first
    ]


def test_left_padded_prompts_fall_back():
    inputs, keep = batch([5, 6, 7], [[11, 12], [13]])
    inputs["attention_mask"][:, 0] = 0
    assert pack_shared_prefix(inputs, keep, torch.float32) is None
