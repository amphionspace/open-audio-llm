import json
from collections import defaultdict
from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from open_audio_llm.integrations.ms_swift import draft_head
from open_audio_llm.integrations.ms_swift.draft_head import (
    DRAFT,
    DraftHead,
    _ChunkedHeadKL,
    capture_aux,
    draft_loss,
    draft_rows,
    enable_draft_head,
    export_eagle,
    unrolled_mask,
)

DIMS = {"hidden_size": 16, "intermediate_size": 24, "num_attention_heads": 4,
        "num_key_value_heads": 2, "head_dim": 8, "rms_norm_eps": 1e-6, "rope_theta": 10000.0}


def _head(seed=0, aux_layers=()):
    torch.manual_seed(seed)
    head = DraftHead(**DIMS, aux_layers=aux_layers)
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.normal_(0, 0.3)
    return head


def test_first_step_is_a_llama_layer_without_input_norm_on_the_fc_output():
    # vLLM's EagleLlamaForCausalLM: fc(cat(embed, hidden)) -> LlamaDecoderLayer whose
    # input_layernorm is Identity -> hidden + residual.
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import (
        LlamaDecoderLayer,
        LlamaRotaryEmbedding,
    )

    head = _head()
    config = LlamaConfig(hidden_size=16, intermediate_size=24, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=8, rms_norm_eps=1e-6, rope_theta=10000.0)
    config._attn_implementation = "sdpa"
    layer = LlamaDecoderLayer(config, layer_idx=0)
    layer.input_layernorm = torch.nn.Identity()
    missing, unexpected = layer.load_state_dict(
        {k.removeprefix("layers.0."): v for k, v in head.state_dict().items() if k.startswith("layers.0.")},
        strict=False)
    assert missing == [] and unexpected == []
    embeds, hidden = torch.randn(2, 7, 16), torch.randn(2, 7, 16)
    x = head.fc(torch.cat((embeds, hidden), -1))
    positions = torch.arange(7).expand(2, -1)
    expected = layer(x, attention_mask=None, position_ids=positions,
                     position_embeddings=LlamaRotaryEmbedding(config)(x, positions))
    expected = expected[0] if isinstance(expected, tuple) else expected
    torch.testing.assert_close(head.unrolled(embeds, hidden, 1)[0], expected)


def _sequential(head, embeds, target, anchor, steps):
    """vLLM's proposer on one sequence: first pass over rows <= anchor, then one row per step.

    Row r of the first pass holds (embed of token r + 1, target state at r) at position r;
    step k >= 2 holds (embed of the drafted token, previous step's carried output) at
    anchor + k - 1. EAGLE-1 (llama_eagle.py): x = fc(cat(embed, hidden)), output feeds both
    lm_head and the next step. EAGLE-3 (llama_eagle3.py): the first pass hidden is fc(aux),
    attention reads cat(input_layernorm(embed), hidden_norm(hidden)) with residual hidden,
    lm_head reads norm(out) and the next step gets out before the norm.
    """
    layer, cfg = head.layers[0], head.config
    heads, kv_heads, dim = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    cos, sin = head._rope(anchor + steps, target.device)

    def project(x, position):
        attention = layer.self_attn
        q = attention.q_proj(x).view(heads, dim)
        k = attention.k_proj(x).view(kv_heads, dim)
        rotate = lambda t: t * cos[position] + draft_head._rotate_half(t) * sin[position]
        return rotate(q), rotate(k), attention.v_proj(x).view(kv_heads, dim)

    def block(embed, hidden, position, cache):
        if head.eagle3:
            residual = hidden
            x = torch.cat((layer.input_layernorm(embed), layer.hidden_norm(hidden)))
        else:
            x = residual = head.fc(torch.cat((embed, hidden)))
        q, k, v = project(x, position)
        cache.append((k, v))
        keys = torch.stack([k for k, _ in cache], 1).repeat_interleave(heads // kv_heads, 0)
        values = torch.stack([v for _, v in cache], 1).repeat_interleave(heads // kv_heads, 0)
        weights = ((q[:, None, :] * keys).sum(-1) / dim ** 0.5).softmax(-1)
        out = (weights[..., None] * values).sum(1).reshape(-1)
        h = residual + layer.self_attn.o_proj(out)
        carried = h + layer.mlp(layer.post_attention_layernorm(h))
        return carried, head.norm(carried) if head.eagle3 else carried

    cache, carried, feature = [], None, None
    for row in range(anchor + 1):
        first = head.fc(target[row]) if head.eagle3 else target[row]
        carried, feature = block(embeds[row + 1], first, row, cache)
    features = [feature]
    for step in range(2, steps + 1):
        position = anchor + step - 1
        carried, feature = block(embeds[position + 1], carried, position, cache)
        features.append(feature)
    return features


@pytest.mark.parametrize("aux_layers", [(), (1, 2, 4)])
def test_unrolled_steps_match_sequential_vllm_drafting(aux_layers):
    head = _head(1, aux_layers)
    length, steps = 9, 3
    tokens_embeds = torch.randn(length + 1, 16)  # embedding of token t at row t
    target = torch.randn(length, 16 * max(1, len(aux_layers)))
    features = head.unrolled(tokens_embeds[1:][None], target[None], steps)
    for anchor in range(length - steps + 1):
        expected = _sequential(head, tokens_embeds, target, anchor, steps)
        for step in range(1, steps + 1):
            torch.testing.assert_close(features[step - 1][0, anchor + step - 1], expected[step - 1],
                                       rtol=1e-4, atol=1e-5)


def test_aux_capture_reads_the_residual_stream_vllm_returns():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(6)
    config = Qwen3Config(vocab_size=13, hidden_size=16, intermediate_size=24, num_hidden_layers=4,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=8)
    config._attn_implementation = "sdpa"
    model = Qwen3ForCausalLM(config).eval()
    store, layers = {}, model.model.layers
    after = {}
    handles = capture_aux(layers, (0, 1, 3, 4), store)
    handles += [layer.register_forward_hook(lambda m, a, o, i=i: after.__setitem__(i, o))
                for i, layer in enumerate(layers)]
    with torch.no_grad():
        out = model.model(input_ids=torch.randint(0, 13, (2, 7)), output_hidden_states=True)
    for handle in handles:
        handle.remove()
    # vLLM's aux j is the stream after layer j - 1 (embeddings for j = 0), before the final norm.
    torch.testing.assert_close(store[0], out.hidden_states[0])
    for j in (1, 3, 4):
        torch.testing.assert_close(store[j], after[j - 1])
    with pytest.raises(ValueError, match="past the decoder"):
        capture_aux(layers, (5,), store)


def test_eagle3_export_has_the_vllm_eagle3_layout(tmp_path):
    head = _head(2, (1, 2, 4))
    export_eagle(head, tmp_path, vocab_size=13)
    config = json.loads((tmp_path / "config.json").read_text())
    assert config["architectures"] == ["LlamaForCausalLMEagle3"]
    assert config["eagle_aux_hidden_state_layer_ids"] == [1, 2, 4] and config["target_hidden_size"] == 16
    weights = load_file(str(tmp_path / "model.safetensors"))
    assert weights["fc.weight"].shape == (16, 48)
    assert weights["layers.0.self_attn.q_proj.weight"].shape == (32, 32)  # reads cat(embed, hidden)
    assert {"layers.0.input_layernorm.weight", "layers.0.hidden_norm.weight", "norm.weight"} <= set(weights)
    with pytest.raises(ValueError, match="increasing"):
        DraftHead(**DIMS, aux_layers=(4, 1))


def test_unrolled_mask_reaches_first_pass_prefix_and_own_proposal_keys():
    mask = unrolled_mask(5, 3)
    # Query q=4 at step 3 belongs to the proposal anchored at 2: K1[0..2], K2[3], K3[4].
    assert mask[4].nonzero().squeeze(1).tolist() == [0, 1, 2, 5 + 3, 10 + 4]
    assert unrolled_mask(4, 1).equal(torch.ones(4, 4, dtype=torch.bool).tril())


def test_draft_rows_keep_both_tokens_inside_the_assistant_turn():
    labels = torch.tensor([[-100, -100, -100, 5, 6, 7, 8, -100]])
    rows = draft_rows(labels, 3)
    # Step 1 at q drafts token q + 2 after the target emitted token q + 1.
    assert rows[0][0].nonzero().squeeze(1).tolist() == [2, 3, 4]
    # Step k at q needs tokens q - k + 2 .. q + 2 supervised.
    assert rows[1][0].nonzero().squeeze(1).tolist() == [3, 4]
    assert rows[2][0].nonzero().squeeze(1).tolist() == [4]


def test_chunked_head_kl_matches_full_vocabulary_kl(monkeypatch):
    monkeypatch.setattr(draft_head, "_LOSS_CHUNK_ROWS", 3)
    torch.manual_seed(2)
    weight = torch.randn(11, 6)
    student = torch.randn(7, 6, requires_grad=True)
    teacher = torch.randn(7, 6)
    kl, agree = _ChunkedHeadKL.apply(student, teacher, weight)
    expected = F.kl_div((student @ weight.t()).log_softmax(-1), (teacher @ weight.t()).log_softmax(-1),
                        reduction="none", log_target=True).sum(-1)
    torch.testing.assert_close(kl, expected)
    assert agree.tolist() == ((student @ weight.t()).argmax(-1) == (teacher @ weight.t()).argmax(-1)).tolist()
    grad_out = torch.randn(7)
    (actual,) = torch.autograd.grad((kl * grad_out).sum(), student)
    (reference,) = torch.autograd.grad((expected * grad_out).sum(), student)
    torch.testing.assert_close(actual, reference)


def test_draft_loss_counts_accepted_tokens_per_complete_proposal():
    head = _head(3)
    weight = torch.randn(13, 16)
    labels = torch.tensor([[-100, -100, 1, 2, 3, 4, 5, 6], [-100, 1, 2, 3, -100, -100, -100, -100]])
    hidden = torch.randn(2, 8, 16)
    loss, stats, accepted = draft_loss(head, weight, torch.randn(2, 8, 16), hidden, labels, 2, 0.8)
    rows = draft_rows(labels, 2)
    assert [len(kl) for kl, _ in stats] == [int(r.sum()) for r in rows]
    expected = (0.8 ** 0 * stats[0][0].mean() + 0.8 * stats[1][0].mean()) / 1.8
    torch.testing.assert_close(loss, expected)
    # Complete two-step proposals: anchors 1..4 in row 0, anchor 0 in row 1.
    assert len(accepted) == 5 and accepted.max() <= 2


class _Metric:
    def __init__(self):
        self.values = []

    def update(self, values):
        self.values.extend(values.tolist())


class _TinyASR(torch.nn.Module):
    """Mirrors Qwen3-ASR: forward in ``thinker``, tied ``lm_head`` fed by the final norm."""

    def __init__(self, layers=2):
        super().__init__()
        from transformers import Qwen3Config, Qwen3ForCausalLM

        config = Qwen3Config(vocab_size=13, hidden_size=16, intermediate_size=24, num_hidden_layers=layers,
                             num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                             tie_word_embeddings=True, rope_theta=10000.0)
        config._attn_implementation = "sdpa"
        self.thinker = Qwen3ForCausalLM(config)
        self.thinker.config.text_config = config
        self.config = config

    def forward(self, input_ids, attention_mask=None, labels=None, use_cache=None):
        return self.thinker(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                            use_cache=use_cache)


def test_enable_draft_head_trains_only_the_head_and_saves_a_vllm_draft(tmp_path):
    torch.manual_seed(4)
    model = _TinyASR().train()
    saved = {}
    trainer = SimpleNamespace(
        model=model, args=SimpleNamespace(output_dir=str(tmp_path)), state=SimpleNamespace(global_step=5),
        custom_metrics={"train": defaultdict(_Metric)})
    trainer.log = MethodType(lambda self, logs: saved.setdefault("logs", dict(logs)), trainer)
    trainer._save_model = MethodType(
        lambda self, output_dir=None, state_dict=None: saved.update(state=state_dict), trainer)
    head = enable_draft_head(trainer, 3, 0.8)
    assert head.fc.weight.dtype == torch.float32 and abs(head.fc.weight.std() - 0.02) < 0.005
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert trainable and all(name.startswith(f"{DRAFT}.") for name in trainable)

    input_ids = torch.randint(0, 13, (2, 10))
    attention_mask = torch.ones(2, 10, dtype=torch.long)
    attention_mask[1, 8:] = 0
    labels = torch.full((2, 10), -100)
    labels[0, 4:] = input_ids[0, 4:]
    labels[1, 3:8] = input_ids[1, 3:8]
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    outputs.loss.backward()
    assert model.thinker.training and all(m.training for m in model.thinker.modules())
    assert all((p.grad is not None) == p.requires_grad for p in model.parameters())

    # Same loss straight from the frozen model's final-norm output.
    with torch.no_grad():
        hidden = model.thinker.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    expected, _, _ = draft_loss(head, model.thinker.lm_head.weight, model.thinker.model.embed_tokens(input_ids),
                                hidden, labels, 3, 0.8)
    torch.testing.assert_close(outputs.loss, expected)

    trainer.log({"loss": 1.0})
    metrics = trainer.custom_metrics["train"]
    assert len(metrics["draft_kl_1"].values) == int(draft_rows(labels, 3)[0].sum())
    assert set(metrics) == {"draft_kl_1", "draft_kl_2", "draft_kl_3", "draft_acc_1", "draft_acc_2",
                            "draft_acc_3", "draft_accepted_of_3"}

    with pytest.raises(ValueError, match="right-padded"):
        model(input_ids=input_ids, attention_mask=attention_mask.flip(-1), labels=labels)

    trainer._save_model(str(tmp_path))
    assert set(saved["state"]) == {f"{DRAFT}.{name}" for name in head.state_dict()}
    weights = load_file(str(tmp_path / "eagle" / "model.safetensors"))
    assert set(weights) == {
        "fc.weight", "layers.0.post_attention_layernorm.weight",
        *(f"layers.0.self_attn.{n}_proj.weight" for n in "qkvo"),
        *(f"layers.0.mlp.{n}_proj.weight" for n in ("gate", "up", "down")),
    }
    assert all(w.dtype == torch.bfloat16 for w in weights.values())
    config = json.loads((tmp_path / "eagle" / "config.json").read_text())
    assert config["architectures"] == ["LlamaForCausalLM"] and config["num_hidden_layers"] == 1
    assert config["rope_theta"] == 10000.0 and config["rope_scaling"] is None and config["vocab_size"] == 13
    assert json.loads((tmp_path / "eagle" / "draft_info.json").read_text())["global_step"] == 5


def test_enable_draft_head_eagle3_feeds_vllm_default_layers(tmp_path):
    torch.manual_seed(8)
    model = _TinyASR(layers=8).train()
    trainer = SimpleNamespace(
        model=model, args=SimpleNamespace(output_dir=str(tmp_path)), state=SimpleNamespace(global_step=1),
        custom_metrics={"train": defaultdict(_Metric)})
    trainer.log = MethodType(lambda self, logs: None, trainer)
    trainer._save_model = MethodType(lambda self, output_dir=None, state_dict=None: None, trainer)
    head = enable_draft_head(trainer, 2, 0.8, arch="eagle3")
    assert head.aux_layers == (2, 4, 5)  # vLLM SupportsEagle3 default (2, L // 2, L - 3)
    input_ids = torch.randint(0, 13, (2, 10))
    attention_mask = torch.ones(2, 10, dtype=torch.long)
    labels = torch.full((2, 10), -100)
    labels[:, 4:] = input_ids[:, 4:]
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    store = {}
    handles = capture_aux(model.thinker.model.layers, head.aux_layers, store)
    with torch.no_grad():
        hidden = model.thinker.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    for handle in handles:
        handle.remove()
    expected, _, _ = draft_loss(head, model.thinker.lm_head.weight, model.thinker.model.embed_tokens(input_ids),
                                hidden, labels, 2, 0.8, aux=torch.cat([store[j] for j in (2, 4, 5)], -1))
    torch.testing.assert_close(outputs.loss, expected)
    outputs.loss.backward()
    assert head.fc.weight.grad is not None and head.norm.weight.grad is not None
    trainer._save_model(str(tmp_path))
    info = json.loads((tmp_path / "eagle" / "draft_info.json").read_text())
    assert info["method"] == "eagle3"
