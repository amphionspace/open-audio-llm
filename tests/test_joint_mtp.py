import json
from collections import defaultdict
from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from open_audio_llm.integrations.ms_swift import draft_head
from open_audio_llm.integrations.ms_swift.draft_head import draft_rows
from open_audio_llm.integrations.ms_swift.joint_mtp import (
    MTP,
    MTPModule,
    enable_joint_mtp,
    mtp_loss,
)

DIMS = {"hidden_size": 16, "intermediate_size": 24, "num_attention_heads": 4,
        "num_key_value_heads": 2, "head_dim": 8, "rms_norm_eps": 1e-6, "rope_theta": 10000.0}


def _module(seed=0):
    torch.manual_seed(seed)
    module = MTPModule(**DIMS)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(0, 0.3)
    return module


def _qwen3_config():
    from transformers import Qwen3Config

    config = Qwen3Config(vocab_size=13, hidden_size=16, intermediate_size=24, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=8, rms_norm_eps=1e-6,
                         tie_word_embeddings=True, rope_theta=10000.0)
    config._attn_implementation = "sdpa"
    return config


def test_first_step_is_a_qwen3_decoder_layer_on_the_fused_input_then_the_final_norm():
    # vLLM's Qwen3-Next MTP layout: fc(cat(pre_fc_norm_embedding(e), pre_fc_norm_hidden(h)))
    # -> decoder layer (with its input_layernorm and q/k norms) -> norm.
    from transformers.models.qwen3.modeling_qwen3 import (
        Qwen3DecoderLayer,
        Qwen3RotaryEmbedding,
    )

    module = _module()
    config = _qwen3_config()
    layer = Qwen3DecoderLayer(config, layer_idx=0)
    missing, unexpected = layer.load_state_dict(
        {k.removeprefix("layers.0."): v for k, v in module.state_dict().items() if k.startswith("layers.0.")})
    assert missing == [] and unexpected == []
    embeds, hidden = torch.randn(2, 7, 16), torch.randn(2, 7, 16)
    x = module.fc(torch.cat((module.pre_fc_norm_embedding(embeds), module.pre_fc_norm_hidden(hidden)), -1))
    positions = torch.arange(7).expand(2, -1)
    mask = torch.full((7, 7), float("-inf")).triu(1)[None, None].expand(2, 1, 7, 7)
    out = layer(x, attention_mask=mask, position_ids=positions,
                position_embeddings=Qwen3RotaryEmbedding(config)(x, positions))
    out = out[0] if isinstance(out, tuple) else out
    torch.testing.assert_close(module.unrolled(embeds, hidden, 1)[0], module.norm(out), rtol=1e-4, atol=1e-5)


def _sequential(module, embeds, hidden, anchor, steps):
    """vLLM's MTP proposer on one sequence: first pass over rows <= anchor, then one row per
    step fed with the module's own (post-norm) output and the drafted token."""
    layer, cfg = module.layers[0], module.config
    heads, kv_heads, dim = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    cos, sin = module._rope(anchor + steps, hidden.device)
    attention = layer.self_attn

    def block(embed, previous, position, cache):
        x = module.fc(torch.cat((module.pre_fc_norm_embedding(embed), module.pre_fc_norm_hidden(previous))))
        normed = layer.input_layernorm(x)
        rotate = lambda t: t * cos[position] + draft_head._rotate_half(t) * sin[position]
        q = rotate(attention.q_norm(attention.q_proj(normed).view(heads, dim)))
        k = rotate(attention.k_norm(attention.k_proj(normed).view(kv_heads, dim)))
        cache.append((k, attention.v_proj(normed).view(kv_heads, dim)))
        keys = torch.stack([k for k, _ in cache], 1).repeat_interleave(heads // kv_heads, 0)
        values = torch.stack([v for _, v in cache], 1).repeat_interleave(heads // kv_heads, 0)
        weights = ((q[:, None, :] * keys).sum(-1) / dim ** 0.5).softmax(-1)
        h = x + attention.o_proj((weights[..., None] * values).sum(1).reshape(-1))
        return module.norm(h + layer.mlp(layer.post_attention_layernorm(h)))

    cache, feature = [], None
    for row in range(anchor + 1):
        feature = block(embeds[row + 1], hidden[row], row, cache)
    features = [feature]
    for step in range(2, steps + 1):
        position = anchor + step - 1
        feature = block(embeds[position + 1], feature, position, cache)
        features.append(feature)
    return features


def test_unrolled_steps_match_sequential_vllm_mtp_drafting():
    module = _module(1)
    length, steps = 9, 3
    token_embeds = torch.randn(length + 1, 16)  # embedding of token t at row t
    hidden = torch.randn(length, 16)
    features = module.unrolled(token_embeds[1:][None], hidden[None], steps)
    for anchor in range(length - steps + 1):
        expected = _sequential(module, token_embeds, hidden, anchor, steps)
        for step in range(1, steps + 1):
            torch.testing.assert_close(features[step - 1][0, anchor + step - 1], expected[step - 1],
                                       rtol=1e-4, atol=1e-5)


class _Metric:
    def __init__(self):
        self.values = []

    def update(self, values):
        self.values.extend(values.reshape(-1).tolist())


class _TinyASR(torch.nn.Module):
    """Mirrors Qwen3-ASR: forward in ``thinker``, tied ``lm_head`` fed by the final norm."""

    def __init__(self):
        super().__init__()
        from transformers import Qwen3ForCausalLM

        config = _qwen3_config()
        self.thinker = Qwen3ForCausalLM(config)
        self.thinker.config.text_config = config
        self.config = config

    def forward(self, input_ids, attention_mask=None, labels=None, use_cache=None):
        return self.thinker(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                            use_cache=use_cache)


def test_from_thinker_starts_as_the_identity_on_the_backbone_output():
    model = _TinyASR()
    module = MTPModule.from_thinker(model.thinker)
    last = model.thinker.model.layers[-1].state_dict()
    for name, value in module.layers[0].state_dict().items():
        if name in ("self_attn.o_proj.weight", "mlp.down_proj.weight"):
            assert not value.any()
        else:
            torch.testing.assert_close(value, last[name])
    with torch.no_grad():
        model.thinker.model.norm.weight.uniform_(0.5, 1.5)
        hidden = model.thinker.model.norm(torch.randn(1, 5, 16) * 3)
        out = module.unrolled(torch.randn(1, 5, 16), hidden, 1)[0]
    # The backbone's own logits up to a positive per-position scale.
    torch.testing.assert_close(out, hidden / hidden.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt(),
                               rtol=1e-4, atol=1e-5)


def _batch():
    torch.manual_seed(3)
    input_ids = torch.randint(0, 13, (2, 10))
    attention_mask = torch.ones(2, 10, dtype=torch.long)
    attention_mask[1, 8:] = 0
    labels = torch.full((2, 10), -100)
    labels[0, 4:] = input_ids[0, 4:]
    labels[1, 3:8] = input_ids[1, 3:8]
    return input_ids, attention_mask, labels


@pytest.mark.parametrize("steps", [1, 3])
def test_joint_loss_adds_weighted_mtp_ce_and_trains_backbone_and_module(tmp_path, steps):
    torch.manual_seed(4)
    model = _TinyASR().train()
    saved = {}
    trainer = SimpleNamespace(
        model=model, args=SimpleNamespace(output_dir=str(tmp_path), acc_strategy="token", should_save=True),
        state=SimpleNamespace(global_step=5), custom_metrics={"train": defaultdict(_Metric)})
    trainer.log = MethodType(lambda self, logs: saved.setdefault("logs", dict(logs)), trainer)
    trainer._save_model = MethodType(
        lambda self, output_dir=None, state_dict=None: saved.update(state=state_dict), trainer)
    module = enable_joint_mtp(trainer, 0.3, steps, 0.8)
    assert all(p.requires_grad for p in model.parameters())

    input_ids, attention_mask, labels = _batch()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    outputs.loss.backward()
    assert all(p.grad is not None for p in model.parameters())
    assert model.thinker.lm_head.weight.grad.abs().sum() > 0

    # Same loss from an ordinary forward: next-token CE + 0.3 * MTP CE on the token after next.
    with torch.no_grad():
        hidden = model.thinker.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        logits = model.thinker.lm_head(hidden)
        main = F.cross_entropy(logits[:, :-1].flatten(0, 1), labels[:, 1:].flatten(), ignore_index=-100)
        auxiliary, _ = mtp_loss(module, model.thinker.lm_head.weight, model.thinker.model.embed_tokens(input_ids),
                                hidden, labels, steps, 0.8)
    torch.testing.assert_close(outputs.loss.detach(), main + 0.3 * auxiliary, rtol=1e-5, atol=1e-6)

    trainer.log({"loss": 1.0})
    metrics = trainer.custom_metrics["train"]
    assert len(metrics["mtp_ce_1"].values) == int(draft_rows(labels, steps)[0].sum())
    assert {f"mtp_acc_{steps}", "main_loss", "token_acc"} <= set(metrics)

    with pytest.raises(ValueError, match="right-padded"):
        model(input_ids=input_ids, attention_mask=attention_mask.flip(-1), labels=labels)

    trainer._save_model(str(tmp_path))
    assert not any(k.startswith(f"{MTP}.") for k in saved["state"]) and saved["state"]
    weights = load_file(str(tmp_path / MTP / "model.safetensors"))
    assert set(weights) == set(module.state_dict())
    assert {"fc.weight", "pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight", "norm.weight",
            "layers.0.self_attn.q_norm.weight"} <= set(weights)
    config = json.loads((tmp_path / MTP / "config.json").read_text())
    assert config["global_step"] == 5 and config["joint_mtp_weight"] == 0.3 and config["vocab_size"] == 13


def test_joint_mtp_init_loads_a_saved_module(tmp_path):
    model = _TinyASR().train()
    trainer = SimpleNamespace(model=model, args=SimpleNamespace(output_dir=str(tmp_path), acc_strategy="token",
                                                                should_save=True),
                              state=SimpleNamespace(global_step=1), custom_metrics={"train": defaultdict(_Metric)})
    trainer.log = MethodType(lambda self, logs: None, trainer)
    trainer._save_model = MethodType(lambda self, output_dir=None, state_dict=None: None, trainer)
    source = _module(7)
    from safetensors.torch import save_file

    save_file(source.state_dict(), str(tmp_path / "init.safetensors"))
    module = enable_joint_mtp(trainer, 0.1, init=str(tmp_path / "init.safetensors"))
    for name, value in source.state_dict().items():
        torch.testing.assert_close(module.state_dict()[name], value)


def test_joint_mtp_lr_moves_module_parameters_into_their_own_groups(tmp_path):
    model = _TinyASR().train()
    trainer = SimpleNamespace(model=model, args=SimpleNamespace(output_dir=str(tmp_path), acc_strategy="token",
                                                                should_save=True),
                              state=SimpleNamespace(global_step=1), custom_metrics={"train": defaultdict(_Metric)})
    trainer.log = MethodType(lambda self, logs: None, trainer)
    trainer._save_model = MethodType(lambda self, output_dir=None, state_dict=None: None, trainer)

    def create_optimizer():
        decay = [p for n, p in model.named_parameters() if "norm" not in n]
        rest = [p for n, p in model.named_parameters() if "norm" in n]
        return torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": rest, "weight_decay": 0.0}],
                                 lr=3e-6)

    trainer.create_optimizer = create_optimizer
    module = enable_joint_mtp(trainer, 0.1, lr=1e-4)
    optimizer = trainer.create_optimizer()
    own = {id(p) for p in module.parameters()}
    groups = [(g["lr"], g["weight_decay"], {id(p) for p in g["params"]}) for g in optimizer.param_groups]
    assert sum(len(ids) for _, _, ids in groups) == len(list(model.parameters()))
    assert {(lr, wd) for lr, wd, ids in groups if ids & own} == {(1e-4, 0.1), (1e-4, 0.0)}
    assert all(not (ids & own) for lr, _, ids in groups if lr == 3e-6)
