from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from audio_data_contract import AudioRecord, AudioRef, AudioSlot

from open_audio_llm.data.sot import TIMESTAMP_FORMAT
from open_audio_llm.data.target_sot import (
    TARGET_FORMAT,
    EnrollmentConfig,
    eligible_candidates,
    parse_target_segments,
    render_target,
    sample_enrollment,
    target_prompt,
)


@pytest.fixture
def record():
    mixture = AudioRef("meeting", "v1", "train", "mix", duration=10)
    candidates = [{"speaker_id": s, "partition": "train", "single_speaker": True,
                       "ref": AudioRef("voices", "v1", "train", f"{s}-{i}", duration=d).to_dict()}
                  for s in ["alice", "bob", "charlie", "absent"]
                  for i, d in enumerate([2., 8.])]
    return AudioRecord("sample", "speaker_attributed_asr", (AudioSlot("mixture", mixture),),
                       "[S1][0.10-2.00] 你好\n[S2][1.00-3.00] hello\n[S1][4.00-5.00] 再见\n[S3][6.00-7.00] 好的",
                       language="zh-en", metadata={"sot_output_format": TIMESTAMP_FORMAT,
                       "primary_language": "zh", "partition": "train",
                       "speaker_identities": {"S1": "alice", "S2": "bob", "S3": "charlie"},
                       "source_spans": [mixture.to_dict()], "enrollment_candidates": candidates})


def test_dynamic_draws_are_reproducible_and_diverse(record):
    config = EnrollmentConfig(probability=1)
    draws = [sample_enrollment(record, config, f"42:0:0:{i}") for i in range(80)]
    assert draws == [sample_enrollment(record, config, f"42:0:0:{i}") for i in range(80)]
    assert {len(r.audio_slots) - 1 for r in draws} == {1, 2, 3}
    assert {r.metadata["enrollment_view"]["mode"] for r in draws} == {"all", "targets_only"}
    refs = [s.ref for r in draws for s in r.audio_slots[:-1]]
    assert len({r.duration for r in refs}) > 100
    assert len({r.cut_id for r in refs}) == 8
    assert all(1 <= r.duration <= 5 and r.start >= 0 for r in refs)
    assert any(not e["present"] for r in draws for e in r.metadata["enrollment_view"]["enrollments"])
    assert record.audio_slots[0].name == "mixture"


def test_modes_and_reference_permutations_preserve_complete_timeline(record):
    all_text = render_target(record, ["bob", "alice", "absent"], "all")
    selected = render_target(record, ["bob", "alice", "absent"], "targets_only")
    assert selected == "\n".join(l for l in all_text.splitlines() if l.startswith("[T"))
    assert all_text.startswith("[T2][0.10-2.00]")
    assert "[T3]" not in all_text and "[S1][6.00-7.00]" in all_text
    swapped = render_target(record, ["alice", "bob", "absent"], "all")
    assert swapped == all_text.replace("[T1]", "[TEMP]").replace("[T2]", "[T1]").replace("[TEMP]", "[T2]")
    assert render_target(record, ["absent"], "targets_only") == ""


def test_reference_provenance_and_short_sources(record):
    candidates = record.metadata["enrollment_candidates"]
    leaking = dict(candidates[0], ref=record.audio_slots[0].ref.to_dict())
    short = dict(candidates[0], ref=replace(record.audio_slots[0].ref, cut_id="short", duration=.5).to_dict())
    modified = replace(record, metadata={**record.metadata, "enrollment_candidates": [leaking, short]})
    assert eligible_candidates(modified, EnrollmentConfig()) == {}
    for field, value in [("single_speaker", False), ("partition", "dev")]:
        modified = replace(record, metadata={**record.metadata,
                           "enrollment_candidates": [{**candidates[0], field: value}]})
        with pytest.raises(ValueError):
            eligible_candidates(modified, EnrollmentConfig())


def test_frozen_reference_ignores_draw_seed(record):
    config = EnrollmentConfig(probability=1)
    first = sample_enrollment(record, config, "first")
    fixed = first.metadata["enrollment_view"]
    assert sample_enrollment(record, config, "second", fixed=fixed) == first
    wrong_channel = {"mode": fixed["mode"], "enrollments": [
        {**e, "ref": {**e["ref"], "channel": 1}} for e in fixed["enrollments"]]}
    with pytest.raises(ValueError, match="eligible reference"):
        sample_enrollment(record, config, "second", fixed=wrong_channel)


def test_prompt_empty_native_protocol_and_segment_parsing(record):
    from open_audio_llm.data.qwen3_asr import native_messages
    empty = replace(record, target="", audio_slots=(
        AudioSlot("enrollment_1", AudioRef("voices", "v1", "train", "absent", duration=2)),
        record.audio_slots[0]), metadata={**record.metadata, "sot_output_format": TARGET_FORMAT,
                                         "enrollment_view": {"mode": "targets_only"}})
    assert native_messages(empty, "N/A")[-1]["content"] == "language None<asr_text>"
    assert "3 秒" not in target_prompt(2, "all")
    assert parse_target_segments("", 3) == []
    with pytest.raises(ValueError):
        parse_target_segments("[T4][0.00-1.00] hi", 3)


def tiny_tower():
    from qwen_asr.core.transformers_backend.configuration_qwen3_asr import (
        Qwen3ASRAudioEncoderConfig,
    )
    from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
        Qwen3ASRAudioEncoder,
    )
    cfg = Qwen3ASRAudioEncoderConfig(d_model=32, encoder_layers=2, encoder_attention_heads=4,
                                    encoder_ffn_dim=64, output_dim=16, downsample_hidden_size=8,
                                    n_window=50, conv_chunksize=3)
    cfg._attn_implementation = "sdpa"
    return Qwen3ASRAudioEncoder(cfg)


def test_segment_encoder_keeps_native_path_and_sep_gradients():
    from open_audio_llm.audio.target_sot import (
        attach_separator,
        encode_segments,
        segment_tokens,
    )
    tower = tiny_tower().train()
    tower.gradient_checkpointing_enable({"use_reentrant": False})
    attach_separator(tower)
    features = torch.randn(128, 401, requires_grad=True)
    native = tower(features[:, :201], feature_lens=torch.tensor([201])).last_hidden_state
    torch.testing.assert_close(encode_segments(tower, features, [201]), native, rtol=0, atol=0)
    result = encode_segments(tower, features, [99, 101, 201])
    assert result.shape == (segment_tokens([99, 101, 201]), 16)
    result.square().mean().backward()
    assert torch.count_nonzero(tower.sep_token.weight.grad) > 0
    assert all(torch.count_nonzero(features.grad[:, a:b]) > 0 for a, b in [(0, 99), (99, 200), (200, 401)])
    before = result.detach()
    with torch.no_grad():
        changed = features.clone()
        changed[:, :99] += 4
        assert not torch.equal(encode_segments(tower, changed, [99, 101, 201])[-1], before[-1])


def test_conditional_template_packs_independent_features(monkeypatch):
    from transformers import WhisperFeatureExtractor

    from open_audio_llm.audio.target_sot import extract_segments, segment_tokens
    from open_audio_llm.integrations.ms_swift import register_qwen3_asr as module
    template = object.__new__(module.Qwen3ASRTemplate)
    template.sampling_rate, template._hop_length = 16000, 160
    template._feature_extractor = WhisperFeatureExtractor(feature_size=128, chunk_length=1)
    wavs = {"e1": np.ones(16001, dtype=np.float32), "e2": np.zeros(33001, dtype=np.float32),
            "mix": np.zeros(16000 * 31 + 1, dtype=np.float32)}
    monkeypatch.setattr(module, "load_audio", lambda p, **kw: wavs[p])
    monkeypatch.setattr(module.Template, "_encode", lambda self, inp: {"tags": self.replace_tag("audio", 0, inp)})
    from swift.template.template_inputs import StdTemplateInputs

    inputs = StdTemplateInputs.from_dict({"audios": ["mix"], "messages": [
        {"role": "user", "content": "<audio>"}], "chat_template_kwargs": {"enroll_wavs": ["e1", "e2"]}})
    output = template._encode(inputs)
    features, lengths = extract_segments(template.feature_extractor, list(wavs.values()))
    torch.testing.assert_close(output["input_features"], features, rtol=0, atol=0)
    assert output["tags"].count("<|audio_pad|>") == segment_tokens(lengths)
    assert output["enrollment_lengths"].tolist() == [[101, 207, 0]]
    assert lengths[-1] == 3101
    monkeypatch.setattr(module.Template, "_data_collator", lambda *args, **kw: {})
    ordinary = {"input_features": torch.zeros(1, 128, 100), "feature_attention_mask": torch.ones(1, 100)}
    batch = template._data_collator([output, ordinary])
    assert batch["enrollment_lengths"].tolist() == [[101, 207, 0], [0, 0, 0]]


@pytest.mark.parametrize("cached", [False, True])
def test_catalog_online_crops_budget_and_frozen_evaluation(record, tmp_path, cached):
    import json
    from io import BytesIO

    import soundfile as sf
    from audio_data_contract import ArtifactRef, DatasetSpec, Split, write_records

    from open_audio_llm.data.catalog_dataset import CatalogSwiftDataset
    from open_audio_llm.data.catalog_sampler import CatalogBatchSampler
    from open_audio_llm.data.prepare_target_sot import freeze_enrollment

    indexes = {"meeting": [record.audio_slots[0].ref], "voices": [
        AudioRef.from_dict(c["ref"]) for c in record.metadata["enrollment_candidates"]]}
    specs = []
    for dataset_id, refs in indexes.items():
        with (tmp_path / f"{dataset_id}.jsonl").open("w") as sink:
            for ref in refs:
                audio = np.random.default_rng(7).normal(0, .01, int(ref.duration * 16000)).astype(np.float32)
                sf.write(tmp_path / f"{ref.cut_id}.wav", audio, 16000)
                sink.write(json.dumps({"cut_id": ref.cut_id, "root_alias": "data", "relative_path": f"{ref.cut_id}.wav",
                                           "sample_rate": 16000, "channels": 1, "num_frames": len(audio), "duration": ref.duration}) + "\n")
        specs.append(DatasetSpec(dataset_id=dataset_id, version="v1", languages=("zh", "en"),
            tasks=("speaker_attributed_asr",), artifacts=(
                ArtifactRef("audio", "audio-index", "data", f"{dataset_id}.jsonl"),
                ArtifactRef("records", "audio-records", "data", "records.jsonl")),
            splits={"train": Split({"records": ("records",), "audio_index": ("audio",)})}))
    write_records([record], tmp_path / "records.jsonl")
    (tmp_path / "catalog.jsonl").write_text("\n".join(json.dumps(s.to_dict()) for s in specs))
    (tmp_path / "roots.json").write_text(json.dumps({"data": str(tmp_path)}))
    source = {"dataset_id": "meeting", "version": "v1", "split": "train", "enrollment": {"probability": 1}}
    config = dict(catalog=str(tmp_path / "catalog.jsonl"), roots=str(tmp_path / "roots.json"),
                  train=[source], validation=[source], augmentation={"speed_prob": 1},
                  **({"metadata_cache": str(tmp_path / "cache")} if cached else {}))
    dataset = CatalogSwiftDataset(config, message_format="qwen3_asr", collect_metrics=True)
    sampler = CatalogBatchSampler(dataset)
    assert sampler.durations[0] >= 25
    one, two = dataset[(0, 0, 0)], dataset[(0, 0, 1)]
    assert one["enrollment_view"] != two["enrollment_view"]
    assert one["audios"] == two["audios"]
    for sample in [one, two]:
        for wav, enrollment in zip(sample["chat_template_kwargs"]["enroll_wavs"], sample["enrollment_view"]["enrollments"]):
            assert abs(sf.info(BytesIO(wav)).duration - enrollment["ref"]["duration"]) < 1 / 16000
        assert sample["_performance"]["enrollment_count"] >= 1
    with pytest.raises(ValueError, match="frozen"):
        CatalogSwiftDataset(config, training=False, message_format="qwen3_asr")[0]
    frozen = freeze_enrollment(record, 42, seconds=2)
    write_records([frozen], tmp_path / "records.jsonl")
    evaluation = CatalogSwiftDataset(config, training=False, message_format="qwen3_asr")
    assert evaluation[(0, 0, 0)]["enrollment_view"] == evaluation[(2, 0, 11)]["enrollment_view"]
    source["enrollment"] = {}
    assert CatalogSwiftDataset(config, training=False, message_format="qwen3_asr")[0]["enrollment_view"]


def test_full_forward_and_checkpoint_round_trip(tmp_path):
    from qwen_asr.core.transformers_backend.configuration_qwen3_asr import (
        Qwen3ASRConfig,
    )
    from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
        Qwen3ASRForConditionalGeneration,
    )

    from open_audio_llm.audio.target_sot import enable_target_audio, segment_tokens

    torch.manual_seed(17)
    config = Qwen3ASRConfig(thinker_config={
        "audio_config": tiny_tower().config.to_dict(),
        "text_config": {"vocab_size": 64, "hidden_size": 16, "intermediate_size": 32,
                         "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 2,
                         "head_dim": 8, "rope_scaling": {"rope_type": "default", "mrope_section": [1, 1, 2]}},
        "audio_token_id": 60, "audio_start_token_id": 61})
    model = Qwen3ASRForConditionalGeneration(config).eval()
    model.thinker.audio_tower.config._attn_implementation = "sdpa"
    enable_target_audio(model)
    lengths = [101, 117]
    tokens = segment_tokens(lengths)
    features = torch.randn(1, 128, sum(lengths))
    inputs = {"input_ids": torch.tensor([[1] + [60] * tokens + [2, 3]]),
                  "attention_mask": torch.ones(1, tokens + 3, dtype=torch.long),
                  "input_features": features, "feature_attention_mask": torch.ones(1, sum(lengths), dtype=torch.long),
                  "enrollment_lengths": torch.tensor([[101, 0, 0]]), "use_cache": False}
    # The public top-level forward delegates to thinker through the existing Swift patch.
    output = model.thinker(**inputs).logits
    output.square().mean().backward()
    assert model.thinker.audio_tower.sep_token.weight.grad.abs().sum() > 0
    model.save_pretrained(tmp_path)
    reloaded = Qwen3ASRForConditionalGeneration.from_pretrained(tmp_path, attn_implementation="sdpa").eval()
    enable_target_audio(reloaded, tmp_path)
    torch.testing.assert_close(reloaded.thinker(**inputs).logits, output, rtol=0, atol=0)
    model.zero_grad(set_to_none=True)
    # An ordinary-only batch creates a zero (not missing) SEP gradient for DDP.
    model.thinker(input_ids=torch.tensor([[1, 2, 3]]), use_cache=False).logits.sum().backward()
    assert model.thinker.audio_tower.sep_token.weight.grad is not None


def test_vllm_request_uses_same_segment_embeddings(monkeypatch):
    import sys

    from transformers import WhisperFeatureExtractor

    from open_audio_llm.audio.target_sot import (
        attach_separator,
        encode_segments,
        extract_segments,
    )
    from open_audio_llm.integrations.vllm import target_sot as module

    calls = []
    engine = SimpleNamespace(generate=lambda prompts, params, **kwargs: (
        calls.append(prompts) or [SimpleNamespace(outputs=[SimpleNamespace(text="", finish_reason="stop")])]))
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=lambda **kwargs: kwargs))
    wavs = {"ref": np.ones(16000, dtype=np.float32), "mixture": np.zeros(32000, dtype=np.float32)}
    monkeypatch.setattr(module, "read_waveform", lambda p: wavs[p])
    runner = object.__new__(module.TargetSOTVLLM)
    runner.tower = tiny_tower().eval()
    attach_separator(runner.tower)
    extractor = WhisperFeatureExtractor(feature_size=128, chunk_length=1)
    tokenizer = SimpleNamespace(
        apply_chat_template=lambda messages, **kwargs: messages[0]["content"] + messages[1]["content"],
        encode=lambda text: [1] * 20)
    runner.processor = SimpleNamespace(feature_extractor=extractor, tokenizer=tokenizer)
    runner.engine, runner.backend, runner.max_model_len = engine, "vllm", 16384
    result = runner.transcribe("mixture", ["ref"], "targets_only")
    assert result == {"prediction": "", "finish_reason": "stop", "backend": "vllm", "duration": 2}
    features, lengths = extract_segments(extractor, list(wavs.values()))
    expected = encode_segments(runner.tower, features[0], lengths)
    request = calls[0][0]
    torch.testing.assert_close(request["multi_modal_data"]["audio"]["audio_embeds"], expected, rtol=0, atol=0)
    assert request["prompt"].startswith(target_prompt(1, "targets_only"))
    assert request["prompt"].count("<|audio_pad|>") == 1
    runner.max_model_len = 25
    with pytest.raises(ValueError, match="no audio was truncated"):
        runner.transcribe("mixture", ["ref"])
    assert len(calls) == 1
