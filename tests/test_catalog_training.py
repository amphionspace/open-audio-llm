import json
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch
from audio_data_contract import (
    ArtifactRef,
    AudioRecord,
    AudioRef,
    AudioSlot,
    DatasetSpec,
    write_records,
)
from lhotse import (
    CutSet,
    MonoCut,
    Recording,
    RecordingSet,
    SupervisionSegment,
    SupervisionSet,
)

from open_audio_llm.data.augment import (
    AugmentConfig,
    augment_features,
    augment_waveform,
)
from open_audio_llm.data.catalog_dataset import CatalogSwiftDataset, read_data_config


@pytest.mark.parametrize("cached", [False, True])
def test_native_ts_reads_registered_audio_index_and_budgets_concat(catalog_config, tmp_path, cached):
    import gzip
    from dataclasses import replace

    from open_audio_llm.data.catalog_sampler import CatalogBatchSampler

    row = {"cut_id": "speech", "root_alias": "data", "relative_path": "speech.wav",
           "sample_rate": 16000, "channels": 1, "num_frames": 16000, "duration": 1.0}
    with gzip.open(tmp_path / "index.jsonl.gz", "wt") as stream:
        stream.write(json.dumps(row) + "\n")
    record = AudioRecord("ts", "ts_asr", tuple(
        AudioSlot(name, AudioRef("ts", "1", "train", "speech", start=0.25))
        for name in ("enrollment", "mixture")
    ), "hello", language="en")
    write_records([record, replace(record, id="negative", target="")], tmp_path / "ts.jsonl.gz")
    spec = DatasetSpec(
        dataset_id="ts", version="1", languages=("en",), tasks=("ts_asr",),
        artifacts=(ArtifactRef("records", "audio-records", "data", "ts.jsonl.gz"),
                   ArtifactRef("index", "audio-index", "data", "index.jsonl.gz")),
        splits={"train": {"records_artifacts": ["records"], "audio_index_artifact": "index"}},
    )
    with Path(catalog_config["catalog"]).open("a") as stream:
        stream.write("\n" + json.dumps(spec.to_dict()))
    catalog_config.update(
        train=[{"dataset_id": "ts", "version": "1", "split": "train", "min_duration": 0.5}],
        augmentation={}, batching={"max_duration": 7},
    )
    if cached:
        catalog_config["metadata_cache"] = str(tmp_path / "cache")
    dataset = CatalogSwiftDataset(catalog_config, message_format="qwen3_asr", collect_metrics=True)
    assert len(dataset) == 2  # Missing inline durations must not filter out TS records.
    sampler = CatalogBatchSampler(dataset)
    assert list(sampler.slots) == [1, 1]
    assert sampler.durations == pytest.approx([6.75, 6.75], abs=0.001)
    sample = dataset[0]
    assert sample["audio_slot_count"] == 1 and len(sample["audios"]) == 1
    assert sample["duration"] == 6.75
    assert sample["_performance"]["audio_seconds"] == 6.75
    audio, _ = sf.read(BytesIO(sample["audios"][0]))
    assert np.count_nonzero(audio[3 * 16000:6 * 16000]) == 0
    assert dataset[1]["solution"] == "language None<asr_text>"
    assert len(dataset.resolver._audio_indexes) == 1


@pytest.fixture
def catalog_config(tmp_path):
    sr = 16000
    samples = np.sin(np.arange(sr, dtype=np.float32) * 0.07) * 0.2
    sf.write(tmp_path / "speech.wav", samples, sr, subtype="FLOAT")
    recording = Recording.from_file(tmp_path / "speech.wav", recording_id="speech")
    segments = [
        SupervisionSegment(
            id=f"s{i}",
            recording_id="speech",
            start=i * 0.3,
            duration=0.25,
            channel=0,
            text="hello",
            language="en",
        )
        for i in range(2)
    ]
    cuts = [
        MonoCut(
            id=s.id,
            start=s.start,
            duration=s.duration,
            channel=0,
            recording=recording,
            supervisions=[s.with_offset(-s.start)],
        )
        for s in segments
    ]
    for i, cut in enumerate(cuts):
        CutSet.from_cuts([cut]).to_file(tmp_path / f"cuts{i}.jsonl.gz")
    RecordingSet.from_recordings([recording]).to_file(tmp_path / "recordings.jsonl.gz")
    SupervisionSet.from_segments(segments).to_file(tmp_path / "supervisions.jsonl.gz")
    specs = [
        DatasetSpec(
            dataset_id="speech",
            version="1",
            languages=("en",),
            tasks=("asr",),
            artifacts=tuple(
                ArtifactRef(f"c{i}", "lhotse-cuts", "data", f"cuts{i}.jsonl.gz")
                for i in range(2)
            )
            + (
                ArtifactRef("r", "lhotse-recordings", "data", "recordings.jsonl.gz"),
                ArtifactRef(
                    "s", "lhotse-supervisions", "data", "supervisions.jsonl.gz"
                ),
            ),
            splits={
                "train": {"cuts_artifacts": ["c0", "c1"]},
                "dev": {"recordings_artifacts": ["r"], "supervisions_artifacts": ["s"]},
            },
        )
    ]
    for name in ("noise", "rir"):
        audio = np.random.default_rng(42).normal(0, 0.1, sr).astype(np.float32)
        if name == "rir":
            audio = np.zeros(100, dtype=np.float32)
            audio[0], audio[50] = 1, 0.5
        sf.write(tmp_path / f"{name}.wav", audio, sr, subtype="FLOAT")
        rec = Recording.from_file(tmp_path / f"{name}.wav", recording_id=name)
        RecordingSet.from_recordings([rec]).to_file(tmp_path / f"{name}.jsonl.gz")
        specs.append(
            DatasetSpec(
                dataset_id=name,
                version="1",
                languages=("und",),
                tasks=("augmentation",),
                artifacts=(
                    ArtifactRef("r", "lhotse-recordings", "data", f"{name}.jsonl.gz"),
                ),
                splits={"train": {"recordings_artifact": "r"}},
            )
        )
    (tmp_path / "catalog.jsonl").write_text(
        "\n".join(json.dumps(s.to_dict()) for s in specs)
    )
    (tmp_path / "roots.json").write_text(json.dumps({"data": str(tmp_path)}))
    return {
        "catalog": str(tmp_path / "catalog.jsonl"),
        "roots": str(tmp_path / "roots.json"),
        "seed": 42,
        "sampling_rate": sr,
        "train": [{"dataset_id": "speech", "version": "1", "split": "train"}],
        "validation": [{"dataset_id": "speech", "version": "1", "split": "dev"}],
        "noise_sources": [{"dataset_id": "noise", "version": "1", "split": "train"}],
        "rir_sources": [{"dataset_id": "rir", "version": "1", "split": "train"}],
        "augmentation": {
            "speed_prob": 1,
            "speed_factors": [2.0],
            "noise_prob": 1,
            "rir_prob": 1,
        },
    }


def test_catalog_audio_is_lazy_and_never_writes_intermediates(
    catalog_config, tmp_path, monkeypatch
):
    before = set(tmp_path.rglob("*"))
    with monkeypatch.context() as patch:
        patch.setattr(
            Recording, "load_audio", lambda *a, **kw: pytest.fail("eager audio decode")
        )
        dataset = CatalogSwiftDataset(catalog_config)
    sample = dataset[0]
    audio, sr = sf.read(BytesIO(sample["audios"][0]))
    assert sr == 16000
    assert len(audio) == 2000  # 0.25 seconds at speed 2.0.
    assert len(dataset) == 2
    assert sample["messages"][-1]["role"] == "assistant"
    assert "hello" in sample["solution"]
    assert set(tmp_path.rglob("*")) == before


def test_epoch_changes_training_but_not_validation(catalog_config):
    train = CatalogSwiftDataset(catalog_config)
    first = train[0]
    assert first == train[0]
    train.set_epoch(1)
    assert first["audios"] != train[0]["audios"]
    train.set_epoch(0)
    assert first == train[0]
    validation = CatalogSwiftDataset(catalog_config, training=False)
    fixed = validation[1]
    validation.set_epoch(9)
    assert fixed == validation[1]
    reference, _ = sf.read(BytesIO(fixed["audios"][0]))
    assert len(reference) == 4000
    original, _ = sf.read(str(Path(catalog_config["roots"]).parent / "speech.wav"))
    np.testing.assert_allclose(reference, original[4800:8800], atol=1e-6)


def test_persistent_workers_receive_epoch(catalog_config):
    dataset = CatalogSwiftDataset(catalog_config)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=None, num_workers=1, persistent_workers=True
    )
    try:
        first_epoch = list(loader)
        first = first_epoch[0]["audios"]
        np.testing.assert_array_equal(sf.read(BytesIO(first[0]))[0],
                                      sf.read(BytesIO(dataset[0]["audios"][0]))[0])
        dataset.set_epoch(2)
        second_epoch = list(loader)
        second = sf.read(BytesIO(second_epoch[0]["audios"][0]))[0]
        np.testing.assert_array_equal(second, sf.read(BytesIO(dataset[0]["audios"][0]))[0])
        assert not np.array_equal(second, sf.read(BytesIO(first[0]))[0])
    finally:
        loader._iterator._shutdown_workers()


def test_grpo_reuses_group_audio_and_keeps_reference_out_of_prompt(catalog_config):
    dataset = CatalogSwiftDataset(catalog_config, grpo=True)
    group = [dataset[0] for _ in range(3)]
    assert group[0] == group[1] == group[2]
    assert all(m["role"] != "assistant" for m in group[0]["messages"])
    assert "hello" in group[0]["solution"]


def test_spec_augment_preserves_lengths_and_padding():
    features = torch.ones(2, 8, 20)
    lengths = torch.tensor([10, 20])
    policy = {
        "seed": "sample:epoch",
        "config": {
            "spec_aug_prob": 1,
            "time_mask_width": 10,
            "frequency_mask_width": 4,
        },
    }
    masked = augment_features(features, lengths, policy)
    assert torch.count_nonzero(masked == 0) > 0
    torch.testing.assert_close(masked, augment_features(features, lengths, policy))
    torch.testing.assert_close(masked[0, :, 10:], features[0, :, 10:])
    assert lengths.tolist() == [10, 20]
    assert torch.all(features == 1)


def test_noise_snr_and_rir_are_real_transforms():
    audio = np.ones(1000, dtype=np.float32)
    noise = np.tile(np.array([-1, 1], dtype=np.float32), 500)
    import random

    mixed = augment_waveform(
        audio,
        16000,
        random.Random(0),
        AugmentConfig(noise_prob=1, noise_snr_db=(10, 10)),
        noise_loader=lambda rng, sr: noise,
    )
    assert np.mean((mixed - audio) ** 2) == pytest.approx(0.1, rel=1e-5)
    reverbed = augment_waveform(
        audio,
        16000,
        random.Random(0),
        AugmentConfig(rir_prob=1),
        rir_loader=lambda rng, sr: np.array([1, 0.5], dtype=np.float32),
    )
    assert reverbed.shape == audio.shape
    assert not np.array_equal(reverbed, audio)


def test_pipeline_encodes_only_at_access(catalog_config):
    from open_audio_llm.integrations.ms_swift.train import CatalogTrainingMixin

    calls = []
    pipeline = CatalogTrainingMixin()
    pipeline.args = SimpleNamespace(_catalog_config=catalog_config)
    pipeline.template = SimpleNamespace(
        encode=lambda row: calls.append(row) or {"encoded": True}
    )
    train, validation = pipeline._prepare_dataset()
    assert calls == []
    assert train[0] == {"encoded": True}
    assert len(calls) == 1
    assert validation is not None


def test_config_uses_environment_catalog_and_roots(
    catalog_config, tmp_path, monkeypatch
):
    monkeypatch.setenv("AUDIO_DATA_CATALOG", catalog_config["catalog"])
    monkeypatch.setenv("AUDIO_DATA_ROOTS_FILE", catalog_config["roots"])
    path = tmp_path / "training.json"
    path.write_text(json.dumps({"train": catalog_config["train"]}))
    assert read_data_config(path)["catalog"] == catalog_config["catalog"]


def test_yaml_config_relative_paths_and_samples(catalog_config, tmp_path, monkeypatch):
    import yaml

    monkeypatch.delenv("AUDIO_DATA_CATALOG", raising=False)
    monkeypatch.delenv("AUDIO_DATA_ROOTS_FILE", raising=False)
    config = dict(catalog_config, catalog="catalog.jsonl", roots="roots.json")
    config["train"][0]["samples"] = 1
    path = tmp_path / "train.yaml"
    path.write_text(yaml.safe_dump(config))
    loaded = read_data_config(path)
    assert loaded["roots"] == str(tmp_path / "roots.json")
    assert len(CatalogSwiftDataset(loaded)) == 1


def test_catalog_mux_rejects_ambiguous_weights(catalog_config):
    catalog_config["train"][0].update(weight=1, reps=2)
    with pytest.raises(ValueError, match="mutually exclusive"):
        CatalogSwiftDataset(catalog_config)


def test_mux_worker_prefetch_resume_preserves_audio(catalog_config):
    from open_audio_llm.data.catalog_sampler import CatalogBatchSampler

    catalog_config["train"][0].update(samples=2, reps=3)
    dataset = CatalogSwiftDataset(catalog_config)
    sampler = CatalogBatchSampler(dataset)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=list,
        num_workers=2,
        persistent_workers=True,
    )
    try:
        iterator = iter(loader)
        next(iterator)
        sampler.mark_consumed()
        state = sampler.state_dict()
        expected = [sample[0]["audios"] for sample in iterator]
        restored = CatalogBatchSampler(dataset)
        restored.load_state_dict(state)
        actual = [dataset[batch[0]]["audios"] for batch in restored]
        assert expected == actual
        # Each repetition has a different deterministic augmentation seed.
        assert len({audios[0] for audios in actual}) == len(actual)
    finally:
        if loader._iterator is not None:
            loader._iterator._shutdown_workers()


def test_catalog_loader_does_not_advance_model_rng(catalog_config):
    from open_audio_llm.integrations.ms_swift.catalog_loader import (
        install_catalog_loader,
    )

    trainer = SimpleNamespace(
        args=SimpleNamespace(
            max_steps=10,
            deepspeed=None,
            process_index=0,
            world_size=1,
            train_dataloader_shuffle=True,
            gradient_accumulation_steps=1,
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
        ),
        template=SimpleNamespace(sequence_parallel_size=1),
        train_dataset=CatalogSwiftDataset(catalog_config),
        _train_batch_size=1,
        data_collator=list,
        accelerator=SimpleNamespace(device=torch.device("cpu")),
        add_callback=lambda callback: None,
        set_initial_training_values=lambda *args: (1, 2, 2, 2, False, 2, 10),
    )
    install_catalog_loader(trainer)
    state = torch.get_rng_state()
    batch = next(iter(trainer.get_train_dataloader()))
    assert len(batch) == 1
    assert torch.equal(state, torch.get_rng_state())


def test_portable_records_artifact_preserves_segment_refs(catalog_config, tmp_path):
    record = AudioRecord(
        "segment",
        "asr",
        (
            AudioSlot(
                "primary",
                AudioRef(
                    "speech",
                    "1",
                    "train",
                    "s0",
                    start=0.0625,
                    duration=0.125,
                ),
            ),
        ),
        "hello",
        language="en",
    )
    write_records([record], tmp_path / "records.jsonl")
    spec = DatasetSpec(
        dataset_id="records",
        version="1",
        languages=("en",),
        tasks=("asr",),
        artifacts=(ArtifactRef("records", "audio-record", "data", "records.jsonl"),),
        splits={"train": {"records_artifact": "records"}},
    )
    with Path(catalog_config["catalog"]).open("a") as stream:
        stream.write("\n" + json.dumps(spec.to_dict()))
    catalog_config["train"].append(
        {"dataset_id": "records", "version": "1", "split": "train"}
    )
    catalog_config["augmentation"] = {}
    dataset = CatalogSwiftDataset(catalog_config)
    sample = dataset[2]
    waveform, _ = sf.read(BytesIO(sample["audios"][0]))
    original, _ = sf.read(tmp_path / "speech.wav")
    np.testing.assert_allclose(waveform, original[1000:3000], atol=1e-6)
    assert sample["duration"] == 0.125


def test_catalog_to_real_template_and_model_backward(catalog_config):
    from swift.template import TemplateMeta
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, WhisperFeatureExtractor

    from open_audio_llm import AudioLLMConfig, AudioLLMForConditionalGeneration
    from open_audio_llm.integrations.ms_swift.template import AudioLLMTemplate

    tokens = [
        "[UNK]",
        "[PAD]",
        "<|im_start|>",
        "<|im_end|>",
        "<start_text>",
        "<end_text>",
        "<start_speech>",
        "<speech>",
        "<end_speech>",
    ]
    vocab = {token: i for i, token in enumerate(tokens)}
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="<|im_end|>",
        additional_special_tokens=tokens[2:],
    )
    config = AudioLLMConfig(
        audio_tower_config={"type": "identity", "input_dim": 128, "output_dim": 128},
        connector_config={
            "type": "mlp_downsample",
            "input_dim": 128,
            "output_dim": 16,
            "downsample_rate": 1,
        },
        text_config={
            "model_type": "gpt2",
            "n_embd": 16,
            "n_layer": 1,
            "n_head": 1,
            "vocab_size": len(tokens),
        },
        default_speech_token_id=vocab["<speech>"],
        start_text_token_id=vocab["<start_text>"],
        end_text_token_id=vocab["<end_text>"],
        start_speech_token_id=vocab["<start_speech>"],
        end_speech_token_id=vocab["<end_speech>"],
    )
    tokenizer.model_info = SimpleNamespace(
        config=config, task_type="causal_lm", max_model_len=512
    )
    tokenizer.model_meta = SimpleNamespace(is_multimodal=True)
    tokenizer.feature_extractor = WhisperFeatureExtractor(
        feature_size=128, chunk_length=1
    )
    template = AudioLLMTemplate(
        tokenizer,
        TemplateMeta(
            "catalog-test",
            prefix=[],
            prompt=["{{QUERY}}"],
            chat_sep=[],
            suffix=["<|im_end|>"],
        ),
    )
    template.set_mode("train")
    dataset = CatalogSwiftDataset(catalog_config, encode=template.encode)
    batch = template.data_collator([dataset[0], dataset[1]])
    model = AudioLLMForConditionalGeneration(config)
    output = model(**batch)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.connector.parameters()
    )


def test_native_qwen_pipeline_renders_before_encoding(catalog_config):
    pytest.importorskip("qwen_asr")
    from open_audio_llm.integrations.ms_swift.train import CatalogTrainingMixin

    pipeline = CatalogTrainingMixin()
    pipeline.args = SimpleNamespace(
        _catalog_config=catalog_config, model_type="amphion_asr_1.7b"
    )
    pipeline.template = SimpleNamespace(encode=lambda row: row)
    train, validation = pipeline._prepare_dataset()
    for dataset in (train, validation):
        row = dataset[0]
        assert row["messages"][1] == {"role": "user", "content": "<audio>"}
        assert row["messages"][-1]["content"] == "language English<asr_text>hello"
        assert row["solution"] == row["messages"][-1]["content"]


@pytest.mark.parametrize("cached,incomplete", [(False, False), (True, False), (True, True)])
def test_hotword_eval_uses_dataset_audio_and_checks_prediction_count(
    catalog_config, tmp_path, monkeypatch, cached, incomplete
):
    qwen_asr = pytest.importorskip("qwen_asr")
    from open_audio_llm.eval.qwen3_asr import main

    config = dict(catalog_config, evaluation=catalog_config["train"])
    if cached:
        config["metadata_cache"] = str(tmp_path / "cache")
    path = tmp_path / "eval.json"
    path.write_text(json.dumps(config))
    output = tmp_path / "evaluation"
    model = torch.nn.Module()
    model.generation_config = SimpleNamespace(use_cache=False, do_sample=True)
    model.thinker = torch.nn.Module()
    model.thinker.generation_config = SimpleNamespace(use_cache=False, do_sample=True)
    model.thinker.model = SimpleNamespace(config=SimpleNamespace(use_cache=False))
    original, rate = sf.read(tmp_path / "speech.wav", dtype="float32")
    expected = [original[:4000], original[4800:8800]]
    calls = []

    def transcribe(audio, context):
        assert len(audio) == len(context) == 2
        for waveform, sampling_rate in audio:
            assert sampling_rate == rate
            assert waveform.dtype == np.float32
            assert any(np.allclose(waveform, segment) for segment in expected)
        assert not model.training and not model.thinker.training
        assert model.thinker.model.config.use_cache
        for module in (model, model.thinker):
            assert module.generation_config.use_cache
            assert not module.generation_config.do_sample
        calls.append(context)
        return [SimpleNamespace(text="hello", language="English")] * (1 if incomplete else 2)

    monkeypatch.setattr(qwen_asr.Qwen3ASRModel, "from_pretrained", lambda *a, **kw: SimpleNamespace(
        model=model, transcribe=transcribe,
    ))
    monkeypatch.setattr("sys.argv", [
        "qwen3_asr", "--model", "unused", "--data_config", str(path),
        "--output_dir", str(output), "--samples_per_source", "2", "--batch_size", "2",
    ])
    if incomplete:
        with pytest.raises(RuntimeError, match="incomplete batch"):
            main()
        assert not (output / "summary.json").exists()
    else:
        main()
        summary = json.loads((output / "summary.json").read_text())
        assert len(calls) == 2
        for condition in ("no_hotwords", "hotwords"):
            assert summary[f"speech/{condition}"]["utterances"] == 2
            assert summary[f"speech/{condition}"]["error_rate"] == 0


def test_performance_logging_preserves_online_sample(catalog_config):
    plain = CatalogSwiftDataset(catalog_config)
    measured = CatalogSwiftDataset(catalog_config, collect_metrics=True)
    expected = plain[(0, 0, 5)]
    actual = measured[(0, 0, 5)]
    metrics = actual.pop('_performance')
    assert actual == expected
    assert metrics['audio_seconds'] == sf.info(BytesIO(expected['audios'][0])).duration
    assert metrics['prepare_s'] >= metrics['decode_s']
    assert metrics['prepare_cpu_s'] >= 0


def test_indexed_metadata_preserves_audio_sampling_and_resume(catalog_config, tmp_path, monkeypatch):
    import pickle

    from open_audio_llm.data.catalog_sampler import CatalogBatchSampler

    original = CatalogSwiftDataset(catalog_config, message_format='qwen3_asr')
    old = CatalogBatchSampler(original)
    old_batches = list(old)
    old.mark_consumed()
    state = old.state_dict()
    catalog_config = {**catalog_config, 'metadata_cache': str(tmp_path / 'cache')}
    indexed = CatalogSwiftDataset(catalog_config, message_format='qwen3_asr')
    new = CatalogBatchSampler(indexed)
    assert new.signature == old.signature
    assert list(new) == old_batches
    new.load_state_dict(state)
    assert list(new) == old_batches[1:]
    np.testing.assert_array_equal(sf.read(BytesIO(indexed[0]['audios'][0]))[0],
                                  sf.read(BytesIO(original[0]['audios'][0]))[0])
    assert pickle.loads(pickle.dumps(indexed.records))[1].record == original.records[1].record
    monkeypatch.setattr(CatalogSwiftDataset, '_source_records', lambda *a: pytest.fail('rebuilt warm index'))
    warm = CatalogSwiftDataset(catalog_config, message_format='qwen3_asr')
    np.testing.assert_array_equal(sf.read(BytesIO(warm[0]['audios'][0]))[0],
                                  sf.read(BytesIO(original[0]['audios'][0]))[0])
    assert not warm._audio_cuts
    # Changing replay quotas must not rescan the audio manifests.
    catalog_config['train'] = [{**source, 'weight': 3} for source in catalog_config['train']]
    reweighted = CatalogSwiftDataset(catalog_config, message_format='qwen3_asr')
    assert reweighted.records[0].record == indexed.records[0].record
