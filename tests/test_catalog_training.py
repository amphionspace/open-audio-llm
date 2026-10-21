import json
import random
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
    Split,
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
from open_audio_llm.data.catalog_dataset import (
    CatalogSwiftDataset,
    choose_from_groups,
    read_data_config,
    source_weights,
)


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("second_clean", [False, None, "true", 1])
def test_clean_training_uses_only_explicit_pass_and_invalidates_cache(catalog_config, tmp_path, cached, second_clean):
    for index, passed in enumerate([True, second_clean]):
        path = tmp_path / f"cuts{index}.jsonl.gz"
        cut = next(iter(CutSet.from_file(path)))
        cut.supervisions[0].text = "clean transcript" if index == 0 else "unchecked transcript"
        cut.supervisions[0].custom = {"clean": {"pass": passed}} if passed is not None else {}
        CutSet.from_cuts([cut]).to_file(path)
    if cached:
        catalog_config["metadata_cache"] = str(tmp_path / "clean-cache")
    unfiltered = CatalogSwiftDataset(
        {**catalog_config, "validation": [{**catalog_config["train"][0], "require_clean_pass": False}]},
        training=False,
    )
    assert len(unfiltered) == 2
    catalog_config["train"][0]["require_clean_pass"] = True
    filtered = CatalogSwiftDataset(catalog_config)
    assert len(filtered) == 1
    assert filtered.records[0].record.target == "clean transcript"


@pytest.mark.parametrize("cached", [False, True])
def test_clean_labels_outside_recording_are_removed_before_trimming(catalog_config, tmp_path, cached):
    segments = list(SupervisionSet.from_file(tmp_path / "supervisions.jsonl.gz"))
    for segment in segments:
        segment.custom = {"clean": {"pass": True}}
    segments[1].start, segments[1].duration = .9, .25  # The recording is only 1 second.
    SupervisionSet.from_segments(segments).to_file(tmp_path / "supervisions.jsonl.gz")
    catalog_config["train"][0].update(split="dev", require_clean_pass=True)
    if cached:
        catalog_config["metadata_cache"] = str(tmp_path / "clean-cache")
    dataset = CatalogSwiftDataset(catalog_config)
    assert len(dataset) == 1 and dataset.records[0].record.audio_slots[0].ref.cut_id == "s0"


@pytest.mark.parametrize("cached", [False, True])
def test_native_ts_reads_registered_audio_index_and_budgets_concat(catalog_config, tmp_path, cached, monkeypatch):
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
    ), "hello", language="en", metadata={"clean": {"pass": True}})
    records = [record, replace(record, id="negative", target="")]
    records += [replace(record, id=f"unchecked-{i}",
                        metadata={"clean": {"pass": value}} if value is not None else {})
                for i, value in enumerate((False, None, "true", 1))]
    write_records(records, tmp_path / "ts.jsonl.gz")
    spec = DatasetSpec(
        dataset_id="ts", version="1", languages=("en",), tasks=("ts_asr",),
        artifacts=(ArtifactRef("records", "audio-records", "data", "ts.jsonl.gz"),
                   ArtifactRef("index", "audio-index", "data", "index.jsonl.gz")),
        splits={"train": Split({"records": ("records",), "audio_index": ("index",)})},
    )
    with Path(catalog_config["catalog"]).open("a") as stream:
        stream.write("\n" + json.dumps(spec.to_dict()))
    catalog_config.update(
        train=[{"dataset_id": "ts", "version": "1", "split": "train", "min_duration": 0.5,
                "require_clean_pass": True}],
        augmentation={}, batching={"max_duration": 7},
    )
    if cached:
        catalog_config["metadata_cache"] = str(tmp_path / "cache")
        # Reproduce the old portable cache: flag=true was recorded in its key
        # while unapproved records were still indexed.
        from open_audio_llm.data import catalog_cache

        identity = catalog_cache.source_identity
        source_records = CatalogSwiftDataset._source_records

        def legacy_identity(*args):
            result = identity(*args)
            result.pop("clean_record_filter")
            return result

        with monkeypatch.context() as patch:
            patch.setattr(catalog_cache, "source_identity", legacy_identity)
            patch.setattr(CatalogSwiftDataset, "_source_records",
                          lambda self, source: source_records(self, {**source, "require_clean_pass": False}))
            legacy = CatalogSwiftDataset(catalog_config, message_format="qwen3_asr")
            assert len(legacy) == 6
    dataset = CatalogSwiftDataset(catalog_config, message_format="qwen3_asr", collect_metrics=True)
    assert len(dataset) == 2  # Missing inline durations must not filter out TS records.
    if cached:
        assert dataset.records.sources[0].path != legacy.records.sources[0].path
    sampler = CatalogBatchSampler(dataset)
    assert list(sampler.slots) == [1, 1]
    assert sampler.durations == pytest.approx([1.5, 1.5], abs=0.001)
    sample = dataset[0]
    assert sample["audio_slot_count"] == 1 and len(sample["audios"]) == 1
    assert sample["duration"] == pytest.approx(1.5)
    assert sample["_performance"]["audio_seconds"] == pytest.approx(1.5)
    enroll, _ = sf.read(BytesIO(sample["audios"][0]))
    mix, _ = sf.read(BytesIO(sample["mix_wav"]))
    assert len(enroll) == int(0.75 * 16000) and len(mix) == int(0.75 * 16000)
    assert dataset[1]["solution"] == "language None<asr_text>"
    assert len(dataset.resolver._audio_indexes) == 1
    # A source without an applicable clean version remains usable, including TS.
    raw_config = {**catalog_config, "train": [{**catalog_config["train"][0], "require_clean_pass": False}]}
    raw = CatalogSwiftDataset(raw_config, message_format="qwen3_asr")
    assert len(raw) == 6 and raw[2]["audio_slot_count"] == 1


def _online_ts_config(catalog_config, tmp_path, online_mix):
    import gzip

    row = {"cut_id": "speech", "root_alias": "data", "relative_path": "speech.wav",
           "sample_rate": 16000, "channels": 1, "num_frames": 16000, "duration": 1.0}
    with gzip.open(tmp_path / "index.jsonl.gz", "wt") as stream:
        stream.write(json.dumps(row) + "\n")
    record = AudioRecord("ts", "ts_asr", tuple(
        AudioSlot(name, AudioRef("ts", "1", "train", "speech", start=0.25))
        for name in ("enrollment", "mixture")
    ), "hello", language="en")
    write_records([record], tmp_path / "ts.jsonl.gz")
    spec = DatasetSpec(
        dataset_id="ts", version="1", languages=("en",), tasks=("ts_asr",),
        artifacts=(ArtifactRef("records", "audio-records", "data", "ts.jsonl.gz"),
                   ArtifactRef("index", "audio-index", "data", "index.jsonl.gz")),
        splits={"train": Split({"records": ("records",), "audio_index": ("index",)})},
    )
    with Path(catalog_config["catalog"]).open("a") as stream:
        stream.write("\n" + json.dumps(spec.to_dict()))
    catalog_config.update(
        train=[{"dataset_id": "ts", "version": "1", "split": "train", "online_mix": online_mix}],
        interference_sources=[{"dataset_id": "speech", "version": "1", "split": "train", "language": "en"}],
        augmentation={"noise_prob": 0.5},
    )
    return catalog_config


def test_online_mix_rebuilds_ts_samples_per_epoch(catalog_config, tmp_path):
    config = _online_ts_config(catalog_config, tmp_path, {
        "absent_prob": 0.0, "enrollment_seconds": [0.2, 0.3], "level_dbfs": [-30, -30],
        "interferer_probs": [0, 1], "speaker_rir_prob": 1.0, "speaker_far_rir_prob": 0.0,
    })
    dataset = CatalogSwiftDataset(config, message_format="qwen3_asr")
    first, again = dataset[0], dataset[0]
    assert first["solution"] == "language English<asr_text>hello"
    assert first["mix_wav"] == again["mix_wav"]  # same epoch and index, same sample
    enroll, _ = sf.read(BytesIO(first["audios"][0]))
    mix, _ = sf.read(BytesIO(first["mix_wav"]))
    assert int(0.2 * 16000) <= len(enroll) <= int(0.3 * 16000)
    assert len(mix) == int(0.75 * 16000)
    assert 20 * np.log10(np.sqrt(np.mean(mix**2))) == pytest.approx(-30, abs=0.1)
    dataset.set_epoch(1)
    assert dataset[0]["mix_wav"] != first["mix_wav"]
    # Validation reads the stored clips unchanged.
    plain = CatalogSwiftDataset({**config, "validation": config["train"]}, training=False,
                                message_format="qwen3_asr")[0]
    assert len(plain["audios"][0]) == int(0.75 * 16000)


def test_online_mix_absent_target_empties_the_transcript(catalog_config, tmp_path):
    config = _online_ts_config(catalog_config, tmp_path, {
        "absent_prob": 1.0, "absent_interferer_probs": [1.0], "speaker_rir_prob": 0.0,
    })
    sample = CatalogSwiftDataset(config, message_format="qwen3_asr")[0]
    assert sample["solution"] == "language None<asr_text>"
    mix, _ = sf.read(BytesIO(sample["mix_wav"]))
    assert np.abs(mix).max() > 0  # an empty window always gets noise
    config["interference_sources"] = []
    config["online_mix"] = None
    with pytest.raises(ValueError, match="interference_sources"):
        CatalogSwiftDataset({**config, "train": [{**config["train"][0], "online_mix": {"absent_prob": 0.0, "speaker_rir_prob": 0.0}}]},
                            message_format="qwen3_asr")[0]


def test_online_mix_nonspeech_window_has_an_empty_transcript_and_its_own_level(catalog_config, tmp_path):
    config = _online_ts_config(catalog_config, tmp_path, {
        "absent_prob": 0.0, "nonspeech_prob": 1.0, "nonspeech_seconds": [0.4, 0.6],
        "nonspeech_level_dbfs": [-55, -55], "level_dbfs": [-30, -30], "enrollment_seconds": [0.2, 0.3],
        "speaker_rir_prob": 0.0,
    })
    sample = CatalogSwiftDataset(config, message_format="qwen3_asr")[0]
    assert sample["solution"] == "language None<asr_text>"
    mix, _ = sf.read(BytesIO(sample["mix_wav"]))
    enroll, _ = sf.read(BytesIO(sample["audios"][0]))
    assert int(0.39 * 16000) <= len(mix) <= int(0.61 * 16000)
    assert 20 * np.log10(np.sqrt(np.mean(mix**2))) == pytest.approx(-55, abs=0.2)
    assert 20 * np.log10(np.sqrt(np.mean(enroll**2))) == pytest.approx(-30, abs=0.2)  # enrollment keeps its range


def test_scene_record_places_the_named_utterances_and_is_never_blanked(catalog_config, tmp_path):
    config = _online_ts_config(catalog_config, tmp_path, {
        "absent_prob": 1.0, "nonspeech_prob": 1.0, "enrollment_seconds": [0.2, 0.3], "speaker_rir_prob": 0.0,
        "alternate_gap_seconds": [0.2, 0.2],
    })
    span = {"cut_id": "speech", "start": 0.0, "duration": 0.4}
    base = {"dataset_id": "ts", "version": "1", "split": "train"}
    record = AudioRecord("scene", "ts_asr", (
        AudioSlot("enrollment", AudioRef(**base, cut_id="speech", start=0.25)),
        AudioSlot("mixture", AudioRef(**base, cut_id="speech", start=0.0, duration=0.4))),
        "hello world", language="en",
        metadata={"scene": {"mode": "alternate", "targets": [{**span, "text": "hello"}, {**span, "text": "world"}],
                            "interferers": [{"speaker": "x", "utts": [span]}]}})
    write_records([record], tmp_path / "ts.jsonl.gz")
    sample = CatalogSwiftDataset(config, message_format="qwen3_asr")[0]
    assert sample["solution"] == "language English<asr_text>hello world"
    mix, _ = sf.read(BytesIO(sample["mix_wav"]))
    assert len(mix) == int(3 * 0.4 * 16000) + 2 * int(0.2 * 16000)  # three turns and two gaps


def test_full_scene_record_uses_the_gap_ranges_the_record_names(catalog_config, tmp_path):
    config = _online_ts_config(catalog_config, tmp_path, {
        "absent_prob": 1.0, "nonspeech_prob": 1.0, "enrollment_seconds": [0.2, 0.3], "speaker_rir_prob": 0.0,
    })
    span = {"cut_id": "speech", "start": 0.0, "duration": 0.4}
    base = {"dataset_id": "ts", "version": "1", "split": "train"}
    record = AudioRecord("scene", "ts_asr", (
        AudioSlot("enrollment", AudioRef(**base, cut_id="speech", start=0.25)),
        AudioSlot("mixture", AudioRef(**base, cut_id="speech", start=0.0, duration=0.4))),
        "hello world", language="en",
        metadata={"scene": {"mode": "full", "target_gap": [1.0, 1.0],
                            "targets": [{**span, "text": "hello"}, {**span, "text": "world"}],
                            "interferers": [{"speaker": "x", "gap": [0.1, 0.1], "utts": [span, span, span]}]}})
    write_records([record], tmp_path / "ts.jsonl.gz")
    sample = CatalogSwiftDataset(config, message_format="qwen3_asr")[0]
    assert sample["solution"] == "language English<asr_text>hello world"
    mix, _ = sf.read(BytesIO(sample["mix_wav"]))
    # the target track is 0.4 + 1.0 + 0.4 s and the other speaker's is 3 x 0.4 + 2 x 0.1 s: the longer one is the window
    assert len(mix) == int(1.8 * 16000)


def test_online_mix_target_timing_changes_the_window(catalog_config, tmp_path):
    config = _online_ts_config(catalog_config, tmp_path, {
        "absent_prob": 0.0, "entry_prob": 1.0, "lead_seconds": [1.0, 1.0], "trail_seconds": [0.5, 0.5],
        "interferer_probs": [1.0], "speaker_rir_prob": 0.0,
    })
    dataset = CatalogSwiftDataset(config, message_format="qwen3_asr")
    lengths = set()
    for epoch in range(24):
        dataset.set_epoch(epoch)
        mix, _ = sf.read(BytesIO(dataset[0]["mix_wav"]))
        lengths.add(round(len(mix) / 16000, 2))
    assert lengths == {1.25, 1.75, 2.25}  # leave, enter, middle around the 0.75 s target


def test_online_mix_activity_labels_follow_the_target_timing(catalog_config, tmp_path):
    from open_audio_llm.data.ts_online import activity_codes

    config = _online_ts_config(catalog_config, tmp_path, {
        "absent_prob": 0.0, "entry_prob": 1.0, "lead_seconds": [1.0, 1.0], "trail_seconds": [0.5, 0.5],
        "interferer_probs": [1.0], "speaker_rir_prob": 0.0,
    })
    assert "ts_activity" not in CatalogSwiftDataset(config, message_format="qwen3_asr")[0]
    dataset = CatalogSwiftDataset({**config, "objective": {"ts_activity_weight": 0.1}}, message_format="qwen3_asr")
    seen = set()
    for epoch in range(12):
        dataset.set_epoch(epoch)
        sample = dataset[0]
        mix, _ = sf.read(BytesIO(sample["mix_wav"]))
        lead = 0 if len(mix) == int(1.25 * 16000) else 16000  # leave, or enter / middle
        expected = activity_codes(len(mix), [(lead, lead + 12000)], [], 16000)
        assert np.array_equal(sample["ts_activity"], expected)
        seen.add(len(mix))
    assert len(seen) == 3


def test_real_window_activity_labels_come_from_its_annotation(catalog_config, tmp_path):
    config = _online_ts_config(catalog_config, tmp_path, {
        "absent_prob": 0.0, "interferer_probs": [1.0], "speaker_rir_prob": 0.0,
    })
    base = {"dataset_id": "ts", "version": "1", "split": "train"}
    record = AudioRecord("window", "ts_asr", (
        AudioSlot("enrollment", AudioRef(**base, cut_id="speech", start=0.0, duration=0.3)),
        AudioSlot("mixture", AudioRef(**base, cut_id="speech", start=0.25))), "hello", language="en",
        metadata={"target_speaker": "a", "window_utts": [
            {"speaker": "a", "start": 0.0, "duration": 0.3}, {"speaker": "b", "start": 0.2, "duration": 0.4}]})
    write_records([record], tmp_path / "ts.jsonl.gz")
    sample = CatalogSwiftDataset({**config, "objective": {"ts_activity_weight": 0.1}}, message_format="qwen3_asr")[0]
    # 0.75 s mixture: the target in frames 0-29, the other speaker in 20-59, overlap in 20-29
    assert sample["ts_activity"].tolist() == [1] * 20 + [3] * 10 + [2] * 30 + [0] * 15


def test_plain_asr_nonspeech_source_trains_on_an_empty_transcript(catalog_config, tmp_path):
    import gzip

    row = {"cut_id": "speech", "root_alias": "data", "relative_path": "speech.wav",
           "sample_rate": 16000, "channels": 1, "num_frames": 16000, "duration": 1.0}
    with gzip.open(tmp_path / "index_n.jsonl.gz", "wt") as stream:
        stream.write(json.dumps(row) + "\n")
    record = AudioRecord("n", "asr", (AudioSlot("primary", AudioRef("nonspeech", "1", "train", "speech")),),
                         "", language="zh")
    write_records([record], tmp_path / "n.jsonl.gz")
    spec = DatasetSpec(
        dataset_id="nonspeech", version="1", languages=("zh",), tasks=("asr",),
        artifacts=(ArtifactRef("records", "audio-records", "data", "n.jsonl.gz"),
                   ArtifactRef("index", "audio-index", "data", "index_n.jsonl.gz")),
        splits={"train": Split({"records": ("records",), "audio_index": ("index",)})},
    )
    with Path(catalog_config["catalog"]).open("a") as stream:
        stream.write("\n" + json.dumps(spec.to_dict()))
    catalog_config["train"] = [{"dataset_id": "nonspeech", "version": "1", "split": "train",
                                "nonspeech": {"level_dbfs": [-52, -52]}}]
    catalog_config["augmentation"] = {}
    sample = CatalogSwiftDataset(catalog_config, message_format="qwen3_asr")[0]
    assert sample["solution"] == "language None<asr_text>"
    audio, _ = sf.read(BytesIO(sample["audios"][0]))
    assert 20 * np.log10(np.sqrt(np.mean(audio**2))) == pytest.approx(-52, abs=0.2)
    with pytest.raises(ValueError, match="empty-transcript"):
        CatalogSwiftDataset({**catalog_config, "train": [{**catalog_config["train"][0], "dataset_id": "speech",
                            "split": "dev", "nonspeech": {"level_dbfs": [-50, -50]}}]}, message_format="qwen3_asr")


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
            custom={"clean": {"pass": True}},
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
                "train": Split({"cuts": ("c0", "c1")}),
                "dev": Split({"recordings": ("r",), "supervisions": ("s",)}),
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
                splits={"train": Split({"recordings": ("r",)})},
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
        "train": [{"dataset_id": "speech", "version": "1", "split": "train", "require_clean_pass": True}],
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


def test_cache_ignores_unused_roots_and_mapping_order(catalog_config, tmp_path, monkeypatch):
    catalog_config['metadata_cache'] = str(tmp_path / 'cache')
    catalog_config['train'][0]['split'] = 'dev'
    first = CatalogSwiftDataset(catalog_config)
    roots = Path(catalog_config['roots'])
    roots.write_text(json.dumps({'unrelated': '/unused/new-data', 'data': str(tmp_path)}))
    catalog = Path(catalog_config['catalog'])
    specs = [json.loads(line) for line in catalog.read_text().splitlines()]
    specs[0]['splits']['dev'] = dict(reversed(list(specs[0]['splits']['dev'].items())))
    catalog.write_text('\n'.join(json.dumps(s, sort_keys=True) for s in specs))
    monkeypatch.setattr(CatalogSwiftDataset, '_source_records',
                        lambda *a: pytest.fail('unchanged source rebuilt'))
    second = CatalogSwiftDataset(catalog_config)
    assert first.records.sources[0].key == second.records.sources[0].key
    assert first.records[0].record == second.records[0].record


@pytest.mark.parametrize('change', ['root', 'file', 'version', 'clean', 'exclude',
                                  'duration', 'speed', 'sampling_rate', 'message_format', 'shard_order'])
def test_cache_identity_changes_for_index_inputs(catalog_config, tmp_path, change):
    import os
    import shutil
    from open_audio_llm.data.catalog_cache import source_identity

    dataset = CatalogSwiftDataset(catalog_config)
    source = dict(dataset.sources[0])
    before = source_identity(dataset, source)
    if change == 'root':
        other = tmp_path / 'other'
        other.mkdir()
        for i in range(2):
            shutil.copy2(tmp_path / f'cuts{i}.jsonl.gz', other)
        dataset.resolver.roots['data'] = other
    elif change == 'file':
        path = tmp_path / 'cuts0.jsonl.gz'
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
    elif change in {'version', 'shard_order'}:
        specs = [json.loads(line) for line in Path(catalog_config['catalog']).read_text().splitlines()]
        if change == 'version':
            specs[0]['version'] = '2'
            source['version'] = '2'
        else:
            specs[0]['splits']['train']['artifacts']['cuts'].reverse()
        Path(catalog_config['catalog']).write_text('\n'.join(json.dumps(s) for s in specs))
        from audio_data_contract import load_catalog
        dataset.resolver.catalog = load_catalog(catalog_config['catalog'])
    elif change == 'clean':
        source['require_clean_pass'] = False
    elif change == 'exclude':
        source['exclude_speakers'] = ['heldout']
    elif change == 'duration':
        source['max_duration'] = .2
    elif change == 'speed':
        dataset.augmentation = AugmentConfig(speed_prob=1, speed_factors=[.5])
    elif change == 'sampling_rate':
        dataset.sampling_rate = 8000
    else:
        dataset.message_format = 'qwen3_asr'
    assert before != source_identity(dataset, source)


def test_verified_legacy_cache_reuse_survives_new_roots_and_order(catalog_config, tmp_path, monkeypatch):
    from copy import deepcopy
    from open_audio_llm.data import catalog_cache

    config = deepcopy(catalog_config)
    config['metadata_cache'] = str(tmp_path / 'cache')
    config['train'][0]['split'] = 'dev'
    Path(config['roots']).write_text(json.dumps({'data': str(tmp_path), 'unused': '/old'}))
    previous = deepcopy(config)
    for field in ['catalog', 'roots']:
        path = tmp_path / ('previous-' + Path(config[field]).name)
        path.write_bytes(Path(config[field]).read_bytes())
        previous[field] = str(path)
    with monkeypatch.context() as patch:
        patch.setattr(catalog_cache, 'source_identity', catalog_cache._legacy_source_identity)
        old = CatalogSwiftDataset(previous)
    Path(config['roots']).write_text(json.dumps({'data': str(tmp_path), 'unused': '/new'}))
    specs = [json.loads(line) for line in Path(config['catalog']).read_text().splitlines()]
    specs[0]['splits']['dev'] = dict(reversed(list(specs[0]['splits']['dev'].items())))
    Path(config['catalog']).write_text('\n'.join(json.dumps(s) for s in specs))
    # Different processing parameters cannot claim the old cache.
    changed = {**config, 'sampling_rate': 8000}
    assert not catalog_cache.reuse_indexes(changed, previous, config['metadata_cache'], 'generic')
    reused = catalog_cache.reuse_indexes(config, previous, config['metadata_cache'], 'generic')
    assert reused
    monkeypatch.setattr(CatalogSwiftDataset, '_source_records',
                        lambda *a: pytest.fail('verified legacy index rebuilt'))
    current = CatalogSwiftDataset(config)
    assert current.records.sources[0].path.resolve() == old.records.sources[0].path.resolve()
    assert current.records[0].record == old.records[0].record


def test_preflight_uses_real_quotas_before_decoding(catalog_config, monkeypatch):
    from open_audio_llm.data.catalog_preflight import check_data

    catalog_config['augmentation'] = {}
    source = catalog_config['train'][0]
    catalog_config['train'] = [{**source, 'weight': 99.99}, {**source, 'weight': .01}]
    catalog_config['replay'] = {'epoch_samples': 10000, 'window_samples': 5000}
    with monkeypatch.context() as patch:
        patch.setattr(CatalogSwiftDataset, '__getitem__', lambda *a: pytest.fail('decoded before quota check'))
        with pytest.raises(ValueError, match='every source'):
            check_data(catalog_config)
    catalog_config['replay']['window_samples'] = 10000
    report = check_data(catalog_config)
    assert report['status'] == 'passed'
    assert [s['quota'] for s in report['sources']] == [9999, 1]
    assert report['sampler_state']['consumed'] == 0
    assert {s['selection'] for s in report['decoded_examples']} == {'train', 'validation'}


def test_preflight_reports_resolved_full_coverage_budget(catalog_config):
    from open_audio_llm.data.catalog_preflight import check_data

    catalog_config['augmentation'] = {}
    for source in catalog_config['train']:
        source['weight'] = 1
    catalog_config['replay'] = {'epoch_samples': 'full_coverage', 'window_samples': 10}
    report = check_data(catalog_config)
    assert report['full_coverage'] is True
    assert report['replay']['epoch_samples'] >= sum(s['records'] for s in report['sources'])
    assert all(s['samples_per_epoch'] >= s['records'] for s in report['sources'])
    assert catalog_config['replay']['epoch_samples'] == 'full_coverage'


def test_previous_recipe_paths_are_not_overridden_by_current_run(catalog_config, tmp_path, monkeypatch):
    path = tmp_path / 'recipe.json'
    path.write_text(json.dumps(catalog_config))
    monkeypatch.setenv('AUDIO_DATA_ROOTS_FILE', '/new/run/roots.json')
    monkeypatch.setenv('AUDIO_DATA_CATALOG', '/new/run/catalog')
    monkeypatch.setenv('AUDIO_DATA_METADATA_CACHE', '/new/run/cache')
    assert read_data_config(path)['roots'] == catalog_config['roots']
    assert read_data_config(path)['catalog'] == catalog_config['catalog']
    assert CatalogSwiftDataset(read_data_config(path)).metadata_cache is None


@pytest.mark.parametrize('config_name', ['ts-replay', 'ts-full', 'ts-joint', 'sot'])
def test_config_preflight_matches_training_data_and_batch(config_name, tmp_path):
    from open_audio_llm.run_config import command, load_config

    root = Path(__file__).resolve().parents[1]
    config = load_config(root / f'examples/configs/train/{config_name}.yaml', tmp_path)
    training = command(config)
    preflight = command({**config, 'task': config['preflight'][0]})
    assert preflight[preflight.index('--data_config') + 1] == training[training.index('--data_config') + 1]
    assert preflight[preflight.index('--batch-size') + 1] == training[training.index('--per_device_train_batch_size') + 1]
    assert preflight[preflight.index('--world-size') + 1] == str(config['runtime']['distributed']['processes'])
    assert '--preflight-report' in preflight


@pytest.mark.parametrize("flag", [None, False])
@pytest.mark.parametrize("cached", [False, True])
def test_nonclean_training_is_allowed_without_clean_requirement(catalog_config, tmp_path, flag, cached):
    source = catalog_config["train"][0]
    if flag is None:
        source.pop("require_clean_pass")
    else:
        source["require_clean_pass"] = flag
    for index in range(2):
        path = tmp_path / f"cuts{index}.jsonl.gz"
        cut = next(iter(CutSet.from_file(path)))
        cut.supervisions[0].custom = {}
        CutSet.from_cuts([cut]).to_file(path)
    path = Path(catalog_config["catalog"])
    specs = [json.loads(line) for line in path.read_text().splitlines()]
    specs[0]["provenance"] = {"quality_status": "source_annotations_preserved_not_reaudited"}
    path.write_text("\n".join(json.dumps(spec) for spec in specs))
    if cached:
        catalog_config["metadata_cache"] = str(tmp_path / "raw-cache")
    dataset = CatalogSwiftDataset(catalog_config)
    assert len(dataset) == 2 and "hello" in dataset[0]["solution"]


def test_single_speaker_format_rejects_portable_records(catalog_config, tmp_path):
    record = AudioRecord(
        "portable",
        "asr",
        (AudioSlot("primary", AudioRef("speech", "1", "train", "s0", duration=0.25)),),
        "hello",
        language="en",
    )
    write_records([record], tmp_path / "portable.jsonl.gz")
    spec = DatasetSpec(
        dataset_id="portable",
        version="1",
        languages=("en",),
        tasks=("asr",),
        artifacts=(ArtifactRef("records", "audio-records", "data", "portable.jsonl.gz"),),
        splits={"train": Split({"records": ("records",)})},
    )
    with Path(catalog_config["catalog"]).open("a") as stream:
        stream.write("\n" + json.dumps(spec.to_dict()))
    catalog_config["train"] = [{
        "dataset_id": "portable", "version": "1", "split": "train",
        "task": "asr", "single_speaker_format": True,
    }]
    with pytest.raises(ValueError, match="requires Lhotse supervision"):
        CatalogSwiftDataset(catalog_config, message_format="qwen3_asr")


@pytest.mark.parametrize("cached", [False, True])
def test_single_speaker_asr_uses_unified_diarization_target(catalog_config, tmp_path, cached):
    for index in range(2):
        path = tmp_path / f"cuts{index}.jsonl.gz"
        cut = next(iter(CutSet.from_file(path)))
        cut.supervisions[0].speaker = "speaker0"
        CutSet.from_cuts([cut]).to_file(path)
    source = catalog_config["train"][0]
    source.update(task="asr", single_speaker_format=True)
    if cached:
        catalog_config["metadata_cache"] = str(tmp_path / "cache")
    dataset = CatalogSwiftDataset(catalog_config, message_format="qwen3_asr")
    record = dataset.records[0].record
    assert record.task == "speaker_attributed_asr"
    assert record.target == "[S1] hello"
    assert record.metadata == {
        "source_task": "asr",
        "single_speaker_verified": True,
        "source_speaker": "speaker0",
    }
    assert dataset[0]["solution"] == "language English<asr_text>[S1] hello"


@pytest.mark.parametrize("flag", ["true", 1])
def test_clean_requirement_must_be_boolean(catalog_config, flag):
    catalog_config["train"][0]["require_clean_pass"] = flag
    with pytest.raises(ValueError, match="require_clean_pass must be a boolean"):
        CatalogSwiftDataset(catalog_config)


def test_cache_preparation_preserves_unfiltered_validation(catalog_config):
    from open_audio_llm.data.catalog_cache import _prepare_source

    source = {**catalog_config["validation"][0], "require_clean_pass": False}
    assert _prepare_source(catalog_config, source, "qwen3_asr", training=False) == ("speech", "dev", 2)
    assert _prepare_source(catalog_config, source, "qwen3_asr", training=True) == ("speech", "dev", 2)


def test_index_preparation_skips_augmentation_resources(catalog_config):
    full = CatalogSwiftDataset(catalog_config)
    index_only = CatalogSwiftDataset(catalog_config, resources=False)
    assert full.rir_groups and full.noise_groups
    assert not index_only.rir_groups and not index_only.noise_groups
    assert len(index_only) == len(full)


@pytest.mark.parametrize('cached', [False, True])
def test_speaker_holdout_is_removed_from_replay_and_invalidates_old_cache(catalog_config, tmp_path, cached):
    for index in range(2):
        path = tmp_path / f'cuts{index}.jsonl.gz'
        cut = next(iter(CutSet.from_file(path)))
        cut.supervisions[0].speaker = f'speaker{index}'
        CutSet.from_cuts([cut]).to_file(path)
    if cached:
        catalog_config['metadata_cache'] = str(tmp_path / 'cache')
    assert len(CatalogSwiftDataset(catalog_config)) == 2
    catalog_config['train'][0]['exclude_speakers'] = ['speaker1']
    filtered = CatalogSwiftDataset(catalog_config)
    assert len(filtered) == 1 and filtered.records[0].record.audio_slots[0].ref.cut_id == 's0'


@pytest.mark.parametrize('cached', [False, True])
def test_sot_catalog_uses_mixture_without_enrollment_and_excludes_teacher_kl(catalog_config, tmp_path, cached):
    target = '[S1] 第一人\n[S2] 第二人\n[S3] 第三人'
    record = AudioRecord('sot-example', 'speaker_attributed_asr',
        (AudioSlot('mixture', AudioRef('speech', '1', 'train', 's0', duration=.25)),), target, language='zh')
    write_records([record], tmp_path / 'sot.jsonl.gz')
    spec = DatasetSpec('sot', '1', ('zh',), ('speaker_attributed_asr',),
        (ArtifactRef('records', 'audio-records', 'data', 'sot.jsonl.gz'),),
        {'train': Split({'records': ('records',)})})
    with Path(catalog_config['catalog']).open('a') as stream:
        stream.write('\n' + json.dumps(spec.to_dict()))
    catalog_config.update(train=[{'dataset_id': 'sot', 'version': '1', 'split': 'train'}],
                          augmentation={}, objective={'sample_mean': True})
    if cached:
        catalog_config['metadata_cache'] = str(tmp_path / 'cache')
    dataset = CatalogSwiftDataset(catalog_config, message_format='qwen3_asr')
    sample = dataset[0]
    assert sample['catalog_task'] == 3 and sample['audio_slot_count'] == 1
    assert sample['duration'] == .25 and sample['solution'].endswith(target)
    assert len(sf.read(BytesIO(sample['audios'][0]))[0]) == 4000
    from open_audio_llm.integrations.ms_swift.retention import sample_losses

    ordinary = torch.tensor([sample['catalog_task']]) == 0
    losses, ce, kl = sample_losses(torch.zeros(1, 4), torch.tensor([0]), torch.tensor([0]),
                                    torch.tensor([1]), ordinary=ordinary)
    assert not ordinary.any() and kl.item() == 0 and torch.equal(losses, ce)


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
    # Validation keeps the waveform instead of encoding a WAV, so compare the arrays.
    np.testing.assert_array_equal(fixed["audios"][0], validation[1]["audios"][0])
    reference = fixed["audios"][0]
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


def test_grpo_deepspeed_scheduler_keeps_parameter_groups_aligned():
    from open_audio_llm.integrations.ms_swift.train import CatalogTrainingMixin

    parameter = torch.nn.Parameter(torch.ones(1))

    def create_optimizer():
        return torch.optim.AdamW([{"params": [parameter]}, {"params": []}], lr=0.01)

    class ParentPipeline:
        def train(self, trainer):
            optimizer = trainer.create_optimizer()
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.5)
            # DeepSpeed drops empty groups when it wraps the existing optimizer.
            optimizer.param_groups[:] = [g for g in optimizer.param_groups if g["params"]]
            parameter.sum().backward()
            optimizer.step()
            scheduler.step()
            assert len(scheduler.base_lrs) == len(optimizer.param_groups) == 1
            assert optimizer.param_groups[0]["lr"] == 0.005
            assert parameter.item() < 1

    class Pipeline(CatalogTrainingMixin, ParentPipeline):
        pass

    pipeline = Pipeline()
    pipeline.args = SimpleNamespace(rlhf_type="grpo", _catalog_config={"train": [{}]})
    trainer = SimpleNamespace(
        args=SimpleNamespace(deepspeed="zero2", optimizer=None),
        train_dataset=[],
        add_callback=lambda callback: None,
        create_optimizer=create_optimizer,
    )
    pipeline.train(trainer)
    assert trainer.create_optimizer is create_optimizer


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


def test_noise_source_weight_selects_wham_thirty_percent():
    sources = [{"dataset_id": name, "weight": weight} for name, weight in (
        ("wham", 30), ("musan", 17.5), ("1spk", 17.5), ("2spk", 17.5), ("3spk", 17.5),
    )]
    weights = source_weights(sources)
    assert weights[0] / sum(weights) == pytest.approx(0.3)
    groups = [[name] for name in ("wham", "musan", "1spk", "2spk", "3spk")]
    counts = {}
    for seed in range(4000):
        chosen = choose_from_groups(random.Random(seed), groups, weights)
        counts[chosen] = counts.get(chosen, 0) + 1
    assert counts["wham"] / 4000 == pytest.approx(0.3, abs=0.02)
    with pytest.raises(ValueError, match="every augmentation source"):
        source_weights([{"weight": 1}, {}])


def test_enrollment_noise_level_is_separate_from_the_mixture_s():
    from dataclasses import replace

    from open_audio_llm.data.augment import for_slot

    config = AugmentConfig(noise_prob=1, noise_snr_db=(0, 20), enrollment_noise_snr_db=(10, 20))
    assert for_slot(config, "enrollment").noise_snr_db == (10, 20)
    assert for_slot(config, "mixture").noise_snr_db == (0, 20)
    assert for_slot(AugmentConfig(noise_snr_db=(0, 20)), "enrollment").noise_snr_db == (0, 20)
    with pytest.raises(ValueError, match="enrollment_noise_snr_db"):
        AugmentConfig(enrollment_noise_snr_db=(20, 10))
    audio = np.ones(1000, dtype=np.float32)
    noise = np.tile(np.array([-1, 1], dtype=np.float32), 500)
    mixed = augment_waveform(audio, 16000, random.Random(0), for_slot(replace(config, enrollment_noise_snr_db=(10, 10)), "enrollment"),
                             noise_loader=lambda rng, sr: noise)
    assert np.mean((mixed - audio) ** 2) == pytest.approx(0.1, rel=1e-5)


def test_far_rir_is_exclusive_with_ordinary_reverb():
    audio = np.ones(8, dtype=np.float32)
    calls = []

    def loader(rng, sr, mode):
        calls.append(mode)
        impulse = [1.0, 0.5] if mode == "ordinary" else [1.0, 0.0, 0.5]
        return np.array(impulse, dtype=np.float32)

    far = augment_waveform(
        audio, 16000, random.Random(0), AugmentConfig(far_rir_prob=1), rir_loader=loader,
    )
    ordinary = augment_waveform(
        audio, 16000, random.Random(0), AugmentConfig(rir_prob=1), rir_loader=loader,
    )
    assert calls == ["far", "ordinary"]
    assert far.shape == audio.shape and ordinary.shape == audio.shape
    assert not np.array_equal(far, ordinary)
    with pytest.raises(ValueError, match="at most 1"):
        AugmentConfig(rir_prob=0.6, far_rir_prob=0.6)


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
        rir_loader=lambda rng, sr, mode: np.array([1, 0.5], dtype=np.float32),
    )
    assert reverbed.shape == audio.shape
    assert not np.array_equal(reverbed, audio)


def test_rir_starts_at_direct_path_and_reverberates_noise():
    import random
    from dataclasses import replace

    from open_audio_llm.data.catalog_dataset import load_rir

    audio = np.zeros(4000, dtype=np.float32)
    audio[1000] = 1
    impulse = np.zeros(2000, dtype=np.float32)
    impulse[300], impulse[1500] = 1, 0.5  # propagation delay, then a late reflection
    reverbed = augment_waveform(
        audio, 16000, random.Random(0), AugmentConfig(rir_prob=1),
        rir_loader=lambda rng, sr, mode: impulse,
    )
    assert int(np.argmax(np.abs(reverbed))) == 1000
    assert reverbed[2200] > 0

    # Late tail: aligned impulse from 50 ms (800 samples) on; the reflection sits at
    # tail index 400. An exact-length clip makes the noise window deterministic.
    tail = len(impulse) - 300 - 800
    noise = np.random.default_rng(0).normal(size=4000 + tail - 1).astype(np.float32)
    clean = np.ones(4000, dtype=np.float32)
    config = AugmentConfig(rir_prob=1, noise_prob=1, noise_snr_db=(10, 10), reverb_noise=True)
    mixed = augment_waveform(
        clean, 16000, random.Random(0), config,
        noise_loader=lambda rng, sr: noise, rir_loader=lambda rng, sr, mode: impulse,
    )
    speech = augment_waveform(
        clean, 16000, random.Random(0), replace(config, noise_prob=0),
        noise_loader=lambda rng, sr: noise, rir_loader=lambda rng, sr, mode: impulse,
    )
    added = mixed - speech
    assert np.mean(added**2) == pytest.approx(np.mean(speech**2) / 10, rel=1e-4)
    # Only the late reflection reaches the noise, which arrives without the direct path.
    expected = noise[tail - 1 - 400 :][:4000]
    expected *= np.sqrt(np.mean(speech**2) / 10 / np.mean(expected**2))
    np.testing.assert_allclose(added, expected, atol=1e-4)

    rir = np.stack([np.ones(10), 2 * np.ones(10)]).astype(np.float32)
    cut = SimpleNamespace(id="array", sampling_rate=16000, load_audio=lambda: rir)
    assert {float(load_rir(cut, random.Random(seed), 16000)[0]) for seed in range(20)} == {1.0, 2.0}


def test_far_level_applies_only_to_reverberated_clips():
    import random
    from dataclasses import replace

    audio = np.random.default_rng(0).normal(0, 0.3, 4000).astype(np.float32)
    impulse = np.array([1, 0, 0.5], dtype=np.float32)
    config = AugmentConfig(rir_prob=1, far_level_dbfs=(-40, -40))
    far = augment_waveform(audio, 16000, random.Random(0), config, rir_loader=lambda rng, sr, mode: impulse)
    assert 10 * np.log10(np.mean(far**2)) == pytest.approx(-40, abs=1e-3)
    near = augment_waveform(audio, 16000, random.Random(0), replace(config, rir_prob=0))
    np.testing.assert_array_equal(near, audio)
    with pytest.raises(ValueError, match="far_level_dbfs"):
        AugmentConfig(far_level_dbfs=(-30, 3))


def test_rir_drr_range_selects_microphones(catalog_config):
    from open_audio_llm.data.augment import direct_to_reverberant_db

    sr = catalog_config["sampling_rate"]
    impulse = np.zeros(sr // 10)
    impulse[0], impulse[sr // 100] = 1, 0.5  # reflection 10 ms after the direct path
    assert direct_to_reverberant_db(impulse, sr) == pytest.approx(10 * np.log10(4))
    # The fixture RIR has the same shape (DRR ~6 dB).
    catalog_config["augmentation"]["rir_drr_db"] = [0, 10]
    dataset = CatalogSwiftDataset(catalog_config)
    assert [channel for _, channel in dataset.rir_choices] == [0]
    assert dataset[0]
    catalog_config["augmentation"]["rir_drr_db"] = [-6, -1]
    with pytest.raises(ValueError, match="DRR within"):
        CatalogSwiftDataset(catalog_config)


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


def test_config_requires_explicit_catalog_and_roots(
    catalog_config, tmp_path, monkeypatch
):
    monkeypatch.setenv("AUDIO_DATA_CATALOG", catalog_config["catalog"])
    monkeypatch.setenv("AUDIO_DATA_ROOTS_FILE", catalog_config["roots"])
    path = tmp_path / "training.json"
    path.write_text(json.dumps({"train": catalog_config["train"]}))
    with pytest.raises(ValueError, match="catalog"):
        read_data_config(path)


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
        # FLOAT WAV's PEAK chunk carries wall-clock time; compare decoded samples.
        for expected_sample, actual_sample in zip(expected, actual, strict=True):
            for expected_audio, actual_audio in zip(expected_sample, actual_sample, strict=True):
                expected_wave, expected_rate = sf.read(BytesIO(expected_audio))
                actual_wave, actual_rate = sf.read(BytesIO(actual_audio))
                assert expected_rate == actual_rate
                np.testing.assert_array_equal(expected_wave, actual_wave)
        # Each repetition has a different deterministic augmentation seed.
        assert len({audios[0] for audios in actual}) == len(actual)
    finally:
        if loader._iterator is not None:
            loader._iterator._shutdown_workers()


class _EvalTemplate:
    def __init__(self):
        self.model = None
        self.dummy_model = None
        self._handles = []
        self._deepspeed_initialize = None

    def encode(self, sample):
        return {"text": sample["text"]}

    def data_collator(self, batch, padding_to=None):
        return {"text": [row["text"] for row in batch], "padding_to": padding_to}


class _EngineWithLocalHook:
    """Same shape as DeepSpeedEngine's unpicklable forward hook."""

    def __init__(self):
        def _module_forward_post_hook():
            return None

        self.hook = _module_forward_post_hook


class _EncodedRows(torch.utils.data.Dataset):
    def __init__(self, encode):
        self.encode = encode
        self.rows = [{"text": "a"}, {"text": "b"}]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.encode(self.rows[index])


def test_eval_workers_do_not_pickle_deepspeed_engine():
    from functools import partial

    from open_audio_llm.integrations.ms_swift.catalog_loader import catalog_eval_dataloader

    template = _EvalTemplate()
    engine = _EngineWithLocalHook()
    template.model = engine
    template._handles = [(engine, engine.hook)]
    trainer = SimpleNamespace(
        args=SimpleNamespace(
            dataloader_num_workers=1,
            dataloader_pin_memory=False,
            dataloader_prefetch_factor=None,
            dataloader_drop_last=False,
            eval_batch_size=2,
        ),
        data_collator=partial(template.data_collator, padding_to=8),
        _get_eval_sampler=lambda eval_dataset: None,
        _get_collator_with_removed_columns=lambda collator, description: collator,
    )
    loader = catalog_eval_dataloader(trainer, _EncodedRows(template.encode))
    try:
        batch = next(iter(loader))
    finally:
        if loader._iterator is not None:
            loader._iterator._shutdown_workers()
    assert batch["text"] == ["a", "b"]
    assert batch["padding_to"] == 8
    assert template.model is engine
    assert template._handles[0][0] is engine


def test_eval_dataloader_keeps_workers_without_forking_cuda_process(catalog_config):
    from open_audio_llm.integrations.ms_swift.catalog_loader import (
        install_catalog_loader,
    )

    dataset = CatalogSwiftDataset(catalog_config, training=False)
    trainer = SimpleNamespace(
        args=SimpleNamespace(
            max_steps=10,
            deepspeed=None,
            process_index=0,
            world_size=1,
            train_dataloader_shuffle=True,
            gradient_accumulation_steps=1,
            dataloader_num_workers=2,
            dataloader_persistent_workers=False,
            dataloader_pin_memory=False,
            dataloader_prefetch_factor=None,
            dataloader_drop_last=False,
            eval_batch_size=1,
        ),
        template=SimpleNamespace(sequence_parallel_size=1),
        train_dataset=CatalogSwiftDataset(catalog_config),
        eval_dataset=dataset,
        _train_batch_size=1,
        data_collator=list,
        accelerator=SimpleNamespace(
            device=torch.device("cpu"), prepare=lambda loader: loader
        ),
        add_callback=lambda callback: None,
        set_initial_training_values=lambda *args: (1, 2, 2, 2, False, 2, 10),
        _get_eval_sampler=lambda eval_dataset: None,
        _get_collator_with_removed_columns=lambda collator, description: collator,
    )
    install_catalog_loader(trainer)
    loader = trainer.get_eval_dataloader()
    assert loader is trainer.get_eval_dataloader()
    assert loader.num_workers == 2
    assert loader.persistent_workers
    assert loader.multiprocessing_context.get_start_method() == "forkserver"


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


def test_sft_pipeline_passes_continuous_options_to_catalog_loader(monkeypatch):
    from open_audio_llm.integrations.ms_swift import catalog_loader
    from open_audio_llm.integrations.ms_swift.train import CatalogTrainingMixin

    installed = []
    monkeypatch.setattr(catalog_loader, 'install_catalog_loader',
                        lambda trainer, checkpoint: installed.append(
                            (trainer.args.continuous_training,
                             trainer.args.reset_catalog_sampler,
                             trainer.args.catalog_sampler_migration, checkpoint)))

    class ParentPipeline:
        def train(self, trainer):
            return 'training'

    class Pipeline(CatalogTrainingMixin, ParentPipeline):
        pass

    pipeline = Pipeline()
    pipeline.args = SimpleNamespace(_catalog_config={}, continuous_training=True,
                                    reset_catalog_sampler=True,
                                    catalog_sampler_migration='{"world_size": 2}')
    pipeline._get_resume_checkpoint = lambda trainer: 'checkpoint-1000'
    trainer = SimpleNamespace(args=SimpleNamespace())
    assert pipeline.train(trainer) == 'training'
    assert installed == [(True, True, '{"world_size": 2}', 'checkpoint-1000')]


def test_continuous_catalog_resume_keeps_optimizer_and_restarts_constant_lr(catalog_config):
    import sys
    from open_audio_llm.integrations.ms_swift.catalog_loader import install_catalog_loader

    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter], lr=1e-5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: max(0, 1 - step / 2))
    for _ in range(2):
        parameter.sum().backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
    assert optimizer.param_groups[0]['lr'] == 0
    moments = optimizer.state[parameter]['exp_avg'].clone()
    trainer = SimpleNamespace(
        args=SimpleNamespace(
            continuous_training=True, reset_catalog_sampler=True, max_steps=-1,
            lr_scheduler_type='constant', warmup_steps=0, warmup_ratio=0,
            deepspeed=None, process_index=0, world_size=1,
            train_dataloader_shuffle=True, gradient_accumulation_steps=1,
            dataloader_num_workers=0, dataloader_pin_memory=False,
            lr_scheduler_kwargs={'timescale': 10000},
        ),
        template=SimpleNamespace(sequence_parallel_size=1),
        train_dataset=CatalogSwiftDataset(catalog_config), _train_batch_size=1,
        data_collator=list, accelerator=SimpleNamespace(device=torch.device('cpu')),
        add_callback=lambda callback: None, set_initial_training_values=lambda *args: (1,),
        optimizer=optimizer, lr_scheduler=scheduler,
        _load_optimizer_and_scheduler=lambda checkpoint: None,
    )
    install_catalog_loader(trainer, 'old-quota-checkpoint')
    trainer._load_optimizer_and_scheduler('old-quota-checkpoint')
    assert trainer.args.max_steps == sys.maxsize
    assert trainer.set_initial_training_values()[0] == sys.maxsize
    assert trainer.catalog_sampler.epoch == trainer.catalog_sampler.consumed == 0
    assert trainer.lr_scheduler.last_epoch == 2
    assert optimizer.param_groups[0]['lr'] == 1e-5
    assert optimizer.state[parameter]['step'] == 2
    assert torch.equal(optimizer.state[parameter]['exp_avg'], moments)
    parameter.sum().backward()
    optimizer.step()
    trainer.lr_scheduler.step()
    assert optimizer.state[parameter]['step'] == trainer.lr_scheduler.last_epoch == 3
    assert optimizer.param_groups[0]['lr'] == 1e-5


def test_continuous_inverse_sqrt_keeps_schedule_on_resume(catalog_config):
    from transformers import get_inverse_sqrt_schedule
    from open_audio_llm.integrations.ms_swift.catalog_loader import install_catalog_loader

    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter], lr=2e-5)
    scheduler = get_inverse_sqrt_schedule(optimizer, num_warmup_steps=2)
    rates = []
    for _ in range(8):
        optimizer.step()
        scheduler.step()
        rates.append(optimizer.param_groups[0]['lr'])
    assert rates[0] < rates[1] and rates[-1] < rates[1]
    trainer = SimpleNamespace(
        args=SimpleNamespace(
            continuous_training=True, reset_catalog_sampler=True, max_steps=-1,
            lr_scheduler_type='inverse_sqrt', warmup_steps=2, warmup_ratio=0,
            lr_scheduler_kwargs={'timescale': 10000},
            get_warmup_steps=lambda steps: 2,
            deepspeed=None, process_index=0, world_size=1,
            train_dataloader_shuffle=True, gradient_accumulation_steps=1,
            dataloader_num_workers=0, dataloader_pin_memory=False,
        ),
        template=SimpleNamespace(sequence_parallel_size=1),
        train_dataset=CatalogSwiftDataset(catalog_config), _train_batch_size=1,
        data_collator=list, accelerator=SimpleNamespace(device=torch.device('cpu')),
        add_callback=lambda callback: None, set_initial_training_values=lambda *args: (1,),
        optimizer=optimizer, lr_scheduler=scheduler,
        create_scheduler=lambda *args: None,
        _load_optimizer_and_scheduler=lambda checkpoint: None,
    )
    install_catalog_loader(trainer)
    trainer.lr_scheduler = None
    trainer.create_scheduler(9223372036854775807, optimizer)
    trainer.lr_scheduler.step(2002)
    assert trainer.lr_scheduler.last_epoch == 2002
    assert optimizer.param_groups[0]['lr'] > 1.8e-5


def test_continuous_inverse_sqrt_rebuilds_scheduler_after_checkpoint(catalog_config):
    from open_audio_llm.integrations.ms_swift.catalog_loader import install_catalog_loader

    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter], lr=2e-5)
    old_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.0)
    old_scheduler.step(1000)
    trainer = SimpleNamespace(
        args=SimpleNamespace(
            continuous_training=True, reset_catalog_sampler=True, max_steps=-1,
            lr_scheduler_type='inverse_sqrt', warmup_steps=500, warmup_ratio=0,
            lr_scheduler_kwargs={'timescale': 10000},
            get_warmup_steps=lambda steps: 500,
            deepspeed=None, process_index=0, world_size=1,
            train_dataloader_shuffle=True, gradient_accumulation_steps=1,
            dataloader_num_workers=0, dataloader_pin_memory=False,
        ),
        template=SimpleNamespace(sequence_parallel_size=1),
        train_dataset=CatalogSwiftDataset(catalog_config), _train_batch_size=1,
        data_collator=list, accelerator=SimpleNamespace(device=torch.device('cpu')),
        add_callback=lambda callback: None, set_initial_training_values=lambda *args: (1,),
        optimizer=optimizer, lr_scheduler=None,
        create_scheduler=lambda *args: None,
        _load_optimizer_and_scheduler=lambda checkpoint: setattr(
            trainer, 'lr_scheduler', old_scheduler
        ),
    )
    install_catalog_loader(trainer, 'old-checkpoint')
    trainer.create_scheduler(9223372036854775807, optimizer)
    trainer._load_optimizer_and_scheduler('old-checkpoint')
    assert trainer.lr_scheduler.last_epoch == 0
    assert optimizer.param_groups[0]['lr'] == 0
    trainer.lr_scheduler.step()
    assert optimizer.param_groups[0]['lr'] == 4e-8


def test_continuous_inverse_sqrt_resume_ignores_scheduler_built_without_timescale(catalog_config):
    import math
    from transformers import get_inverse_sqrt_schedule
    from open_audio_llm.integrations.ms_swift.catalog_loader import install_catalog_loader

    parameter = torch.nn.Parameter(torch.ones(1))
    saved_optimizer = torch.optim.AdamW([parameter], lr=2e-5)
    saved = get_inverse_sqrt_schedule(saved_optimizer, num_warmup_steps=500, timescale=10000)
    saved.step(8000)
    optimizer_state, scheduler_state = saved_optimizer.state_dict(), saved.state_dict()
    optimizer = torch.optim.AdamW([parameter], lr=2e-5)
    trainer = SimpleNamespace(
        args=SimpleNamespace(
            continuous_training=True, reset_catalog_sampler=False, max_steps=-1,
            lr_scheduler_type='inverse_sqrt', warmup_steps=500, warmup_ratio=0,
            lr_scheduler_kwargs={'timescale': 10000},
            get_warmup_steps=lambda steps: 500,
            deepspeed=None, process_index=0, world_size=1,
            train_dataloader_shuffle=True, gradient_accumulation_steps=1,
            dataloader_num_workers=0, dataloader_pin_memory=False,
        ),
        template=SimpleNamespace(sequence_parallel_size=1),
        train_dataset=CatalogSwiftDataset(catalog_config), _train_batch_size=1,
        data_collator=list, accelerator=SimpleNamespace(device=torch.device('cpu')),
        add_callback=lambda callback: None, set_initial_training_values=lambda *args: (1,),
        optimizer=optimizer,
        # ms-swift 4.5 creates this through the unbound HF create_scheduler.
        lr_scheduler=get_inverse_sqrt_schedule(optimizer, num_warmup_steps=500),
        create_scheduler=lambda *args: None,
        _load_optimizer_and_scheduler=lambda checkpoint: (
            optimizer.load_state_dict(optimizer_state),
            trainer.lr_scheduler.load_state_dict(scheduler_state),
        ),
    )
    trainer.args.catalog_sampler_migration = None
    install_catalog_loader(trainer)
    trainer._load_optimizer_and_scheduler('checkpoint-8000')
    assert trainer.lr_scheduler.last_epoch == 8000
    assert math.isclose(optimizer.param_groups[0]['lr'], 2e-5 / math.sqrt(1.75))
    optimizer.step()
    trainer.lr_scheduler.step()
    assert math.isclose(optimizer.param_groups[0]['lr'], 2e-5 / math.sqrt(1.7501))


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
        metadata={"clean": {"pass": True}},
    )
    write_records([record], tmp_path / "records.jsonl")
    spec = DatasetSpec(
        dataset_id="records",
        version="1",
        languages=("en",),
        tasks=("asr",),
        artifacts=(ArtifactRef("records", "audio-record", "data", "records.jsonl"),),
        splits={"train": Split({"records": ("records",)})},
    )
    with Path(catalog_config["catalog"]).open("a") as stream:
        stream.write("\n" + json.dumps(spec.to_dict()))
    catalog_config["train"].append(
        {"dataset_id": "records", "version": "1", "split": "train", "require_clean_pass": True}
    )
    catalog_config["augmentation"] = {}
    dataset = CatalogSwiftDataset(catalog_config)
    sample = dataset[2]
    waveform, _ = sf.read(BytesIO(sample["audios"][0]))
    original, _ = sf.read(tmp_path / "speech.wav")
    np.testing.assert_allclose(waveform, original[1000:3000], atol=1e-6)
    assert sample["duration"] == 0.125


def test_mixture_channel_ref_reads_one_microphone(catalog_config, tmp_path):
    sr = 16000
    channels = np.stack([np.full(sr, 0.1), np.full(sr, 0.5)], axis=1).astype(np.float32)
    sf.write(tmp_path / "array.wav", channels, sr, subtype="FLOAT")
    (tmp_path / "array-index.jsonl").write_text(json.dumps({
        "cut_id": "meeting", "root_alias": "data", "relative_path": "array.wav",
        "sample_rate": sr, "channels": 2, "num_frames": sr, "duration": 1.0,
    }) + "\n")
    record = AudioRecord(
        "view",
        "speaker_attributed_asr",
        (AudioSlot("mixture", AudioRef("array", "1", "train", "meeting",
                                        channel=1, start=0.25, duration=0.5)),),
        "[S1][0.00-0.50] hello",
        language="en",
    )
    write_records([record], tmp_path / "array-records.jsonl")
    spec = DatasetSpec(
        dataset_id="array",
        version="1",
        languages=("en",),
        tasks=("speaker_attributed_asr",),
        artifacts=(
            ArtifactRef("records", "audio-records", "data", "array-records.jsonl"),
            ArtifactRef("index", "jsonl-metadata", "data", "array-index.jsonl"),
        ),
        splits={"train": Split({"records": ("records",), "audio_index": ("index",)})},
    )
    with Path(catalog_config["catalog"]).open("a") as stream:
        stream.write("\n" + json.dumps(spec.to_dict()))
    catalog_config["train"] = [{"dataset_id": "array", "version": "1", "split": "train"}]
    catalog_config["augmentation"] = {}
    sample = CatalogSwiftDataset(catalog_config)[0]
    waveform, _ = sf.read(BytesIO(sample["audios"][0]))
    # The original second microphone, not the 0.3 channel mean.
    np.testing.assert_allclose(waveform, np.full(sr // 2, 0.5), atol=1e-6)


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


def test_exclude_records_drops_listed_ids_and_keys_the_cache(catalog_config, tmp_path):
    catalog_config['metadata_cache'] = str(tmp_path / 'cache')
    full = CatalogSwiftDataset(catalog_config)
    ids = [full.records[i].record.id for i in range(len(full))]
    listed = tmp_path / 'excluded.txt'
    listed.write_text(ids[0] + '\n')
    catalog_config['train'][0]['exclude_records'] = str(listed)
    kept = CatalogSwiftDataset(catalog_config)
    assert [kept.records[i].record.id for i in range(len(kept))] == ids[1:]
    listed.write_text('\n'.join(ids) + '\n')
    with pytest.raises(ValueError, match='Empty source'):
        CatalogSwiftDataset(catalog_config)
    catalog_config['train'][0]['exclude_records'] = str(tmp_path / 'missing.txt')
    with pytest.raises(ValueError, match='exclude_records'):
        CatalogSwiftDataset(catalog_config)


def test_exclude_media_drops_listed_augmentation_files(catalog_config, tmp_path):
    from open_audio_llm.data.far_rir import media_path

    full = CatalogSwiftDataset(catalog_config)
    paths = [media_path(item) for item in full.rir_groups[0]]
    listed = tmp_path / 'held-out-rirs.txt'
    listed.write_text('/elsewhere/held-out.wav\n')
    catalog_config['rir_sources'][0]['exclude_media'] = str(listed)
    kept = CatalogSwiftDataset(catalog_config)
    assert [media_path(item) for item in kept.rir_groups[0]] == paths
    listed.write_text('\n'.join(paths) + '\n')
    with pytest.raises(ValueError, match='empty augmentation source'):
        CatalogSwiftDataset(catalog_config)
    catalog_config['rir_sources'][0]['exclude_media'] = str(tmp_path / 'missing.txt')
    with pytest.raises(ValueError, match='exclude_media'):
        CatalogSwiftDataset(catalog_config)


def test_noise_exclude_datasets_keeps_listed_audio_clean(catalog_config):
    catalog_config['augmentation'] = {'noise_prob': 1}
    noisy = sf.read(BytesIO(CatalogSwiftDataset(catalog_config)[0]['audios'][0]))[0]
    catalog_config['augmentation']['noise_exclude_datasets'] = ['speech']
    clean = sf.read(BytesIO(CatalogSwiftDataset(catalog_config)[0]['audios'][0]))[0]
    reference_audio = CatalogSwiftDataset(catalog_config, training=False)[0]['audios'][0]
    reference = reference_audio if isinstance(reference_audio, np.ndarray) else sf.read(BytesIO(reference_audio))[0]
    assert not np.allclose(noisy, reference, atol=1e-4)
    np.testing.assert_allclose(clean, reference, atol=1e-4)


def test_cache_identity_ignores_declared_roles_the_loader_never_reads(catalog_config, tmp_path):
    from open_audio_llm.data.catalog_cache import source_identity

    specs = [json.loads(line) for line in Path(catalog_config['catalog']).read_text().splitlines()]
    specs[0]['artifacts'].append({'name': 'punc', 'kind': 'lhotse-supervisions',
                                  'root_alias': 'data', 'relative_path': 'missing-punc.jsonl.gz'})
    specs[0]['splits']['train']['artifacts']['punctuated_supervisions'] = ['punc']
    Path(catalog_config['catalog']).write_text('\n'.join(json.dumps(s) for s in specs))
    dataset = CatalogSwiftDataset(catalog_config)
    identity = source_identity(dataset, dict(dataset.sources[0]))
    assert not any('missing-punc' in path for path, *_ in identity['files'])


def test_validation_subset_keeps_source_share_and_drops_long_audio():
    from open_audio_llm.data.catalog_dataset import _subset_counts, limit_validation_records

    assert _subset_counts([3, 2], 4) == [2, 2]
    assert _subset_counts([10, 0], 3) == [3, 0]

    class Rows:
        def __init__(self):
            self.durations = np.array([1.0, 1.0, 1.0, 400.0, 2.0, 2.0])
            self.slots = np.ones(6, dtype=np.uint16)
            self.hotword_pool = ["a"]
            self.items = [f"row{i}" for i in range(6)]

        def __getitem__(self, index):
            return self.items[index]

        def __len__(self):
            return 6

    limited, ranges = limit_validation_records(
        Rows(),
        [range(0, 4), range(4, 6)],
        {"validation_samples": 4, "validation_max_duration": 150, "seed": 42},
    )
    assert len(limited) == 4
    assert len(ranges[0]) == 2 and len(ranges[1]) == 2
    assert "row3" not in [limited[i] for i in range(len(limited))]
    assert limited.hotword_pool == ["a"]
    with pytest.raises(ValueError, match="validation_max_duration"):
        limit_validation_records(
            Rows(),
            [range(0, 4), range(4, 6)],
            {"validation_samples": 1, "validation_max_duration": 0.1, "seed": 42},
        )


def test_validation_audio_stays_a_waveform(catalog_config):
    catalog_config["validation"] = [dict(catalog_config["train"][0])]
    catalog_config["validation_samples"] = 1
    dataset = CatalogSwiftDataset(catalog_config, training=False, message_format="qwen3_asr")
    assert len(dataset) == 1
    sample = dataset[0]
    assert isinstance(sample["audios"][0], np.ndarray)
    if sample.get("mix_wav") is not None:
        assert isinstance(sample["mix_wav"], np.ndarray)


def test_train_end_stops_persistent_loader_workers():
    from torch.utils.data import DataLoader
    from open_audio_llm.integrations.ms_swift.catalog_loader import StopLoaderWorkersCallback

    loader = DataLoader(range(8), batch_size=2, num_workers=2, persistent_workers=True)
    list(loader)
    workers = loader._iterator._workers
    callback = StopLoaderWorkersCallback()
    callback.loader = loader
    # Without this, torch's per-worker atexit join waits 5 s for each worker.
    callback.on_train_end(None, None, None)
    assert workers and not any(worker.is_alive() for worker in workers)
