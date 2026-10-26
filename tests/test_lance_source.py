import hashlib
import json
from io import BytesIO

import numpy as np
import pytest
import soundfile as sf
from audio_data_contract import ArtifactRef, DatasetSpec, Split

from open_audio_llm.data.catalog_dataset import CatalogSwiftDataset

lance = pytest.importorskip("lance")
pa = pytest.importorskip("pyarrow")

SOURCE_RATE = 8000


def wav_bytes(seconds):
    buffer = BytesIO()
    samples = np.sin(np.arange(int(SOURCE_RATE * seconds)) * 0.1).astype(np.float32)
    sf.write(buffer, samples, SOURCE_RATE, format="WAV")
    return buffer.getvalue()


@pytest.fixture
def lance_config(tmp_path):
    rows = [("a", "你好", "train"), ("b", "hello", "train"), ("c", "测试集", "test")]
    count = len(rows)
    table = pa.table({
        "sample_id": [row[0] for row in rows],
        "text": [row[1] for row in rows],
        "text_kind": ["source_transcript"] * count,
        "language": ["zh", "en", "zh"],
        "speaker_id": ["spk"] * count,
        "speaker_scope": ["demo"] * count,
        "duration_seconds": [1.0, 2.0, 1.0],
        "sample_rate": pa.array([SOURCE_RATE] * count, pa.int32()),
        "channels": pa.array([1] * count, pa.int16()),
        "metadata_json": [json.dumps({"original_split": row[2]}) for row in rows],
        "audio": [{"bytes": wav_bytes(d), "path": None} for d in (1.0, 2.0, 1.0)],
    })
    release = tmp_path / "demo" / "v0.1"
    release.mkdir(parents=True)
    dataset = lance.write_dataset(table, str(release / "samples.lance"))
    dataset.create_scalar_index("sample_id", "BTREE")
    # Creating the index is a new Lance version; pin the one the release serves.
    manifest = {"status": "complete", "rows": count, "table_path": "samples.lance",
                "lance_version": lance.dataset(str(release / "samples.lance")).version}
    path = release / "manifest.json"
    path.write_text(json.dumps(manifest))
    spec = DatasetSpec(
        dataset_id="demo",
        version="tts-unified-v0.1",
        languages=("zh",),
        tasks=("tts", "asr"),
        artifacts=(ArtifactRef.from_dict({
            "name": "samples", "kind": "tts-lance-release", "root_alias": "tts",
            "relative_path": "demo/v0.1/manifest.json",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "metadata": {"lance_version": manifest["lance_version"], "rows": count},
        }),),
        splits={"train": Split({"samples": ("samples",)})},
    )
    (tmp_path / "catalog.jsonl").write_text(json.dumps(spec.to_dict()) + "\n")
    (tmp_path / "roots.json").write_text(json.dumps({"tts": str(tmp_path)}))
    return {
        "catalog": str(tmp_path / "catalog.jsonl"),
        "roots": str(tmp_path / "roots.json"),
        "sampling_rate": 16000,
        "train": [{"dataset_id": "demo", "version": "tts-unified-v0.1", "split": "train"}],
    }


@pytest.mark.parametrize("cached", [False, True])
def test_lance_release_trains_with_embedded_audio(lance_config, tmp_path, cached):
    if cached:
        lance_config["metadata_cache"] = str(tmp_path / "cache")
    dataset = CatalogSwiftDataset(lance_config)

    # The original test split is excluded by the contract's ASR read rules.
    assert len(dataset) == 2
    records = [dataset.records[i].record for i in range(2)]
    assert [r.id for r in records] == [
        "demo@tts-unified-v0.1:train:a", "demo@tts-unified-v0.1:train:b"]
    assert [s.name for s in records[1].audio_slots] == ["primary"]
    assert records[1].audio_slots[0].ref.duration == 2.0

    sample = dataset[1]
    audio, rate = sf.read(BytesIO(sample["audios"][0]))
    assert rate == 16000 and len(audio) == 2 * 16000
    assert sample["duration"] == pytest.approx(2.0)
    assert "hello" in json.dumps(sample["messages"], ensure_ascii=False)


def test_release_reads_go_through_a_bounded_cache_session(lance_config, monkeypatch):
    from open_audio_llm.data import lance_source

    created, bounded = [], lance_source.bounded_session
    monkeypatch.setattr(lance_source, "_RELEASES", {})
    monkeypatch.setattr(lance_source, "bounded_session", lambda: created.append(bounded()) or created[-1])
    dataset = CatalogSwiftDataset(lance_config)
    dataset[1]
    (release,) = lance_source._RELEASES.values()
    assert any(release.dataset.session().is_same_as(session) for session in created)

def test_lance_filter_is_pushed_into_the_scan(lance_config):
    lance_config["train"][0]["lance_filter"] = "language = 'en'"
    dataset = CatalogSwiftDataset(lance_config)
    assert [dataset.records[i].record.id for i in range(len(dataset))] == [
        "demo@tts-unified-v0.1:train:b"]


def test_lance_filter_is_rejected_for_other_sources(lance_config):
    lance_config["train"][0]["lance_filter"] = ""
    with pytest.raises(ValueError, match="lance_filter"):
        CatalogSwiftDataset(lance_config)


def test_supervision_span_at_the_start_of_a_long_file_loads_only_the_span(tmp_path):
    import numpy as np
    import soundfile as sf

    from open_audio_llm.data.lance_index import supervision_cut

    sf.write(tmp_path / "long.wav", np.zeros(160000, dtype="float32"), 16000)
    for start in (0, 16000):
        row = {"cut_id": "c", "sample_rate": 16000, "start_frame": start, "num_frames": 48000, "channels": 1, "channel": None}
        assert supervision_cut(row, tmp_path / "long.wav").load_audio().shape == (1, 48000)


def _pool(lance_config, tmp_path, ref):
    from open_audio_llm.data.interference_pool import pool_clips

    records = tmp_path / "pool-zh.jsonl"
    records.write_text(json.dumps({
        "schema_version": "audio-record/1.0", "id": f"pool:{ref['cut_id']}", "task": "ts_asr_interference",
        "audio_slots": [{"name": "primary", "ref": ref}], "target": "", "language": "zh",
        "labels": {}, "hotwords": [], "metadata": {}}) + "\n")
    pool = DatasetSpec(
        dataset_id="pool", version="v3", languages=("zh",), tasks=("ts_asr_interference",),
        artifacts=(ArtifactRef.from_dict({
            "name": "zh-records", "kind": "audio-records", "root_alias": "tts",
            "relative_path": records.name, "sha256": hashlib.sha256(records.read_bytes()).hexdigest(),
        }),),
        splits={"zh": Split({"records": ("zh-records",)}), "en": Split({"records": ("zh-records",)})},
    )
    with open(lance_config["catalog"], "a") as catalog:
        catalog.write(json.dumps(pool.to_dict()) + "\n")
    dataset = CatalogSwiftDataset(lance_config)
    return list(pool_clips(dataset.resolver, dataset.resolver.catalog.get("pool", "v3"), "zh"))


def test_a_pool_clip_naming_a_lance_sample_is_read_from_the_release(lance_config, tmp_path):
    from open_audio_llm.data.catalog_dataset import load_mono
    from open_audio_llm.data.interference_pool import LanceClip

    (clip,) = _pool(lance_config, tmp_path, {
        "dataset_id": "demo", "version": "tts-unified-v0.1", "split": "train", "cut_id": "b"})
    assert isinstance(clip, LanceClip) and clip.id == "b"
    assert load_mono(clip, 16000).shape == (2 * 16000,)


def test_a_pool_clip_naming_another_non_lance_split_is_rejected(lance_config, tmp_path):
    with pytest.raises(ValueError, match="neither the pool nor a Lance release"):
        _pool(lance_config, tmp_path, {"dataset_id": "pool", "version": "v3", "split": "en", "cut_id": "x"})
