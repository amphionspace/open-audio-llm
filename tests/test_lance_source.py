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
    rows = [("a", "你好", "train"), ("b", "世界", "train"), ("c", "测试集", "test")]
    count = len(rows)
    table = pa.table({
        "sample_id": [row[0] for row in rows],
        "text": [row[1] for row in rows],
        "text_kind": ["source_transcript"] * count,
        "language": ["zh"] * count,
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
    assert "世界" in json.dumps(sample["messages"], ensure_ascii=False)
