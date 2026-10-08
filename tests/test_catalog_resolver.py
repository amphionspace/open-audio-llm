import json

import numpy as np
import soundfile as sf
from audio_data_contract import ArtifactRef, AudioRef, DatasetSpec, Split
from lhotse import CutSet, MonoCut, Recording

from open_audio_llm.data.catalog_resolver import LhotseCatalogAudioResolver


def test_catalog_resolver_reads_exact_cut_without_writing(tmp_path):
    audio_path = tmp_path / "audio.wav"
    sf.write(audio_path, np.arange(16000, dtype=np.float32) / 16000, 16000)
    recording = Recording.from_file(audio_path, recording_id="rec")
    cut = MonoCut(
        id="cut",
        start=0.25,
        duration=0.5,
        channel=0,
        recording=recording,
    )
    cuts_path = tmp_path / "manifests/cuts.jsonl.gz"
    cuts_path.parent.mkdir(parents=True)
    CutSet.from_cuts([cut]).to_file(cuts_path)

    spec = DatasetSpec(
        dataset_id="demo",
        version="1.0",
        languages=("en",),
        tasks=("asr",),
        artifacts=(
            ArtifactRef(
                "cuts",
                "lhotse-cuts",
                "data",
                "manifests/cuts.jsonl.gz",
            ),
        ),
        splits={"train": Split({"cuts": ("cuts",)})},
    )
    catalog_path = tmp_path / "catalog.jsonl"
    catalog_path.write_text(json.dumps(spec.to_dict()) + "\n", encoding="utf-8")
    roots_path = tmp_path / "roots.json"
    roots_path.write_text(json.dumps({"data": str(tmp_path)}), encoding="utf-8")

    resolver = LhotseCatalogAudioResolver(
        catalog_path,
        roots_path,
    )
    before = set(tmp_path.rglob("*"))
    resolved = resolver.get_cut(AudioRef("demo", "1.0", "train", "cut"))
    assert resolved.sampling_rate == 16000
    assert resolved.load_audio().shape == (1, 8000)
    segment = resolver.get_cut(
        AudioRef("demo", "1.0", "train", "cut", start=0.1, duration=0.2)
    )
    np.testing.assert_allclose(
        segment.load_audio(), resolved.load_audio()[:, 1600:4800]
    )
    assert set(tmp_path.rglob("*")) == before
