import gzip
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from open_audio_llm.data.catalog_dataset import CatalogSwiftDataset


def _list(tmp_path):
    (tmp_path / "audio").mkdir()
    sf.write(tmp_path / "audio/c.flac", np.full(8000, 0.25, np.float32), 16000)
    with gzip.open(tmp_path / "utterances.jsonl.gz", "wt") as handle:
        handle.write(json.dumps({"speaker": "aishellmix:C0001", "path": "audio/c.flac"}) + "\n")
    return {"list": str(tmp_path / "utterances.jsonl.gz"), "prob": 1.0}


def _record(source="C0001_0003", dataset="aishellmix"):
    slot = SimpleNamespace(ref=SimpleNamespace(dataset_id=dataset))
    return SimpleNamespace(metadata={"source_record_id": source} if source else {}, slot=lambda name: slot)


def _dataset(spec):
    return SimpleNamespace(tts_enrollments=CatalogSwiftDataset._load_tts_enrollments(spec), sampling_rate=16000)


def test_enrollment_becomes_the_speakers_tts_clip_without_touching_the_sample_stream(tmp_path):
    dataset, rng = _dataset(_list(tmp_path)), random.Random(5)
    state = rng.getstate()
    out = CatalogSwiftDataset._tts_enrollment(dataset, _record(), np.zeros(16000, np.float32), rng)
    assert out.shape == (8000,) and np.allclose(out, 0.25, atol=1e-3)
    assert rng.getstate() == state


def test_no_swap_without_a_clip_a_source_id_or_the_option(tmp_path):
    enrollment = np.zeros(16000, np.float32)
    dataset = _dataset(_list(tmp_path))
    for record in (_record(source="C0002_0001"), _record(dataset="librimix"), _record(source=None)):
        assert CatalogSwiftDataset._tts_enrollment(dataset, record, enrollment, random.Random(0)) is enrollment
    assert CatalogSwiftDataset._tts_enrollment(_dataset(None), _record(), enrollment, random.Random(0)) is enrollment


def test_prob_is_validated(tmp_path):
    with pytest.raises(ValueError, match="prob"):
        CatalogSwiftDataset._load_tts_enrollments({**_list(tmp_path), "prob": 1.5})
