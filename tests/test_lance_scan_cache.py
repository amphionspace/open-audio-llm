from types import SimpleNamespace

import numpy as np

from open_audio_llm.data import lance_index


class Probe(lance_index.LanceSourceIndex):
    def __init__(self, dataset, source, spec, scan):
        self.scan = scan


class InlinePool:
    def __init__(self, processes, initializer=None, initargs=()):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def map(self, function, items, chunksize=1):
        return [function(item) for item in items]


def _dataset(tmp_path, sha="index-a", cache=True, recipe_parameters=None):
    artifact = SimpleNamespace(sha256=sha)
    spec = SimpleNamespace(dataset_id="demo", version="v1", key="demo@v1",
                           splits={"train": SimpleNamespace(artifacts={lance_index.ROLE: ("records",)})},
                           artifact=lambda name: artifact, recipe_parameters=recipe_parameters or {})
    catalog = SimpleNamespace(get=lambda dataset_id, version: spec)
    resolver = SimpleNamespace(catalog=catalog, catalog_path="catalog", roots_path="roots")
    return SimpleNamespace(resolver=resolver, augmentation=SimpleNamespace(speed_prob=0.0),
                           metadata_cache=str(tmp_path / "cache") if cache else None)


def _open(dataset, sources, monkeypatch, scanned):
    def scan(request):
        scanned.append(request[3])
        return {"positions": np.arange(3), "duration_sum": np.ones(3), "split": request[3]}

    monkeypatch.setattr(lance_index, "_scan_one", scan)
    monkeypatch.setattr(lance_index.multiprocessing, "get_context", lambda method: SimpleNamespace(Pool=InlinePool))
    return Probe.open_many(dataset, sources)


def test_a_cached_scan_is_reused_until_the_index_or_a_filter_changes(tmp_path, monkeypatch):
    (tmp_path / "cache").mkdir()
    excluded = tmp_path / "excluded.txt"
    excluded.write_text("x\n")
    sources = [{"dataset_id": "demo", "version": "v1", "split": "train", "exclude_records": str(excluded)}]
    scanned = []
    first = _open(_dataset(tmp_path), sources, monkeypatch, scanned)
    second = _open(_dataset(tmp_path), sources, monkeypatch, scanned)
    assert scanned == ["train"]
    np.testing.assert_array_equal(second[0].scan["positions"], first[0].scan["positions"])

    excluded.write_text("x\ny\n")
    _open(_dataset(tmp_path), sources, monkeypatch, scanned)
    _open(_dataset(tmp_path, sha="index-b"), sources, monkeypatch, scanned)
    assert scanned == ["train"] * 3
    assert len(list((tmp_path / "cache").glob("lance-scan-*.pkl"))) == 3
    assert not list((tmp_path / "cache").glob("*.tmp"))


def test_without_a_metadata_cache_every_open_scans(tmp_path, monkeypatch):
    sources = [{"dataset_id": "demo", "version": "v1", "split": "train"}]
    scanned = []
    _open(_dataset(tmp_path, cache=False), sources, monkeypatch, scanned)
    _open(_dataset(tmp_path, cache=False), sources, monkeypatch, scanned)
    assert scanned == ["train", "train"]
    assert not (tmp_path / "cache").exists()


def test_require_clean_pass_uses_the_index_only_for_a_clean_pass_only_version(tmp_path, monkeypatch):
    sources = [{"dataset_id": "demo", "version": "v1", "split": "train", "require_clean_pass": True}]
    scanned = []
    found = _open(_dataset(tmp_path, cache=False), sources, monkeypatch, scanned)
    assert found == [None] and scanned == []  # markers are not in the index: keep the JSONL cache
    for declared in ("true", 1, False):
        found = _open(_dataset(tmp_path, cache=False, recipe_parameters={"clean_pass_only": declared}),
                      sources, monkeypatch, scanned)
        assert found == [None] and scanned == []
    found = _open(_dataset(tmp_path, cache=False, recipe_parameters={"clean_pass_only": True}),
                  sources, monkeypatch, scanned)
    assert found[0] is not None and scanned == ["train"]
