import json
import random
from types import SimpleNamespace

import numpy as np
import pytest

from open_audio_llm.data.lookalike import (
    LookalikeTable,
    item_key,
    lookalike_loader,
    pool_clip_key,
)
from open_audio_llm.data.ts_online import OnlineMixConfig


def _table(tmp_path, enroll_ids=("e1", "e2"), neighbors=((1, 0, -1), (-1, -1, -1))):
    path = tmp_path / "table.npz"
    neighbors = np.array(neighbors, dtype=np.int32)
    np.savez(path, enroll_ids=np.array(enroll_ids), pool_ids=np.array(["p0", "p1", "p2"]), neighbors=neighbors,
             cosines=np.zeros(neighbors.shape, dtype=np.float16), meta=np.array(json.dumps({"top_k": 3})))
    return LookalikeTable(path)


def test_candidates_are_the_listed_neighbors_without_padding(tmp_path):
    table = _table(tmp_path)
    assert table.candidates("e1").tolist() == [1, 0]
    assert table.candidates("e2").size == 0
    assert table.candidates("missing").size == 0


def test_unsorted_enrollments_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="sorted"):
        _table(tmp_path, enroll_ids=("e2", "e1"))


def test_loader_draws_lookalikes_with_the_given_probability(tmp_path):
    table = _table(tmp_path)
    items = {"p0": "clip-p0", "p1": "clip-p1", "p2": "clip-p2"}
    base = lambda rng, sr: "random"
    load = lambda item, sr: item
    loader = lookalike_loader(base, table, items, "e1", 0.5, load)
    drawn = [loader(random.Random(seed), 16000) for seed in range(2000)]
    assert set(drawn) == {"random", "clip-p0", "clip-p1"}  # never p2, which is not a neighbour of e1
    assert drawn.count("random") / len(drawn) == pytest.approx(0.5, abs=0.05)


def test_loader_falls_back_without_lookalikes(tmp_path):
    table = _table(tmp_path)
    base = lambda rng, sr: "random"
    assert lookalike_loader(base, table, {"p0": 1}, "e2", 1.0, None) is base  # no neighbours listed
    assert lookalike_loader(base, table, {}, "e1", 1.0, None) is base  # neighbours not in this pool
    assert lookalike_loader(base, table, {"p0": 1}, "e1", 0.0, None) is base


def test_lookalike_prob_is_validated():
    assert OnlineMixConfig(lookalike_prob=0.3).lookalike_prob == 0.3
    with pytest.raises(ValueError, match="lookalike_prob"):
        OnlineMixConfig(lookalike_prob=1.5)


def test_pool_clip_key_tells_spans_of_one_cut_apart():
    assert pool_clip_key("c") == "c"
    assert pool_clip_key("c", 1.5, 2.0) != pool_clip_key("c", 4.0, 2.0)
    assert item_key(SimpleNamespace(id="c", start=1.5, duration=2.0)) == pool_clip_key("c", 1.5, 2.0)
    assert item_key(SimpleNamespace(id="c")) == "c"  # a LanceClip has no span


def test_duplicate_pool_keys_are_rejected(tmp_path):
    path = tmp_path / "table.npz"
    np.savez(path, enroll_ids=np.array(["e1"]), pool_ids=np.array(["p0", "p0"]), neighbors=np.zeros((1, 2), np.int32),
             cosines=np.zeros((1, 2), np.float16), meta=np.array("{}"))
    with pytest.raises(ValueError, match="unique"):
        LookalikeTable(path)
