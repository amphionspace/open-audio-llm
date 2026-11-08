"""Look-alike interferers for online target-speaker mixing.

False alarms of the TS-ASR model on target-absent windows are mostly the interferer's own words
when the interferer sounds like the enrolled speaker (docs/experiments/2026-11-07-whitebox, task
09). Random interferers from the pool rarely sound alike, so a table built offline
(``scripts/build_lookalike_table.py``) lists, per enrollment clip, the pool clips whose speaker
embedding is nearest to it while staying below a same-speaker cosine, and online mixing draws
interferers from that list with ``OnlineMixConfig.lookalike_prob``.

Table (``.npz``): ``enroll_ids`` (sorted str), ``pool_ids`` (str, ``pool_clip_key``), ``neighbors`` (int32,
enrollments x K, indices into ``pool_ids``, -1 pads), ``cosines`` (float16, same shape), and
``meta`` (JSON str: embedding model, K, cosine cap, pool and record splits).
"""

from __future__ import annotations

import json

import numpy as np


def pool_clip_key(cut_id, start=None, duration=None):
    """Key of one pool clip. A pool can hold several spans of one source cut, so the cut id alone
    is not unique; spans add their start and duration."""
    if start is None and duration is None:
        return cut_id
    return f"{cut_id}@{start}+{duration}"


def item_key(item):
    """``pool_clip_key`` of a pool item as the training loader holds it (PoolClip, LanceClip, cut)."""
    return pool_clip_key(item.id, getattr(item, "start", None), getattr(item, "duration", None))


class LookalikeTable:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as data:
            self.enroll_ids = data["enroll_ids"]
            self.pool_ids = data["pool_ids"]
            self.neighbors = data["neighbors"]
            self.cosines = data["cosines"]
            self.meta = json.loads(str(data["meta"]))
        if self.neighbors.shape != self.cosines.shape or len(self.neighbors) != len(self.enroll_ids):
            raise ValueError(f"{path}: neighbors, cosines and enroll_ids disagree in shape")
        if not np.all(self.enroll_ids[:-1] < self.enroll_ids[1:]):
            raise ValueError(f"{path}: enroll_ids must be sorted and unique")
        if len(np.unique(self.pool_ids)) != len(self.pool_ids):
            raise ValueError(f"{path}: pool_ids must be unique (pool_clip_key)")

    def candidates(self, enroll_id):
        """Pool indices of ``enroll_id``'s look-alikes, nearest first; empty if it has none."""
        row = int(np.searchsorted(self.enroll_ids, enroll_id))
        if row >= len(self.enroll_ids) or self.enroll_ids[row] != enroll_id:
            return np.zeros(0, dtype=np.int32)
        found = self.neighbors[row]
        return found[found >= 0]


def lookalike_loader(base, table, items, enroll_id, prob, load):
    """A speech loader that, with ``prob``, draws one of ``enroll_id``'s look-alikes instead.

    ``base(rng, sr)`` is the ordinary random interferer loader; ``items`` maps a ``pool_clip_key`` to the
    pool item ``load(item, sr)`` reads. Enrollments missing from the table fall back to ``base``.
    """
    if not prob:
        return base
    found = [items[pool_id] for pool_id in table.pool_ids[table.candidates(enroll_id)] if pool_id in items]
    if not found:
        return base

    def draw(rng, sampling_rate):
        if rng.random() < prob:
            return load(found[rng.randrange(len(found))], sampling_rate)
        return base(rng, sampling_rate)

    return draw
