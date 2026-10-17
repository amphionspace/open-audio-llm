"""Sharded AudioRecord streams from a pinned Lance snapshot, for training loaders.

Each shard reads whole fragments, so shards differ in size, more so under selective
queries; distributed training should not assume equal batch counts per rank.

    class Records(torch.utils.data.IterableDataset):
        def __iter__(self):
            shard, shards = worker_shard(rank, world_size)
            yield from iter_records(artifact, query, shard=shard, num_shards=shards,
                                    seed=seed, epoch=self.epoch)
"""

from __future__ import annotations

import json
import random

from .errors import ContractError
from .lance import RecordQuery, open_artifact
from .types import AudioRecord


def worker_shard(rank=0, world_size=1):
    """(shard, num_shards) for this process, split further across DataLoader workers."""
    try:
        from torch.utils.data import get_worker_info
    except ImportError:
        info = None
    else:
        info = get_worker_info()
    workers, worker = (info.num_workers, info.id) if info else (1, 0)
    return rank * workers + worker, world_size * workers


def iter_records(
    artifact,
    query=None,
    *,
    shard=0,
    num_shards=1,
    seed=0,
    epoch=0,
    batch_size=4096,
):
    """Yield this shard's records; fragment order is reshuffled per (seed, epoch)."""
    if not 0 <= shard < num_shards:
        raise ContractError("shard must be in [0, num_shards)")
    dataset = open_artifact(artifact)
    fragments = sorted(dataset.get_fragments(), key=lambda f: f.fragment_id)
    random.Random(f"{seed}:{epoch}").shuffle(fragments)
    expression = (query or RecordQuery()).expression()
    for fragment in fragments[shard::num_shards]:
        for batch in fragment.to_batches(
            columns=["record_json"], filter=expression, batch_size=batch_size
        ):
            for value in batch.column(0).to_pylist():
                yield AudioRecord.from_dict(json.loads(value))
