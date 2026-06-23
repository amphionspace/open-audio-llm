"""Dataset mux and duration-aware batching utilities."""

from __future__ import annotations

from collections import defaultdict
import random
from typing import Iterable, Iterator

from torch.utils.data import Sampler


class WeightedMuxSampler(Sampler[int]):
    """Sample indexes according to per-dataset weights."""

    def __init__(self, records, weights: dict[str, float] | None = None, seed: int = 42):
        self.records = records
        self.weights = weights or {}
        self.seed = seed
        self.by_dataset: dict[str, list[int]] = defaultdict(list)
        for idx, record in enumerate(records):
            self.by_dataset[record.dataset_id].append(idx)

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed)
        keys = list(self.by_dataset)
        probs = [float(self.weights.get(k, len(self.by_dataset[k]))) for k in keys]
        for _ in range(len(self.records)):
            key = rng.choices(keys, weights=probs, k=1)[0]
            yield rng.choice(self.by_dataset[key])

    def __len__(self) -> int:
        return len(self.records)


def duration_buckets(records, bucket_size: float = 2.0) -> dict[int, list[int]]:
    buckets: dict[int, list[int]] = defaultdict(list)
    for idx, record in enumerate(records):
        duration = record.duration or 0.0
        buckets[int(duration // bucket_size)].append(idx)
    return dict(buckets)


def batch_by_duration(
    indexes: Iterable[int],
    records,
    max_duration: float,
) -> Iterator[list[int]]:
    batch, total = [], 0.0
    for idx in indexes:
        duration = float(records[idx].duration or 0.0)
        if batch and total + duration > max_duration:
            yield batch
            batch, total = [], 0.0
        batch.append(idx)
        total += duration
    if batch:
        yield batch
