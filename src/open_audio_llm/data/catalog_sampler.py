"""Finite source mux with duration/slot batching and a trained-batch cursor."""

from __future__ import annotations

import hashlib
import json
import math
import random
from itertools import islice

from torch.utils.data import Sampler


class CatalogBatchSampler(Sampler):
    def __init__(self, dataset, *, batch_size=1, rank=0, world_size=1, shuffle=True):
        self.dataset = dataset
        self.rank, self.world_size = rank, world_size
        self.shuffle = shuffle
        self.options = dataset.config.get("batching", {})
        unknown = set(self.options) - {
            "max_duration",
            "max_samples",
            "num_buckets",
            "buffer_size",
            "drop_last",
        }
        if unknown:
            raise ValueError(f"Unknown batching options: {sorted(unknown)}")
        self.max_duration = self.options.get("max_duration")
        self.max_samples = self.options.get(
            "max_samples", None if self.max_duration else batch_size
        )
        self.num_buckets = self.options.get("num_buckets", 30)
        self.buffer_size = self.options.get("buffer_size", 10000)
        for name, value in (
            ("max_samples", self.max_samples),
            ("num_buckets", self.num_buckets),
            ("buffer_size", self.buffer_size),
        ):
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"batching.{name} must be a positive integer")
        if self.max_duration is not None and (
            not math.isfinite(self.max_duration) or self.max_duration <= 0
        ):
            raise ValueError("batching.max_duration must be positive and finite")
        self.drop_last = self.options.get("drop_last", False)
        speed = dataset.augmentation
        slowest = min(1.0, *speed.speed_factors) if speed.speed_prob else 1.0
        self.durations, self.slots = [], []
        for row in dataset.records:
            total = 0.0
            for slot in row.record.audio_slots:
                duration = slot.ref.duration
                if duration is None:
                    duration = dataset.resolver.get_cut(slot.ref).duration
                total += duration / slowest + 1 / dataset.sampling_rate
            if total <= 0 or not math.isfinite(total):
                raise ValueError(f"Invalid sampling duration: {row.record.id}")
            if self.max_duration and total > self.max_duration:
                raise ValueError(
                    f"{row.record.id}: augmented duration {total:.3f}s exceeds batching.max_duration"
                )
            self.durations.append(total)
            self.slots.append(len(row.record.audio_slots))
        identity = {
            "config": dataset.config,
            "batch_size": batch_size,
            "shuffle": shuffle,
            "world_size": world_size,
            "records": [
                (r.record.id, d, s)
                for r, d, s in zip(dataset.records, self.durations, self.slots)
            ],
        }
        self.signature = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest()
        self.epoch, self.consumed = 0, 0
        self._start = 0
        self._plan = None
        self._resume_epoch = None
        self._epoch_offset = 0

    def set_epoch(self, epoch):
        # HF infers loop epochs from the initial loader length. Restore the
        # actual data epoch even when duration batching changes that length.
        if self._resume_epoch is not None:
            self._epoch_offset = self._resume_epoch - epoch
            self._resume_epoch = None
        actual = epoch + self._epoch_offset
        if actual != self.epoch:
            self.epoch, self.consumed, self._plan = actual, 0, None
            self._start = 0
        self.dataset.set_epoch(actual)

    def _mux(self, rng):
        pools, weights = [], []
        for source, indexes in zip(self.dataset.sources, self.dataset.source_ranges):
            indexes = list(indexes)
            cap = min(source.get("samples", len(indexes)), len(indexes))
            if source.get("shard_rotation", False):
                if "samples" not in source:
                    raise ValueError("shard_rotation requires samples")
                # Logical shards avoid offline manifests; wrap the last shard
                # to preserve the epoch quota while covering the full source.
                start = (self.epoch % math.ceil(len(indexes) / cap)) * cap
                indexes = [indexes[(start + i) % len(indexes)] for i in range(cap)]
            else:
                indexes = indexes[:cap]
            indexes = indexes * source.get("reps", 1)
            if self.shuffle:
                rng.shuffle(indexes)
            pools.append(indexes)
            weights.append(source.get("weight", len(indexes)))
        active = list(range(len(pools)))
        occurrence = 0
        while active:
            source = rng.choices(active, weights=[weights[i] for i in active], k=1)[0]
            idx = pools[source].pop()
            # Explicit epoch survives worker prefetch; repetitions get distinct
            # augmentation seeds, reproducible after restoring the cursor.
            yield (self.epoch, idx, occurrence)
            occurrence += 1
            if not pools[source]:
                active.remove(source)

    def _batches(self):
        if self._plan is not None:
            return self._plan
        rng = random.Random(f"{self.dataset.seed}:{self.epoch}:mux")
        stream = iter(self._mux(rng))
        result = []
        scale = max(self.durations) / self.num_buckets
        while buffer := list(islice(stream, self.buffer_size)):
            buckets = {}
            for key in buffer:
                idx = key[1]
                bucket = (
                    self.slots[idx],
                    min(self.num_buckets - 1, int(self.durations[idx] / scale)),
                )
                buckets.setdefault(bucket, []).append(key)
            chunk = []
            for keys in buckets.values():
                batch, duration = [], 0.0
                for key in keys:
                    cost = self.durations[key[1]]
                    if batch and (
                        (self.max_samples and len(batch) >= self.max_samples)
                        or (self.max_duration and duration + cost > self.max_duration)
                    ):
                        chunk.append(batch)
                        batch, duration = [], 0.0
                    batch.append(key)
                    duration += cost
                if batch:
                    chunk.append(batch)
            result.extend(chunk)
        remainder = len(result) % self.world_size
        if remainder:
            if self.drop_last:
                result = result[:-remainder]
            else:
                result.extend(
                    [
                        result[i % len(result)]
                        for i in range(self.world_size - remainder)
                    ]
                )
        if not result:
            raise ValueError(
                "No complete distributed batch; disable batching.drop_last or select more samples"
            )
        self._plan = result[self.rank :: self.world_size]
        return self._plan

    def __iter__(self):
        yield from self._batches()[self.consumed :]

    def __len__(self):
        return len(self._batches()) - self._start

    def mark_consumed(self):
        self.consumed += 1
        if self.consumed > len(self._batches()):
            raise RuntimeError("Consumed more batches than the epoch plan")

    def state_dict(self):
        return {
            "signature": self.signature,
            "epoch": self.epoch,
            "consumed": self.consumed,
        }

    def load_state_dict(self, state):
        if state["signature"] != self.signature:
            raise ValueError(
                "Cannot resume Catalog sampler: recipe, records, batching or world size changed"
            )
        self.epoch, self.consumed = state["epoch"], state["consumed"]
        self._plan = None
        if not 0 <= self.consumed <= len(self._batches()):
            raise ValueError("Invalid Catalog sampler checkpoint cursor")
        if self.consumed == len(self._batches()):
            self.epoch, self.consumed, self._plan = self.epoch + 1, 0, None
        self._resume_epoch = self.epoch
        self._start = self.consumed
        self.dataset.set_epoch(self.epoch)
