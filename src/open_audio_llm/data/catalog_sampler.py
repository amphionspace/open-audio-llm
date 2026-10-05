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
            "batch_merge",
            "merge_window",
        }
        if unknown:
            raise ValueError(f"Unknown batching options: {sorted(unknown)}")
        self.batch_merge = self.options.get("batch_merge", 1)
        self.merge_window = self.options.get("merge_window", self.batch_merge)
        self._replay_pools = {}
        self.max_duration = self.options.get("max_duration")
        self.max_samples = self.options.get(
            "max_samples", None if self.max_duration else batch_size
        )
        self.num_buckets = self.options.get("num_buckets", 30)
        self.buffer_size = self.options.get("buffer_size", 10000)
        for name, value in (
            ("batch_merge", self.batch_merge),
            ("merge_window", self.merge_window),
            ("max_samples", self.max_samples),
            ("num_buckets", self.num_buckets),
            ("buffer_size", self.buffer_size),
        ):
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"batching.{name} must be a positive integer")
        if self.merge_window % self.batch_merge:
            raise ValueError("batching.merge_window must be divisible by batch_merge")
        if self.max_duration is not None and (
            not math.isfinite(self.max_duration) or self.max_duration <= 0
        ):
            raise ValueError("batching.max_duration must be positive and finite")
        self.drop_last = self.options.get("drop_last", False)
        self.replay = dataset.config.get("replay")
        self.full_coverage = False
        if self.replay is not None:
            self.replay = dict(self.replay)
            if set(self.replay) != {"epoch_samples", "window_samples"}:
                raise ValueError("replay requires epoch_samples and window_samples")
            self.full_coverage = self.replay["epoch_samples"] == "full_coverage"
            if self.full_coverage and self.drop_last:
                raise ValueError("full_coverage requires batching.drop_last=false")
            for name, value in self.replay.items():
                if name == "epoch_samples" and self.full_coverage:
                    continue
                if type(value) is not int or value <= 0:
                    raise ValueError(f"replay.{name} must be a positive integer"
                                     " (epoch_samples also accepts full_coverage)")
            window = self.replay["window_samples"]
            if not self.full_coverage and self.replay["epoch_samples"] % window:
                raise ValueError("replay.epoch_samples must be divisible by window_samples")
            weights = []
            for source in dataset.sources:
                if any(k in source for k in ("samples", "reps", "shard_rotation")):
                    raise ValueError("replay uses weight, not samples/reps/shard_rotation")
                if self.full_coverage and "max_samples" in source:
                    raise ValueError("full_coverage cannot use source max_samples")
                weight = source.get("weight", 0)
                if not math.isfinite(weight) or weight <= 0:
                    raise ValueError("replay requires positive weight for every source")
                weights.append(weight)
            exact = [window * w / sum(weights) for w in weights]
            self.quotas = [math.floor(q) for q in exact]
            order = sorted(range(len(exact)), key=lambda i: exact[i] - self.quotas[i], reverse=True)
            for i in order[:window - sum(self.quotas)]:
                self.quotas[i] += 1
            if min(self.quotas) == 0:
                raise ValueError("Increase replay.window_samples to include every source")
            if self.full_coverage:
                # Cover the slowest source; smaller sources cycle without replacement.
                windows = max((len(indices) + quota - 1) // quota
                              for indices, quota in zip(dataset.source_ranges, self.quotas))
                self.replay["epoch_samples"] = windows * window
            # Each bucketing window must retain the configured source quotas.
            self.buffer_size = window
        indexed = hasattr(dataset.records, "signature")
        if indexed:
            self.durations = dataset.records.durations
            self.slots = dataset.records.slots
            self._max_duration = float(self.durations.max())
        else:
            from .catalog_cache import sampling_cost

            self.durations, self.slots = [], []
            for row in dataset.records:
                total, slots = sampling_cost(row, dataset)
                if total <= 0 or not math.isfinite(total):
                    raise ValueError(f"Invalid sampling duration: {row.record.id}")
                self.durations.append(total)
                self.slots.append(slots)
            self._max_duration = max(self.durations)
        if self.max_duration and self._max_duration > self.max_duration:
            raise ValueError("augmented duration exceeds batching.max_duration")
        config = {k: v for k, v in dataset.config.items() if k != "metadata_cache"}
        if "batching" in config:
            config["batching"] = {k: v for k, v in config["batching"].items() if k not in {"batch_merge", "merge_window"}}
        identity = {
            "config": config,
            "batch_size": batch_size,
            "shuffle": shuffle,
            "world_size": world_size,
        }
        if indexed:
            self.signature = dataset.records.signature(identity)
        else:
            identity["records"] = [
                (r.record.id, d, s)
                for r, d, s in zip(dataset.records, self.durations, self.slots)
            ]
            self.signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
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
        if self.replay is not None:
            yield from self._replay_mux(rng)
            return
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

    def _replay_mux(self, rng):
        windows = self.replay["epoch_samples"] // self.replay["window_samples"]
        positions = [self.epoch * windows * quota for quota in self.quotas]
        if self.full_coverage:
            # Start each data epoch at a fresh permutation. A tail of one shuffle
            # plus a prefix of another need not cover all records, even at N draws.
            positions = [self.epoch * ((windows * quota + len(indices) - 1) // len(indices))
                         * len(indices)
                         for indices, quota in zip(self.dataset.source_ranges, self.quotas)]
        occurrence = 0
        for _ in range(windows):
            window = []
            for i, (indexes, quota) in enumerate(zip(self.dataset.source_ranges, self.quotas)):
                for _ in range(quota):
                    cycle, offset = divmod(positions[i], len(indexes))
                    cached_cycle, pool = self._replay_pools.get(i, (None, None))
                    if cached_cycle != cycle:
                        pool = self._replay_pool(i, cycle, indexes)
                        self._replay_pools[i] = (cycle, pool)
                    window.append((self.epoch, int(pool[offset]), occurrence))
                    positions[i] += 1
                    occurrence += 1
            if self.shuffle:
                rng.shuffle(window)
            yield from window

    def _replay_pool(self, source, cycle, indexes):
        def build():
            pool = list(indexes)
            if self.shuffle:
                random.Random(f"{self.dataset.seed}:replay:{source}:{cycle}").shuffle(pool)
            return pool

        root = getattr(self.dataset.records, "root", None)
        if root is None:
            return build()
        import os

        import numpy as np

        from .catalog_cache import lock_exclusive

        key = hashlib.sha256(json.dumps([
            self.dataset.seed, source, cycle, self.shuffle, indexes.start, indexes.stop,
        ]).encode()).hexdigest()
        path = root / f"permutation-{key}.npy"
        with path.with_suffix(".lock").open("a") as lock:
            lock_exclusive(lock)
            if not path.exists():
                with path.with_suffix(".tmp").open("wb") as stream:
                    np.save(stream, np.asarray(build(), dtype=np.int64))
                os.replace(path.with_suffix(".tmp"), path)
        return np.load(path, mmap_mode="r")

    def _batches(self):
        if self._plan is not None:
            return self._plan
        rng = random.Random(f"{self.dataset.seed}:{self.epoch}:mux")
        stream = iter(self._mux(rng))
        result = []
        scale = self._max_duration / self.num_buckets
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
            if self.replay is not None and self.shuffle:
                # Bucketing otherwise turns short ASR and long TS into task blocks.
                rng.shuffle(chunk)
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
        batches = self._batches()
        for start in range(self.consumed, len(batches), self.merge_window):
            window = batches[start:start + self.merge_window]
            if self.merge_window > self.batch_merge:
                # Pair similar durations within an optimizer update, preserving
                # its complete sample set and the original checkpoint cursor.
                window = sorted(
                    window,
                    key=lambda batch: max(self.durations[key[1]] for key in batch),
                )
            for offset in range(0, len(window), self.batch_merge):
                yield [key for batch in window[offset:offset + self.batch_merge] for key in batch]

    def __len__(self):
        return math.ceil((len(self._batches()) - self._start) / self.batch_merge)

    def mark_consumed(self):
        size = len(self._batches())
        if self.consumed >= size:
            raise RuntimeError("Consumed more batches than the epoch plan")
        # Persist the original microbatch cursor, independent of batch merging.
        self.consumed = min(size, self.consumed + self.batch_merge)

    def state_dict(self):
        if (
            self.merge_window > self.batch_merge
            and self.consumed != len(self._batches())
            and (self.consumed - self._start) % self.merge_window
        ):
            raise ValueError("Save merged sampling state at an optimizer-update boundary")
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
