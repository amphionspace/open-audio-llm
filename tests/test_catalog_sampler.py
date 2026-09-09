from collections import Counter
from types import SimpleNamespace

import pytest

from open_audio_llm.data.augment import AugmentConfig
from open_audio_llm.data.catalog_sampler import CatalogBatchSampler


def dataset(sources, sizes, durations=None, slots=None, batching=None):
    total = sum(sizes)
    durations = durations or [1.0] * total
    slots = slots or [1] * total
    rows = [
        SimpleNamespace(
            record=SimpleNamespace(
                id=str(i),
                audio_slots=[
                    SimpleNamespace(ref=SimpleNamespace(duration=durations[i]))
                ]
                * slots[i],
            )
        )
        for i in range(total)
    ]
    ranges, start = [], 0
    for size in sizes:
        ranges.append(range(start, start + size))
        start += size
    ds = SimpleNamespace(
        config={"train": sources, "batching": batching or {}},
        sources=sources,
        source_ranges=ranges,
        records=rows,
        seed=42,
        sampling_rate=16000,
        augmentation=AugmentConfig(),
        epoch=0,
    )
    ds.set_epoch = lambda epoch: setattr(ds, "epoch", epoch)
    return ds


def indexes(sampler):
    return [key[1] for batch in sampler for key in batch]


def test_samples_reps_and_rotation_cover_source():
    ds = dataset(
        [{"samples": 3, "reps": 2, "shard_rotation": True}, {"samples": 2}], [8, 4]
    )
    sampler = CatalogBatchSampler(ds)
    seen = set()
    for epoch in range(3):
        sampler.set_epoch(epoch)
        ids = indexes(sampler)
        assert len(ids) == 8
        assert sum(i < 8 for i in ids) == 6
        assert all(n == 2 for i, n in Counter(ids).items() if i < 8)
        seen.update(i for i in ids if i < 8)
    assert seen == set(range(8))


def test_weight_controls_prefix_but_mux_exhausts_all_sources():
    ds = dataset(
        [{"weight": 1000}, {"weight": 1}], [100, 100], batching={"num_buckets": 1}
    )
    ids = indexes(CatalogBatchSampler(ds))
    assert sum(i < 100 for i in ids[:50]) >= 48
    assert sorted(ids) == list(range(200))


def test_duration_slots_and_distributed_partition():
    ds = dataset(
        [{}],
        [24],
        durations=[0.2, 0.8, 1.8] * 8,
        slots=[1, 2] * 12,
        batching={"max_duration": 4.1, "num_buckets": 2},
    )
    samplers = [CatalogBatchSampler(ds, rank=r, world_size=2) for r in range(2)]
    assert len(samplers[0]) == len(samplers[1])
    all_batches = [batch for sampler in samplers for batch in sampler]
    assert len({len(batch) for batch in all_batches}) > 1
    for batch in all_batches:
        assert len({samplers[0].slots[k[1]] for k in batch}) == 1
        assert sum(samplers[0].durations[k[1]] for k in batch) <= 4.1
    assert set(indexes(samplers[0]) + indexes(samplers[1])) == set(range(24))


def test_resume_ignores_prefetch_and_maps_actual_epoch():
    ds = dataset([{"samples": 4, "shard_rotation": True}], [12])
    sampler = CatalogBatchSampler(ds)
    sampler.set_epoch(2)
    plan = list(sampler)  # Simulates workers prefetching the entire epoch.
    sampler.mark_consumed()
    state = sampler.state_dict()
    restored = CatalogBatchSampler(ds)
    restored.load_state_dict(state)
    restored.set_epoch(9)  # HF's inferred epoch need not match the data epoch.
    assert restored.epoch == 2
    assert list(restored) == plan[1:]
    assert len(restored) == 3
    restored.set_epoch(10)
    sampler.set_epoch(3)
    assert list(restored) == list(sampler)
    changed = CatalogBatchSampler(ds, world_size=2)
    with pytest.raises(ValueError, match="world size changed"):
        changed.load_state_dict(state)


def test_resume_at_epoch_boundary():
    ds = dataset([{}], [3])
    sampler = CatalogBatchSampler(ds)
    for _ in list(sampler):
        sampler.mark_consumed()
    restored = CatalogBatchSampler(ds)
    restored.load_state_dict(sampler.state_dict())
    restored.set_epoch(1)
    assert restored.epoch == 1
    assert len(restored) == 3


def test_duration_budget_accounts_for_speed_and_all_slots():
    ds = dataset([{}], [1], slots=[2], batching={"max_duration": 2.1})
    ds.augmentation = AugmentConfig(speed_prob=1, speed_factors=[0.5])
    with pytest.raises(ValueError, match="exceeds"):
        CatalogBatchSampler(ds)
