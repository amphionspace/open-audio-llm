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
                metadata={},
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


def test_replay_keeps_window_quotas_after_small_source_exhausts():
    # Vastly different corpus sizes must not change the 50/20/30 mix.
    ds = dataset([{"weight": 50}, {"weight": 20}, {"weight": 30}], [3, 80, 120],
                 batching={"num_buckets": 4, "max_samples": 3},
                 durations=[1.0] * 3 + [2.0] * 80 + [8.0] * 120)
    ds.config["replay"] = {"epoch_samples": 200, "window_samples": 20}
    ids = indexes(CatalogBatchSampler(ds))
    assert len(ids) == 200
    for start in range(0, len(ids), 20):
        groups = Counter(0 if i < 3 else 1 if i < 83 else 2 for i in ids[start:start + 20])
        assert groups == {0: 10, 1: 4, 2: 6}


def test_replay_covers_source_before_repeating_across_epochs_and_resumes():
    ds = dataset([{"weight": 1}], [10])
    ds.config["replay"] = {"epoch_samples": 4, "window_samples": 2}
    sampler = CatalogBatchSampler(ds)
    seen = []
    for epoch in range(3):
        sampler.set_epoch(epoch)
        seen.extend(indexes(sampler))
    assert set(seen[:8]) < set(range(10))
    assert len(set(seen[:10])) == 10
    assert seen[:4] != list(range(4))  # Random full-source selection, not a prefix.
    plan = list(sampler)
    sampler.mark_consumed()
    restored = CatalogBatchSampler(ds)
    restored.load_state_dict(sampler.state_dict())
    restored.set_epoch(0)
    assert list(restored) == plan[1:]
    assert len({key for batch in plan for key in batch}) == 4


def test_replay_ddp_reconstructs_global_plan():
    ds = dataset([{"weight": 1}, {"weight": 1}], [2, 18])
    ds.config["replay"] = {"epoch_samples": 20, "window_samples": 4}
    ranks = [list(CatalogBatchSampler(ds, rank=r, world_size=2)) for r in range(2)]
    assert [batch for pair in zip(*ranks) for batch in pair] == list(CatalogBatchSampler(ds))


def test_full_coverage_keeps_mix_and_covers_every_record_in_every_epoch():
    ds = dataset([{"weight": 2}, {"weight": 4}, {"weight": 4}], [3, 18, 51],
                 batching={"max_samples": 2, "num_buckets": 3},
                 durations=[1.0] * 3 + [5.0] * 18 + [10.0] * 51)
    ds.config["replay"] = {"epoch_samples": "full_coverage", "window_samples": 10}
    sampler = CatalogBatchSampler(ds)
    assert ds.config["replay"]["epoch_samples"] == "full_coverage"
    assert sampler.replay["epoch_samples"] == 130
    previous = None
    for epoch in range(3):
        sampler.set_epoch(epoch)
        ids = indexes(sampler)
        assert set(ids) == set(range(72))
        for start in range(0, len(ids), 10):
            counts = Counter(0 if i < 3 else 1 if i < 21 else 2 for i in ids[start:start + 10])
            assert counts == {0: 2, 1: 4, 2: 4}
        assert ids != previous
        previous = ids
    plan = list(sampler)
    sampler.mark_consumed()
    resumed = CatalogBatchSampler(ds)
    resumed.load_state_dict(sampler.state_dict())
    resumed.set_epoch(0)
    assert resumed.epoch == 2
    assert list(resumed) == plan[1:]


def test_full_coverage_ddp_preserves_all_records_and_only_pads_last_batch():
    ds = dataset([{"weight": 1}, {"weight": 1}], [2, 19],
                 batching={"max_samples": 3, "num_buckets": 1})
    ds.config["replay"] = {"epoch_samples": "full_coverage", "window_samples": 4}
    single = list(CatalogBatchSampler(ds))
    ranks = [list(CatalogBatchSampler(ds, rank=r, world_size=3)) for r in range(3)]
    combined = [batch for triplet in zip(*ranks) for batch in triplet]
    assert combined[:len(single)] == single
    assert len(combined) - len(single) < 3
    assert set(key[1] for batch in combined for key in batch) == set(range(21))


@pytest.mark.parametrize("source,batching,match", [
    ({"weight": 1, "max_samples": 1}, {}, "max_samples"),
    ({"weight": 1}, {"drop_last": True}, "drop_last"),
])
def test_full_coverage_rejects_record_dropping(source, batching, match):
    ds = dataset([source], [3], batching=batching)
    ds.config["replay"] = {"epoch_samples": "full_coverage", "window_samples": 2}
    with pytest.raises(ValueError, match=match):
        CatalogBatchSampler(ds)


def test_full_coverage_shuffles_duration_batches_without_losing_records():
    ds = dataset([{"weight": 1}], [80], durations=[1.0] * 40 + [10.0] * 40,
                 batching={"max_samples": 2, "num_buckets": 2})
    ds.config["replay"] = {"epoch_samples": "full_coverage", "window_samples": 80}
    sampler = CatalogBatchSampler(ds)
    batches = list(sampler)
    assert sorted(key[1] for batch in batches for key in batch) == list(range(80))
    kinds = [sampler.durations[batch[0][1]] for batch in batches]
    assert sum(a != b for a, b in zip(kinds, kinds[1:])) > 1
    assert batches == list(CatalogBatchSampler(ds))


@pytest.mark.parametrize("replay,weights,match", [
    ({"epoch_samples": 5, "window_samples": 2}, [1, 1], "divisible"),
    ({"epoch_samples": 10, "window_samples": 2}, [1000, 1], "every source"),
    ({"epoch_samples": 10, "window_samples": 2}, [0, 1], "positive weight"),
])
def test_replay_rejects_unrepresentable_quotas(replay, weights, match):
    ds = dataset([{"weight": w} for w in weights], [2, 2])
    ds.config["replay"] = replay
    with pytest.raises(ValueError, match=match):
        CatalogBatchSampler(ds)


def test_merge_preserves_legacy_cursor_and_replay_order():
    ds = dataset([{'weight': 1}], [80], batching={'max_samples': 2, 'num_buckets': 1})
    ds.config['replay'] = {'epoch_samples': 20, 'window_samples': 10}
    old = CatalogBatchSampler(ds)
    batches = list(old)
    for _ in range(3):
        old.mark_consumed()
    state = old.state_dict()
    ds.config['batching']['batch_merge'] = 2
    merged = CatalogBatchSampler(ds)
    assert merged.signature == old.signature
    merged.load_state_dict(state)
    assert [k for b in merged for k in b] == [k for b in batches[3:] for k in b]
    merged.mark_consumed()
    assert merged.state_dict()['consumed'] == 5
    pool = merged._replay_pools[0][1]
    merged.set_epoch(1)
    list(merged)
    assert merged._replay_pools[0][1] is pool


def test_duration_pairing_preserves_update_samples_and_checkpoint_cursor():
    ds = dataset([{}], [32], durations=[1.0]*8+[10.0]*8+[2.0]*8+[9.0]*8,
                 batching={'max_samples': 8, 'num_buckets': 1, 'batch_merge': 2, 'merge_window': 4})
    sampler = CatalogBatchSampler(ds, shuffle=False)
    # Explicit native batches isolate the merge scheduling from source shuffling.
    sampler._plan = [[(0, i, i) for i in range(start, start+8)] for start in [0,8,16,24]]
    batches = list(sampler)
    assert {k[1] for k in batches[0]} == set(range(8)) | set(range(16,24))
    assert {k[1] for k in batches[1]} == set(range(8,16)) | set(range(24,32))
    assert sorted(k[1] for b in batches for k in b) == list(range(32))
    sampler.mark_consumed()
    with pytest.raises(ValueError, match='optimizer-update boundary'):
        sampler.state_dict()
    sampler.mark_consumed()
    assert sampler.state_dict()['consumed'] == 4
