"""Time __getitem__ on the real rank-0 sampler batches of a recipe (warm process, like a training worker)."""
import json, sys, time
from collections import defaultdict
import qwen_asr  # noqa: F401  the training main process imports it before forking workers
from open_audio_llm.data.catalog_dataset import CatalogSwiftDataset, read_data_config
from open_audio_llm.data.catalog_sampler import CatalogBatchSampler

config, first, count, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
ds = CatalogSwiftDataset(read_data_config(config), training=True, message_format="qwen3_asr", collect_metrics=True)
t = time.perf_counter()
sampler = CatalogBatchSampler(ds, batch_size=12, world_size=8)
sampler.rank, sampler._plan = 0, None
batches = sampler._batches()
print(f"plan {time.perf_counter()-t:.1f}s batches {len(batches)}", flush=True)
names = {}
for number, indices in enumerate(ds.source_ranges):
    names[number] = (indices, ds.sources[number]["dataset_id"])
def source(idx):
    for indices, name in names.values():
        if indices.start <= idx < indices.stop:
            return name
keys = [k for b in batches[first:first + count] for k in b]
ds[keys[0]]  # warm-up: lazy opens and first-call costs are per worker, not per sample
rows = []
for key in keys[1:]:
    t = time.perf_counter()
    item = ds[key]
    wall = time.perf_counter() - t
    perf = item.get("_performance") or {}
    rows.append({"wall": wall, "source": source(key[1]), "task": item.get("task"), "duration": item.get("duration"),
                 **{k: v for k, v in perf.items() if isinstance(v, (int, float))}})
json.dump(rows, open(out, "w"))
tot = sum(r["wall"] for r in rows)
print(f"samples {len(rows)} total {tot:.1f}s mean {tot/len(rows):.3f}s")
for k in ("decode_s", "wave_augment_s", "encode_s", "prepare_s", "prepare_cpu_s"):
    print(f"  {k} mean {sum(r.get(k, 0) for r in rows)/len(rows):.3f}")
by = defaultdict(list)
for r in rows: by[r["source"]].append(r["wall"])
for k, v in sorted(by.items(), key=lambda x: -sum(x[1])):
    print(f"  {k:22s} n={len(v):3d} share={sum(v)/tot:5.1%} mean={sum(v)/len(v):.3f}s")
