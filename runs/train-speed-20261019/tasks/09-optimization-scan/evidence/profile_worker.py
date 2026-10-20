"""Cold-cache data-worker profile: per-sample time, directory listings, hotword sampling."""
import functools, json, os, random, sys, time
from collections import Counter, defaultdict
import open_audio_llm.data.catalog_resolver as cr
import open_audio_llm.data.online_dataset as od
from open_audio_llm.data.catalog_dataset import CatalogSwiftDataset, read_data_config

listings = []
original = cr._resolved_directory.__wrapped__

@functools.lru_cache(maxsize=1 << 16)
def timed_directory(directory):
    t = time.perf_counter()
    out = original(directory)
    n = sum(1 for _ in os.scandir(directory)) if os.path.isdir(directory) else 0
    listings.append((directory, n, time.perf_counter() - t))
    return out

cr._resolved_directory = timed_directory
hot = []
orig_hot = od.sample_hotwords
def timed_hot(*a, **k):
    t = time.perf_counter(); out = orig_hot(*a, **k); hot.append(time.perf_counter() - t); return out
od.sample_hotwords = timed_hot

t0 = time.perf_counter()
ds = CatalogSwiftDataset(read_data_config(sys.argv[1]), training=True, message_format="qwen3_asr")
print(f"init {time.perf_counter()-t0:.1f}s records {len(ds)} hotword_pool {len(ds.hotword_pool)}", flush=True)
rng = random.Random(0)
idx = rng.sample(range(len(ds)), int(sys.argv[2]))
per = []
for i in idx:
    before = len(listings); t = time.perf_counter()
    item = ds[i]
    per.append((time.perf_counter() - t, sum(x[2] for x in listings[before:]), item.get("dataset_id")))
tot = sum(p[0] for p in per)
print(f"samples {len(per)} total {tot:.1f}s mean {tot/len(per):.3f}s listing {sum(p[1] for p in per):.1f}s hotwords {sum(hot):.1f}s mean {1000*sum(hot)/max(1,len(hot)):.1f}ms")
big = sorted(listings, key=lambda x: -x[2])[:12]
for d, n, s in big: print(f"  {s:6.2f}s {n:7d} entries {d}")
by = defaultdict(lambda: [0, 0.0])
for t, l, d in per: by[d][0] += 1; by[d][1] += l
print("listing seconds by dataset:", sorted(((round(v[1],1), v[0], k) for k, v in by.items()), reverse=True)[:10])
