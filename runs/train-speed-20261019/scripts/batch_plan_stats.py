"""Compare sorted-packing plans of a recipe across sort-window sizes (no audio, no GPU).

Costs are the sampler's own sort keys (upper-bound seconds), so the padding reported
here is only the part a larger window can remove; real token padding also includes the
gap between that bound and the realized length (online mixing, prompt and text tokens).
"""
import argparse
import copy
import json
from collections import Counter

import numpy as np

from open_audio_llm.data.catalog_dataset import CatalogSwiftDataset, read_data_config
from open_audio_llm.data.catalog_sampler import CatalogBatchSampler


def family(source):
    if "online_mix" in source:
        return "ts_online"
    if source["dataset_id"] in {"hw_sft", "hard_hw_sft"}:
        return "hotword"
    if "task" in source:
        return source["task"]
    return "ts_record"


def global_plan(dataset, world_size, batch_size):
    sampler = CatalogBatchSampler(dataset, batch_size=batch_size, world_size=world_size)
    plans = []
    for rank in range(world_size):
        sampler.rank, sampler._plan = rank, None
        plans.append(sampler._batches())
    order = [plans[i % world_size][i // world_size] for i in range(sum(map(len, plans)))]
    return sampler, order


def stats(dataset, sampler, order, group_size):
    owner = np.empty(len(sampler.durations), dtype=np.int32)
    for number, indices in enumerate(dataset.source_ranges):
        owner[indices.start:indices.stop] = number
    families = [family(source) for source in dataset.sources]
    names = [source["dataset_id"] for source in dataset.sources]
    cost = np.asarray(sampler.durations)
    sizes, padded, real = [], [], []
    for batch in order:
        values = cost[[key[1] for key in batch]]
        sizes.append(len(batch))
        padded.append(len(batch) * values.max())
        real.append(values.sum())
    total = Counter(families[owner[key[1]]] for batch in order for key in batch)
    share = {k: v / sum(total.values()) for k, v in total.items()}
    deviation, dominated, distinct = [], 0, []
    for start in range(0, len(order) - group_size + 1, group_size):
        keys = [key for batch in order[start:start + group_size] for key in batch]
        counts = Counter(families[owner[key[1]]] for key in keys)
        local = {k: counts.get(k, 0) / len(keys) for k in share}
        deviation.append(0.5 * sum(abs(local[k] - share[k]) for k in share))
        dominated += max(local.values()) > 0.9
        distinct.append(len({names[owner[key[1]]] for key in keys}))
    batch_sources = [len({owner[key[1]] for key in batch}) for batch in order]
    updates = len(deviation)
    return {
        "batches": len(order),
        "updates": updates,
        "samples_per_batch": float(np.mean(sizes)),
        "full_batches": float(np.mean(np.asarray(sizes) == sampler.max_samples)),
        "cost_padding": 1 - float(np.sum(real) / np.sum(padded)),
        "padded_seconds_per_sample": float(np.sum(padded) / np.sum(sizes)),
        "family_share": share,
        "update_family_tv_distance_mean": float(np.mean(deviation)),
        "update_family_tv_distance_p90": float(np.quantile(deviation, 0.9)),
        "updates_one_family_over_90pct": dominated / updates,
        "update_distinct_datasets_mean": float(np.mean(distinct)),
        "batch_distinct_sources_mean": float(np.mean(batch_sources)),
        "single_source_batches": float(np.mean(np.asarray(batch_sources) == 1)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_config", required=True)
    parser.add_argument("--windows", type=int, nargs="+", required=True)
    parser.add_argument("--epoch_samples", type=int, required=True)
    parser.add_argument("--world_size", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = read_data_config(args.data_config)
    dataset = CatalogSwiftDataset(config, message_format="qwen3_asr")
    accumulation = config["batching"].get("accumulation", 1)
    report = {"data_config": args.data_config, "epoch_samples": args.epoch_samples,
              "world_size": args.world_size, "batch_size": args.batch_size,
              "accumulation": accumulation, "results": {}}
    for window in args.windows:
        dataset.config = copy.deepcopy(config)
        dataset.config["replay"] = {"epoch_samples": args.epoch_samples, "window_samples": window}
        dataset.config["batching"]["buffer_size"] = window
        sampler, order = global_plan(dataset, args.world_size, args.batch_size)
        report["results"][window] = stats(dataset, sampler, order, args.world_size * accumulation)
        print(window, json.dumps(report["results"][window]), flush=True)
    with open(args.output, "w") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
