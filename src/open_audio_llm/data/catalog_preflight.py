"""Check the real training sampler and audio reader without loading a model."""

from collections import Counter
import hashlib
import json

import numpy as np

from .catalog_dataset import CatalogSwiftDataset
from .catalog_sampler import CatalogBatchSampler
from .sot import TIMESTAMP_FORMAT


def check_data(config, *, message_format="qwen3_asr", batch_size=8, world_size=2):
    dataset = CatalogSwiftDataset(config, message_format=message_format)
    sampler = CatalogBatchSampler(dataset, batch_size=batch_size, world_size=world_size)
    sources, examples = [], []
    quotas = Counter()
    for number, (source, indices) in enumerate(zip(dataset.sources, dataset.source_ranges)):
        quota = sampler.quotas[number] if sampler.replay else None
        if quota is not None:
            quotas[source["dataset_id"]] += quota
        sources.append({"dataset_id": source["dataset_id"], "version": source["version"],
                        "split": source["split"], "records": len(indices),
                        "weight": source.get("weight"), "quota": quota})
    # Construct the sampler before decoding: a zero quota must fail preflight.
    datasets = [("train", dataset)]
    if config.get("validation"):
        datasets.append(("validation", CatalogSwiftDataset(
            config, training=False, message_format=message_format)))
    for split, selected in datasets:
        for source, indices in zip(selected.sources, selected.source_ranges):
            if hasattr(selected.records, "durations"):
                durations = selected.records.durations[indices.start:indices.stop]
            else:
                durations = [selected.records[i].duration for i in indices]
            # Exercise both duration extremes, including long event recordings.
            picks = {indices.start + int(np.argmin(durations)),
                     indices.start + int(np.argmax(durations))}
            for index in sorted(picks):
                row, sample = selected.records[index], selected[index]
                if not sample["audios"] or sample["duration"] <= 0:
                    raise ValueError(f"Unreadable audio: {row.record.id}")
                if row.record.metadata.get("sot_output_format") == TIMESTAMP_FORMAT:
                    if abs(sample["duration"] - row.duration) > 1 / selected.sampling_rate:
                        raise ValueError(f"Timestamped audio duration changed: {row.record.id}")
                    if row.record.target not in sample["solution"]:
                        raise ValueError(f"Timestamp target changed: {row.record.id}")
                examples.append({"selection": split, "dataset_id": source["dataset_id"],
                                 "split": source["split"], "id": row.record.id,
                                 "duration": sample["duration"]})
    return {"status": "passed", "check": "data_only_no_model_or_optimizer",
            "effective_config_sha256": hashlib.sha256(
                json.dumps(config, sort_keys=True).encode()).hexdigest(),
            "world_size": world_size, "batch_size": batch_size,
            "replay": sampler.replay, "sources": sources,
            "quotas_by_dataset": dict(quotas), "decoded_examples": examples,
            "sampler_state": sampler.state_dict()}
