"""Check the real training sampler and audio reader without loading a model."""

from collections import Counter
import hashlib
import json

import numpy as np

from .augment import speed_ratio
from .catalog_dataset import CatalogSwiftDataset
from .catalog_sampler import CatalogBatchSampler
from .sot import TIMESTAMP_FORMAT, scale_timestamps


def augmentation_in_force(dataset, row, dataset_id):
    """Waveform augmentation probabilities a source's records actually get, after the per-record overrides."""
    if row.record.metadata.get("online_mix") is not None:
        return {"online_mix": True}
    config = dataset.record_augmentation(dataset.augmentation, row.record, dataset_id, "mixture")
    speed = config.speed_prob
    if row.record.metadata.get("sot_output_format") == TIMESTAMP_FORMAT:
        # Drawn per sample by _prepare_record, which retimes the target.
        speed = dataset.augmentation.speed_prob
    return {"speed_prob": speed, "noise_prob": config.noise_prob, "rir_prob": config.rir_prob,
            "far_rir_prob": config.far_rir_prob, "spec_aug_prob": dataset.augmentation.spec_aug_prob}


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
                        "weight": source.get("weight"), "quota": quota,
                        "samples_per_epoch": (quota * sampler.replay["epoch_samples"]
                                              // sampler.replay["window_samples"]
                                              if quota is not None else None),
                        "augmentation": augmentation_in_force(
                            dataset, dataset.records[indices.start], source["dataset_id"]) if len(indices) else None})
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
                    # Speed perturbation stretches the audio; the target must be retimed by the same ratio.
                    ratio, target = 1, row.record.target
                    if selected.training and selected.augmentation.speed_prob:
                        ratio = min((speed_ratio(f) for f in selected.augmentation.speed_factors),
                                    key=lambda r: abs(row.duration * r - sample["duration"]))
                        target = scale_timestamps(target, float(ratio))
                    if abs(sample["duration"] - row.duration * ratio) > 2 / selected.sampling_rate:
                        raise ValueError(f"Timestamped audio duration changed: {row.record.id}")
                    if not sample.get("enrollment_view") and target not in sample["solution"]:
                        raise ValueError(f"Timestamp target changed: {row.record.id}")
                examples.append({"selection": split, "dataset_id": source["dataset_id"],
                                 "split": source["split"], "id": row.record.id,
                                 "duration": sample["duration"]})
    return {"status": "passed", "check": "data_only_no_model_or_optimizer",
            "effective_config_sha256": hashlib.sha256(
                json.dumps(config, sort_keys=True).encode()).hexdigest(),
            "world_size": world_size, "batch_size": batch_size,
            "replay": sampler.replay, "full_coverage": sampler.full_coverage,
            "sources": sources,
            "quotas_by_dataset": dict(quotas), "decoded_examples": examples,
            "sampler_state": sampler.state_dict()}
