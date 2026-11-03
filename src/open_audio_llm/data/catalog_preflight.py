"""Check the real training sampler and audio reader without loading a model."""

from collections import Counter, defaultdict
import hashlib
import io
import json
import random

import numpy as np
import soundfile as sf

from .augment import speed_ratio
from .catalog_dataset import CatalogSwiftDataset
from .catalog_sampler import CatalogBatchSampler
from .sot import TIMESTAMP_FORMAT, scale_timestamps

# Clips rendered for the level report, split across sources by their share of an epoch; 1000 leaves
# about 100 clips below the p10 of a task that fills the epoch.
LEVEL_SAMPLES = 1000
# Loudness as heard: 95th percentile of 25 ms frame energy. Whole-clip RMS (what level_dbfs and a
# nonspeech level_dbfs set) reads lower on speech with pauses than on steady noise.
LEVEL_FRAME_SECONDS = 0.025
LEVEL_FRAME_PERCENTILE = 95
LEVEL_PERCENTILES = (10, 25, 50, 75, 90)
# Keeps log10 finite on digital silence.
_ENERGY_FLOOR = 1e-12


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
            "far_rir_prob": config.far_rir_prob, "level_prob": config.level_prob,
            "spec_aug_prob": dataset.augmentation.spec_aug_prob}


def clip_levels(audio, sampling_rate):
    """Whole-clip RMS and loud-frame level of one clip, both in dBFS; None when shorter than a frame."""
    frame = round(LEVEL_FRAME_SECONDS * sampling_rate)
    frames = audio[: len(audio) // frame * frame].reshape(-1, frame)
    if not len(frames):
        return None
    energy = 10 * np.log10(np.mean(frames**2, axis=1) + _ENERGY_FLOOR)
    rms = 10 * np.log10(np.mean(audio**2) + _ENERGY_FLOOR)
    return float(rms), float(np.percentile(energy, LEVEL_FRAME_PERCENTILE))


def _percentiles(values):
    if not values:
        return {"n": 0}
    return {"n": len(values), **{f"p{p}": round(float(np.percentile(values, p)), 1) for p in LEVEL_PERCENTILES}}


def level_report(dataset, shares, samples):
    """Input level of training clips by task and target presence, rendered as training renders them.

    Only a report: a level that tells empty targets from text ones is a shortcut the model can learn
    (quiet speech emptied on CHiME-4, 2026-11-03), and the recipe owner decides what to change.
    """
    total = sum(shares)
    rng = random.Random(0)
    levels = defaultdict(list)
    by_source = []
    for source, indices, share in zip(dataset.sources, dataset.source_ranges, shares):
        count = min(len(indices), round(samples * share / total)) if total else 0
        found = defaultdict(list)
        for index in sorted(rng.sample(range(indices.start, indices.stop), count)):
            # A draw through the training path; the sampler numbers occurrences across the epoch,
            # so this is a representative draw rather than the exact one training will make.
            sample = dataset[(0, index, 0)]
            audio = sample["mix_wav"] if sample.get("mix_wav") is not None else sample["audios"][-1]
            measured = clip_levels(sf.read(io.BytesIO(audio), dtype="float32")[0], dataset.sampling_rate)
            if measured is None:
                continue
            target = sample.get("solution", sample["messages"][-1]["content"])
            presence = "text" if target.split("<asr_text>", 1)[-1].strip() else "empty"
            levels[(sample["task"], presence)].append(measured)
            found[presence].append(measured[1])
        by_source.append({"dataset_id": source["dataset_id"], "version": source["version"],
                          "split": source["split"], "samples": count,
                          **{f"{presence}_loud_p50": round(float(np.median(found[presence])), 1)
                             for presence in ("text", "empty") if found[presence]}})
    by_task = {}
    for task in sorted({task for task, _ in levels}):
        row = {presence: {"rms_dbfs": _percentiles([rms for rms, _ in levels[(task, presence)]]),
                          "loud_dbfs": _percentiles([loud for _, loud in levels[(task, presence)]])}
               for presence in ("text", "empty")}
        text, empty = levels[(task, "text")], levels[(task, "empty")]
        if text and empty:
            floor = np.percentile([loud for _, loud in text], 10)
            row["empty_below_text_p10"] = round(float(np.mean([loud < floor for _, loud in empty])), 3)
        by_task[task] = row
    return {"samples": samples, "frame_seconds": LEVEL_FRAME_SECONDS,
            "frame_percentile": LEVEL_FRAME_PERCENTILE, "by_task": by_task, "by_source": by_source}


def check_data(config, *, message_format="qwen3_asr", batch_size=8, world_size=2, level_samples=LEVEL_SAMPLES):
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
    levels = None
    if level_samples:
        shares = [s["samples_per_epoch"] if s["samples_per_epoch"] is not None else s["records"] for s in sources]
        levels = level_report(dataset, shares, level_samples)
    return {"status": "passed", "check": "data_only_no_model_or_optimizer",
            "effective_config_sha256": hashlib.sha256(
                json.dumps(config, sort_keys=True).encode()).hexdigest(),
            "world_size": world_size, "batch_size": batch_size,
            "replay": sampler.replay, "full_coverage": sampler.full_coverage,
            "sources": sources,
            "quotas_by_dataset": dict(quotas), "decoded_examples": examples, "levels": levels,
            "sampler_state": sampler.state_dict()}
