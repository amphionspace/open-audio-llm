"""Catalog-backed training samples: load and augment audio only when sampled."""

from __future__ import annotations

import os
import time
import hashlib
from dataclasses import asdict, replace
from fractions import Fraction
from io import BytesIO
from pathlib import Path

import numpy as np
import yaml
from audio_data_contract import (
    AudioRecord,
    AudioRef,
    AudioSlot,
    load_records,
    resolve_artifact,
)

from .augment import AugmentConfig, augment_waveform
from .catalog_resolver import LhotseCatalogAudioResolver
from .online_dataset import TASK_TEMPLATES, OnlineAudioDataset
from .records import ResolvedAudioRecord


def in_ts_partition(record, selection):
    """Keep all targets and 2/3-speaker derivatives of a seed utterance together."""
    if record.task != "ts_asr":
        raise ValueError("TS holdout applies only to ts_asr records")
    source_id = record.metadata["source_record_id"]
    mixture_id = source_id.split("|")[-2]
    dataset_id = record.audio_slots[-1].ref.dataset_id
    if dataset_id == "aishellmix":
        family = "_".join(mixture_id.split("_")[:2])
    elif dataset_id == "librimix":
        family = mixture_id.split("_")[0]
    else:
        raise ValueError(f"Unknown TS mixture family format: {dataset_id}")
    modulus = selection["modulus"]
    part = selection["part"]
    if type(modulus) is not int or modulus < 2 or part not in {"train", "dev"}:
        raise ValueError("TS holdout requires modulus >= 2 and part train/dev")
    key = f"{selection['seed']}:{dataset_id}:{family}".encode()
    held_out = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % modulus == 0
    return held_out if part == "dev" else not held_out


def read_data_config(path):
    path = Path(path).expanduser().resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError("data config must be a YAML mapping")
    for field, env in (
        ("catalog", "AUDIO_DATA_CATALOG"),
        ("roots", "AUDIO_DATA_ROOTS_FILE"),
    ):
        value = os.environ.get(env) or config.get(field)
        if not value:
            raise ValueError(f"Set {env} or {field!r} in {path}")
        selected = Path(value).expanduser()
        config[field] = str(
            selected if selected.is_absolute() else path.parent / selected
        )
    if not config.get("train"):
        raise ValueError("data config requires at least one train source")
    return config


def load_mono(cut, sampling_rate):
    from scipy.signal import resample_poly

    audio = cut.load_audio()
    if audio is None or audio.size == 0:
        raise ValueError(f"cut {cut.id!r} contains no audio")
    audio = np.asarray(audio, dtype=np.float32).mean(axis=0)
    if cut.sampling_rate != sampling_rate:
        ratio = Fraction(sampling_rate, cut.sampling_rate)
        audio = resample_poly(audio, ratio.numerator, ratio.denominator)
    return np.asarray(audio, dtype=np.float32)


class CatalogSwiftDataset(OnlineAudioDataset):
    """Read facts from memory or a shared index; decode audio on each __getitem__."""

    def __init__(self, config, *, training=True, encode=None, grpo=False, message_format="generic", collect_metrics=False):
        self.collect_metrics = collect_metrics
        self.metadata_cache = os.environ.get("AUDIO_DATA_METADATA_CACHE") or config.get("metadata_cache")
        self.resolver = LhotseCatalogAudioResolver(
            config["catalog"],
            config["roots"],
            index_cache=self.metadata_cache,
        )
        self.sampling_rate = int(config.get("sampling_rate", 16000))
        self.augmentation = AugmentConfig(**config.get("augmentation", {}))
        self.encode = encode
        self.grpo = grpo
        self.training = training
        self.message_format = message_format
        self._audio_cuts = {}
        self.config = config
        self.source_ranges = []
        records = []
        sources = config["train" if training else "validation"]
        self.sources = sources
        explicit = any("weight" in s for s in sources)
        if explicit and any("samples" in s or "reps" in s for s in sources):
            raise ValueError("weight is mutually exclusive with samples/reps")
        if explicit and not all("weight" in s for s in sources):
            raise ValueError("Specify weight for every source")
        for source in sources:
            excluded = source.get("exclude_speakers", [])
            if not isinstance(excluded, list) or not all(isinstance(item, str) for item in excluded):
                raise ValueError("exclude_speakers must be a list of speaker IDs")
            if type(source.get("require_clean_pass", False)) is not bool:
                raise ValueError("require_clean_pass must be a boolean")
            for key in ("samples", "reps", "max_samples"):
                value = source.get(key)
                if value is not None and (type(value) is not int or value <= 0):
                    raise ValueError(f"{key} must be a positive integer")
            if "samples" in source and "max_samples" in source:
                raise ValueError("Use samples instead of max_samples, not both")
            if source.get("shard_rotation") and "samples" not in source:
                raise ValueError("shard_rotation requires samples")
            if "weight" in source:
                import math

                if not math.isfinite(source["weight"]) or source["weight"] <= 0:
                    raise ValueError("weight must be positive and finite")
            if not self.metadata_cache:
                start = len(records)
                records.extend(self._source_records(source))
                if len(records) == start:
                    raise ValueError(f"Empty source: {source}")
                self.source_ranges.append(range(start, len(records)))
        if self.metadata_cache:
            from .catalog_cache import CatalogRecordIndex

            records = CatalogRecordIndex(self, self.metadata_cache)
            self.source_ranges = [range(start, end) for start, end in zip([0, *records.ends[:-1]], records.ends)]
        if not records:
            raise ValueError(
                "Catalog selection contains no training/validation samples"
            )
        self.noise_cuts = (
            self._resource_cuts(config.get("noise_sources", [])) if training else []
        )
        self.rir_cuts = (
            self._resource_cuts(config.get("rir_sources", [])) if training else []
        )
        if training and self.augmentation.noise_prob and not self.noise_cuts:
            raise ValueError("noise_prob requires non-empty noise_sources")
        if training and self.augmentation.rir_prob and not self.rir_cuts:
            raise ValueError("rir_prob requires non-empty rir_sources")
        if grpo and self.augmentation.spec_aug_prob:
            raise ValueError(
                "GRPO supports waveform augmentation; SpecAugment is SFT-only"
            )
        super().__init__(
            records=records,
            training=training,
            message_format=message_format,
            seed=int(config.get("seed", 42)),
            **config.get("hotwords", {}),
        )

    def _resource_cuts(self, sources):
        return [
            cut
            for source in sources
            for cut in self.resolver.iter_cuts(
                source["dataset_id"], source["version"], source["split"]
            )
        ]

    def _source_records(self, source):
        spec = self.resolver.catalog.get(source["dataset_id"], source["version"])
        split_name = source["split"]
        split = spec.splits[split_name]
        if "records_artifact" in split or "records_artifacts" in split:
            if source.get("exclude_speakers"):
                raise ValueError("exclude_speakers requires Lhotse supervision speaker IDs")
            names = split.get("records_artifacts") or [split["records_artifact"]]
            rows = (
                ResolvedAudioRecord(record)
                for name in names
                for record in load_records(resolve_artifact(
                    self.resolver.catalog, spec.dataset_id, spec.version,
                    name, self.resolver.roots,
                ))
            )
            if source.get("require_clean_pass"):
                # Portable records must attest the entire example, including
                # every audio slot; a dataset name or file-integrity check is insufficient.
                rows = (row for row in rows
                        if (row.record.metadata.get("clean") or {}).get("pass") is True)
        else:
            cuts = self.resolver.iter_cuts(spec.dataset_id, spec.version, split_name)
            rows = self._cut_records(cuts, source, spec)
        count = 0
        for row in rows:
            if source.get("ts_holdout") and not in_ts_partition(row.record, source["ts_holdout"]):
                continue
            # Portable TS records omit durations: read indexed metadata, not audio.
            if row.duration is None:
                slots = tuple(
                    replace(slot, ref=replace(
                        slot.ref, duration=self.resolver.get_duration(slot.ref),
                    )) if slot.ref.duration is None else slot
                    for slot in row.record.audio_slots
                )
                row = ResolvedAudioRecord(replace(row.record, audio_slots=slots))
            duration = row.duration
            if duration < source.get("min_duration", 0) or duration > source.get(
                "max_duration", float("inf")
            ):
                continue
            if row.record.task not in TASK_TEMPLATES:
                raise ValueError(f"Unsupported task template: {row.record.task}")
            yield row
            count += 1
            limit = source.get("max_samples")
            if not source.get("shard_rotation"):
                limit = source.get("samples", limit)
            if limit is not None and count >= limit:
                break

    def _cut_records(self, cuts, source, spec):
        excluded = set(source.get("exclude_speakers", []))
        if excluded:
            def exclude_heldout(cut):
                return cut.filter_supervisions(lambda supervision: supervision.speaker not in excluded)
            cuts = cuts.map(exclude_heldout)
        if source.get("require_clean_pass"):
            def clean_supervisions(cut):
                def eligible(supervision):
                    custom = {**(cut.custom or {}), **(supervision.custom or {})}
                    clean = custom.get("clean") or {}
                    return (clean.get("pass") is True and supervision.duration > 0
                            and supervision.start >= 0 and supervision.end <= cut.duration)
                return cut.filter_supervisions(eligible)

            # A cleaned WenetSpeech W label can extend past its recording. Filter
            # before trimming: Lhotse otherwise drops the label and asserts on an empty cut.
            cuts = cuts.map(clean_supervisions)
        for cut in cuts.trim_to_supervisions(keep_overlapping=False):
            supervision = cut.supervisions[0]
            custom = {**(cut.custom or {}), **(supervision.custom or {})}
            task = source.get("task", custom.get("task", "asr"))
            name = "mixture" if task == "ts_asr" else "primary"
            ref = AudioRef(
                spec.dataset_id,
                spec.version,
                source["split"],
                supervision.id,
                duration=cut.duration,
                purpose=name,
            )
            if not self.metadata_cache:
                self._audio_cuts[(ref.dataset_id, ref.version, ref.split, ref.cut_id)] = cut
            slots = []
            if custom.get("enrollment_cut_id"):
                slots.append(
                    AudioSlot(
                        "enrollment",
                        AudioRef(
                            custom.get("enrollment_dataset_id", spec.dataset_id),
                            custom.get("enrollment_dataset_version", spec.version),
                            custom.get("enrollment_split", source["split"]),
                            custom["enrollment_cut_id"],
                            purpose="enrollment",
                        ),
                    )
                )
            slots.append(AudioSlot(name, ref))
            if task == "ts_asr" and len(slots) != 2:
                raise ValueError(f"{supervision.id}: TS-ASR requires enrollment_cut_id")
            yield ResolvedAudioRecord(
                AudioRecord(
                    id=f"{spec.key}:{source['split']}:{supervision.id}",
                    task=task,
                    audio_slots=tuple(slots),
                    target=supervision.text or "",
                    language=supervision.language
                    or custom.get("language")
                    or spec.languages[0],
                    hotwords=tuple(custom.get("hotwords", [])),
                    labels=custom.get(
                        "labels",
                        {k: v for k, v in custom.items() if k.endswith("_label")},
                    ),
                ),
                cuts={name: cut} if self.metadata_cache else None,
            )

    def _resolve_audio(self, resolved, rng):
        import soundfile as sf

        result = {}
        for slot in resolved.record.audio_slots:
            ref = slot.ref
            cut = (resolved.cuts or {}).get(slot.name) or self._audio_cuts.get(
                (ref.dataset_id, ref.version, ref.split, ref.cut_id)
            )
            if cut is None:
                cut = self.resolver.get_cut(ref)
            elif ref.start is not None or ref.duration is not None:
                cut = cut.truncate(offset=ref.start or 0.0, duration=ref.duration)
            started = time.perf_counter()
            audio = load_mono(cut, self.sampling_rate)
            if self.collect_metrics:
                self._sample_metrics["decode_s"] += time.perf_counter() - started
            started = time.perf_counter()
            if self.training:
                audio = augment_waveform(
                    audio,
                    self.sampling_rate,
                    rng,
                    self.augmentation,
                    noise_loader=lambda r, sr: load_mono(r.choice(self.noise_cuts), sr),
                    rir_loader=lambda r, sr: load_mono(r.choice(self.rir_cuts), sr),
                )
            if self.collect_metrics:
                self._sample_metrics["wave_augment_s"] += time.perf_counter() - started
                self._sample_metrics["audio_seconds"] += len(audio) / self.sampling_rate
            # ms-swift accepts WAV bytes on both training and rollout paths.
            # FLOAT preserves augmentation values without PCM clipping/quantization.
            buffer = BytesIO()
            sf.write(buffer, audio, self.sampling_rate, format="WAV", subtype="FLOAT")
            result[slot.name] = buffer.getvalue()
        return result

    def __getitem__(self, idx):
        import soundfile as sf

        started = time.perf_counter()
        cpu_started = time.process_time()
        if self.collect_metrics:
            self._sample_metrics = dict(decode_s=0.0, wave_augment_s=0.0, audio_seconds=0.0)
        sample = super().__getitem__(idx)
        sample["duration"] = sf.info(BytesIO(sample["audios"][-1])).duration
        if self.collect_metrics and self.message_format == "qwen3_asr" and sample["task"] == "ts_asr":
            self._sample_metrics["audio_seconds"] = sample["duration"]
        if self.grpo:
            sample["messages"] = [
                m for m in sample["messages"] if m["role"] != "assistant"
            ]
        elif self.training and self.augmentation.spec_aug_prob:
            sample["audio_augmentation"] = {
                "seed": f"{self.seed}:{idx if isinstance(idx, tuple) else (self.epoch, idx)}:features",
                "config": asdict(self.augmentation),
            }
        encode_started = time.perf_counter()
        result = self.encode(sample) if self.encode is not None else sample
        if self.config.get("objective", {}).get("sample_mean"):
            # Metadata is consumed by the training loss, never model.forward.
            if sample["task"] == "asr":
                result["catalog_task"] = 0
            elif sample["task"] == "ts_asr":
                target = sample["solution"].split("<asr_text>", 1)[1]
                result["catalog_task"] = 1 if target.strip() else 2
            elif sample["task"] == "speaker_attributed_asr":
                result["catalog_task"] = 3
            else:
                raise ValueError("Replay objective supports ordinary, target-speaker and speaker-attributed ASR")
        if self.collect_metrics:
            result["_performance"] = {
                **self._sample_metrics,
                "encode_s": time.perf_counter() - encode_started,
                "prepare_s": time.perf_counter() - started,
                "prepare_cpu_s": time.process_time() - cpu_started,
                "dataset_id": sample["dataset_id"],
            }
        return result
