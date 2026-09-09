"""Catalog-backed training samples: load and augment audio only when sampled."""

from __future__ import annotations

import os
from dataclasses import asdict
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
    """Keep sample facts in memory; decode fresh audio on each __getitem__."""

    def __init__(self, config, *, training=True, encode=None, grpo=False):
        self.resolver = LhotseCatalogAudioResolver(
            config["catalog"],
            config["roots"],
        )
        self.sampling_rate = int(config.get("sampling_rate", 16000))
        self.augmentation = AugmentConfig(**config.get("augmentation", {}))
        self.encode = encode
        self.grpo = grpo
        self.training = training
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
            start = len(records)
            records.extend(self._source_records(source))
            if len(records) == start:
                raise ValueError(f"Empty source: {source}")
            self.source_ranges.append(range(start, len(records)))
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
        if "records_artifact" in split:
            path = resolve_artifact(
                self.resolver.catalog,
                spec.dataset_id,
                spec.version,
                split["records_artifact"],
                self.resolver.roots,
            )
            rows = (ResolvedAudioRecord(record) for record in load_records(path))
        else:
            cuts = self.resolver.iter_cuts(spec.dataset_id, spec.version, split_name)
            rows = self._cut_records(cuts, source, spec)
        count = 0
        for row in rows:
            duration = row.duration or 0
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
                )
            )

    def _resolve_audio(self, resolved, rng):
        import soundfile as sf

        result = {}
        for slot in resolved.record.audio_slots:
            ref = slot.ref
            cut = self._audio_cuts.get(
                (ref.dataset_id, ref.version, ref.split, ref.cut_id)
            )
            if cut is None:
                cut = self.resolver.get_cut(ref)
            elif ref.start is not None or ref.duration is not None:
                cut = cut.truncate(offset=ref.start or 0.0, duration=ref.duration)
            audio = load_mono(cut, self.sampling_rate)
            if self.training:
                audio = augment_waveform(
                    audio,
                    self.sampling_rate,
                    rng,
                    self.augmentation,
                    noise_loader=lambda r, sr: load_mono(r.choice(self.noise_cuts), sr),
                    rir_loader=lambda r, sr: load_mono(r.choice(self.rir_cuts), sr),
                )
            # ms-swift accepts WAV bytes on both training and rollout paths.
            # FLOAT preserves augmentation values without PCM clipping/quantization.
            buffer = BytesIO()
            sf.write(buffer, audio, self.sampling_rate, format="WAV", subtype="FLOAT")
            result[slot.name] = buffer.getvalue()
        return result

    def __getitem__(self, idx):
        import soundfile as sf

        sample = super().__getitem__(idx)
        sample["duration"] = sf.info(BytesIO(sample["audios"][-1])).duration
        if self.grpo:
            sample["messages"] = [
                m for m in sample["messages"] if m["role"] != "assistant"
            ]
        elif self.training and self.augmentation.spec_aug_prob:
            sample["audio_augmentation"] = {
                "seed": f"{self.seed}:{idx if isinstance(idx, tuple) else (self.epoch, idx)}:features",
                "config": asdict(self.augmentation),
            }
        return self.encode(sample) if self.encode is not None else sample
