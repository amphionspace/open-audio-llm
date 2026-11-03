"""Measure which augmentations the training dataset actually applies, per source (tasks/06-augmentation-audit).

Builds the training dataset from an attempt's effective data config (data-effective.yaml) with the training code
snapshot, keeps the smallest split of every dataset version, draws samples per split and records, for each sample: the
augmentation config in force after the per-record overrides (speed off for timestamped SOT, noise exclusions), whether
the waveform came back changed (length → speed, values → noise/reverb), and whether a SpecAugment policy reaches the
template. Usage (snapshot code on PYTHONPATH):
    audit_augmentation.py DATA_EFFECTIVE_YAML OUT_JSON [samples_per_split]
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from open_audio_llm.data import catalog_dataset
from open_audio_llm.data.augment import AugmentConfig, augment_features

PER = int(sys.argv[3]) if len(sys.argv) > 3 else 20


def main(config_path, out_path):
    config = yaml.safe_load(open(config_path))
    # Record counts from the same attempt's data preflight, to keep the smallest split per dataset version.
    preflight = json.load(open(Path(config_path).parent / "artifacts/data-preflight.json"))
    counts = {(s["dataset_id"], s["version"], s["split"]): s["records"] for s in preflight["sources"]}
    smallest = {}
    for source in config["train"]:
        key = (source["dataset_id"], source["version"])
        size = counts[(*key, source["split"])]
        if key not in smallest or size < smallest[key][0]:
            smallest[key] = (size, source)
    config["train"] = [source for _, source in smallest.values()]
    calls = []
    original = catalog_dataset.augment_waveform

    def recorded(audio, sampling_rate, rng, augmentation, **kwargs):
        out = original(audio, sampling_rate, rng, augmentation, **kwargs)
        same_length = len(out) == len(audio)
        calls.append({
            "speed_prob": augmentation.speed_prob, "noise_prob": augmentation.noise_prob,
            "rir_prob": augmentation.rir_prob, "far_rir_prob": augmentation.far_rir_prob,
            "length_changed": not same_length,
            "values_changed": bool(same_length and not np.allclose(out, np.asarray(audio, dtype=np.float32))),
        })
        return out

    catalog_dataset.augment_waveform = recorded
    dataset = catalog_dataset.CatalogSwiftDataset(config, training=True, message_format="qwen3_asr")
    report = defaultdict(lambda: defaultdict(float))
    for source, indices in zip(config["train"], dataset.source_ranges):
        name = f"{source['dataset_id']}@{source['version']}:{source['split']}"
        picks = np.random.default_rng(0).choice(list(indices), size=min(PER, len(indices)), replace=False)
        for i, index in enumerate(picks):
            calls.clear()
            sample = dataset[(0, int(index), None)]
            row = report[name]
            row["samples"] += 1
            for call in calls:
                row["waveform_calls"] += 1
                for key in ("speed_prob", "noise_prob", "rir_prob", "far_rir_prob"):
                    row[f"configured_{key}"] += call[key]
                row["length_changed"] += call["length_changed"]
                row["values_changed"] += call["values_changed"]
            policy = sample.get("audio_augmentation")
            row["spec_policy_present"] += bool(policy)
            if policy:
                features = np.ones((1, 128, 3000), dtype=np.float32)
                import torch
                masked = augment_features(torch.from_numpy(features), torch.tensor([3000]), policy)
                row["spec_masked"] += bool((masked == 0).any())
    result = {}
    for name, row in report.items():
        n, calls_n = row["samples"], max(row["waveform_calls"], 1)
        result[name] = {
            "samples": int(n),
            "waveform_calls": int(row["waveform_calls"]),
            "speed_prob_in_force": round(row["configured_speed_prob"] / calls_n, 3),
            "noise_prob_in_force": round(row["configured_noise_prob"] / calls_n, 3),
            "rir_prob_in_force": round((row["configured_rir_prob"] + row["configured_far_rir_prob"]) / calls_n, 3),
            "speed_applied": round(row["length_changed"] / calls_n, 3),
            "noise_or_reverb_applied": round(row["values_changed"] / calls_n, 3),
            "spec_policy_present": round(row["spec_policy_present"] / n, 3),
            "spec_masked": round(row["spec_masked"] / n, 3),
        }
        print(name, result[name], flush=True)
    json.dump(result, open(out_path, "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
