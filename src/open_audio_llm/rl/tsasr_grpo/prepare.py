"""Draw on-the-fly TS-ASR samples (enrollment + mixture + target) for GRPO.

Sources are picked from an on-the-fly training data config (``base``) by
``dataset_id``/``version``/``split``; when the same split appears several
times with different ``online_mix`` settings, ``select`` names the values that
identify the wanted entry, and ``override`` changes ``online_mix`` fields of
the copy. Every sample is rendered by ``CatalogSwiftDataset`` in training mode,
exactly as SFT sees it at the given epoch, and written as two WAV files.
``samples.jsonl`` stores paths relative to itself, so the output directory can
be copied to another machine as a whole.

    python -m open_audio_llm.rl.tsasr_grpo.prepare --config CONFIG --split heldout --out DIR
    python -m open_audio_llm.rl.tsasr_grpo.prepare --config CONFIG --split train --out DIR \
        --epoch 1 --exclude HELDOUT/samples.jsonl
"""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from open_audio_llm.data.catalog_dataset import CatalogSwiftDataset, read_data_config

# Base-config fields that only matter for SFT batching, validation or the
# metadata cache of the original machine.
_DROP = ("validation", "validation_samples", "validation_max_duration", "replay", "batching",
         "hotwords", "metadata_cache")

_DATASET = None
_AUDIO_DIR = None


def build_data_config(config_path: Path) -> tuple[dict, list[dict]]:
    spec = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base = read_data_config(config_path.parent / spec["base"])
    for key in ("catalog", "roots"):
        if spec.get(key):
            base[key] = str((config_path.parent / spec[key]).resolve())
    cap = spec.get("max_enrollment_seconds")
    train, wanted = [], []
    for want in spec["sources"]:
        matches = [
            s for s in base["train"]
            if (s["dataset_id"], s["version"], s["split"]) == (want["dataset_id"], want["version"], want["split"])
            and all((s.get("online_mix") or {}).get(k) == v for k, v in (want.get("select") or {}).items())
        ]
        if len(matches) != 1:
            raise ValueError(f"{want['label']}: {len(matches)} base sources match; refine select")
        source = copy.deepcopy(matches[0])
        source.pop("weight", None)
        if want.get("override"):
            source.setdefault("online_mix", {}).update(want["override"])
        if cap and source.get("online_mix"):
            lo, hi = source["online_mix"].get("enrollment_seconds", [1.0, 20.0])
            source["online_mix"]["enrollment_seconds"] = [lo, min(hi, cap)]
        train.append(source)
        wanted.append(want)
    data = {k: v for k, v in base.items() if k not in _DROP and k != "train"}
    data["record_index"] = spec.get("record_index", "jsonl")
    data["train"] = train
    return data, wanted


def _decode(value) -> np.ndarray:
    import io

    audio, rate = sf.read(io.BytesIO(value), dtype="float32", always_2d=False)
    if rate != 16000:
        raise ValueError(f"expected 16 kHz audio, got {rate}")
    return audio if audio.ndim == 1 else audio.mean(axis=-1)


def _render(job):
    label, index, epoch = job
    record = _DATASET.records[index].record
    sample = _DATASET[(epoch, index, None)]
    enroll, mix = _decode(sample["audios"][0]), _decode(sample["mix_wav"])
    stem = f"{label}_{index}"
    sf.write(_AUDIO_DIR / f"{stem}.enroll.wav", enroll, 16000, subtype="FLOAT")
    sf.write(_AUDIO_DIR / f"{stem}.mix.wav", mix, 16000, subtype="FLOAT")
    target = sample["solution"].split("<asr_text>", 1)[1]
    return {
        "id": f"{label}:{index}", "label": label, "record_id": record.id, "epoch": epoch,
        "language": record.language, "target": target, "present": bool(target.strip()),
        "record_labels": record.labels,
        "enroll_seconds": len(enroll) / 16000, "mix_seconds": len(mix) / 16000,
        "enroll_wav": f"audio/{stem}.enroll.wav", "mix_wav": f"audio/{stem}.mix.wav",
    }


def main(argv=None):
    global _DATASET, _AUDIO_DIR
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--split", required=True, help="key of each source's count mapping, e.g. train or heldout")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epoch", type=int, default=0, help="rendering epoch (interferers, levels, RIR draws)")
    p.add_argument("--tag", default="", help="extra seed for which records are drawn")
    p.add_argument("--exclude", type=Path, help="samples.jsonl whose records must not be drawn")
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args(argv)

    data, wanted = build_data_config(args.config)
    _AUDIO_DIR = args.out / "audio"
    _AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    (args.out / "data-effective.yaml").write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False))
    _DATASET = CatalogSwiftDataset(data, training=True, message_format="qwen3_asr")
    skip = set()
    if args.exclude:
        skip = {json.loads(line)["record_id"] for line in args.exclude.open(encoding="utf-8")}
    jobs = []
    for want, indexes in zip(wanted, _DATASET.source_ranges):
        pool = [i for i in indexes if _DATASET.records[i].record.id not in skip]
        picked = random.Random(f"{want['label']}:{args.split}:{args.tag}").sample(pool, want["count"][args.split])
        jobs += [(want["label"], i, args.epoch) for i in picked]
    print(f"rendering {len(jobs)} samples", flush=True)
    with mp.get_context("fork").Pool(args.workers) as workers, \
            (args.out / "samples.jsonl").open("w", encoding="utf-8") as out:
        for k, row in enumerate(workers.imap_unordered(_render, jobs, chunksize=4), 1):
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            if k % 500 == 0:
                print(f"rendered {k}/{len(jobs)}", flush=True)
    print(f"done: {args.out / 'samples.jsonl'}", flush=True)


if __name__ == "__main__":
    main()
