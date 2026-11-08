"""Verify a look-alike table and its use by the training loader.

A. Table: for a sample of enrollments, re-embed the enrollment, its stored nearest pool clip and a
   random pool clip with the builder's own code; the stored cosine must reproduce, and look-alikes
   must be far more similar than random clips.
B. Loader: a CatalogSwiftDataset with lookalike_prob 1 on a target-absent online source must draw
   every interferer from that record's enrollment look-alikes.
Usage (onnxruntime on PYTHONPATH, see scripts/build_lookalike_table.py):
  verify_lookalike.py TABLE POOL RECORDS OUT.json [--samples 300]
"""
import argparse
import json
import random
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import build_lookalike_table as builder  # noqa: E402

from open_audio_llm.data import catalog_dataset  # noqa: E402
from open_audio_llm.data.lookalike import LookalikeTable, item_key  # noqa: E402

CATALOG, ROOTS = "/workspace/data/registry/catalog", "/workspace/data/registry/roots.json"
MODEL = "/workspace/model/wespeaker-ecapa-tdnn512-LM/voxceleb_ECAPA512_LM.onnx"


def part_a(table, pool, records, samples, rng):
    builder._init(CATALOG, ROOTS, MODEL, table.meta["max_seconds"])
    pool_tasks = {t[1]: t for t in builder._pool_tasks(CATALOG, ROOTS, pool)}
    enroll_tasks = {t[1]: t for t in builder._enrollment_tasks(CATALOG, ROOTS, records)}
    rows = [i for i in rng.sample(range(len(table.enroll_ids)), samples) if table.neighbors[i, 0] >= 0]
    pool_keys = list(pool_tasks)
    stored, recomputed, random_cos = [], [], []
    for i in rows:
        _, enroll = builder._job(enroll_tasks[str(table.enroll_ids[i])])
        _, near = builder._job(pool_tasks[str(table.pool_ids[table.neighbors[i, 0]])])
        _, other = builder._job(pool_tasks[rng.choice(pool_keys)])
        if not all(isinstance(v, np.ndarray) for v in (enroll, near, other)):
            continue
        stored.append(float(table.cosines[i, 0]))
        recomputed.append(float(enroll.astype(np.float32) @ near.astype(np.float32)))
        random_cos.append(float(enroll.astype(np.float32) @ other.astype(np.float32)))
    stored, recomputed, random_cos = map(np.array, (stored, recomputed, random_cos))
    return {"checked": len(stored), "max_abs_diff_stored_vs_recomputed": float(np.abs(stored - recomputed).max()),
            "lookalike_cos_mean": float(recomputed.mean()), "random_cos_mean": float(random_cos.mean()),
            "lookalike_cos_p10_p50_p90": np.percentile(recomputed, [10, 50, 90]).round(3).tolist(),
            "random_cos_p10_p50_p90": np.percentile(random_cos, [10, 50, 90]).round(3).tolist(),
            "lookalike_above_random_p90": float((recomputed > np.percentile(random_cos, 90)).mean())}


def part_b(table_path, pool, records, samples, rng):
    dataset_id, version, split = records[0]
    language = pool[2]
    config = {
        "catalog": CATALOG, "roots": ROOTS, "sampling_rate": 16000,
        "train": [{"dataset_id": dataset_id, "version": version, "split": split, "weight": 1.0,
                   "online_mix": {"absent_prob": 1.0, "absent_interferer_probs": [0.0, 1.0], "lookalike_prob": 1.0,
                                  "speaker_rir_prob": 0.0}}],
        "interference_sources": [{"dataset_id": pool[0], "version": pool[1], "split": pool[2], "language": language,
                                  "lookalike_table": str(table_path)}],
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        json.dump(config, handle)
    dataset = catalog_dataset.CatalogSwiftDataset(catalog_dataset.read_data_config(handle.name),
                                                  message_format="qwen3_asr")
    table = LookalikeTable(table_path)
    drawn = []
    real_load = catalog_dataset.load_mono
    catalog_dataset.load_mono = lambda item, sr: (drawn.append(item_key(item)), np.full(16000, 0.01, np.float32))[1]
    try:
        checked = inside = skipped = 0
        for index in rng.sample(range(len(dataset.records)), samples):
            record = dataset.records[index].record
            enroll_id = record.slot("enrollment").ref.cut_id
            allowed = set(table.pool_ids[table.candidates(enroll_id)].tolist())
            if not allowed:
                skipped += 1
                continue
            record = replace(record, metadata={**record.metadata, "online_absent": True})
            drawn.clear()
            dataset._online_slot(record, "mixture", np.full(32000, 0.01, np.float32), random.Random(index))
            checked += len(drawn)
            inside += sum(clip in allowed for clip in drawn)
    finally:
        catalog_dataset.load_mono = real_load
    return {"records": samples, "skipped_without_lookalikes": skipped, "interferers_drawn": checked,
            "drawn_from_lookalikes": inside}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("table", type=Path)
    parser.add_argument("pool", type=builder._split)
    parser.add_argument("records", type=builder._split)
    parser.add_argument("out", type=Path)
    parser.add_argument("--samples", type=int, default=300)
    args = parser.parse_args()
    table = LookalikeTable(args.table)
    result = {"table": str(args.table), "meta": table.meta,
              "A_embedding_check": part_a(table, args.pool, [args.records], args.samples, random.Random(0)),
              "B_loader_check": part_b(args.table, args.pool, [args.records], 100, random.Random(1))}
    args.out.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(json.dumps(result, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
