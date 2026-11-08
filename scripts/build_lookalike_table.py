"""Build a look-alike interferer table for online TS mixing (``open_audio_llm.data.lookalike``).

Embeds every clip of an interference pool split and every enrollment clip of the given record
splits with a WeSpeaker ECAPA ONNX model (the one the TS benchmark used to pick look-alike
negatives), then keeps, per enrollment, the ``--top-k`` pool clips with the highest cosine below
``--max-cos``. Pairs at or above the cap are most likely the same person and would make a
target-absent mixture wrong, so they are skipped.

Needs onnxruntime next to the training env (lance, torchaudio), e.g. an onnxruntime package directory on
``PYTHONPATH`` in front of ``src``. Example::

    python scripts/build_lookalike_table.py --pool ts_interference_pool:v3-20261018:zh \\
        --records aishellmix:pack3-current-audio-record-20260910-punc-20261102:train_1spk \\
        --model /workspace/model/wespeaker-ecapa-tdnn512-LM/voxceleb_ECAPA512_LM.onnx --out zh.npz
"""
import argparse
import gzip
import json
import multiprocessing
import os
import time
from pathlib import Path

import numpy as np

from open_audio_llm.data.lookalike import pool_clip_key

SAMPLING_RATE = 16000
_worker = {}


def _split(value):
    dataset_id, version, split = value.split(":")
    return dataset_id, version, split


def _init(catalog, roots, model, max_seconds):
    import onnxruntime
    import torch

    torch.set_num_threads(1)  # one thread per worker; fbank and resampling otherwise oversubscribe the CPU

    from open_audio_llm.data.catalog_resolver import LhotseCatalogAudioResolver

    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 1
    session = onnxruntime.InferenceSession(model, sess_options=options, providers=["CPUExecutionProvider"])
    _worker.update(resolver=LhotseCatalogAudioResolver(catalog, roots), session=session,
                   input=session.get_inputs()[0].name, max_samples=int(max_seconds * SAMPLING_RATE))


def _embed(audio):
    import torch
    from torchaudio.compliance import kaldi

    audio = np.asarray(audio, dtype=np.float32)
    limit = _worker["max_samples"]
    if len(audio) > limit:  # the middle of a long clip
        start = (len(audio) - limit) // 2
        audio = audio[start:start + limit]
    features = kaldi.fbank(torch.from_numpy(audio).unsqueeze(0) * 32768, num_mel_bins=80, frame_length=25,
                           frame_shift=10, dither=0.0, sample_frequency=SAMPLING_RATE, window_type="hamming",
                           use_energy=False)
    features = features - features.mean(0, keepdim=True)
    embedding = _worker["session"].run(None, {_worker["input"]: features.unsqueeze(0).numpy()})[0].reshape(-1)
    return (embedding / np.linalg.norm(embedding)).astype(np.float16)


def _job(task):
    from audio_data_contract import AudioRef

    from open_audio_llm.data.catalog_dataset import load_mono
    from open_audio_llm.data.interference_pool import (
        LanceClip,
        PoolClip,
        _lance_release_key,
    )

    kind, key, base, start, duration, cut_id = task
    try:
        if kind == "pool":
            # Same clip objects as interference_pool.pool_clips gives the training loader; ``key`` is their item_key.
            pool, owner = base
            resolver = _worker["resolver"]
            item = (PoolClip(resolver, pool, cut_id, start, duration) if owner == pool
                    else LanceClip(resolver, _lance_release_key(resolver, *owner), cut_id))
        else:
            item = _worker["resolver"].get_cut(AudioRef(*base, key, start=start, duration=duration))
        audio = load_mono(item, SAMPLING_RATE)
        if len(audio) < SAMPLING_RATE // 2:
            return key, None
        return key, _embed(audio)
    except Exception as error:  # noqa: BLE001 - one unreadable clip must not stop the build
        return key, repr(error)


def _pool_tasks(catalog, roots, pool):
    from audio_data_contract import resolve_split

    from open_audio_llm.data.catalog_resolver import LhotseCatalogAudioResolver

    resolver = LhotseCatalogAudioResolver(catalog, roots)
    tasks = []
    for path in resolve_split(resolver.catalog, *pool, "records", resolver.roots):
        with (gzip.open if path.suffix == ".gz" else open)(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    ref = json.loads(line)["audio_slots"][0]["ref"]
                    owner = (ref["dataset_id"], ref["version"], ref["split"])
                    tasks.append(("pool", pool_clip_key(ref["cut_id"], ref.get("start"), ref.get("duration")),
                                  (pool, owner), ref.get("start"), ref.get("duration"), ref["cut_id"]))
    return tasks


def _enrollment_tasks(catalog, roots, splits, speaker_pattern=None):
    """Enrollment tasks, and with ``speaker_pattern`` each clip's speaker: group 1 of the pattern searched in
    the record's ``metadata.source_record_id`` (or its id)."""
    import re

    from audio_data_contract import resolve_split

    from open_audio_llm.data.catalog_resolver import LhotseCatalogAudioResolver

    resolver = LhotseCatalogAudioResolver(catalog, roots)
    tasks, speakers = {}, {}
    pattern = re.compile(speaker_pattern) if speaker_pattern else None
    for split in splits:
        for path in resolve_split(resolver.catalog, *split, "records", resolver.roots):
            with (gzip.open if path.suffix == ".gz" else open)(path, "rt", encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    slots = {slot["name"]: slot["ref"] for slot in record["audio_slots"]}
                    ref = slots["enrollment"]
                    if pattern:
                        name = record.get("metadata", {}).get("source_record_id") or record["id"]
                        match = pattern.search(name)
                        if not match:
                            raise ValueError(f"speaker pattern does not match record {name!r}")
                        if speakers.setdefault(ref["cut_id"], match.group(1)) != match.group(1):
                            raise ValueError(f"enrollment {ref['cut_id']} belongs to two speakers")
                    if ref.get("start") is not None or ref.get("duration") is not None:
                        raise ValueError("enrollment spans are not supported: the table is keyed by cut id")
                    tasks.setdefault(ref["cut_id"], ("enrollment", ref["cut_id"],
                                                     (ref["dataset_id"], ref["version"], ref["split"]),
                                                     None, None, ref["cut_id"]))
    return list(tasks.values()), speakers


def _embed_all(tasks, args, label):
    ids, vectors, failed = [], [], 0
    started = time.time()
    with multiprocessing.get_context("spawn").Pool(
            args.workers, initializer=_init, initargs=(args.catalog, args.roots, args.model, args.max_seconds)) as pool:
        for done, (key, value) in enumerate(pool.imap_unordered(_job, tasks, chunksize=64), 1):
            if isinstance(value, np.ndarray):
                ids.append(key)
                vectors.append(value)
            else:
                failed += 1
            if done % 20000 == 0:
                print(f"{label}: {done}/{len(tasks)} in {time.time() - started:.0f}s, {failed} skipped", flush=True)
    print(f"{label}: {len(ids)} embedded, {failed} skipped (too short or unreadable)", flush=True)
    return np.array(ids), np.stack(vectors)


def main():
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(name, "1")  # inherited by the spawned workers
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="/workspace/data/registry/catalog")
    parser.add_argument("--roots", default="/workspace/data/registry/roots.json")
    parser.add_argument("--pool", type=_split, help="interference pool dataset:version:split")
    parser.add_argument("--same-corpus", action="store_true",
                        help="the pool is the enrollment clips themselves (register their audio release as the "
                             "interference source); needs --speaker-pattern so no speaker is its own look-alike")
    parser.add_argument("--speaker-pattern",
                        help="regex whose group 1 is the speaker in a record's source_record_id; same-speaker pairs "
                             "are skipped")
    parser.add_argument("--records", type=_split, nargs="+", required=True,
                        help="record splits whose enrollment clips get look-alikes")
    parser.add_argument("--model", required=True, help="WeSpeaker ECAPA ONNX model")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--max-cos", type=float, default=0.75,
                        help="pairs at or above this cosine are treated as the same speaker and skipped")
    parser.add_argument("--max-seconds", type=float, default=6.0, help="embed at most this much of each clip")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0, help="only the first N pool and enrollment clips (smoke test)")
    args = parser.parse_args()

    if args.same_corpus == bool(args.pool):
        parser.error("give exactly one of --pool and --same-corpus")
    if args.same_corpus and not args.speaker_pattern:
        parser.error("--same-corpus needs --speaker-pattern")
    enroll_tasks, speakers = _enrollment_tasks(args.catalog, args.roots, args.records, args.speaker_pattern)
    if args.limit:
        enroll_tasks = enroll_tasks[:args.limit]
    enroll_ids, enroll_vectors = _embed_all(enroll_tasks, args, "enrollment")
    if args.same_corpus:
        pool_ids, pool_vectors = enroll_ids, enroll_vectors
    else:
        pool_tasks = _pool_tasks(args.catalog, args.roots, args.pool)
        if args.limit:
            pool_tasks = pool_tasks[:args.limit]
        pool_ids, pool_vectors = _embed_all(pool_tasks, args, "pool")
    if len(set(pool_ids.tolist())) != len(pool_ids):
        raise ValueError("pool clips must have unique keys")

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pool_matrix = torch.from_numpy(pool_vectors).to(device, torch.float16 if device == "cuda" else torch.float32)
    order = np.argsort(enroll_ids)
    enroll_ids, enroll_vectors = enroll_ids[order], enroll_vectors[order]
    neighbors = np.full((len(enroll_ids), args.top_k), -1, dtype=np.int32)
    cosines = np.zeros((len(enroll_ids), args.top_k), dtype=np.float16)
    if args.speaker_pattern:
        # Same-speaker pairs are skipped by id; a pool clip of an unknown speaker gets its own index.
        index_of = {}
        code = lambda key: index_of.setdefault(speakers.get(key, f"?{key}"), len(index_of))
        pool_speaker = torch.tensor([code(k) for k in pool_ids.tolist()], device=device)
        enroll_speaker = np.array([code(k) for k in enroll_ids.tolist()])
    for begin in range(0, len(enroll_ids), 4096):
        block = torch.from_numpy(enroll_vectors[begin:begin + 4096]).to(device, pool_matrix.dtype)
        scores = (block @ pool_matrix.T).float()
        scores[scores >= args.max_cos] = float("-inf")
        if args.speaker_pattern:
            own = torch.from_numpy(enroll_speaker[begin:begin + len(block)]).to(device)
            scores[own[:, None] == pool_speaker[None, :]] = float("-inf")
        values, index = scores.topk(min(args.top_k, scores.shape[1]), dim=1)
        index[torch.isinf(values)] = -1
        neighbors[begin:begin + len(block), :index.shape[1]] = index.cpu().numpy()
        cosines[begin:begin + len(block), :values.shape[1]] = values.clamp(min=-1).cpu().numpy()
    meta = {"model": args.model, "top_k": args.top_k, "max_cos": args.max_cos, "max_seconds": args.max_seconds,
            "pool": "same-corpus" if args.same_corpus else ":".join(args.pool),
            "speaker_pattern": args.speaker_pattern, "records": [":".join(s) for s in args.records],
            "pool_clips": len(pool_ids), "enrollments": len(enroll_ids)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, enroll_ids=enroll_ids, pool_ids=pool_ids, neighbors=neighbors, cosines=cosines,
             meta=np.array(json.dumps(meta)))
    valid = neighbors >= 0
    print(json.dumps({**meta, "out": str(args.out), "mean_top1_cos": float(cosines[:, 0][valid[:, 0]].mean()),
                      "mean_topk_cos": float(cosines[valid].mean()),
                      "enrollments_without_lookalike": int((~valid[:, 0]).sum())}), flush=True)


if __name__ == "__main__":
    main()
