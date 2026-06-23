#!/usr/bin/env python3
"""Convert Voices-in-the-Wild-2M parquet shards to ShareGPT JSONL.

The converter extracts embedded HuggingFace audio bytes to WAV files on disk
and writes ms-swift compatible ShareGPT rows:

    python examples/data/convert_vitw.py \
        --data-dir /path/to/Voices-in-the-Wild-2M/data \
        --out-dir output/sharegpt/vitw \
        --audio-dir output/audio/vitw \
        --num 3000 \
        --num-workers 16
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import random
import re
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import soundfile as sf

try:
    import ujson as json
except ImportError:  # pragma: no cover - stdlib fallback
    import json  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

INSTRUCTION = "Transcribe the following audio.<audio>"
MAX_TEXT_CHARS = 256
MAX_SAME_CHAR_RUN = 32
MAX_TOP_CHAR_RATIO = 0.85
MIN_AUDIO_DUR_S = 0.1

_worker_audio_dir: str | None = None
_worker_seed: int = 42


def longest_same_char_run(text: str) -> int:
    if not text:
        return 0
    best = cur = 1
    prev = text[0]
    for ch in text[1:]:
        if ch == prev:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
            prev = ch
    return best


def is_abnormal_text(text: str) -> bool:
    if not text or len(text) > MAX_TEXT_CHARS:
        return True
    if longest_same_char_run(text) > MAX_SAME_CHAR_RUN:
        return True
    counts: dict[str, int] = {}
    for ch in text:
        counts[ch] = counts.get(ch, 0) + 1
    return max(counts.values()) / len(text) > MAX_TOP_CHAR_RATIO


def write_wav(dst_path: Path, audio_bytes: bytes) -> bool:
    if dst_path.exists() and dst_path.stat().st_size > 0:
        return True
    try:
        data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
        if data.size == 0 or (data.shape[0] / max(sr, 1)) < MIN_AUDIO_DUR_S:
            return False
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(dst_path, data, sr, subtype="PCM_16")
        return True
    except Exception as exc:  # noqa: BLE001 - converter should skip bad rows
        logger.warning("failed to write WAV %s: %s", dst_path, exc)
        return False


def discover_parquet_files(data_dir: Path, splits: list[str] | None) -> dict[str, list[tuple[Path, int]]]:
    pattern = re.compile(r"^(.+)-(\d+)-of-\d+\.parquet$")
    out: dict[str, list[tuple[Path, int]]] = {}
    for path in sorted(data_dir.iterdir()):
        match = pattern.match(path.name)
        if not match:
            continue
        split_name = match.group(1)
        if splits and split_name not in splits:
            continue
        out.setdefault(split_name, []).append((path, int(match.group(2))))
    for items in out.values():
        items.sort(key=lambda item: item[1])
    return out


def worker_init(audio_dir: str, seed: int) -> None:
    global _worker_audio_dir, _worker_seed
    _worker_audio_dir = audio_dir
    _worker_seed = seed


def clean_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", value)


def row_audio_bytes(audio_cell: Any) -> bytes | None:
    if isinstance(audio_cell, dict):
        data = audio_cell.get("bytes")
        return bytes(data) if data else None
    if isinstance(audio_cell, (bytes, bytearray)):
        return bytes(audio_cell)
    return None


def process_parquet_file(task: tuple[str, str, int, int]) -> tuple[list[str], int, int]:
    parquet_path_str, split_name, shard_idx, num_per_file = task
    assert _worker_audio_dir is not None
    parquet_path = Path(parquet_path_str)
    try:
        import pyarrow.parquet as pq

        table = pq.read_table(parquet_path)
    except Exception as exc:  # noqa: BLE001
        logger.error("cannot read parquet %s: %s", parquet_path, exc)
        return [], 0, 0

    indices = list(range(table.num_rows))
    if 0 < num_per_file < table.num_rows:
        rng = random.Random(_worker_seed ^ hash(parquet_path_str))
        indices = rng.sample(indices, num_per_file)

    names = set(table.schema.names)
    if "audio" not in names:
        logger.error("no audio column in %s; skipping", parquet_path)
        return [], 0, 0

    lines: list[str] = []
    skipped = 0
    audio_col = table.column("audio")
    answer_col = table.column("answer") if "answer" in names else None
    text_col = table.column("text") if "text" in names else None
    name_col = table.column("name") if "name" in names else None
    index_col = table.column("index") if "index" in names else None

    for row_idx in indices:
        answer = ""
        if answer_col is not None and answer_col[row_idx].as_py():
            answer = str(answer_col[row_idx].as_py()).strip()
        if not answer and text_col is not None and text_col[row_idx].as_py():
            answer = str(text_col[row_idx].as_py()).strip()
        if is_abnormal_text(answer):
            skipped += 1
            continue

        audio_cell = audio_col[row_idx].as_py()
        audio_bytes = row_audio_bytes(audio_cell)
        if not audio_bytes:
            skipped += 1
            continue

        orig_path = audio_cell.get("path") if isinstance(audio_cell, dict) else None
        if orig_path:
            stem = Path(orig_path).stem
        elif name_col is not None and index_col is not None:
            stem = f"{name_col[row_idx].as_py() or ''}_{index_col[row_idx].as_py()}"
        else:
            stem = f"{split_name}_{shard_idx}_{row_idx}"

        wav_path = (
            Path(_worker_audio_dir)
            / split_name
            / f"shard{shard_idx:04d}"
            / f"{clean_stem(stem)}.wav"
        )
        if not write_wav(wav_path, audio_bytes):
            skipped += 1
            continue

        sample = {
            "messages": [
                {"role": "user", "content": INSTRUCTION},
                {"role": "assistant", "content": answer},
            ],
            "audios": [str(wav_path)],
            "sample_type": "positive",
        }
        lines.append(json.dumps(sample, ensure_ascii=False))

    return lines, len(lines), skipped


def convert(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    audio_dir = Path(args.audio_dir)
    split_map = discover_parquet_files(data_dir, args.splits)
    if not split_map:
        raise SystemExit(f"no parquet files found under {data_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    logger.info("found splits: %s", ", ".join(split_map))

    merged_file = None
    if not args.one_file_per_split:
        merged_file = (out_dir / f"{args.data_name}.jsonl").open("w", encoding="utf-8")

    total_written = 0
    total_skipped = 0
    try:
        for split_name, shards in sorted(split_map.items()):
            num_per_file = 0
            if args.num > 0:
                num_per_file = max(1, (args.num + len(shards) - 1) // len(shards))
            tasks = [(str(path), split_name, shard_idx, num_per_file) for path, shard_idx in shards]

            split_file = None
            if args.one_file_per_split:
                split_file = (out_dir / f"{args.data_name}_{split_name}.jsonl").open(
                    "w", encoding="utf-8"
                )
            try:
                fout = split_file or merged_file
                assert fout is not None
                split_written = 0
                split_skipped = 0
                with Pool(
                    args.num_workers,
                    initializer=worker_init,
                    initargs=(str(audio_dir), args.seed),
                ) as pool:
                    for lines, written, skipped in pool.imap_unordered(
                        process_parquet_file, tasks, chunksize=1
                    ):
                        for line in lines:
                            fout.write(line + "\n")
                        split_written += written
                        split_skipped += skipped
                total_written += split_written
                total_skipped += split_skipped
                logger.info(
                    "split=%s written=%d skipped=%d",
                    split_name,
                    split_written,
                    split_skipped,
                )
            finally:
                if split_file is not None:
                    split_file.close()
    finally:
        if merged_file is not None:
            merged_file.close()

    logger.info("done. total written=%d skipped=%d out_dir=%s", total_written, total_skipped, out_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Directory containing parquet shards")
    parser.add_argument("--out-dir", required=True, help="Output directory for JSONL files")
    parser.add_argument("--audio-dir", required=True, help="Directory for extracted WAV files")
    parser.add_argument("--splits", nargs="*", default=None, help="Optional split names to process")
    parser.add_argument("--num", type=int, default=0, help="Max samples per split; 0 means all")
    parser.add_argument("--num-workers", type=int, default=8, help="Parallel worker processes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subsampling")
    parser.add_argument("--data-name", default="vitw", help="Output JSONL stem name")
    parser.add_argument("--one-file-per-split", action="store_true", help="Write one JSONL per split")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    convert(build_parser().parse_args())


if __name__ == "__main__":
    main()
