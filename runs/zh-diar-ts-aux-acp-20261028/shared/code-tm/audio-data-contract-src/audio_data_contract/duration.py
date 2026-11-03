"""Parallel duration measurements for manifests, audio files, and catalog splits."""

from __future__ import annotations

import io
import json
import math
import multiprocessing
import os
import resource
import sys
import time
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    wait,
)
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from queue import Empty, Full

from .catalog import load_catalog, resolve_artifact
from .declarations import (
    declaration_files,
    editable_declarations,
    write_declarations,
)
from .overview import update_data_overview
from .roots import load_roots

BLOCK_BYTES = 8 * 1024 * 1024
PIPELINE_BYTES = 32 * 1024 * 1024
AUDIO_EXTENSIONS = {
    ".wav",
    ".flac",
    ".ogg",
    ".opus",
    ".mp3",
    ".m4a",
    ".aac",
    ".wma",
    ".aiff",
    ".aif",
    ".sph",
    ".amr",
    ".webm",
    ".mp4",
    ".au",
}


def cpu_resources() -> dict:
    available = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else os.cpu_count() or 1
    )
    quota = None
    for directory in _cgroups():
        path = directory / "cpu.max"
        if path.exists():
            limit, period = path.read_text().split()
            if limit != "max":
                value = int(limit) / int(period)
                quota = value if quota is None else min(quota, value)
    return {
        "logical_cpus": os.cpu_count(),
        "available_cpus": available,
        "quota_cpus": quota,
    }


def _cgroups():
    root = Path("/sys/fs/cgroup")
    membership = Path("/proc/self/cgroup")
    if membership.exists():
        for line in membership.read_text().splitlines():
            if line.startswith("0::"):
                current = root / line[3:].lstrip("/")
                while current.is_relative_to(root):
                    yield current
                    current = current.parent
                return
    yield root


def _throttling():
    result = {}
    for directory in _cgroups():
        path = directory / "cpu.stat"
        if path.exists():
            values = dict(line.split() for line in path.read_text().splitlines())
            result[str(directory)] = {
                k: int(values.get(k, 0)) for k in ("nr_throttled", "throttled_usec")
            }
    return result


def _seconds(row: dict) -> float:
    value = row.get("duration")
    if value is None and row.get("type") == "MixedCut":
        value = max(
            float(track["offset"]) + _seconds(track["cut"]) for track in row["tracks"]
        )
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise TypeError("duration must be a number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError("duration must be finite and positive")
    return float(value)


def _sum_lines(lines, clean: bool) -> dict:
    import orjson

    count = skipped = 0

    def values():
        nonlocal count, skipped
        for number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                row = orjson.loads(line)
                if clean:
                    passed = row.get("custom", {}).get("clean", {}).get("pass")
                    if not isinstance(passed, bool):
                        raise ValueError("missing boolean custom.clean.pass")
                    if not passed:
                        skipped += 1
                        continue
                value = _seconds(row)
                count += 1
                yield value
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                raise ValueError(f"line within block {number}: {exc}") from exc

    seconds = math.fsum(values())
    return {"count": count, "skipped": skipped, "seconds": seconds, "errors": []}


def _run_job(job):
    key, mode, payload, clean = job
    try:
        if mode == "error":
            raise ValueError(payload)
        if mode == "block":
            return key, _sum_lines(payload.splitlines(), clean)
        if mode == "shared":
            name, size = payload
            memory = SharedMemory(name=name)
            try:
                return key, _sum_lines(bytes(memory.buf[:size]).splitlines(), clean)
            finally:
                memory.close()
        if mode == "range":
            path, start, end = payload
            with open(path, "rb") as stream:
                if start:
                    stream.seek(start - 1)
                    if stream.read(1) != b"\n":
                        stream.readline()

                def lines():
                    while stream.tell() < end:
                        line = stream.readline()
                        if not line:
                            break
                        yield line

                return key, _sum_lines(lines(), clean)
        if mode == "gzip":
            import rapidgzip

            with (
                rapidgzip.open(payload, parallelization=1) as raw,
                io.BufferedReader(raw, buffer_size=BLOCK_BYTES) as stream,
            ):
                return key, _sum_lines(stream, clean)
        if mode == "audio":
            import soundfile

            values, errors = [], []
            for path in payload:
                try:
                    info = soundfile.info(path)
                    seconds = info.frames / info.samplerate
                    values.append(_seconds({"duration": seconds}))
                except (OSError, RuntimeError, ValueError, ZeroDivisionError) as exc:
                    errors.append(f"{path}: {exc}")
            return key, {
                "count": len(values),
                "skipped": 0,
                "seconds": math.fsum(values),
                "errors": errors,
            }
        raise ValueError(f"unknown job mode {mode}")
    except (
        OSError,
        RuntimeError,
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        OverflowError,
    ) as exc:
        return key, {
            "count": 0,
            "skipped": 0,
            "seconds": 0,
            "errors": [f"{mode}: {exc}"],
        }


def _execute_job(job):
    start = time.process_time()
    key, result = _run_job(job)
    result["cpu_seconds"] = time.process_time() - start
    return key, result


def _jobs(sources, decompressors, pipeline):
    seen_audio = set()
    for key, source in sources.items():
        path, clean = source["path"], source.get("clean", False)
        try:
            if source["mode"] == "audio":
                if not Path(path).is_dir():
                    raise ValueError(f"audio directory is unavailable: {path}")
                batch = []
                errors = []
                for directory, _, names in os.walk(path, onerror=errors.append):
                    for name in names:
                        audio = Path(directory) / name
                        if audio.suffix.lower() not in AUDIO_EXTENSIONS:
                            continue
                        resolved = str(audio.resolve())
                        identity = (key, resolved)
                        if identity in seen_audio:
                            continue
                        seen_audio.add(identity)
                        batch.append(resolved)
                        if len(batch) >= 64:
                            yield key, "audio", batch, False
                            batch = []
                if batch:
                    yield key, "audio", batch, False
                for error in errors:
                    yield key, "error", str(error), False
            elif not Path(path).is_file():
                raise FileNotFoundError(f"manifest is unavailable: {path}")
            elif not path.endswith(".gz"):
                size = Path(path).stat().st_size
                for start in range(0, size, BLOCK_BYTES):
                    yield (
                        key,
                        "range",
                        (path, start, min(size, start + BLOCK_BYTES)),
                        clean,
                    )
            elif not pipeline or (
                len(sources) > 1 and Path(path).stat().st_size < PIPELINE_BYTES
            ):
                yield key, "gzip", path, clean
            else:
                import rapidgzip

                with (
                    rapidgzip.open(path, parallelization=decompressors) as raw,
                    io.BufferedReader(raw, buffer_size=BLOCK_BYTES) as stream,
                ):
                    while block := stream.read(BLOCK_BYTES):
                        if not block.endswith(b"\n"):
                            block += stream.readline()
                        yield key, "block", block, clean
        except (OSError, RuntimeError, ValueError) as exc:
            yield key, "error", f"{path}: {exc}", clean


def _cpu_time():
    return sum(
        getattr(resource.getrusage(who), attr)
        for who in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN)
        for attr in ("ru_utime", "ru_stime")
    )


def _producer_init(ready, stopped, slots, use_shared):
    global _ready, _stopped, _slots, _use_shared
    _ready, _stopped, _slots, _use_shared = ready, stopped, slots, use_shared


def _release_shared(job, slots):
    if job[1] == "shared":
        memory = SharedMemory(name=job[2][0])
        memory.close()
        memory.unlink()
        slots.release()


def _produce_gzip(key, source, threads):
    for job in _jobs({key: source}, threads, True):
        if _stopped.is_set():
            return
        key, mode, payload, clean = job
        if mode == "block" and _use_shared and len(payload) <= BLOCK_BYTES * 2:
            while not _slots.acquire(timeout=0.1):
                if _stopped.is_set():
                    return
            try:
                memory = SharedMemory(create=True, size=len(payload))
                memory.buf[: len(payload)] = payload
                job = key, "shared", (memory.name, len(payload)), clean
                memory.close()
            except OSError:
                _slots.release()
                raise
        while not _stopped.is_set():
            try:
                _ready.put(job, timeout=0.1)
                break
            except Full:
                continue
        else:
            _release_shared(job, _slots)
            return


def _large_gzip(source, source_count):
    path = Path(source["path"])
    return (
        source["mode"] == "manifest"
        and path.suffix == ".gz"
        and path.is_file()
        and (source_count == 1 or path.stat().st_size >= PIPELINE_BYTES)
    )


def _parallel_jobs(sources, decompressors, parsers, context, slots, use_shared):
    """Independent decoder processes feed shared blocks to the parsing pool."""
    large, other = {}, {}
    for key, source in sources.items():
        selected = _large_gzip(source, len(sources))
        (large if selected else other)[key] = source
    readers = min(len(large), max(1, decompressors // 2), 8)
    if not readers:
        yield from _jobs(other, 0, False)
        return
    ready = context.Queue(maxsize=max(2, parsers))
    stopped = context.Event()
    with ProcessPoolExecutor(
        max_workers=readers,
        mp_context=context,
        initializer=_producer_init,
        initargs=(ready, stopped, slots, use_shared),
    ) as pool:
        futures = [
            pool.submit(
                _produce_gzip, key, source, max(1, decompressors // readers - 1)
            )
            for key, source in sorted(
                large.items(),
                key=lambda item: Path(item[1]["path"]).stat().st_size,
                reverse=True,
            )
        ]
        try:
            while True:
                try:
                    yield ready.get(timeout=0.1)
                except Empty:
                    if all(future.done() for future in futures):
                        for future in futures:
                            future.result()
                        break
            yield from _jobs(other, 0, False)
        finally:
            stopped.set()
            while True:
                try:
                    _release_shared(ready.get(timeout=0.1), slots)
                except Empty:
                    if all(future.done() for future in futures):
                        break
            ready.close()
            ready.join_thread()


def measure(sources: dict, workers: int) -> dict:
    """Measure unique sources with bounded outstanding work and a shared CPU budget."""
    if workers < 1:
        raise ValueError("workers must be positive")
    start, cpu_start = time.monotonic(), _cpu_time()
    parent_cpu_start, throttling_start = time.process_time(), _throttling()
    pipeline = workers > 1 and any(
        _large_gzip(s, len(sources)) for s in sources.values()
    )
    decompressors = max(1, workers // 4) if pipeline else 0
    parsers = max(1, workers - decompressors)
    results = {
        key: {"count": 0, "skipped": 0, "errors": [], "parts": []} for key in sources
    }
    last_progress, completed_cpu = start, 0.0

    def consume(result):
        nonlocal last_progress, completed_cpu
        key, part = result
        target = results[key]
        for field in ("count", "skipped"):
            target[field] += part[field]
        target["errors"].extend(part["errors"])
        target["parts"].append(part["seconds"])
        completed_cpu += part["cpu_seconds"]
        now = time.monotonic()
        if now - last_progress >= 5:
            count = sum(r["count"] for r in results.values())
            cpu = time.process_time() - parent_cpu_start
            if workers > 1:
                cpu += completed_cpu
            print(
                f"duration: {count:,} records, {count / (now - start):,.0f}/s, "
                f"{now - start:.1f}s elapsed, completed CPU={cpu:.1f}s",
                file=sys.stderr,
                flush=True,
            )
            last_progress = now

    if workers == 1:
        for job in _jobs(sources, 0, False):
            consume(_execute_job(job))
    else:
        # Use shared memory where available; bound occupancy to half the free space.
        shared_slots = 0
        if Path("/dev/shm").is_dir():
            storage = os.statvfs("/dev/shm")
            shared_slots = min(
                parsers * 2, storage.f_bavail * storage.f_frsize // (2 * BLOCK_BYTES)
            )
        context = multiprocessing.get_context("spawn")
        slots = context.Semaphore(max(1, shared_slots))
        jobs = (
            _parallel_jobs(
                sources, decompressors, parsers, context, slots, bool(shared_slots)
            )
            if pipeline
            else _jobs(sources, 0, False)
        )
        with ProcessPoolExecutor(max_workers=parsers, mp_context=context) as pool:
            pending = {}

            def drain():
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    job = pending.pop(future)
                    try:
                        consume(future.result())
                    finally:
                        _release_shared(job, slots)

            try:
                for job in jobs:
                    pending[pool.submit(_execute_job, job)] = job
                    # Shared slots can be the tighter bound: drain before awaiting
                    # another producer block or every producer could wait on us.
                    limit = (
                        min(parsers * 2, shared_slots) if shared_slots else parsers * 2
                    )
                    if len(pending) >= limit:
                        drain()
                while pending:
                    drain()
            finally:
                jobs.close()
                for job in pending.values():
                    _release_shared(job, slots)
    for key, result in results.items():
        seconds = math.fsum(result.pop("parts"))
        if not result["count"] and not result["errors"]:
            result["errors"].append("no accepted records or audio files")
        result.update(
            path=sources[key]["path"],
            seconds=seconds,
            hours=seconds / 3600,
            status="error" if result["errors"] else "ok",
        )
    elapsed, cpu = time.monotonic() - start, _cpu_time() - cpu_start
    count = sum(r["count"] for r in results.values())
    throttling = _throttling()
    for path, values in throttling.items():
        for name in values:
            values[name] -= throttling_start.get(path, {}).get(name, 0)
    return {
        "resources": cpu_resources(),
        "workers": workers,
        "cgroup_throttling_delta": throttling,
        "elapsed_seconds": elapsed,
        "cpu_seconds": cpu,
        "average_cpu_cores": cpu / elapsed,
        "records_per_second": count / elapsed,
        "sources": results,
    }


def catalog_sources(catalog_path, roots):
    catalog = load_catalog(catalog_path)
    sources, targets = {}, []
    for spec in catalog:
        if (
            spec.provenance.get("inventory_status") == "download_planned"
            or spec.provenance.get("inventory_category") == "training_mixture"
        ):
            continue
        for split_name, split in spec.splits.items():
            if split.group or any(
                split.statistics.get(k) is not None
                for k in ("duration_hours", "hours")
            ):
                continue
            target = {"dataset": spec.key, "split": split_name, "sources": []}
            targets.append(target)
            try:
                filtering = spec.recipe_parameters.get("filters")
                if filtering not in (None, "custom.clean.pass=false"):
                    raise ValueError(f"unsupported filter: {filtering}")
                clean = filtering is not None
                paths = []
                for role in ("cuts", "supervisions", "recordings"):
                    refs = split.artifacts.get(role, ())
                    if refs:
                        paths = [
                            resolve_artifact(
                                catalog, spec.dataset_id, spec.version, ref, roots
                            )
                            for ref in refs
                        ]
                        break
                else:
                    refs = split.artifacts.get("source", ())
                    if (
                        not refs
                        or split.provenance.get("split_policy")
                        or clean
                        or any(
                            spec.artifact(ref).kind != "source-directory"
                            for ref in refs
                        )
                        or any(
                            set(refs) & set(other.artifacts.get("source", ()))
                            for name, other in spec.splits.items()
                            if name != split_name
                        )
                    ):
                        raise ValueError(
                            "no supported manifest or unambiguous split audio directory"
                        )
                    role = "audio"
                    paths = [
                        resolve_artifact(
                            catalog, spec.dataset_id, spec.version, ref, roots
                        )
                        for ref in refs
                    ]
                if clean and role == "recordings":
                    raise ValueError("clean subset requires segment manifests")
                target["basis"] = f"sum of {role} durations"
                if role in ("cuts", "supervisions"):
                    target["basis"] += (
                        "; overlapping segments may count speech more than once"
                    )
                if clean:
                    target["basis"] += "; custom.clean.pass=true only"
                target["basis"] += "; sources: " + ", ".join(
                    str(p)
                    if not roots
                    else next(
                        (
                            f"{alias}:{p.relative_to(root)}"
                            for alias, root in roots.items()
                            if p.is_relative_to(root)
                        ),
                        p.name,
                    )
                    for p in paths
                )
                for path in paths:
                    key = json.dumps([str(path), role == "audio", clean])
                    sources[key] = {
                        "path": str(path),
                        "mode": "audio" if role == "audio" else "manifest",
                        "clean": clean,
                    }
                    if key not in target["sources"]:
                        target["sources"].append(key)
            except (ValueError, KeyError) as exc:
                target["error"] = str(exc)
    return sources, targets


def fill_catalog(catalog_path, targets):
    updates = {(r["dataset"], r["split"]): r for r in targets if r["status"] == "ok"}
    for path in declaration_files(catalog_path):
        lines, changed = [], False
        if path.suffix == ".jsonl":
            entries = [
                (line, json.loads(line))
                if line.strip() and not line.lstrip().startswith("#") else (line, None)
                for line in path.read_text().splitlines(keepends=True)
            ]
        else:
            rows = editable_declarations(path)
            entries = [(None, row) for row in rows]
        for line, row in entries:
            if row is None:
                lines.append(line)
                continue
            dirty = False
            for name, split in row["splits"].items():
                result = updates.get((f"{row['dataset_id']}@{row['version']}", name))
                stats = split.get("statistics", {})
                if result and not any(
                    stats.get(k) is not None for k in ("hours", "duration_hours")
                ):
                    split.setdefault("statistics", {}).update(
                        duration_hours=result["hours"], duration_basis=result["basis"]
                    )
                    dirty = True
            if line is not None:
                lines.append(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                    if dirty else line
                )
            changed |= dirty
        if changed:
            if path.suffix == ".jsonl":
                temporary = path.with_suffix(path.suffix + ".duration.tmp")
                temporary.write_text("".join(lines))
                temporary.replace(path)
            else:
                write_declarations(path, rows)


def run(args) -> int:
    try:
        import orjson  # noqa: F401
        import rapidgzip  # noqa: F401
        import soundfile  # noqa: F401
    except ImportError as exc:
        raise SystemExit("Install duration support: pip install '.[duration]'") from exc
    workers = (
        args.workers if args.workers is not None else cpu_resources()["available_cpus"]
    )
    if workers < 1:
        raise SystemExit("--workers must be positive")
    if args.write and not args.catalog:
        raise SystemExit("--write requires --catalog")
    targets = None
    if args.catalog:
        sources, targets = catalog_sources(args.catalog, load_roots(args.roots))
    else:
        mode = "manifest" if args.manifest else "audio"
        sources = {
            str(Path(p).resolve()): {"path": str(Path(p).resolve()), "mode": mode}
            for p in (args.manifest or args.audio_dir)
        }
    print(
        f"duration: workers={workers}, resources={cpu_resources()}, "
        f"unique sources={len(sources)}",
        file=sys.stderr,
        flush=True,
    )
    report = measure(sources, workers)
    if targets is not None:
        for target in targets:
            parts = [report["sources"][key] for key in target["sources"]]
            errors = ([target["error"]] if "error" in target else []) + [
                error for part in parts for error in part["errors"]
            ]
            target["status"] = "error" if errors else "ok"
            target["errors"] = errors
            if not errors:
                target["hours"] = math.fsum(p["seconds"] for p in parts) / 3600
        report["splits"] = targets
        if args.write:
            fill_catalog(args.catalog, targets)
            update_data_overview("docs/datasets/data-overview.md", catalog_path=args.catalog)
            report["written_splits"] = sum(t["status"] == "ok" for t in targets)
    failed = sum(
        r["status"] == "error"
        for r in (targets if targets is not None else report["sources"].values())
    )
    report["failed"] = failed
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(rendered + "\n")
        print(
            json.dumps(
                {
                    "output": args.output,
                    "failed": failed,
                    "elapsed_seconds": report["elapsed_seconds"],
                    "average_cpu_cores": report["average_cpu_cores"],
                }
            )
        )
    else:
        print(rendered)
    return 1 if failed else 0
