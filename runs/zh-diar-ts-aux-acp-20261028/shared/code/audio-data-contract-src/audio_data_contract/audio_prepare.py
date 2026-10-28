"""Opt-in, byte-preserving preparation of Lhotse audio sources."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import shlex
import shutil
import tarfile
import tempfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import islice
from pathlib import Path


@dataclass
class Dependency:
    kind: str
    path: Path
    selector: str | tuple[int, int] | None
    target: Path
    context: dict
    status: str = "pending"
    size: int = 0

    @property
    def identity(self):
        return [self.kind, str(self.path), self.selector]


def _open(path, mode):
    if str(path).endswith(".gz"):
        return gzip.open(path, mode + "t", encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def recording_sources(item):
    """Visit only Lhotse's known audio-bearing structures."""
    kind = item.get("type")
    if kind == "MixedCut":
        for track in item["tracks"]:
            yield from recording_sources(track["cut"])
    elif kind == "PaddingCut":
        return
    elif kind in {"MonoCut", "MultiCut"}:
        if "recording" not in item:
            raise ValueError("cut has no raw recording")
        yield from recording_sources(item["recording"])
    elif kind is None and "sources" in item and "sampling_rate" in item:
        if not item["sources"]:
            raise ValueError("recording has no audio sources")
        yield from item["sources"]
        for transform in item.get("transforms", []):
            if transform["name"] == "ReverbWithImpulseResponse":
                rir = transform.get("kwargs", {}).get("rir")
                if rir is not None:
                    yield from recording_sources(rir)
    else:
        raise ValueError(f"unsupported manifest item: {kind!r}")


def _path(value, root):
    if not isinstance(value, str) or not value or "://" in value:
        raise ValueError(f"unsupported file path: {value!r}")
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def parse_audio_source(source, root, extract):
    if source["type"] == "file":
        return "file", _path(source["source"], root), None
    if source["type"] != "command" or not extract:
        raise ValueError(f"unsupported audio source type: {source['type']}")
    tokens = shlex.split(source["source"])
    if tokens[:1] == ["timeout"]:
        if not (
            len(tokens) >= 5
            and tokens[1] == "--signal=TERM"
            and re.fullmatch(r"--kill-after=[0-9]+s", tokens[2])
            and re.fullmatch(r"[0-9]+s", tokens[3])
        ):
            raise ValueError("unsupported timeout command template")
        tokens = tokens[4:]
    if len(tokens) == 4 and tokens[:2] == ["tar", "-xOf"]:
        if tokens[2].startswith("-") or tokens[3].startswith("-"):
            raise ValueError("tar options are not audio paths")
        return "tar", _path(tokens[2], root), tokens[3]
    if (
        len(tokens) == 6
        and tokens[0] == "dd"
        and tokens[1].startswith("if=")
        and tokens[2] == "iflag=skip_bytes,count_bytes"
        and re.fullmatch(r"skip=[0-9]+", tokens[3])
        and re.fullmatch(r"count=[0-9]+", tokens[4])
        and tokens[5] == "status=none"
    ):
        offset, size = int(tokens[3][5:]), int(tokens[4][6:])
        if size <= 0:
            raise ValueError("dd count must be positive")
        return "dd", _path(tokens[1][3:], root), (offset, size)
    raise ValueError("unsupported command template (commands are never executed)")


def _signature(path):
    stat = path.stat()
    if not path.is_file():
        raise ValueError(f"not a regular file: {path}")
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _receipt(dep):
    return dep.target.with_name(dep.target.name + ".complete.json")


def _reuse(dep, signature):
    try:
        record = json.loads(_receipt(dep).read_text(encoding="utf-8"))
        identity = json.loads(json.dumps(dep.identity))
        if (
            record["source"] == identity
            and record["signature"] == signature
            and dep.target.is_file()
            and dep.target.stat().st_size == record["bytes"]
        ):
            dep.status, dep.size = "reused", record["bytes"]
            return True
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return False


def _atomic_write(path, write):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".partial-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _materialize(dep, stream, size, signature):
    def copy(output):
        remaining = size
        while remaining:
            block = stream.read(min(1024 * 1024, remaining))
            if not block:
                raise ValueError("short audio read")
            output.write(block)
            remaining -= len(block)
        if _signature(dep.path) != signature:
            raise ValueError("source changed during preparation")

    _atomic_write(dep.target, copy)
    record = {"source": dep.identity, "signature": signature, "bytes": size}
    _atomic_write(
        _receipt(dep),
        lambda out: out.write(json.dumps(record).encode("utf-8")),
    )
    dep.status = "copied" if dep.kind == "file" else "extracted"
    dep.size = size


def _prepare_group(deps, extract):
    failures = []

    def fail(dep, error):
        failures.append({**dep.context, "error": str(error)})

    try:
        signature = _signature(deps[0].path)
        if deps[0].kind == "tar":
            requested = {dep.selector: dep for dep in deps}
            seen = set()
            # Scan to the end even when all targets are found: duplicate names
            # would make a single-member extraction semantically ambiguous.
            with tarfile.open(deps[0].path, "r:*") as archive:
                for member in archive:
                    dep = requested.get(member.name)
                    if dep is None:
                        continue
                    if member.name in seen:
                        raise ValueError(f"duplicate tar member: {member.name}")
                    seen.add(member.name)
                    if not member.isfile():
                        raise ValueError(f"not a regular tar member: {member.name}")
                    if _reuse(dep, signature):
                        continue
                    with archive.extractfile(member) as stream:
                        _materialize(dep, stream, member.size, signature)
            for name in requested.keys() - seen:
                fail(requested[name], f"missing tar member: {name}")
        else:
            with deps[0].path.open("rb") as stream:
                for dep in deps:
                    try:
                        if dep.kind == "file" and extract:
                            dep.status, dep.size = "unchanged", signature["size"]
                            continue
                        if _reuse(dep, signature):
                            continue
                        offset, size = (
                            dep.selector if dep.kind == "dd" else (0, signature["size"])
                        )
                        if offset + size > signature["size"]:
                            raise ValueError("dd range exceeds archive size")
                        stream.seek(offset)
                        _materialize(dep, stream, size, signature)
                    except (OSError, ValueError) as error:
                        fail(dep, error)
        if _signature(deps[0].path) != signature:
            raise ValueError("source changed during preparation")
    except (OSError, ValueError, tarfile.TarError, EOFError) as error:
        for dep in deps:
            fail(dep, error)
    return failures


def run(args):
    """Prepare an entire invocation before publishing any output manifest."""
    inputs = [Path(p).resolve() for p in args.manifest]
    cache, output = Path(args.cache_dir).resolve(), Path(args.output_dir).resolve()
    root = Path(args.source_root or Path.cwd()).resolve()
    extract = args.command == "extract-audio"
    summary = {
        "inputs": [str(p) for p in inputs],
        "cache_dir": str(cache),
        "output_dir": str(output),
        "outputs": [],
        "references": 0,
        "unique_files": 0,
        "failures": [],
    }
    dependencies = {}
    stage = None
    prepared = False
    context = {}
    try:
        if args.workers < 1:
            raise ValueError("workers must be positive")
        if output.exists():
            raise ValueError(f"output directory already exists: {output}")
        if cache == output or cache.is_relative_to(output):
            raise ValueError("cache directory must not be inside output directory")
        if len({p.name for p in inputs}) != len(inputs):
            raise ValueError("input manifest names collide; run separately")
        for path in inputs:
            if not (path.name.endswith(".jsonl") or path.name.endswith(".jsonl.gz")):
                raise ValueError(f"unsupported manifest format: {path}")
        output.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=".audio-manifests-", dir=output.parent))
        for path in inputs:
            context = {"manifest": str(path)}
            with _open(path, "r") as source, _open(stage / path.name, "w") as dest:
                for line_number, line in enumerate(source, 1):
                    if not line.strip():
                        continue
                    context = {"manifest": str(path), "line": line_number}
                    item = json.loads(line)
                    context["record_id"] = item.get("id")
                    for audio in recording_sources(item):
                        context["source"] = audio.get("source")
                        summary["references"] += 1
                        kind, origin, selector = parse_audio_source(audio, root, extract)
                        key = (kind, origin, selector)
                        if key not in dependencies:
                            digest = hashlib.sha256(
                                json.dumps([kind, str(origin), selector]).encode()
                            ).hexdigest()
                            suffix = (
                                Path(selector).suffix
                                if kind == "tar"
                                else origin.suffix
                            )
                            if kind == "dd":
                                suffix = (
                                    ".wav"  # Existing dialect builder reads WAV bytes.
                                )
                            target = (cache / digest[:2] / (digest + suffix)).resolve()
                            if target == origin:
                                raise ValueError("cache target aliases original audio")
                            dependencies[key] = Dependency(
                                kind, origin, selector, target, dict(context)
                            )
                        dep = dependencies[key]
                        if kind != "file" or not extract:
                            audio["type"], audio["source"] = "file", str(dep.target)
                    dest.write(json.dumps(item, ensure_ascii=False) + "\n")
            summary["outputs"].append(
                {"input": str(path), "output": str(output / path.name)}
            )
        context = {}
        protected = set(inputs) | {dep.path for dep in dependencies.values()}
        for dep in dependencies.values():
            if dep.kind == "file" and extract:
                continue
            if dep.target in protected or _receipt(dep).resolve() in protected:
                raise ValueError("cache output aliases an input audio or manifest")
        groups = {}
        for dep in dependencies.values():
            groups.setdefault((dep.kind, dep.path), []).append(dep)
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            pending = iter(groups.values())
            while batch := list(islice(pending, args.workers * 2)):
                for failures in executor.map(
                    lambda group: _prepare_group(group, extract), batch
                ):
                    summary["failures"].extend(failures)
        prepared = True
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        AttributeError,
        EOFError,
        zlib.error,
    ) as error:
        summary["failures"].append({**context, "error": str(error)})
    except KeyboardInterrupt:
        summary["failures"].append(
            {"error": "interrupted; completed audio can be reused"}
        )
    finally:
        if not prepared and not summary["failures"]:
            summary["failures"].append({"error": "preparation aborted"})
        summary["unique_files"] = len(dependencies)
        for status in ("copied", "extracted", "reused", "unchanged"):
            matches = [d for d in dependencies.values() if d.status == status]
            summary[status] = len(matches)
            summary[status + "_bytes"] = sum(d.size for d in matches)
        summary["status"] = "failed" if summary["failures"] else "complete"
        if not summary["failures"]:
            try:
                (stage / "summary.json").write_text(
                    json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                if output.exists():
                    raise ValueError(f"output directory already exists: {output}")
                stage.rename(output)
            except (OSError, ValueError) as error:
                summary["failures"].append({"error": str(error)})
                summary["status"] = "failed"
        if stage is not None and stage.exists():
            shutil.rmtree(stage)
        print(json.dumps(summary, ensure_ascii=False))
    return 1 if summary["failures"] else 0
