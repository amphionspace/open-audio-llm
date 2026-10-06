"""Mirror local run artifacts to object storage through the AmphionBucket `ab` CLI."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

# --json transfer results and symlink-safe state directories first shipped in 0.5.0.
MIN_AB_VERSION = (0, 5, 0)

# Trainer/DeepSpeed state needed only to resume optimisation, not to use the weights.
RESUME_STATE_FILES = {"optimizer.pt", "scheduler.pt", "scaler.pt", "latest", "zero_to_fp32.py"}
RESUME_STATE_PREFIXES = ("rng_state", "global_step")


def check(storage):
    for key in ("executable", "remote", "local_root"):
        if not storage.get(key):
            raise ValueError(f"storage.{key} is required")
    if ":" not in storage["remote"]:
        raise ValueError("storage.remote must be bucket:prefix")


def remote_path(local, storage):
    relative = Path(os.path.relpath(os.path.abspath(local), os.path.abspath(storage["local_root"])))
    if relative.parts and relative.parts[0] == "..":
        raise ValueError(f"{local} is outside storage.local_root")
    return storage["remote"].rstrip("/") + "/" + relative.as_posix()


def regular_files(path):
    path = Path(path)
    if path.is_symlink():
        return 0
    if path.is_file():
        return 1
    # `ab` skips symlinks inside directories; os.walk does not follow linked dirs.
    return sum(
        not os.path.islink(os.path.join(root, name))
        for root, _, names in os.walk(path) for name in names
    )


def preflight_problems(storage, local):
    """Problems that would make every transfer of this task fail, checked before it starts."""
    executable = storage["executable"]
    if not (os.path.isfile(executable) and os.access(executable, os.X_OK)):
        return [f"AmphionBucket ab is not executable: {executable}"]
    problems = []
    try:
        remote_path(local, storage)
    except ValueError as exc:
        problems.append(str(exc))
    code, version = run([executable, "--version"], timeout=60)
    found = re.search(r"(\d+)\.(\d+)\.(\d+)", version)
    if code != 0 or not found or tuple(map(int, found.groups())) < MIN_AB_VERSION:
        wanted = ".".join(map(str, MIN_AB_VERSION))
        return problems + [f"AmphionBucket >= {wanted} is required; found {version.strip()!r}"]
    bucket = storage["remote"].split(":", 1)[0]
    code, listed = run([executable, "--json", "credentials"], timeout=60)
    sources = {row.get("bucket"): row.get("source") for row in json_lines(listed)}
    if code != 0 or sources.get(bucket) in (None, "missing"):
        found = sources.get(bucket, tail(listed, 5) if code else "not configured")
        problems.append(f"ab has no bucket {bucket} with credentials (credential source: {found})")
    return problems


class StorageError(RuntimeError):
    """A failed transfer together with the diagnosed cause."""

    def __init__(self, problem):
        super().__init__(f"{problem['cause']}: {problem['detail']}")
        self.problem = problem


def run(cmd, timeout=None):
    try:
        process = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    return process.returncode, (process.stdout + process.stderr).strip()


def tail(output, lines=20):
    return "\n".join(output.splitlines()[-lines:])


def json_lines(output):
    """`ab --json` results; progress and notices on the same streams are plain text."""
    rows = []
    for line in output.splitlines():
        if line.startswith("{"):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def transfer(cmd):
    code, output = run(cmd)
    rows = json_lines(output)
    summary = next((row for row in reversed(rows) if "summary" in row), None)
    return code, output, [row for row in rows if "op" in row], summary


def diagnose(storage, remote, output, local=None, items=(), summary=None, expected=None, pulling=False):
    """Find why a transfer failed, checking the cheapest decisive cause first."""
    def problem(cause, detail, evidence=output):
        return {"cause": cause, "detail": detail, "remote": remote, "evidence": tail(evidence)}

    executable = storage["executable"]
    if local is not None and not Path(local).exists():
        return problem("local-missing", f"本地路径不存在：{local}")
    if not (os.path.isfile(executable) and os.access(executable, os.X_OK)):
        return problem("ab-missing", f"找不到可执行的 AmphionBucket：{executable}；需安装 ab 或修正 storage.executable")
    code, remotes = run([executable, "--json", "remotes"], timeout=60)
    if code != 0:
        if "--json" in remotes:
            return problem("ab-no-json", "ab 不支持 --json 结果输出；需要 AmphionBucket 0.5.0 及以上", remotes)
        return problem("ab-unusable", "ab remotes 无法运行", remotes)
    bucket = remote.split(":", 1)[0]
    if bucket not in {row.get("bucket") for row in json_lines(remotes)}:
        return problem("bucket-not-configured", f"本机 ab 未配置桶 {bucket}；需安装含该桶配置的 ab 版本", remotes)
    code, credentials = run([executable, "--json", "credentials"], timeout=60)
    entry = next((row for row in json_lines(credentials) if row.get("bucket") == bucket), {})
    if entry.get("source") == "missing":
        profile = entry.get("profile", "")
        return problem(
            "credentials-missing",
            f"桶 {bucket} 没有凭证：在 ~/.config/amphion-bucket/credentials.ini 的 [{profile}] 节"
            f"或 AMPHION_BUCKET_{profile.upper()}_ACCESS_KEY/SECRET_KEY 中配置",
            credentials,
        )
    if summary:
        counts = summary["summary"]
        if counts.get("conflict"):
            lines = [f"{row.get('target')}: {row.get('error', '')}" for row in items if row.get("status") == "conflict"]
            return problem("remote-conflict", f"{counts['conflict']} 个远端文件已存在且内容不同；未覆盖",
                           "\n".join(lines) or output)
        confirmed = counts.get("done", 0) + counts.get("skipped", 0)
        if not counts.get("failed") and "error" not in summary and confirmed != expected:
            return problem("unconfirmed-files", f"ab 只确认了 {confirmed}/{expected} 个文件（符号链接会被跳过）")
    # Listing a missing prefix succeeds empty, so this separates access from absence.
    prefix = remote.split(":", 1)[1].rstrip("/") if pulling else remote.split(":", 1)[1].rstrip("/").rsplit("/", 1)[0]
    code, listing = run([executable, "ls", f"{bucket}:{prefix}/" if prefix else f"{bucket}:"], timeout=120)
    if code != 0:
        return problem("bucket-unreachable", f"无法访问 {bucket}：网络、内网地址或凭证问题；可用 ab --reprobe ls {bucket}: 重新探测", listing)
    if pulling and not listing:
        return problem("remote-missing", f"远端没有 {remote}；该文件可能从未上传成功")
    if summary is None:
        return problem("unrecognized-output", "ab 退出或输出与预期不符，无法确认传输结果")
    reasons = sorted({row["error"] for row in items if row.get("status") == "failed" and row.get("error")})
    if summary.get("error"):
        reasons.append(summary["error"])
    return problem("transfer-failed", "桶可访问，但 ab 传输失败：" + ("；".join(reasons) or "未给出原因"))


def push(source, destination, storage, overwrite=False):
    """Upload and return the transfer summary; raise StorageError unless every regular file is confirmed."""
    source = Path(source)
    if source.is_dir():
        destination = destination.rstrip("/") + "/"
    cmd = [storage["executable"], "--json", "push", str(source), destination]
    if overwrite:
        cmd.append("--overwrite")
    code, output, items, summary = transfer(cmd)
    expected = regular_files(source)
    counts = summary["summary"] if summary else {}
    if (code == 0 and summary and "error" not in summary and not counts.get("conflict")
            and not counts.get("failed") and counts.get("done", 0) + counts.get("skipped", 0) == expected):
        return {"source": str(source), "destination": destination,
                "uploaded": counts["done"], "unchanged": counts["skipped"]}
    raise StorageError(diagnose(storage, destination, output, source, items, summary, expected))


def pull(source, destination, storage):
    code, output, items, summary = transfer(
        [storage["executable"], "--json", "pull", source.rstrip("/") + "/", str(destination)])
    if code != 0:
        raise StorageError(diagnose(storage, source, output, items=items, summary=summary, pulling=True))


def is_resume_state(path):
    return path.name in RESUME_STATE_FILES or path.name.startswith(RESUME_STATE_PREFIXES)


def upload_checkpoint(checkpoint, storage, resume_state):
    """Upload a saved checkpoint; resume_state=False keeps optimizer/RNG state local-only."""
    checkpoint = Path(checkpoint)
    destination = remote_path(checkpoint, storage)
    if resume_state:
        return [push(checkpoint, destination, storage)]
    return [
        push(entry, f"{destination}/{entry.name}" if entry.is_dir() else destination + "/", storage)
        for entry in sorted(checkpoint.iterdir())
        if not entry.is_symlink() and not is_resume_state(entry)
    ]
