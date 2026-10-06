"""Configuration-only public launcher and experiment execution records."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .run_config import absolute, child_environment, command, load_config, preview
from .storage import StorageError, preflight_problems, pull, push, remote_path


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def handoff_line(status):
    handoff = status.get("checkpoint_handoff")
    if not handoff:
        return ""
    location = handoff["checkpoint_remote"] or f"{handoff['checkpoint']}（未上传，仅本地）"
    return f"交付 checkpoint（{handoff['selection']}，step {handoff['global_step']}）：`{location}`。\n\n"


def storage_line(status):
    storage = status.get("storage")
    if not storage:
        return ""
    problems = [storage["problem"]] if "problem" in storage else []
    problems += [p["problem"] for p in storage.get("checkpoint_problems", [])]
    if not problems:
        return f"对象存储：已同步到 `{storage['remote']}`。\n\n"
    causes = "；".join(f"{p['cause']}（{p['detail']}）" for p in problems)
    return f"对象存储：同步有问题：{causes}。\n\n"


def update_readme(path, status):
    start, end = "<!-- execution-status -->", "<!-- /execution-status -->"
    content = (
        path.read_text()
        if path.exists()
        else (
            "# 执行记录\n\n[生效配置](effective.yaml)、[状态](status.json)、[日志](logs/)、[产物](artifacts/)。\n"
        )
    )
    block = (
        f"{start}\n\n执行状态：{status['execution']}；质量状态：{status.get('quality', '未核实')}。\n\n"
        f"开始：{status.get('started', '未知')}；结束：{status.get('ended', '未结束')}；"
        f"退出码：{status.get('exit_code', '未知')}。\n\n{handoff_line(status)}{storage_line(status)}{end}"
    )
    if start in content and end in content:
        left, rest = content.split(start, 1)
        content = left + block + rest.split(end, 1)[1]
    else:
        content = content.rstrip() + "\n\n" + block + "\n"
    path.write_text(content)


def next_attempt(directory):
    numbers = [int(p.name) for p in directory.glob("[0-9]*") if p.name.isdigit()]
    return directory / f"{max(numbers, default=0) + 1:03d}"


def read_status(attempt):
    path = attempt / "status.json"
    return json.loads(path.read_text()) if path.exists() else {}


def allocate(config_path, resume=None, dry_run=False):
    raw = yaml.safe_load(config_path.read_text())
    recording = raw.get("recording", {})
    directory = Path(absolute(recording["attempts_dir"]["path"], config_path.parent))
    attempt = Path(resume).absolute() if resume else next_attempt(directory)
    if resume:
        if attempt.parent != directory or not (attempt / "effective.yaml").is_file():
            raise ValueError("--resume must select an existing attempt of this task")
        status = read_status(attempt)
        if status.get("execution") == "completed":
            raise ValueError("A completed attempt cannot be resumed")
        config = yaml.safe_load((attempt / "effective.yaml").read_text())
        if not config["task"].get("resume_supported"):
            raise ValueError(
                "This task does not support in-place resume; select --task for a new attempt"
            )
        config["recording"]["resuming"] = True
    else:
        config = load_config(config_path, attempt)
    if not dry_run:
        directory.mkdir(parents=True, exist_ok=True)
        if not resume:
            attempt.mkdir(exist_ok=False)
        (attempt / "logs").mkdir(exist_ok=True)
        (attempt / "artifacts").mkdir(exist_ok=True)
    return config, attempt


def sync_attempt(attempt, storage, status):
    """Mirror the whole execution record; a storage failure never changes the exit code."""
    remote = remote_path(attempt, storage)
    status["storage"] = {"remote": remote}
    # Training uploads checkpoints itself; surface what it could not upload.
    failed = [
        {"checkpoint": record["checkpoint"], "problem": record["problem"]}
        for log in sorted(attempt.glob("artifacts/**/storage-uploads.json"))
        for record in json.loads(log.read_text())
        if "problem" in record
    ]
    if failed:
        status["storage"]["checkpoint_problems"] = failed
    try:
        # Records (status, logs, README) legitimately change, so the remote mirrors local.
        transfer = push(attempt, remote, storage, overwrite=True)
        status["storage"].update(synced=now(), uploaded=transfer["uploaded"], unchanged=transfer["unchanged"])
    except StorageError as exc:
        status["storage"]["problem"] = exc.problem
    line = storage_line(status).strip()
    if "problem" in status["storage"] or failed:
        print(line, file=sys.stderr)
    write_json(attempt / "status.json", status)
    update_readme(attempt / "README.md", status)
    if "problem" not in status["storage"]:
        for name in ("status.json", "README.md"):
            try:
                push(attempt / name, remote + "/", storage, overwrite=True)
            except StorageError as exc:
                print(f"对象存储：{name} 同步失败：{exc}", file=sys.stderr)
    return "problem" not in status["storage"]


def backend_preflight(config, env):
    task = config["task"]
    if task.get("backend") == "vllm" and not task.get("server_url"):
        subprocess.run(
            [
                config["runtime"]["python"],
                "-c",
                "import importlib.util; assert importlib.util.find_spec('vllm'), 'vLLM is unavailable'",
            ],
            env=env,
            cwd=config["runtime"]["cwd"],
            check=True,
        )
    if task.get("server_url"):
        import urllib.request

        with urllib.request.urlopen(
            task["server_url"].rstrip("/") + "/version", timeout=10
        ) as response:
            info = json.load(response)
        if not info.get("version"):
            raise ValueError("vLLM server did not return its runtime version")
        return {
            "backend": "vllm",
            "server": task["server_url"],
            "version": info["version"],
        }
    return {"backend": task.get("backend", "not-applicable")}


def tracked_command(config, attempt):
    tracking = config["tracking"]
    python = tracking["python"]
    credentials = tracking.get("credentials_file")
    cmd = [python, "-m", "open_audio_llm.run_tracking", "--attempt", str(attempt)]
    if credentials:
        # Positional arguments keep paths and credentials out of shell interpolation.
        cmd = [
            "bash",
            "-c",
            'source "$1" >/dev/null 2>&1; shift; exec "$@"',
            "tracking",
            credentials,
            *cmd,
        ]
    return cmd


def execute(config, attempt):
    env = child_environment(config)
    effective = attempt / "effective.yaml"
    effective.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    status = {
        "execution": "starting",
        "quality": "未核实",
        "started": now(),
        "config_file": config["config_file"],
        "attempt": str(attempt),
        "command": command(config, effective),
        "history": read_status(attempt).get("history", []),
    }
    if (attempt / "status.json").exists():
        previous = read_status(attempt)
        status["history"] = [
            *previous.get("history", []),
            {k: v for k, v in previous.items() if k != "history"},
        ]
    write_json(attempt / "status.json", status)
    update_readme(attempt / "README.md", status)
    print(
        json.dumps(
            {
                "attempt": str(attempt),
                "execution": "starting",
                "log": str(attempt / "logs/process.log"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    children = []
    interrupted = None

    def forward(signum, frame):
        nonlocal interrupted
        interrupted = signum
        # Signal the complete process group, including torchrun workers.
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signum)

    original = {s: signal.signal(s, forward) for s in (signal.SIGINT, signal.SIGTERM)}
    code = 1
    tracker = None
    storage = config.get("storage")
    restored = []
    try:
        if storage:
            # Fail before training rather than at the first checkpoint upload.
            problems = preflight_problems(storage, attempt)
            if problems:
                raise ValueError("Object storage is unusable: " + "; ".join(problems))
        for target in (storage or {}).get("restore", []):
            if not Path(target).exists():
                # Restored inputs are a task-scoped copy; object storage stays authoritative.
                restored.append(target)
                pull(remote_path(target, storage), target, storage)
        if restored:
            status["storage_restored"] = restored
        if not config["recording"].get("resuming"):
            for source, destination in (
                config.get("experiment", {}).get("copies", {}).items()
            ):
                shutil.copy2(source, destination)
        data = config.get("data", {})
        if data.get("config") and not config["recording"].get("resuming"):
            source = Path(data["config"])
            raw = source.read_bytes()
            selected = yaml.safe_load(raw)
            for key in ("catalog", "roots", "metadata_cache", "experiment_plan"):
                if selected.get(key):
                    selected[key] = absolute(selected[key], source.parent)
            for item in (
                selected.get("train", [])
                + selected.get("validation", [])
                + selected.get("evaluation", [])
            ):
                if item.get("sot_alignment_index"):
                    item["sot_alignment_index"] = absolute(
                        item["sot_alignment_index"], source.parent
                    )
            data.update(
                source_config=str(source), source_sha256=hashlib.sha256(raw).hexdigest()
            )
            frozen = attempt / "data-effective.yaml"
            frozen.write_text(
                yaml.safe_dump(selected, allow_unicode=True, sort_keys=False)
            )
            data["config"] = str(frozen)
            for step in config.get("preflight", []):
                if step.get("arguments", {}).get("--data_config") == str(source):
                    step["arguments"]["--data_config"] = str(frozen)
            effective.write_text(
                yaml.safe_dump(config, allow_unicode=True, sort_keys=False)
            )
            status["command"] = command(config, effective)
        write_json(attempt / "backend.json", backend_preflight(config, env))
        if config.get("tracking", {}).get("enabled"):
            tracking_env = dict(env)
            tracking_env.pop("WANDB_DISABLED", None)
            tracking_env.pop("WANDB_MODE", None)
            tracking_env["CUDA_VISIBLE_DEVICES"] = ""
            if config["tracking"].get("pythonpath"):
                tracking_env["PYTHONPATH"] = os.pathsep.join(
                    config["tracking"]["pythonpath"]
                )
            for marker in (
                "wandb-start-verification.json",
                "wandb-verification.json",
                "tracking-stop.json",
            ):
                stale = attempt / marker
                if stale.exists():
                    stale.rename(attempt / f"{stale.stem}-{time.time_ns()}.json")
            with (attempt / "logs/wandb.log").open("a") as log:
                tracker = subprocess.Popen(
                    tracked_command(config, attempt),
                    env=tracking_env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            children.append(tracker)
            deadline = time.monotonic() + config["tracking"].get("startup_timeout", 180)
            verification = attempt / "wandb-start-verification.json"
            while not verification.exists():
                if (
                    interrupted
                    or tracker.poll() is not None
                    or time.monotonic() > deadline
                ):
                    raise RuntimeError(
                        "W&B startup metrics were not verified remotely; see logs/wandb.log"
                    )
                time.sleep(0.2)
        for preparation in config.get("preflight", []):
            prepared = {**config, "task": preparation}
            prepared_env = {**env, "CUDA_VISIBLE_DEVICES": ""}
            with (attempt / "logs/preflight.log").open("a") as log:
                process = subprocess.Popen(
                    command(prepared, effective),
                    cwd=config["runtime"]["cwd"],
                    env=prepared_env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            children.append(process)
            code = process.wait()
            if code:
                raise subprocess.CalledProcessError(code, process.args)
        if interrupted:
            raise KeyboardInterrupt()
        with (attempt / "logs/process.log").open("a") as log:
            process = subprocess.Popen(
                command(config, effective),
                env=env,
                cwd=config["runtime"]["cwd"],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            children.append(process)
            status.update(execution="running", pid=process.pid)
            write_json(attempt / "status.json", status)
            while process.poll() is None:
                if (
                    tracker is not None
                    and tracker.poll() is not None
                    and not interrupted
                ):
                    os.killpg(process.pid, signal.SIGTERM)
                    raise RuntimeError("W&B recorder stopped during execution")
                time.sleep(0.1)
            code = process.returncode
            status["process_exit_code"] = code
            if code < 0:
                code = 128 - code
        result_path = config.get("recording", {}).get("result")
        if result_path and Path(result_path).exists():
            result = json.loads(Path(result_path).read_text())
            status["result"] = result
            quality_field = config["recording"].get("quality_field", "passed")
            if quality_field in result:
                status["quality"] = "通过" if result[quality_field] else "未达标"
        handoff = config["task"].get("arguments", {}).get("--checkpoint_handoff")
        if handoff and Path(handoff).is_file():
            delivered = json.loads(Path(handoff).read_text())
            status["checkpoint_handoff"] = {
                key: delivered.get(key)
                for key in ("checkpoint", "checkpoint_remote", "selection", "global_step")
            }
        status["execution"] = "completed" if code == 0 else "failed"
    except (
        OSError,
        ValueError,
        RuntimeError,
        KeyError,
        yaml.YAMLError,
        subprocess.SubprocessError,
        KeyboardInterrupt,
    ) as exc:
        if code == 0:
            code = 1
        status.update(execution="failed", error=f"{type(exc).__name__}: {exc}")
        print(status["error"], file=sys.stderr)
    finally:
        if interrupted:
            code = 128 + interrupted
            status["execution"] = "interrupted"
        status.update(ended=now(), exit_code=code)
        write_json(attempt / "status.json", status)
        if tracker is not None and tracker.poll() is None:
            write_json(attempt / "tracking-stop.json", {"exit_code": code})
            try:
                tracker.wait(timeout=config["tracking"].get("finish_timeout", 90))
            except subprocess.TimeoutExpired:
                os.killpg(tracker.pid, signal.SIGTERM)
            if (
                tracker.poll() != 0
                or not (attempt / "wandb-verification.json").exists()
            ):
                status["tracking"] = "远端结束状态未核实"
                if code == 0:
                    code = 1
                    status.update(execution="failed", exit_code=code)
            else:
                status["tracking"] = json.loads(
                    (attempt / "wandb-verification.json").read_text()
                )
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
        for sig, handler in original.items():
            signal.signal(sig, handler)
        for target in restored:
            shutil.rmtree(target, ignore_errors=True)
        write_json(attempt / "status.json", status)
        update_readme(attempt / "README.md", status)
        if storage:
            sync_attempt(attempt, storage, status)
        print(
            json.dumps(
                {
                    "attempt": str(attempt),
                    "execution": status["execution"],
                    "exit_code": code,
                    **({"storage": status["storage"]} if "storage" in status else {}),
                },
                ensure_ascii=False,
            )
        )
    return code


def run_recipe(path, resume=None, dry_run=False):
    if dry_run:
        config, attempt = allocate(path, resume, dry_run)
        print(
            yaml.safe_dump(
                preview(config, attempt / "effective.yaml"),
                allow_unicode=True,
                sort_keys=False,
            )
        )
        return 0
    raw = yaml.safe_load(path.read_text())
    directory = Path(absolute(raw["recording"]["attempts_dir"]["path"], path.parent))
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".execution.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config, attempt = allocate(path, resume, dry_run)
        return execute(config, attempt)


def experiment(options):
    path = Path(options.config).absolute()
    manifest = yaml.safe_load(path.read_text())
    tasks = manifest["tasks"]
    identifiers = {task["id"] for task in tasks}
    if options.task and options.task not in identifiers:
        raise ValueError(f"Unknown task: {options.task}")
    if options.resume and not options.task:
        raise ValueError(
            "Experiment resume requires --task and an explicit attempt path"
        )
    completed = set()
    for task in tasks:
        recipe = Path(absolute(task["config"], path.parent))
        raw = yaml.safe_load(recipe.read_text())
        directory = Path(
            absolute(raw["recording"]["attempts_dir"]["path"], recipe.parent)
        )
        statuses = [read_status(p) for p in sorted(directory.glob("[0-9]*"))]
        finished = bool(statuses and statuses[-1].get("execution") == "completed")
        if finished:
            completed.add(task["id"])
        if options.task and task["id"] != options.task:
            continue
        if finished and not options.resume:
            print(f"{task['id']}: 已完成，跳过")
            continue
        if not set(task.get("depends_on", [])) <= completed:
            raise ValueError(f"{task['id']}: dependencies have not completed")
        if not options.task and statuses and not options.dry_run:
            raise ValueError(
                f"{task['id']}: select --task for a new attempt or --task and --resume to resume"
            )
        code = run_recipe(recipe, options.resume, options.dry_run)
        if not options.dry_run:
            latest = read_status(
                Path(options.resume).absolute()
                if options.resume
                else max(directory.glob("[0-9]*"))
            )
            update_readme(directory.parent / "README.md", latest)
            states = []
            for item in tasks:
                source = Path(absolute(item["config"], path.parent))
                settings = yaml.safe_load(source.read_text())
                records = Path(
                    absolute(
                        settings["recording"]["attempts_dir"]["path"], source.parent
                    )
                )
                entries = sorted(records.glob("[0-9]*"))
                states.append(
                    read_status(entries[-1]).get("execution") if entries else "pending"
                )
            summary = {
                "execution": "completed"
                if all(s == "completed" for s in states)
                else "partial",
                "quality": "未核实",
                "last_task": task["id"],
            }
            update_readme(path.parent / "README.md", summary)
        if code:
            return code
        completed.add(task["id"])
    return 0


def storage_sync(config_path, attempt):
    """Upload an existing attempt, then drop local checkpoints the upload confirmed."""
    config_path, attempt = config_path.absolute(), attempt.absolute()
    raw = yaml.safe_load(config_path.read_text())
    directory = Path(absolute(raw["recording"]["attempts_dir"]["path"], config_path.parent))
    if attempt.parent != directory or not (attempt / "status.json").is_file():
        raise ValueError("--attempt must select an existing attempt of this task")
    storage = load_config(config_path, attempt).get("storage")
    if not storage:
        raise ValueError("The config has no storage section")
    with (directory / ".execution.lock").open("w") as lock:
        # run_recipe holds this lock for the whole execution.
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = read_status(attempt)
        if not sync_attempt(attempt, storage, status):
            return 1
        for checkpoint in sorted(attempt.glob("artifacts/**/checkpoint-[0-9]*")):
            if checkpoint.is_dir() and not checkpoint.is_symlink():
                shutil.rmtree(checkpoint)
    print(json.dumps({"attempt": str(attempt), "storage": status["storage"]}, ensure_ascii=False))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in (
        "train",
        "eval",
        "serve",
        "rollout",
        "prepare",
        "synthesize",
        "model",
        "deploy",
    ):
        command_parser = sub.add_parser(action)
        command_parser.add_argument("--config", type=Path, required=True)
        command_parser.add_argument("--dry-run", action="store_true")
        command_parser.add_argument("--resume", type=Path)
    experiment_parser = sub.add_parser("experiment")
    run_parser = experiment_parser.add_subparsers(
        dest="operation", required=True
    ).add_parser("run")
    run_parser.add_argument("--config", type=Path, required=True)
    run_parser.add_argument("--task")
    run_parser.add_argument("--resume", type=Path)
    run_parser.add_argument("--dry-run", action="store_true")
    sync_parser = sub.add_parser("storage").add_subparsers(
        dest="operation", required=True
    ).add_parser("sync")
    sync_parser.add_argument("--config", type=Path, required=True)
    sync_parser.add_argument("--attempt", type=Path, required=True)
    options = parser.parse_args(argv)
    try:
        if options.action == "storage":
            return storage_sync(options.config, options.attempt)
        if options.action == "experiment":
            return experiment(options)
        if options.action == "deploy":
            from .deployment import deploy

            return deploy(options.config, options.dry_run)
        raw = yaml.safe_load(options.config.read_text())
        if raw["task"]["kind"] != options.action:
            raise ValueError("Command does not match task.kind in config")
        return run_recipe(options.config.absolute(), options.resume, options.dry_run)
    except (
        ValueError,
        KeyError,
        OSError,
        yaml.YAMLError,
        subprocess.CalledProcessError,
    ) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
