"""Submit training attempts to the SenseCore ACP pool and run them inside the job."""

from __future__ import annotations

import fcntl
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

import yaml

from .run_config import child_environment, command

# sco accepts the shared SenseCore key from AmphionKeys under its own names.
SCO_CREDENTIALS = {
    "SCO_ACCESS_KEY_ID": "AMPHION_BUCKET_SENSECORE_ACCESS_KEY",
    "SCO_ACCESS_KEY_SECRET": "AMPHION_BUCKET_SENSECORE_SECRET_KEY",
}
REQUIRED = ("sco_home", "credentials", "workspace", "aec2", "image", "worker_spec",
            "nodes", "mounts")
# Statuses after which the main launcher will never start torchrun for this attempt.
FINISHED = {"failed", "completed", "interrupted"}


def mount_points(cluster):
    # VOLUME_ID[/SUB_DIR]:MOUNT_PATH; the pool mounts the same AFS paths as this host.
    return [mount.rsplit(":", 1)[1] for mount in cluster["mounts"]]


def local_paths(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from local_paths(item)
    elif isinstance(value, list):
        for item in value:
            yield from local_paths(item)
    elif isinstance(value, str) and value.startswith("/") and os.path.exists(value):
        yield value


def check(config, attempt=None):
    """Reject jobs that would reference files the container cannot see."""
    cluster = config["cluster"]
    missing = [key for key in REQUIRED if key not in cluster]
    if missing:
        raise ValueError(f"cluster is missing {', '.join(missing)}")
    if cluster.get("kind") != "acp":
        raise ValueError("cluster.kind must be acp")
    if config["task"]["kind"] != "train" or not config["runtime"].get("distributed"):
        raise ValueError("Cluster jobs run distributed training tasks")
    for section in ("runtime", "tracking", "evaluation"):
        python = config.get(section, {}).get("python")
        if python and not os.path.isabs(python):
            # A bare name would resolve inside the image, not to the configured env.
            raise ValueError(f"{section}.python must be an absolute path for cluster jobs")
    roots = mount_points(cluster)
    candidates = list(local_paths(config)) + ([str(attempt)] if attempt else [])
    hidden = sorted(
        {p for p in candidates
         if not any(p == root or p.startswith(root.rstrip("/") + "/") for root in roots)}
    )
    if hidden:
        raise ValueError("Paths outside cluster mounts: " + ", ".join(hidden))


def job_name(config, attempt):
    return f"{config.get('tracking', {}).get('name', 'open-audio-llm')}-{attempt.name}"


def container_command(config, attempt):
    """The job's startup script: load AmphionKeys, then run this attempt."""
    runtime, cluster = config["runtime"], config["cluster"]
    credentials = Path(cluster["credentials"])
    # sco's --env rejects more than one variable, so the startup script sets them.
    environment = [f"{k}={v}" for k, v in cluster.get("environment", {}).items()]
    if runtime.get("pythonpath"):
        environment.append(f"PYTHONPATH={os.pathsep.join(runtime['pythonpath'])}")
    launcher = [
        str(credentials.parent / "load.sh"), credentials.stem, "--", "env", *environment,
        runtime["python"], "-m", "open_audio_llm.cli", "cluster", "run",
        "--attempt", str(attempt),
    ]
    return f"cd {shlex.quote(runtime['cwd'])} && exec {shlex.join(launcher)}"


def submit_command(config, attempt):
    cluster = config["cluster"]
    return [
        str(Path(cluster["sco_home"]) / "bin/sco"), "acp", "jobs", "create",
        f"--workspace-name={cluster['workspace']}",
        f"--aec2-name={cluster['aec2']}",
        f"--job-name={job_name(config, attempt)}",
        f"--container-image-url={cluster['image']}",
        "--training-framework=pytorch",
        f"--worker-spec={cluster['worker_spec']}",
        f"--worker-nodes={cluster['nodes']}",
        f"--priority={cluster.get('priority', 'NORMAL')}",
        f"--quota-type={cluster.get('quota_type', 'reserved')}",
        f"--storage-mount={','.join(cluster['mounts'])}",
        *(["--wait"] if cluster.get("wait", True) else []),
        f"--command={container_command(config, attempt)}",
    ]


def sco_environment(cluster):
    """Pass the AmphionKeys secret to sco without storing it in a profile."""
    values = {}
    # Same format load.sh reads: KEY=VALUE lines, comments and blanks skipped.
    for line in Path(cluster["credentials"]).read_text().splitlines():
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    home = cluster["sco_home"]
    env = dict(os.environ, SCO_HOME=home, SCO_CONFIG=f"{home}/.config",
               SCO_DATA_HOME=f"{home}/.data")
    for target, source in SCO_CREDENTIALS.items():
        if source not in values:
            raise ValueError(f"{cluster['credentials']} does not define {source}")
        env[target] = values[source]
    return env


def submit(config, attempt):
    """Create the ACP job; the job's rank 0 launcher records the execution."""
    from .cli import now, read_status, update_readme, write_json

    previous = read_status(attempt)
    effective = attempt / "effective.yaml"
    effective.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    cluster = config["cluster"]
    status = {
        "execution": "submitting",
        "quality": "未核实",
        "config_file": config["config_file"],
        "attempt": str(attempt),
        "cluster": {"kind": "acp", "workspace": cluster["workspace"],
                    "aec2": cluster["aec2"], "display_name": job_name(config, attempt)},
        "history": [*previous.get("history", []),
                    *([{k: v for k, v in previous.items() if k != "history"}]
                      if previous else [])],
    }
    write_json(attempt / "status.json", status)
    try:
        check(config, attempt)
        process = subprocess.run(submit_command(config, attempt),
                                 env=sco_environment(cluster), capture_output=True,
                                 text=True, timeout=300, check=False)
        output = (process.stdout + process.stderr).strip()
        status["cluster"].update(submitted=now(), output=output[-2000:])
        created = re.search(r"job (\S+) submitted successfully", output)
        if process.returncode or not created:
            raise RuntimeError(f"ACP submission failed: {output[-1000:]}")
        status.update(execution="submitted")
        status["cluster"]["job"] = created.group(1)
        code = 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        code = 1
        status.update(execution="failed", ended=now(), exit_code=code,
                      error=f"{type(exc).__name__}: {exc}")
        print(status["error"], file=sys.stderr)
    write_json(attempt / "status.json", status)
    update_readme(attempt / "README.md", status)
    print(json.dumps({"attempt": str(attempt), "execution": status["execution"],
                      "job": status["cluster"].get("job")}, ensure_ascii=False))
    return code


def placement():
    """Node layout injected by ACP into every worker of a PyTorch job."""
    return {
        "nodes": int(os.environ["SENSECORE_PYTORCH_NNODES"]),
        "node_rank": int(os.environ["SENSECORE_PYTORCH_NODE_RANK"]),
        "devices": int(os.environ["SENSECORE_ACCELERATE_DEVICE_COUNT"]),
        "master_addr": os.environ["MASTER_ADDR"],
        "port": int(os.environ["MASTER_PORT"]),
    }


def run(attempt):
    """Job entry point: rank 0 records the attempt, other nodes join its torchrun."""
    from .cli import execute, read_status

    attempt = Path(attempt).absolute()
    effective = attempt / "effective.yaml"
    config = yaml.safe_load(effective.read_text())
    layout = placement()
    distributed = config["runtime"]["distributed"]
    if layout["nodes"] != config["cluster"]["nodes"]:
        raise ValueError(f"ACP started {layout['nodes']} nodes, config asks for "
                         f"{config['cluster']['nodes']}")
    if layout["devices"] != distributed["processes"]:
        raise ValueError(f"Worker has {layout['devices']} GPUs, runtime.distributed."
                         f"processes is {distributed['processes']}")
    del layout["devices"]
    if layout["node_rank"] == 0:
        distributed.update(layout)
        with (attempt.parent / ".execution.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return execute(config, attempt)
    # Rank 0 freezes data and verifies W&B before torchrun; join only once it runs.
    while (state := read_status(attempt).get("execution")) != "running":
        if state in FINISHED:
            print(f"Main launcher ended as {state} before training started", file=sys.stderr)
            return 1
        time.sleep(2)
    config = yaml.safe_load(effective.read_text())
    config["runtime"]["distributed"].update(layout)
    return subprocess.run(command(config, effective), env=child_environment(config),
                          cwd=config["runtime"]["cwd"], check=False).returncode
