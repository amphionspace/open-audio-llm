"""Render concrete Compose files from a deployment YAML, without env interpolation."""

import subprocess
from pathlib import Path

import yaml

from .run_config import resolve_values


def render(path):
    path = Path(path).absolute()
    config = yaml.safe_load(path.read_text())
    if config.get("version") != 1:
        raise ValueError("Deployment config requires version: 1")
    config = resolve_values(config, path.parent)
    compose = config["compose"]
    for service in compose["services"].values():
        for mount in service.get("volumes", []):
            if (
                mount.get("target", "").startswith("/models/")
                and mount.get("read_only") is not True
            ):
                raise ValueError("Model mounts must remain read-only")
        if not service.get("gpus"):
            raise ValueError("Serving deployment must specify GPUs")
    if "${" in yaml.safe_dump(compose):
        raise ValueError("Business environment interpolation is not supported")
    return config, compose


def deploy(path, dry_run=False):
    config, compose = render(path)
    output = Path(config["output"])
    command = [
        "docker",
        "compose",
        "-f",
        str(output),
        *config.get("operation", ["config"]),
    ]
    if dry_run:
        print(
            yaml.safe_dump(
                {"output": str(output), "compose": compose, "command": command},
                sort_keys=False,
                allow_unicode=True,
            )
        )
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(compose, sort_keys=False))
    return subprocess.run(command, check=False).returncode
