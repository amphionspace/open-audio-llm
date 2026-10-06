"""Resolve YAML recipes without importing model runtimes or reading business env vars."""

from __future__ import annotations

import json
import os
import shlex
from pathlib import Path

import yaml

# Retired launcher variables must never leak into historical training adapters.
BUSINESS_ENV = {
    "MODEL",
    "DATA_CONFIG",
    "OUTPUT_DIR",
    "MAX_STEPS",
    "PYTHON",
    "DEEPSPEED",
    "RESUME_FROM_CHECKPOINT",
    "RETENTION_TEACHER",
    "PREVIOUS_DATA_CONFIG",
    "AUDIO_DATA_CONTRACT_ROOT",
    "AUDIO_DATA_CATALOG",
    "AUDIO_DATA_ROOTS_FILE",
    "AUDIO_DATA_METADATA_CACHE",
    "CUDA_VISIBLE_DEVICES",
    "NPROC_PER_NODE",
    "MASTER_PORT",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "TOKENIZERS_PARALLELISM",
    "OPEN_AUDIO_LLM_MODEL",
    "PORT",
    "ROLLOUT_MODEL",
    "TP_SIZE",
    "DP_SIZE",
    "GPU_MEMORY_UTILIZATION",
    "MAX_MODEL_LEN",
    "ENCODER_LR",
    "ALIGNER_LR",
    "LLM_LR",
    "USE_VLLM",
    "MODEL_IMAGE_TAG",
    "OPEN_AUDIO_LLM_IMAGE_TAG",
    "QWEN_GPU",
    "AMPHION_SPEC_GPU",
    "QWEN3_ASR_PORT",
    "AMPHION_SPEC_PORT",
    "WANDB_MODE",
    "WANDB_DISABLED",
}


def absolute(value, base):
    path = Path(value).expanduser()
    # Keep symlink names: resolving them would erase snapshot provenance.
    return str(Path(os.path.abspath(path if path.is_absolute() else base / path)))


def resolve_values(value, base, attempt=None):
    if isinstance(value, dict):
        if set(value) == {"path"}:
            raw = value["path"]
            if "{attempt}" in raw:
                if attempt is None:
                    raise ValueError("{attempt} requires an execution record")
                raw = raw.replace("{attempt}", str(attempt))
            return absolute(raw, base)
        return {k: resolve_values(v, base, attempt) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_values(v, base, attempt) for v in value]
    return value


def load_config(path, attempt=None):
    path = Path(path).expanduser().absolute()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("version") != 1:
        raise ValueError(f"{path}: expected YAML mapping with version: 1")
    config = resolve_values(config, path.parent, attempt)
    experiment = config.get("experiment", {})
    if experiment:
        root = Path(experiment["root"])
        selected_producers = set()
        for name, producer in experiment.get("producers", {}).items():
            for record in sorted(
                (root / "tasks" / producer / "attempts").glob("[0-9]*"), reverse=True
            ):
                status, effective = record / "status.json", record / "effective.yaml"
                if not status.is_file() or not effective.is_file():
                    continue
                if json.loads(status.read_text()).get("execution") != "completed":
                    continue
                original = yaml.safe_load(effective.read_text())
                value = original.get("experiment", {}).get("files", {}).get(name)
                if value:
                    experiment["files"][name] = value
                    selected_producers.add(name)
                    break
        arm = experiment.get("training_arm")
        if arm:
            if arm in selected_producers:
                config["data"]["config"] = str(
                    Path(experiment["files"][arm]) / "train-data.yaml"
                )
            experiment["files"][arm] = str(attempt / "artifacts")
        if config["task"].get("kind") == "eval" and config.get("parameters", {}).get(
            "models"
        ):
            for name in ("control", "treatment"):
                if name in selected_producers:
                    config["parameters"]["models"][name] = str(
                        Path(experiment["files"][name]) / "training/checkpoint-1000"
                    )
                    if "storage" in config:
                        # Uploaded training checkpoints no longer exist locally.
                        config["storage"].setdefault("restore", []).append(
                            config["parameters"]["models"][name]
                        )
        copies = {}
        for name in experiment.get("copy_inputs", []):
            destination = str(attempt / "artifacts" / name)
            copies[experiment["files"][name]] = destination
            experiment["files"][name] = destination
        experiment["copies"] = copies
    runtime = config.get("runtime", {})
    if not runtime.get("python") or not runtime.get("cwd"):
        raise ValueError(f"{path}: runtime.python and runtime.cwd are required")
    task = config.get("task", {})
    if not task.get("kind"):
        raise ValueError(f"{path}: task.kind is required")
    if task["kind"] in {"eval", "serve", "rollout"} and task.get("backend") != "vllm":
        raise ValueError("ASR evaluation, serving and rollout require backend: vllm")
    if task["kind"] in {"train", "eval"} and not config.get("tracking", {}).get(
        "enabled"
    ):
        raise ValueError("Training and evaluation require W&B tracking")
    if "storage" in config:
        from .storage import check

        check(config["storage"])
    evaluation = config.get("evaluation")
    if evaluation is not None:
        if task["kind"] != "train" or not evaluation.get("python") or not evaluation.get("config"):
            raise ValueError("evaluation needs python and config on a train task")
    config["config_file"] = str(path)
    return config


def arguments(options):
    result = []
    for flag, value in options.items():
        if not flag.startswith("-"):
            raise ValueError(f"Argument must be an explicit CLI flag: {flag}")
        if value is None:
            continue
        result.append(flag)
        for item in value if isinstance(value, list) else [value]:
            if isinstance(item, bool):
                result.append(str(item).lower())
            elif isinstance(item, dict):
                result.append(json.dumps(item, separators=(",", ":")))
            else:
                result.append(str(item))
    return result


def runtime_environment(config):
    runtime = config["runtime"]
    settings = {
        k: str(v).lower() if isinstance(v, bool) else str(v)
        for k, v in runtime.get("environment", {}).items()
    }
    if any(
        "KEY" in k or "TOKEN" in k or "SECRET" in k or "PASSWORD" in k
        for k in settings
        if k != "TOKENIZERS_PARALLELISM"
    ):
        raise ValueError("Credentials belong in the inherited environment, not YAML")
    settings["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, runtime.get("gpus", [])))
    if "pythonpath" in runtime:
        settings["PYTHONPATH"] = os.pathsep.join(runtime["pythonpath"])
    for key, env in (
        ("omp", "OMP_NUM_THREADS"),
        ("mkl", "MKL_NUM_THREADS"),
        ("openblas", "OPENBLAS_NUM_THREADS"),
    ):
        if key in runtime.get("threads", {}):
            settings[env] = str(runtime["threads"][key])
    distributed = runtime.get("distributed", {})
    if distributed:
        settings["NPROC_PER_NODE"] = str(distributed["processes"])
        settings["MASTER_PORT"] = str(distributed["port"])
    if config["task"]["kind"] == "train":
        # The CPU recorder owns W&B; DDP workers only write their existing metrics.
        settings.update(WANDB_MODE="disabled", WANDB_DISABLED="true")
    return settings


def child_environment(config):
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in BUSINESS_ENV
        and (not k.startswith("VLLM_") or k == "VLLM_API_KEY")
        and not k.startswith("OPEN_AUDIO_LLM_ENABLE_")
        and not k.startswith("AMPHION_TSASR_")
        and k not in {"COT_AUDIO_CHUNKED_ATTN", "COT_TSASR_ENROLL_RANDOM"}
    }
    env.update(runtime_environment(config))
    return env


def command(config, effective_file=None):
    runtime, task = config["runtime"], config["task"]
    python = runtime["python"]
    if task["kind"] == "serve":
        cmd = [python, "-m", "vllm.entrypoints.cli.main", "serve", task["model"]]
    elif task.get("executable"):
        cmd = [task["executable"]]
    elif task.get("script"):
        cmd = [python, task["script"]]
    elif task.get("module"):
        if task["kind"] == "train":
            dist = runtime["distributed"]
            cmd = [
                python,
                "-m",
                "torch.distributed.run",
                "--nproc_per_node",
                str(dist["processes"]),
                "--master_port",
                str(dist["port"]),
                "--module",
                task["module"],
            ]
        else:
            cmd = [python, "-m", task["module"]]
    else:
        raise ValueError("task requires module, script or executable")
    cmd += [str(x) for x in task.get("positional", [])]
    options = dict(task.get("arguments", {}))
    data = config.get("data", {})
    if data.get("config"):
        options["--data_config"] = data["config"]
    if config.get("evaluation") and task["kind"] == "train":
        # Written by the launcher's pre-training AmphionEval check.
        options["--amphion_eval_check"] = (
            str(Path(effective_file).parent / "amphion-eval-check.json")
            if effective_file else "<attempt>/amphion-eval-check.json"
        )
    storage = config.get("storage")
    if storage and task["kind"] == "train":
        options["--storage_sync"] = {
            k: storage[k] for k in ("executable", "remote", "local_root")
        }
    if task.get("accepts_config"):
        options["--config"] = str(effective_file or "<effective.yaml>")
    cmd += arguments(options) + task.get("flags", [])
    for flag, values in task.get("repeated_arguments", {}).items():
        for value in values:
            cmd += arguments({flag: value})
    if runtime.get("credentials_file"):
        cmd = [
            "bash",
            "-c",
            'source "$1" >/dev/null 2>&1; shift; exec "$@"',
            "credentials",
            runtime["credentials_file"],
            *cmd,
        ]
    return cmd


def preview(config, effective_file=None):
    return {
        "config": config,
        "command": command(config, effective_file),
        "shell_command": shlex.join(command(config, effective_file)),
        "runtime_settings": runtime_environment(config),
        "preflight_commands": [
            command({**config, "task": task}, effective_file)
            for task in config.get("preflight", [])
        ],
    }
