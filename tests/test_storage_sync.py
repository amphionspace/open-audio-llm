import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from open_audio_llm.cli import main
from open_audio_llm.integrations.ms_swift.checkpoint_handoff import CheckpointHandoffCallback
from open_audio_llm.integrations.ms_swift.checkpoint_upload import CheckpointUploadCallback
from open_audio_llm.run_config import command, load_config
from open_audio_llm.storage import StorageError, push

# Minimal stand-in for `ab --json` (AmphionBucket 0.5) on a local directory acting
# as the bucket; FAKE_AB_FAIL selects a failure mode.
FAKE_AB = r'''
import filecmp, json, os, shutil, sys
from pathlib import Path
bucket = Path(__file__).parent / "bucket"
mode = os.environ.get("FAKE_AB_FAIL")
args = sys.argv[1:]
if mode == "no-json" and "--json" in args:
    print("ab: error: unrecognized arguments: --json")
    raise SystemExit(2)
args = [arg for arg in args if arg != "--json"]
if args[0] == "--version":
    print("AmphionBucket " + ("0.4.0" if mode == "old-ab" else "0.5.0"))
    raise SystemExit(0)
def target(spec):
    return bucket / spec.split(":", 1)[1]
def result(status, source, destination, error=None):
    row = {"op": args[0], "status": status, "source": str(source), "target": destination, "bytes": 1}
    if error:
        row["error"] = error
    print(json.dumps(row, ensure_ascii=False))
def summary(error=None, **counts):
    row = {"summary": {key: counts.get(key, 0) for key in ("done", "skipped", "preview", "conflict", "failed")}}
    if error:
        row["error"] = error
    print(json.dumps(row, ensure_ascii=False))
    raise SystemExit(1 if error or counts.get("conflict") or counts.get("failed") else 0)
if args[0] == "remotes":
    print(json.dumps({"bucket": "whai", "provider": "sensecore"}))
    raise SystemExit(0)
if args[0] == "credentials":
    source = "missing" if mode == "no-credentials" else "file"
    print(json.dumps({"bucket": "whai", "profile": "sensecore", "source": source}))
    raise SystemExit(0)
if any(":" in arg and not arg.startswith("whai:") for arg in args):
    summary("远端路径应为真实桶名:path；使用 remotes 查看存储桶")
if mode == "symlinked-state" and args[0] == "push":
    print("进度 not json")
    result("failed", args[1], args[2], "不跟随符号链接: /home/u/.cache")
    summary(failed=1)
if mode in {"unreachable", "no-credentials"}:
    summary("Could not connect to the endpoint URL")
if args[0] == "ls":
    path = target(args[1])
    print("\n".join(p.name for p in path.iterdir()) if path.is_dir() else "", end="")
    raise SystemExit(0)
if args[0] == "pull":
    if not target(args[1]).is_dir():
        summary("远端文件或目录不存在（或该前缀下没有对象）")
    shutil.copytree(target(args[1]), args[2])
    summary()
source, destination, overwrite = Path(args[1]), args[2], "--overwrite" in args
if source.is_dir():
    pairs = [(p, target(destination) / p.relative_to(source)) for p in source.rglob("*")
             if p.is_file() and not p.is_symlink()]
else:
    pairs = [(source, target(destination) / source.name if destination.endswith("/") else target(destination))]
counts = {"done": 0, "skipped": 0, "conflict": 0}
for local, remote in pairs:
    spec = "whai:" + str(remote.relative_to(bucket))
    print(f"进度 {spec}")
    if remote.exists() and filecmp.cmp(local, remote, shallow=False):
        counts["skipped"] += 1
        result("skipped", local, spec)
    elif remote.exists() and not overwrite:
        counts["conflict"] += 1
        result("conflict", local, spec, "目标已存在，内容不同或无法确认一致；覆盖需 --overwrite")
    else:
        remote.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local, remote)
        counts["done"] += 1
        result("done", local, spec)
summary(**counts)
'''


def fake_storage(tmp_path):
    script = tmp_path / "ab.py"
    script.write_text(FAKE_AB)
    executable = tmp_path / "ab"
    executable.write_text(f"#!/bin/sh\nexec {sys.executable} {script} \"$@\"\n")
    executable.chmod(0o755)
    storage = {"executable": str(executable), "remote": "whai:open-audio-llm", "local_root": str(tmp_path)}
    return storage, tmp_path / "bucket/open-audio-llm"


def save(output, step):
    checkpoint = output / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_model.safetensors").write_text(f"weights {step}")
    (checkpoint / "optimizer.pt").write_text(f"optimizer {step}")
    (checkpoint / f"global_step{step}").mkdir()
    (checkpoint / f"global_step{step}" / "optim_states.pt").write_text("zero")
    return checkpoint


def test_checkpoints_move_to_storage_and_handoff_points_at_remote(tmp_path):
    storage, bucket = fake_storage(tmp_path)
    output = tmp_path / "runs/exp/attempt/artifacts/training"
    uploads = CheckpointUploadCallback(storage)
    handoff = CheckpointHandoffCallback(tmp_path / "handoff.json", base_model="base",
                                        tuner_type="lora", uploads=uploads)
    args = SimpleNamespace(output_dir=str(output))
    state = SimpleNamespace(is_world_process_zero=True, best_model_checkpoint=None)
    for step in (1, 2):
        save(output, step)
        state.global_step = step
        uploads.on_save(args, state, None)
        handoff.on_save(args, state, None)
    for callback in (uploads, handoff):
        callback.on_train_end(args, state, None)

    remote = bucket / "runs/exp/attempt/artifacts/training"
    assert (remote / "checkpoint-1/adapter_model.safetensors").read_text() == "weights 1"
    assert not (remote / "checkpoint-1/optimizer.pt").exists()
    assert not (remote / "checkpoint-1/global_step1").exists()
    assert (remote / "checkpoint-2/optimizer.pt").read_text() == "optimizer 2"
    assert (remote / "checkpoint-2/global_step2/optim_states.pt").exists()
    assert not list(output.glob("checkpoint-*"))
    saved = json.loads((tmp_path / "handoff.json").read_text())
    assert saved["schema_version"] == 2
    assert saved["checkpoint_remote"] == "whai:open-audio-llm/runs/exp/attempt/artifacts/training/checkpoint-2"
    records = json.loads((output / "storage-uploads.json").read_text())
    assert [r["resume_state"] for r in records] == [False, False, True]


def test_failed_upload_keeps_local_checkpoint(tmp_path, monkeypatch):
    storage, _ = fake_storage(tmp_path)
    monkeypatch.setenv("FAKE_AB_FAIL", "unreachable")
    output = tmp_path / "training"
    uploads = CheckpointUploadCallback(storage)
    handoff = CheckpointHandoffCallback(tmp_path / "handoff.json", base_model="base",
                                        tuner_type="full", uploads=uploads)
    args = SimpleNamespace(output_dir=str(output))
    state = SimpleNamespace(is_world_process_zero=True, best_model_checkpoint=None, global_step=1)
    save(output, 1)
    uploads.on_save(args, state, None)
    handoff.on_save(args, state, None)
    for callback in (uploads, handoff):
        callback.on_train_end(args, state, None)
    assert (output / "checkpoint-1/optimizer.pt").exists()
    assert json.loads((tmp_path / "handoff.json").read_text())["checkpoint_remote"] is None
    problem = json.loads((output / "storage-uploads.json").read_text())[0]["problem"]
    assert problem["cause"] == "bucket-unreachable"
    assert "Could not connect" in problem["evidence"]


@pytest.mark.parametrize("case, cause", [
    ("missing-ab", "ab-missing"),
    ("other-bucket", "bucket-not-configured"),
    ("conflict", "remote-conflict"),
    ("unreachable", "bucket-unreachable"),
    ("symlinked-state", "transfer-failed"),
    ("no-credentials", "credentials-missing"),
    ("no-json", "ab-no-json"),
])
def test_push_failures_report_their_cause(tmp_path, monkeypatch, case, cause):
    storage, bucket = fake_storage(tmp_path)
    source = tmp_path / "result.json"
    source.write_text("new")
    destination = "whai:open-audio-llm/runs/"
    if case == "missing-ab":
        storage["executable"] = str(tmp_path / "absent-ab")
    elif case == "other-bucket":
        destination = "elsewhere:runs/"
    elif case == "conflict":
        (bucket / "runs").mkdir(parents=True)
        (bucket / "runs/result.json").write_text("old")
    else:
        monkeypatch.setenv("FAKE_AB_FAIL", case)
    with pytest.raises(StorageError) as error:
        push(source, destination, storage)
    assert error.value.problem["cause"] == cause
    if case == "symlinked-state":
        assert error.value.problem["detail"].endswith("不跟随符号链接: /home/u/.cache")
    if case == "conflict":
        assert "whai:open-audio-llm/runs/result.json" in error.value.problem["evidence"]


def test_restore_reports_missing_remote_and_failed_sync_is_recorded(tmp_path, monkeypatch, capsys):
    storage, _ = fake_storage(tmp_path)
    path = recipe(tmp_path, storage, restore=[{"path": "runs/train/checkpoint-5"}])
    assert main(["prepare", "--config", str(path)]) == 1
    status = json.loads((tmp_path / "runs/prepare/attempts/001/status.json").read_text())
    assert "remote-missing" in status["error"]
    monkeypatch.setenv("FAKE_AB_FAIL", "unreachable")
    assert main(["prepare", "--config", str(path)]) == 1
    status = json.loads((tmp_path / "runs/prepare/attempts/002/status.json").read_text())
    assert status["storage"]["problem"]["cause"] == "bucket-unreachable"
    assert "bucket-unreachable" in (tmp_path / "runs/prepare/attempts/002/README.md").read_text()
    assert "对象存储：同步有问题" in capsys.readouterr().err


def recipe(tmp_path, storage, **extra):
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import sys\nfrom pathlib import Path\n"
        "Path(sys.argv[1]).write_text(str(Path(sys.argv[2]).exists()))\n"
    )
    config = {
        "version": 1,
        "runtime": {"python": sys.executable, "cwd": {"path": "."}, "gpus": []},
        "task": {"kind": "prepare", "script": {"path": "worker.py"},
                 "positional": [{"path": "{attempt}/artifacts/seen.txt"}, {"path": "runs/train/checkpoint-5"}]},
        "storage": {**storage, **extra},
        "recording": {"attempts_dir": {"path": "runs/prepare/attempts"}},
    }
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_execution_restores_inputs_and_mirrors_the_record(tmp_path):
    storage, bucket = fake_storage(tmp_path)
    (bucket / "runs/train/checkpoint-5").mkdir(parents=True)
    (bucket / "runs/train/checkpoint-5/model.safetensors").write_text("w")
    path = recipe(tmp_path, storage, restore=[{"path": "runs/train/checkpoint-5"}])
    assert main(["prepare", "--config", str(path)]) == 0
    attempt = tmp_path / "runs/prepare/attempts/001"
    assert (attempt / "artifacts/seen.txt").read_text() == "True"
    assert not (tmp_path / "runs/train/checkpoint-5").exists()
    status = json.loads((attempt / "status.json").read_text())
    assert status["storage"]["remote"] == "whai:open-audio-llm/runs/prepare/attempts/001"
    remote = bucket / "runs/prepare/attempts/001"
    assert (remote / "artifacts/seen.txt").read_text() == "True"
    assert json.loads((remote / "status.json").read_text())["storage"] == status["storage"]


def test_storage_sync_uploads_attempt_then_drops_local_checkpoints(tmp_path):
    storage, bucket = fake_storage(tmp_path)
    path = recipe(tmp_path, storage)
    attempt = tmp_path / "runs/prepare/attempts/001"
    attempt.mkdir(parents=True)
    (attempt / "status.json").write_text(json.dumps({"execution": "failed"}))
    save(attempt / "artifacts/training", 7)
    assert main(["storage", "sync", "--config", str(path), "--attempt", str(attempt)]) == 0
    remote = bucket / "runs/prepare/attempts/001/artifacts/training/checkpoint-7"
    assert (remote / "optimizer.pt").read_text() == "optimizer 7"
    assert not (attempt / "artifacts/training/checkpoint-7").exists()


def test_training_command_receives_storage_from_yaml(tmp_path):
    storage, _ = fake_storage(tmp_path)
    config = {
        "version": 1,
        "runtime": {"python": "python", "cwd": {"path": "."}, "distributed": {"processes": 1, "port": 1}},
        "task": {"kind": "train", "module": "trainer"},
        "storage": storage,
        "tracking": {"enabled": True},
    }
    path = tmp_path / "train.yaml"
    path.write_text(yaml.safe_dump(config))
    cmd = command(load_config(path))
    assert json.loads(cmd[cmd.index("--storage_sync") + 1]) == storage


def test_readme_shows_the_delivered_checkpoint_location(tmp_path):
    storage, _ = fake_storage(tmp_path)
    handoff = tmp_path / "handoff.json"
    handoff.write_text(json.dumps({
        "checkpoint": "/local/checkpoint-30", "selection": "final", "global_step": 30,
        "checkpoint_remote": "whai:open-audio-llm/runs/train/checkpoint-30",
    }))
    path = recipe(tmp_path, storage)
    config = yaml.safe_load(path.read_text())
    config["task"]["arguments"] = {"--checkpoint_handoff": {"path": str(handoff)}}
    path.write_text(yaml.safe_dump(config))
    (tmp_path / "runs/train/checkpoint-5").mkdir(parents=True)
    assert main(["prepare", "--config", str(path)]) == 0
    attempt = tmp_path / "runs/prepare/attempts/001"
    status = json.loads((attempt / "status.json").read_text())
    assert status["checkpoint_handoff"]["checkpoint_remote"] == "whai:open-audio-llm/runs/train/checkpoint-30"
    assert "交付 checkpoint（final，step 30）：`whai:open-audio-llm/runs/train/checkpoint-30`" in (attempt / "README.md").read_text()


@pytest.mark.parametrize("mode, message", [
    ("old-ab", "AmphionBucket >= 0.5.0 is required"),
    ("no-credentials", "ab has no bucket whai with credentials"),
])
def test_task_refuses_to_start_without_usable_storage(tmp_path, monkeypatch, mode, message):
    storage, _ = fake_storage(tmp_path)
    monkeypatch.setenv("FAKE_AB_FAIL", mode)
    path = recipe(tmp_path, storage)
    assert main(["prepare", "--config", str(path)]) == 1
    attempt = tmp_path / "runs/prepare/attempts/001"
    status = json.loads((attempt / "status.json").read_text())
    assert message in status["error"]
    assert not (attempt / "artifacts/seen.txt").exists()
