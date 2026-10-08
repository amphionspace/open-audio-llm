import json
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from open_audio_llm.cli import main
from open_audio_llm.run_config import child_environment, command, load_config


def recipe(tmp_path, *, code=0):
    script = tmp_path / "worker.py"
    script.write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        'Path(sys.argv[1]).write_text(json.dumps({"args":sys.argv[2:], '
        '"gpus":os.environ.get("CUDA_VISIBLE_DEVICES"), '
        '"model_env":os.environ.get("MODEL")}))\n'
        f"raise SystemExit({code})\n"
    )
    config = {
        "version": 1,
        "runtime": {
            "python": sys.executable,
            "cwd": {"path": "."},
            "gpus": [],
            "threads": {"omp": 2},
        },
        "task": {
            "kind": "prepare",
            "script": {"path": "worker.py"},
            "positional": [
                {"path": "{attempt}/artifacts/observed.json"},
                "two words",
                "--literal",
            ],
        },
        "recording": {"attempts_dir": {"path": "tasks/prepare/attempts"}},
    }
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(config))
    return path, config


@pytest.mark.parametrize("code", [0, 37])
def test_execution_records_exact_arguments_environment_and_exit(
    tmp_path, monkeypatch, code
):
    path, _config = recipe(tmp_path, code=code)
    monkeypatch.setenv("MODEL", "ambient-must-not-win")
    monkeypatch.setenv("MAX_STEPS", "999999")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    assert main(["prepare", "--config", str(path)]) == code
    attempt = tmp_path / "tasks/prepare/attempts/001"
    observed = json.loads((attempt / "artifacts/observed.json").read_text())
    assert observed == {
        "args": ["two words", "--literal"],
        "gpus": "",
        "model_env": None,
    }
    status = json.loads((attempt / "status.json").read_text())
    assert status["exit_code"] == code
    assert status["execution"] == ("completed" if code == 0 else "failed")
    assert status["quality"] == "未核实"
    effective = yaml.safe_load((attempt / "effective.yaml").read_text())
    assert effective["runtime"]["cwd"] == str(tmp_path)
    assert effective["task"]["script"] == str(tmp_path / "worker.py")
    assert "退出码" in (attempt / "README.md").read_text()


def test_dry_run_does_not_create_records_or_start_processes(
    tmp_path, monkeypatch, capsys
):
    path, _ = recipe(tmp_path)
    monkeypatch.setattr(
        subprocess, "Popen", lambda *a, **k: pytest.fail("process started")
    )
    assert main(["prepare", "--config", str(path), "--dry-run"]) == 0
    preview = yaml.safe_load(capsys.readouterr().out)
    assert preview["command"][1] == str(tmp_path / "worker.py")
    assert not (tmp_path / "tasks").exists()


def test_config_paths_are_based_on_config_location_and_ids_stay_literal(tmp_path):
    path, config = recipe(tmp_path)
    config["task"]["arguments"] = {
        "--model": "Qwen/Qwen3-ASR-1.7B",
        "--input": {"path": "audio.wav"},
        "--bool": False,
        "--list": ["a", "b"],
        "--json": {"audio": 4},
    }
    path.write_text(yaml.safe_dump(config))
    loaded = load_config(path, tmp_path / "attempt")
    args = command(loaded)
    assert args[args.index("--model") + 1] == "Qwen/Qwen3-ASR-1.7B"
    assert args[args.index("--input") + 1] == str(tmp_path / "audio.wav")
    assert args[args.index("--bool") + 1] == "false"
    assert args[args.index("--json") + 1] == '{"audio":4}'
    assert args[args.index("--list") + 1 : args.index("--list") + 3] == ["a", "b"]


def test_credentials_remain_inherited_but_cannot_be_saved_in_config(
    tmp_path, monkeypatch
):
    path, config = recipe(tmp_path)
    monkeypatch.setenv("WANDB_API_KEY", "test-only-token")
    assert (
        child_environment(load_config(path, tmp_path / "attempt"))["WANDB_API_KEY"]
        == "test-only-token"
    )
    config["runtime"]["environment"] = {"WANDB_API_KEY": "forbidden"}
    path.write_text(yaml.safe_dump(config))
    assert main(["prepare", "--config", str(path), "--dry-run"]) == 2
    assert not (tmp_path / "tasks").exists()


@pytest.mark.parametrize("kind", ["eval", "serve", "rollout"])
def test_inference_does_not_fall_back_to_transformers(tmp_path, kind):
    path, config = recipe(tmp_path)
    config["task"].update(kind=kind, backend="transformers")
    path.write_text(yaml.safe_dump(config))
    assert main([kind, "--config", str(path), "--dry-run"]) == 2


def test_tsasr_configuration_preserves_sep_and_global_attention_without_ambient_overrides(
    monkeypatch,
):
    from open_audio_llm.tsasr.global_audio_attn import GLOBAL_N_WINDOW_INFER

    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("AMPHION_TSASR_INSERT_SEP", "1")
    monkeypatch.setenv("COT_AUDIO_CHUNKED_ATTN", "1")
    plain = load_config(
        root / "examples/configs/serve/vllm.yaml", root / "runs/preview"
    )
    assert "AMPHION_TSASR_INSERT_SEP" not in child_environment(plain)
    ts = load_config(root / "examples/configs/serve/tsasr.yaml", root / "runs/preview")
    env = child_environment(ts)
    assert env["AMPHION_TSASR_INSERT_SEP"] == "1"
    assert env["COT_AUDIO_CHUNKED_ATTN"] == "0"
    args = command(ts)
    overrides = json.loads(args[args.index("--hf-overrides") + 1])
    assert (
        overrides["thinker_config"]["audio_config"]["n_window_infer"]
        == GLOBAL_N_WINDOW_INFER
    )


def test_train_requires_tracking_and_generates_distributed_command(tmp_path):
    path, config = recipe(tmp_path)
    config["task"] = {
        "kind": "train",
        "module": "open_audio_llm.integrations.ms_swift.train",
        "positional": ["sft"],
        "arguments": {"--max_steps": 1000},
    }
    config["runtime"]["distributed"] = {"processes": 2, "port": 29781}
    config["runtime"]["gpus"] = [0, 1]
    path.write_text(yaml.safe_dump(config))
    assert main(["train", "--config", str(path), "--dry-run"]) == 2
    config["tracking"] = {"enabled": True}
    path.write_text(yaml.safe_dump(config))
    loaded = load_config(path, tmp_path / "attempt")
    assert command(loaded) == [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        "2",
        "--master_port",
        "29781",
        "--module",
        "open_audio_llm.integrations.ms_swift.train",
        "sft",
        "--max_steps",
        "1000",
    ]


def test_preflight_failure_prevents_main_process(tmp_path):
    path, config = recipe(tmp_path)
    preflight = tmp_path / "preflight.py"
    preflight.write_text("raise SystemExit(37)\n")
    config["preflight"] = [{"kind": "prepare", "script": {"path": "preflight.py"}}]
    path.write_text(yaml.safe_dump(config))
    assert main(["prepare", "--config", str(path)]) == 37
    assert not (
        tmp_path / "tasks/prepare/attempts/001/artifacts/observed.json"
    ).exists()


def test_http_evaluation_checks_server_without_importing_local_model_runtime(
    tmp_path, monkeypatch
):
    import io
    import urllib.request

    from open_audio_llm.cli import backend_preflight

    path, config = recipe(tmp_path)
    config["task"].update(
        kind="eval", backend="vllm", server_url="http://localhost:8000"
    )
    config["tracking"] = {"enabled": True}
    path.write_text(yaml.safe_dump(config))
    loaded = load_config(path, tmp_path)
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: pytest.fail("local model runtime imported")
    )
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b'{"version":"0.18.0"}')
    )
    assert backend_preflight(loaded, {})["version"] == "0.18.0"


def test_missing_data_input_is_recorded_as_failed_startup(tmp_path):
    path, config = recipe(tmp_path)
    config["data"] = {"config": {"path": "missing.yaml"}}
    path.write_text(yaml.safe_dump(config))
    assert main(["prepare", "--config", str(path)]) == 1
    status = json.loads(
        (tmp_path / "tasks/prepare/attempts/001/status.json").read_text()
    )
    assert status["execution"] == "failed"
    assert "FileNotFoundError" in status["error"]


def test_resume_requires_explicit_failed_attempt_and_keeps_history(tmp_path):
    path, config = recipe(tmp_path, code=37)
    config["task"]["resume_supported"] = True
    path.write_text(yaml.safe_dump(config))
    assert main(["prepare", "--config", str(path)]) == 37
    attempt = tmp_path / "tasks/prepare/attempts/001"
    script = tmp_path / "worker.py"
    script.write_text(script.read_text().replace("SystemExit(37)", "SystemExit(0)"))
    assert main(["prepare", "--config", str(path), "--resume", str(attempt)]) == 0
    status = json.loads((attempt / "status.json").read_text())
    assert status["history"][0]["exit_code"] == 37
    assert (
        yaml.safe_load((attempt / "effective.yaml").read_text())["recording"][
            "resuming"
        ]
        is True
    )
    assert main(["prepare", "--config", str(path), "--resume", str(attempt)]) == 2
    assert not (attempt.parent / "002").exists()


def test_experiment_stops_at_failed_task_and_never_replays_completed_task(tmp_path):
    _first, config = recipe(tmp_path)
    second_config = dict(
        config, recording={"attempts_dir": {"path": "tasks/second/attempts"}}
    )
    second = tmp_path / "second.yaml"
    second_config["preflight"] = [{"kind": "prepare", "executable": "/bin/false"}]
    second.write_text(yaml.safe_dump(second_config))
    manifest = tmp_path / "experiment.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "tasks": [
                    {"id": "first", "config": "recipe.yaml"},
                    {"id": "second", "config": "second.yaml", "depends_on": ["first"]},
                ],
            }
        )
    )
    args = ["experiment", "run", "--config", str(manifest)]
    assert main(args) == 1
    assert "partial" in (tmp_path / "README.md").read_text()
    assert main(args) == 2
    second_config.pop("preflight")
    second.write_text(yaml.safe_dump(second_config))
    assert main([*args, "--task", "second"]) == 0
    assert (tmp_path / "tasks/second/attempts/002/status.json").exists()
    assert not (tmp_path / "tasks/prepare/attempts/002").exists()
    assert main(args) == 0


def test_sigterm_reaches_worker_and_is_recorded(tmp_path):
    path, _config = recipe(tmp_path)
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import signal, time\nfrom pathlib import Path\n"
        f"root = Path({str(tmp_path)!r})\n"
        "def stop(sig, frame):\n"
        '    (root / "terminated").write_text(str(sig))\n'
        "    raise SystemExit(128 + sig)\n"
        "signal.signal(signal.SIGTERM, stop)\n"
        '(root / "ready").touch()\n'
        "while True: time.sleep(0.1)\n"
    )
    process = subprocess.Popen(
        [sys.executable, "-m", "open_audio_llm", "prepare", "--config", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 10
        while not (tmp_path / "ready").exists():
            assert time.monotonic() < deadline
            assert process.poll() is None
            time.sleep(0.05)
        process.send_signal(signal.SIGTERM)
        process.communicate(timeout=10)
        assert process.returncode == 143
        assert (tmp_path / "terminated").read_text() == "15"
        status = json.loads(
            (tmp_path / "tasks/prepare/attempts/001/status.json").read_text()
        )
        assert status["execution"] == "interrupted"
        assert status["exit_code"] == 143
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()


def test_wandb_verification_checks_real_remote_values():
    from types import SimpleNamespace

    from open_audio_llm.run_tracking import verify

    remote = SimpleNamespace(
        summary={"uploaded": 3}, state="finished", url="https://example.test/run"
    )
    wandb = SimpleNamespace(Api=lambda: SimpleNamespace(run=lambda _: remote))
    run = SimpleNamespace(entity="entity", project="project", id="run")
    assert (
        verify(wandb, run, {"uploaded": 3}, 0, terminal="finished")["url"] == remote.url
    )
    with pytest.raises(RuntimeError, match="did not match"):
        verify(wandb, run, {"uploaded": 4}, 0, terminal="finished")


def test_wandb_verification_retries_busy_service_until_deadline():
    from types import SimpleNamespace

    from open_audio_llm.run_tracking import verify

    class CommError(Exception):
        pass

    remote = SimpleNamespace(summary={"uploaded": 3}, state="running", url="https://example.test/run")
    calls = []

    def lookup(_):
        calls.append(1)
        if len(calls) == 1:
            raise CommError("the service process is busy and did not respond in time")
        return remote

    wandb = SimpleNamespace(Api=lambda: SimpleNamespace(run=lookup),
                            errors=SimpleNamespace(CommError=CommError))
    run = SimpleNamespace(entity="entity", project="project", id="run")
    assert verify(wandb, run, {"uploaded": 3}, 5)["url"] == remote.url
    assert len(calls) == 2
    calls.clear()
    with pytest.raises(CommError):
        verify(wandb, run, {"uploaded": 3}, 0)


@pytest.mark.parametrize('actual,expected,passes', [
    (0.18294914013904137, 0.18294914013904134, True),
    (0.18295014013904134, 0.18294914013904134, False),
    (None, 0.18294914013904134, False),
    (3.0000000000000004, 3, False),
])
def test_wandb_verification_allows_only_float_round_trip_error(actual, expected, passes):
    from types import SimpleNamespace
    from open_audio_llm.run_tracking import verify

    remote = SimpleNamespace(summary={'metric': actual}, state='finished',
                             url='https://example.test/run')
    wandb = SimpleNamespace(Api=lambda: SimpleNamespace(run=lambda _: remote))
    run = SimpleNamespace(entity='entity', project='project', id='run')
    if passes:
        assert verify(wandb, run, {'metric': expected}, 0, terminal='finished')['url'] == remote.url
    else:
        with pytest.raises(RuntimeError, match='did not match'):
            verify(wandb, run, {'metric': expected}, 0, terminal='finished')


def test_quality_is_separate_from_successful_execution(tmp_path):
    path, config = recipe(tmp_path)
    worker = tmp_path / "worker.py"
    worker.write_text(
        "from pathlib import Path\nimport sys\n"
        "Path(sys.argv[1]).write_text('{\"passed\": false}')\n"
    )
    config["recording"]["result"] = {"path": "{attempt}/artifacts/observed.json"}
    path.write_text(yaml.safe_dump(config))
    assert main(["prepare", "--config", str(path)]) == 0
    status = json.loads(
        (tmp_path / "tasks/prepare/attempts/001/status.json").read_text()
    )
    assert status["execution"] == "completed"
    assert status["quality"] == "未达标"


def test_repeated_arguments_preserve_each_input_flag(tmp_path):
    path, config = recipe(tmp_path)
    config["task"]["repeated_arguments"] = {
        "--input": [{"path": "one"}, {"path": "two"}]
    }
    path.write_text(yaml.safe_dump(config))
    args = command(load_config(path, tmp_path))
    assert args[-4:] == [
        "--input",
        str(tmp_path / "one"),
        "--input",
        str(tmp_path / "two"),
    ]


def test_deployment_renders_paths_readonly_models_and_no_environment_overrides(
    monkeypatch,
):
    from open_audio_llm.deployment import render

    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("OPEN_AUDIO_LLM_MODEL", "/ambient/model")
    monkeypatch.setenv("VLLM_PORT", "1234")
    _, compose = render(root / "examples/configs/deploy/vllm.yaml")
    service = compose["services"]["vllm-serving"]
    assert service["ports"] == ["8009:8009"]
    assert service["volumes"][0]["source"] == str(root / "models/qwen3-asr-1.7b")
    assert service["volumes"][0]["read_only"]
    assert service["gpus"] == "all"
    assert "${" not in yaml.safe_dump(compose)


def cluster_recipe(tmp_path):
    path, config = recipe(tmp_path)
    keys = tmp_path / "keys"
    keys.mkdir()
    (keys / "sensecore.env").write_text(
        "# test-only\nAMPHION_BUCKET_SENSECORE_ACCESS_KEY=ak-test\n"
        "AMPHION_BUCKET_SENSECORE_SECRET_KEY=sk-test\n"
    )
    (keys / "load.sh").write_text("")
    sco = tmp_path / "sco"
    (sco / "bin").mkdir(parents=True)
    (sco / "bin/sco").write_text(
        "#!/bin/sh\n"
        'printf "%s\\n" "$@" > "$SCO_HOME/args"\n'
        'echo "$SCO_ACCESS_KEY_ID $SCO_ACCESS_KEY_SECRET" > "$SCO_HOME/keys"\n'
        "echo 'job pt-test1234 submitted successfully, please wait for scheduling!'\n"
    )
    (sco / "bin/sco").chmod(0o755)
    config["task"] = {
        "kind": "train",
        "module": "open_audio_llm.integrations.ms_swift.train",
        "positional": ["sft"],
    }
    config["runtime"].update(
        gpus=[0, 1], distributed={"processes": 2, "port": 29781},
        pythonpath=[{"path": "."}],
    )
    config["tracking"] = {"enabled": True, "name": "acp-test"}
    config["cluster"] = {
        "kind": "acp",
        "sco_home": {"path": "sco"},
        "credentials": {"path": "keys/sensecore.env"},
        "workspace": "workspace-test",
        "aec2": "cluster-test",
        "image": "registry.example/image:tag",
        "worker_spec": "N3lS.Ii.I60.94c944g",
        "nodes": 2,
        "mounts": [f"volume-id:{tmp_path}", f"env-volume:{Path(sys.executable).parent}"],
        "environment": {"NCCL_IB_TIMEOUT": 22},
    }
    path.write_text(yaml.safe_dump(config))
    return path, config


def test_cluster_train_submits_job_and_records_it(tmp_path):
    path, _ = cluster_recipe(tmp_path)
    assert main(["train", "--config", str(path)]) == 0
    attempt = tmp_path / "tasks/prepare/attempts/001"
    status = json.loads((attempt / "status.json").read_text())
    assert status["execution"] == "submitted"
    assert status["cluster"]["job"] == "pt-test1234"
    args = (tmp_path / "sco/args").read_text().splitlines()
    assert args[:3] == ["acp", "jobs", "create"]
    assert "--worker-nodes=2" in args and "--wait" in args
    assert f"--storage-mount=volume-id:{tmp_path},env-volume:" in "\n".join(args)
    startup = next(a for a in args if a.startswith("--command="))
    assert "sensecore -- env NCCL_IB_TIMEOUT=22 PYTHONPATH=" in startup
    assert f"--attempt {attempt}" in startup
    # The secret reaches sco through its environment, never its arguments.
    assert (tmp_path / "sco/keys").read_text().split() == ["ak-test", "sk-test"]
    assert "sk-test" not in "".join(args)


def test_cluster_rejects_paths_the_container_cannot_see(tmp_path):
    path, config = cluster_recipe(tmp_path)
    mounts = config["cluster"]["mounts"]
    config["cluster"]["mounts"] = ["volume-id:/elsewhere", mounts[1]]
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="outside cluster mounts"):
        load_config(path)
    config["cluster"]["mounts"] = mounts
    config["runtime"]["python"] = "python"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="absolute path"):
        load_config(path)


def test_cluster_nodes_join_torchrun_with_platform_layout(tmp_path, monkeypatch):
    from open_audio_llm import cluster

    path, _ = cluster_recipe(tmp_path)
    attempt = tmp_path / "tasks/prepare/attempts/001"
    attempt.mkdir(parents=True)
    config = load_config(path, attempt)
    (attempt / "effective.yaml").write_text(yaml.safe_dump(config))
    (attempt / "status.json").write_text(json.dumps({"execution": "running"}))
    # Workers stay up until rank 0 has finished recording.
    (attempt / cluster.DONE_MARKER).write_text("{}")
    monkeypatch.setenv("SENSECORE_PYTORCH_NNODES", "2")
    monkeypatch.setenv("SENSECORE_PYTORCH_NODE_RANK", "1")
    monkeypatch.setenv("SENSECORE_ACCELERATE_DEVICE_COUNT", "2")
    monkeypatch.setenv("MASTER_ADDR", "10.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "23456")
    launched = []
    monkeypatch.setattr(
        subprocess, "run",
        lambda cmd, **k: launched.append(cmd) or subprocess.CompletedProcess(cmd, 0),
    )
    assert cluster.run(attempt) == 0
    cmd = launched[0]
    assert cmd[cmd.index("--nnodes") + 1] == "2"
    assert cmd[cmd.index("--node_rank") + 1] == "1"
    assert cmd[cmd.index("--master_addr") + 1] == "10.0.0.1"
    assert cmd[cmd.index("--master_port") + 1] == "23456"
    (attempt / "status.json").write_text(json.dumps({"execution": "failed"}))
    launched.clear()
    assert cluster.run(attempt) == 1 and not launched
    monkeypatch.setenv("SENSECORE_ACCELERATE_DEVICE_COUNT", "8")
    with pytest.raises(ValueError, match="GPUs"):
        cluster.run(attempt)
    monkeypatch.setenv("SENSECORE_ACCELERATE_DEVICE_COUNT", "2")
    monkeypatch.setenv("SENSECORE_PYTORCH_NODE_RANK", "0")
    from open_audio_llm import cli

    monkeypatch.setattr(cli, "execute", lambda config, attempt: 3)
    assert cluster.run(attempt) == 3
    assert json.loads((attempt / cluster.DONE_MARKER).read_text()) == {"exit_code": 3}


def test_cluster_submission_failure_is_recorded(tmp_path):
    path, _ = cluster_recipe(tmp_path)
    (tmp_path / "sco/bin/sco").write_text("#!/bin/sh\necho 'quota denied' >&2\nexit 3\n")
    assert main(["train", "--config", str(path)]) == 1
    status = json.loads(
        (tmp_path / "tasks/prepare/attempts/001/status.json").read_text()
    )
    assert status["execution"] == "failed"
    assert "quota denied" in status["error"]
