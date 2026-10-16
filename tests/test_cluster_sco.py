import subprocess

import pytest

from open_audio_llm import cluster


def make_home(tmp_path, version):
    (tmp_path / ".data").mkdir()
    (tmp_path / ".data/product.yaml").write_text(f'registry_version: "{version}"\n')
    return {"sco_home": str(tmp_path)}


def fake_registry(monkeypatch, versions="versions:\n- version: \"20260930\"\n"):
    def fetch(url, timeout=60):
        if url.endswith("versions.yaml"):
            if versions is None:
                raise OSError("offline")
            return versions
        return "echo installer"
    monkeypatch.setattr(cluster, "fetch_text", fetch)


def record_runs(monkeypatch, returncode=0):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, returncode, stdout="", stderr="boom")
    monkeypatch.setattr(cluster.subprocess, "run", run)
    return calls


def test_stale_release_is_reinstalled_into_the_same_home(tmp_path, monkeypatch):
    config = make_home(tmp_path, "20260830")
    fake_registry(monkeypatch)
    calls = record_runs(monkeypatch)
    assert cluster.refresh_sco(config) is True
    command, kwargs = calls[0]
    assert command == ["sh", "-s"] and kwargs["input"] == "echo installer"
    assert kwargs["env"]["SCO_HOME"] == str(tmp_path)
    assert kwargs["env"]["SCO_DATA_HOME"] == f"{tmp_path}/.data"


def test_current_release_is_left_alone(tmp_path, monkeypatch):
    config = make_home(tmp_path, "20260930")
    fake_registry(monkeypatch)
    calls = record_runs(monkeypatch)
    assert cluster.refresh_sco(config) is False and not calls


def test_unreachable_registry_does_not_block_submission(tmp_path, monkeypatch):
    config = make_home(tmp_path, "20260830")
    fake_registry(monkeypatch, versions=None)
    calls = record_runs(monkeypatch)
    assert cluster.refresh_sco(config) is False and not calls


def test_failed_reinstall_is_reported(tmp_path, monkeypatch):
    config = make_home(tmp_path, "20260830")
    fake_registry(monkeypatch)
    record_runs(monkeypatch, returncode=1)
    with pytest.raises(RuntimeError, match="20260830 -> 20260930"):
        cluster.refresh_sco(config)


def test_missing_installation_is_not_created(tmp_path, monkeypatch):
    fake_registry(monkeypatch)
    calls = record_runs(monkeypatch)
    assert cluster.refresh_sco({"sco_home": str(tmp_path)}) is False and not calls


def test_job_names_start_with_the_asr_prefix(tmp_path):
    config = {"tracking": {"name": "zh-demo-20261016"}}
    assert cluster.job_name(config, tmp_path / "001") == "ASR-zh-demo-20261016-001"
    assert cluster.job_name({"tracking": {"name": "ASR-x"}}, tmp_path / "002") == "ASR-x-002"
