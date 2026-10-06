import json
import os

import pytest
import yaml

from open_audio_llm.cli import amphion_eval_check
from open_audio_llm.handoff import HANDOFF_SCHEMA_VERSION
from open_audio_llm.run_config import command, load_config


def evaluator(tmp_path, output):
    # Stands in for the evaluation env's python running `ae open-audio-llm check`.
    script = tmp_path / "eval-python"
    script.write_text(f"#!/bin/sh\necho \"$PYTHONPATH\" > {tmp_path}/pythonpath\ncat <<'EOF'\n{output}\nEOF\n")
    script.chmod(0o755)
    return str(script)


def config(tmp_path, output):
    return {
        "runtime": {"cwd": str(tmp_path)},
        "evaluation": {"python": evaluator(tmp_path, output), "config": "after-training.yaml"},
    }


def report(**changes):
    return json.dumps({"ok": True, "amphion_eval_version": "0.3.0",
                       "handoff_schema": HANDOFF_SCHEMA_VERSION, "problems": [], **changes})


def test_passing_check_is_recorded_without_project_pythonpath(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/repo/src")
    amphion_eval_check(config(tmp_path, report()), tmp_path)
    assert json.loads((tmp_path / "amphion-eval-check.json").read_text())["amphion_eval_version"] == "0.3.0"
    assert (tmp_path / "pythonpath").read_text().strip() == ""


@pytest.mark.parametrize("output, message", [
    (report(ok=False, problems=["W&B credentials are missing"]), "W&B credentials are missing"),
    (report(handoff_schema=1), "reads handoff schema 1, training writes 2"),
    ("usage: ae open-audio-llm: invalid choice: 'check'", "needs amphion-eval >= 0.3.0"),
])
def test_training_is_refused_before_it_starts(tmp_path, output, message):
    with pytest.raises((ValueError, RuntimeError), match=message):
        amphion_eval_check(config(tmp_path, output), tmp_path)


def test_trainer_receives_the_check_report(tmp_path):
    (tmp_path / "train.yaml").write_text(yaml.safe_dump({
        "version": 1,
        "runtime": {"python": "python", "cwd": {"path": "."}, "distributed": {"processes": 1, "port": 1}},
        "task": {"kind": "train", "module": "trainer"},
        "evaluation": {"python": {"path": "eval/bin/python"}, "config": {"path": "after-training.yaml"}},
        "tracking": {"enabled": True},
    }))
    cmd = command(load_config(tmp_path / "train.yaml"), tmp_path / "001" / "effective.yaml")
    assert cmd[cmd.index("--amphion_eval_check") + 1] == str(tmp_path / "001" / "amphion-eval-check.json")
    assert os.path.isabs(load_config(tmp_path / "train.yaml")["evaluation"]["config"])
