import json
from types import SimpleNamespace

import pytest

from open_audio_llm.integrations.ms_swift.checkpoint_handoff import (
    CheckpointHandoffCallback,
)


def test_final_checkpoint_is_explicit_and_includes_lora_base(tmp_path):
    checkpoint = tmp_path / "checkpoint-10"
    checkpoint.mkdir()
    # A newer-looking directory must never affect selection.
    (tmp_path / "checkpoint-999").mkdir()
    output = tmp_path / "handoff.json"
    callback = CheckpointHandoffCallback(output, base_model="org/base@revision", tuner_type="lora")
    args = SimpleNamespace(output_dir=str(tmp_path))
    state = SimpleNamespace(global_step=10, is_world_process_zero=True)
    callback.on_save(args, state, None)
    callback.on_train_end(args, state, None)
    saved = json.loads(output.read_text())
    assert saved["checkpoint"] == str(checkpoint)
    assert saved["base_model"] == "org/base@revision"
    assert saved["tuner_type"] == "lora"
    assert saved["selection"] == "final"
    assert saved["framework"] == "open-audio-llm"
    assert saved["schema_version"] == 1


def test_no_directory_scan_when_final_step_was_not_saved(tmp_path):
    (tmp_path / "checkpoint-10").mkdir()
    callback = CheckpointHandoffCallback(tmp_path / "handoff.json", base_model="base", tuner_type="full")
    args = SimpleNamespace(output_dir=str(tmp_path))
    state = SimpleNamespace(global_step=10, is_world_process_zero=True)
    callback.on_save(args, state, None)
    state.global_step = 11
    with pytest.raises(RuntimeError, match="No saved checkpoint"):
        callback.on_train_end(args, state, None)
    assert not (tmp_path / "handoff.json").exists()


def test_best_requires_explicit_selection_and_only_rank_zero_writes(tmp_path):
    checkpoint = tmp_path / "checkpoint-2"
    checkpoint.mkdir()
    output = tmp_path / "handoff.json"
    callback = CheckpointHandoffCallback(output, base_model="base", tuner_type="full", selection="best")
    args = SimpleNamespace(output_dir=str(tmp_path))
    state = SimpleNamespace(global_step=10, is_world_process_zero=False, best_model_checkpoint=str(checkpoint))
    callback.on_train_end(args, state, None)
    assert not output.exists()
    state.is_world_process_zero = True
    callback.on_train_end(args, state, None)
    assert json.loads(output.read_text())["checkpoint"] == str(checkpoint)
