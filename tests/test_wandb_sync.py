import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from open_audio_llm.scripts.sync_wandb import (
    aggregate_metrics, collect_events, evaluation_metrics, read_jsonl,
    run_config, upload_events,
)


def metric(units=10, errors=2, scored=4, correct=3):
    return {"reference_units": units, "errors": errors, "utterances": 2,
            "metric": "cpWER", "language": "en", "speaker_count_accuracy": .5,
            "format_valid_rate": 1, "empty_output_rate": 0,
            "speaker_attribution": {"method": "unique_reference_units_v1",
                                    "scored_units": scored, "correct_units": correct}}


def test_partial_log_record_is_read_only_after_writer_finishes(tmp_path):
    path = tmp_path / "logging.jsonl"
    path.write_bytes(b'{"loss":1}\n{"loss":')
    assert read_jsonl(path) == [{"loss": 1}]
    with path.open("ab") as stream:
        stream.write(b'2}\n')
    assert read_jsonl(path) == [{"loss": 1}, {"loss": 2}]


@pytest.mark.parametrize("events", [{}, {"performance:0": {"global_step": 0}}])
def test_sync_can_start_before_the_first_training_loss(tmp_path, monkeypatch, events):
    from open_audio_llm.scripts import sync_wandb

    run = SimpleNamespace(entity="team", project="test", id="run", url="local",
                          summary={}, define_metric=lambda *a, **k: None,
                          log=lambda row: None)
    client = MagicMock()
    client.init.return_value.__enter__.return_value = run
    client.Api.return_value.run.return_value.scan_history.return_value = []
    monkeypatch.setitem(sys.modules, "wandb", client)
    monkeypatch.setattr(sync_wandb, "run_config", lambda root: {})
    monkeypatch.setattr(sync_wandb, "collect_events", lambda root: events)
    monkeypatch.setattr(sys, "argv", ["sync_wandb", "--run-dir", str(tmp_path),
                                     "--entity", "team"])
    sync_wandb.main()
    assert run.summary["latest_training_step"] == 0
    assert run.summary["latest_evaluated_step"] == 0
    assert run.summary["sync_event_count"] == len(events)


def test_aggregation_uses_reference_and_scorable_unit_counts():
    result = aggregate_metrics([metric(), metric(90, 9, 36, 36)])
    assert result["cpWER"] == 11
    assert result["attribution_accuracy"] == 97.5
    assert result["attribution_coverage"] == 40
    assert "attribution_accuracy" not in aggregate_metrics([metric(scored=0, correct=0)])


def test_asr_dialects_are_aggregated_without_overwriting_each_other():
    summary = {"metrics": {"kespeech@v1:dev/asr/Mandarin": metric(10, 2),
                           "kespeech@v1:dev/asr/Jiang-Huai": metric(90, 9)}}
    for row in summary["metrics"].values():
        row["metric"] = "CER"
    assert evaluation_metrics(summary, "asr")["eval/asr/kespeech/CER"] == 11


def test_sot_preserves_speaker_count_overlap_and_language_conditions():
    summary = {"metrics": {"sot@v1:dev_en_3spk_dense/speaker_attributed_asr/en": metric()}}
    result = evaluation_metrics(summary, "sot")
    assert result["eval/sot/en/3spk/dense/cpWER"] == 20
    assert result["eval/sot/en/attribution_accuracy"] == 75


def test_late_evaluation_uses_custom_step_and_existing_events_are_not_resent():
    events = {"train:110": {"global_step": 110, "train/loss": .2}}
    rows = []
    run = SimpleNamespace(log=lambda row: rows.append(row))
    seen = set()
    assert upload_events(run, events, seen) == 1
    events["evaluation:100"] = {"global_step": 100, "eval/sot/en/cpWER": 20}
    assert upload_events(run, events, seen) == 1
    assert [row["global_step"] for row in rows] == [110, 100]
    assert upload_events(run, events, seen) == 0


def test_incomplete_or_failed_evaluations_are_not_uploaded(tmp_path):
    folder = tmp_path / "training/retention-evaluations/checkpoint-100"
    for task in ("sot", "asr"):
        (folder / task).mkdir(parents=True)
        (folder / task / "summary.json").write_text(json.dumps({"metrics": {"test": metric()}}))
    assert collect_events(tmp_path) == {}
    (folder / "runner-exit.json").write_text('{"returncode":1}')
    assert collect_events(tmp_path) == {}
    (folder / "runner-exit.json").write_text('{"returncode":0}')
    assert "evaluation:100" in collect_events(tmp_path)


def write_config(tmp_path):
    (tmp_path / "training").mkdir()
    (tmp_path / "training/args.json").write_text(json.dumps({
        "learning_rate": 1e-5, "api_key": "secret-sentinel", "model": "/private/model"}))
    (tmp_path / "training/audit-startup-rank0.json").write_text(json.dumps({
        "world_size": 2, "global_step": 100, "optimizer_steps": []}))
    (tmp_path / "train-data.yaml").write_text(json.dumps({
        "train": [{"dataset_id": "sot", "version": "v1", "split": "train",
                   "weight": 60, "exclude_speakers": ["private-person"]}],
        "objective": {"replay_kl_weight": 2}, "replay": {}, "batching": {}}))


def test_config_excludes_secrets_and_private_sample_metadata(tmp_path):
    write_config(tmp_path)
    config = run_config(tmp_path)
    assert config["learning_rate"] == 1e-5
    assert config["resume_step"] == 100 and config["optimizer_reinitialized"]
    for private in ("secret-sentinel", "/private/model", "private-person"):
        assert private not in json.dumps(config)


@pytest.mark.parametrize("provenance", [
    {"source_origin": "/frozen/source", "files": {"source/train.py": "file-digest"}},
    {"base_git_commit": "base-revision", "files": {"source/train.py": "file-digest"}},
    {"git_commit": "exact-revision", "files_sha256": {"source/train.py": "file-digest"},
     "run_files_sha256": {"track.py": "tracker-digest"}},
])
def test_frozen_provenance_does_not_require_or_invent_a_commit(tmp_path, provenance):
    import hashlib

    write_config(tmp_path)
    path = tmp_path / "source-provenance.json"
    path.write_text(json.dumps(provenance))
    config = run_config(tmp_path)
    assert config["source_git_commit"] == provenance.get("git_commit")
    assert config["source_base_git_commit"] == provenance.get("base_git_commit")
    assert config["source_file_hashes"]["source/train.py"] == "file-digest"
    assert config["source_snapshot_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    if "source_origin" in provenance:
        assert config["source_origin"] == provenance["source_origin"]


def test_restart_reads_remote_acknowledgements_before_backfill(tmp_path, monkeypatch):
    from open_audio_llm.scripts import sync_wandb

    rows = []
    run = SimpleNamespace(entity="team", project="test", id="run", url="local",
                          summary={}, define_metric=lambda *a, **k: None, log=rows.append)
    client = MagicMock()
    client.init.return_value.__enter__.return_value = run
    client.Api.return_value.run.return_value.scan_history.return_value = [{"sync_event": "train:5"}]
    monkeypatch.setitem(sys.modules, "wandb", client)
    monkeypatch.setattr(sync_wandb, "run_config", lambda root: {})
    monkeypatch.setattr(sync_wandb, "collect_events", lambda root: {
        "train:5": {"global_step": 5, "train/loss": .4},
        "train:10": {"global_step": 10, "train/loss": .3}})
    monkeypatch.setattr(sys, "argv", ["sync_wandb", "--run-dir", str(tmp_path), "--entity", "team"])
    sync_wandb.main()
    assert rows == [{"sync_event": "train:10", "global_step": 10, "train/loss": .3}]
    assert run.summary["sync_event_count"] == 2


@pytest.mark.parametrize('local_snapshot', [False, True])
def test_recovery_prefers_its_actual_snapshot_over_parent(tmp_path, local_snapshot):
    import hashlib

    write_config(tmp_path)
    parent = tmp_path / 'parent'
    parent.mkdir()
    inherited = parent / 'source-provenance.json'
    inherited.write_text(json.dumps({'git_commit': 'parent-revision',
                                    'files': {'source/train.py': 'parent-digest'}}))
    (tmp_path / 'recovery.json').write_text(json.dumps({
        'source_run': str(parent), 'source_checkpoint': str(parent / 'checkpoint-500')}))
    expected = inherited
    if local_snapshot:
        expected = tmp_path / 'source-provenance.json'
        expected.write_text(json.dumps({'base_git_commit': 'parent-revision',
                                       'files': {'source/train.py': 'local-digest'}}))
    config = run_config(tmp_path)
    assert config['parent_checkpoint'] == 'checkpoint-500'
    assert config['source_git_commit'] == (None if local_snapshot else 'parent-revision')
    assert config['source_file_hashes']['source/train.py'] == ('local-digest' if local_snapshot else 'parent-digest')
    assert config['source_snapshot_sha256'] == hashlib.sha256(expected.read_bytes()).hexdigest()
    assert config['source_origin'] == str(expected.parent / 'source')


def test_attribution_methods_must_match():
    first, second = metric(), metric()
    second["speaker_attribution"]["method"] = "different_method"
    with pytest.raises(ValueError, match="different"):
        aggregate_metrics([first, second])


def test_timestamp_aggregation_uses_segment_counts_and_keeps_seconds():
    first, second = metric(), metric()
    first['timestamps'] = dict(method='text_assignment_utterance_boundaries_v1', collar_seconds=.5,
        reference_segments=2, predicted_segments=1, matched_segments=1, within_collar_segments=1,
        boundary_absolute_error_seconds=.4)
    second['timestamps'] = dict(method='text_assignment_utterance_boundaries_v1', collar_seconds=.5,
        reference_segments=8, predicted_segments=9, matched_segments=8, within_collar_segments=5,
        boundary_absolute_error_seconds=5.)
    result = aggregate_metrics([first, second])
    assert result['timestamp_precision'] == result['timestamp_recall'] == result['timestamp_f1'] == 60
    assert result['timestamp_matched_reference_fraction'] == 90
    assert result['timestamp_boundary_mae_seconds'] == pytest.approx(.3)
    second['timestamps']['collar_seconds'] = .25
    with pytest.raises(ValueError, match='different timestamp'):
        aggregate_metrics([first, second])


def test_timestamp_reference_coverage_uses_record_counts():
    first, second = metric(), metric()
    second['utterances'] = 8
    first['timestamp_reference_coverage'] = dict(available_utterances=1, total_utterances=2, fraction=.5)
    second['timestamp_reference_coverage'] = dict(available_utterances=8, total_utterances=8, fraction=1.)
    result = aggregate_metrics([first, second])
    assert result['timestamp_reference_records'] == 9
    assert result['timestamp_total_records'] == 10
    assert result['timestamp_reference_coverage'] == 90
