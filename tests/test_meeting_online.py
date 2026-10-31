import gzip
import json
import random
import re
from io import BytesIO

import numpy as np
import pytest
import soundfile as sf
from audio_data_contract import ArtifactRef, AudioRecord, AudioRef, AudioSlot, DatasetSpec, Split, write_records
from lhotse import Recording, RecordingSet

from open_audio_llm.data.meeting_online import (
    OnlineMeetingConfig,
    fill_turns,
    load_material,
    meeting_target,
    plan_meeting,
    render_meeting,
)

SR = 16000
LINE = re.compile(r"\[(S\d+)\]\[(\d+\.\d\d)-(\d+\.\d\d)\] (.+)")


def _config(**overrides):
    return OnlineMeetingConfig(**{"material": {"dataset_id": "m", "version": "1", "split": "material"},
                                  "noise_sources": ({"dataset_id": "noise", "version": "1", "split": "train"},),
                                  "dry_prob": 1.0, "corpus_weights": {"c": 1.0}, **overrides})


def _rows(meeting, speaker, durations, chars=None, overlap=0.0):
    rows, offset = [], 0.0
    for index, duration in enumerate(durations):
        count = chars[index] if chars else 5
        rows.append({"corpus": "c", "meeting": meeting, "speaker": speaker, "index": index, "offset": offset,
                     "duration": duration, "text": f"{meeting}{speaker}{index}" + "字" * count, "chars": count,
                     "origin": "human", "overlap": overlap, "source_start": 10.0 * index})
        offset += duration + 0.1
    return rows


def _material(config, conversations):
    rows = [row for meeting, speakers in conversations.items() for speaker, durations in speakers.items()
            for row in _rows(meeting, speaker, durations)]
    return load_material(rows, config, lambda row: f"{row['meeting']}/{row['speaker']}")


def test_turns_are_filled_without_self_overlap_and_inside_the_window():
    config = _config()
    voice = _material(config, {"a": {"x": [1.0, 2.0, 3.0, 4.0, 0.5, 6.0] * 4}})["c"][0][0]
    turns = [(0.0, 3.0, 9), (3.2, 12.0, 9), (12.1, 14.0, 9)]
    placed = fill_turns(random.Random(0), turns, 14.0, voice, 0, config)
    ends = [(start, start + voice.durations[index]) for start, index in placed]
    assert all(b <= 14.0 - config.min_gap_seconds + 1e-9 for _, b in ends)
    assert all(b1 <= a2 for (_, b1), (a2, _) in zip(ends, ends[1:]))
    assert len({index for _, index in placed}) == len(placed)  # no utterance twice
    # the 8.8 s turn is filled by more than one utterance of the voice
    assert sum(3.2 <= start < 12.0 for start, _ in placed) >= 2


def test_backchannel_turns_get_backchannels_and_longer_turns_do_not():
    config = _config()
    rows = _rows("a", "x", [0.4, 2.0, 0.3, 2.2, 0.5, 1.8], chars=[1, 8, 2, 9, 1, 7])
    voice = load_material(rows, config, lambda row: "t")["c"][0][0]
    placed = fill_turns(random.Random(1), [(0.0, 0.4, 1), (1.0, 3.0, 8), (4.0, 4.3, 2)], 6.0, voice, 0, config)
    kinds = [bool(voice.backchannel[index]) for _, index in placed]
    assert kinds == [True, False, True]


def test_plan_voices_each_template_speaker_and_the_target_is_timestamped():
    config = _config()
    material = _material(config, {"a": {"x": [1.0, 1.5, 2.0] * 6, "y": [0.8, 1.2, 2.5] * 6},
                                  "b": {"z": [1.1, 1.9] * 8}})
    template = [["p", 0.0, 2.0, 6], ["q", 1.0, 3.0, 6], ["r", 2.5, 4.5, 6], ["p", 5.0, 7.0, 6]]
    corpus, placed = plan_meeting(random.Random(3), template, 8.0, material, config)
    assert corpus == "c" and {speaker for _, _, speaker, _, _ in placed} == {"p", "q", "r"}
    # three template speakers need two conversations' voices, one voice each
    assert len({voice.key for _, _, _, voice, _ in placed}) == 3
    target, mapping = meeting_target(placed)
    lines = [LINE.fullmatch(line).groups() for line in target.splitlines()]
    starts = [float(start) for _, start, _, _ in lines]
    assert starts == sorted(starts) and [label for label, *_ in lines][0] == "S1"
    assert sorted(mapping.values()) == ["S1", "S2", "S3"]
    again = plan_meeting(random.Random(3), template, 8.0, material, config)[1]
    assert [(s, d, k) for s, d, k, _, _ in again] == [(s, d, k) for s, d, k, _, _ in placed]


def test_render_keeps_the_timeline_sets_the_snr_and_the_level():
    config = _config(snr_db=(20.0, 20.0), speech_dbfs=(-26.0, -26.0), level_db_std=0.0)
    length = SR * 2
    track = np.zeros(length, dtype=np.float32)
    track[SR // 2 : SR] = np.sin(np.arange(SR // 2) * 0.1).astype(np.float32)
    noise = np.random.default_rng(0).normal(0, 1, SR).astype(np.float32)
    mixture = render_meeting(random.Random(0), [track], [[(SR // 2, SR)]], length, config, rooms=[],
                             load_rir=None, noise_loader=lambda r: noise, sampling_rate=SR)
    assert len(mixture) == length
    speech, quiet = mixture[SR // 2 : SR], mixture[: SR // 2]
    assert 10 * np.log10(np.mean(quiet**2)) == pytest.approx(-46.0, abs=0.5)
    assert 10 * np.log10(np.mean(speech**2)) == pytest.approx(-26.0, abs=0.5)


def test_render_reverberates_from_the_direct_path():
    config = _config(dry_prob=0.0, rir_sources=({"dataset_id": "rir", "version": "1", "split": "train"},),
                     snr_db=(80.0, 80.0), level_db_std=0.0)
    length = SR
    track = np.zeros(length, dtype=np.float32)
    track[1000] = 1.0
    impulse = np.zeros(400, dtype=np.float32)
    impulse[100], impulse[300] = 1.0, 0.5  # 100 samples of pre-delay before the direct path
    mixture = render_meeting(random.Random(0), [track], [[(1000, 1001)]], length, config, rooms=[["room"]],
                             load_rir=lambda item: impulse, noise_loader=lambda r: np.ones(10, np.float32),
                             sampling_rate=SR)
    assert int(np.argmax(np.abs(mixture))) == 1000
    assert abs(mixture[1200]) == pytest.approx(0.5 * abs(mixture[1000]), rel=1e-3)


def test_config_is_validated():
    with pytest.raises(ValueError, match="material"):
        OnlineMeetingConfig(material={"dataset_id": "m"}, noise_sources=({},), dry_prob=1.0)
    with pytest.raises(ValueError, match="rir_sources"):
        OnlineMeetingConfig(material={"dataset_id": "m", "version": "1", "split": "s"}, noise_sources=({},))
    with pytest.raises(ValueError, match="noise_sources"):
        OnlineMeetingConfig(material={"dataset_id": "m", "version": "1", "split": "s"}, dry_prob=1.0)


@pytest.fixture
def meeting_catalog(tmp_path):
    rows, index = [], []
    for speaker, durations in {"x": [1.0, 1.5, 2.0] * 4, "y": [0.9, 1.4, 2.2] * 4}.items():
        speaker_rows = _rows("a", speaker, durations)
        audio = np.concatenate([np.concatenate([np.full(round(r["duration"] * SR), 0.1 * (i + 1), np.float32),
                                                np.zeros(round(0.1 * SR), np.float32)])
                                for i, r in enumerate(speaker_rows)])
        relative = f"material/{speaker}.flac"
        (tmp_path / "material").mkdir(exist_ok=True)
        sf.write(tmp_path / relative, audio, SR, format="FLAC", subtype="PCM_16")
        index.append({"cut_id": f"c/a/{speaker}", "root_alias": "data", "relative_path": relative,
                      "sample_rate": SR, "channels": 1, "num_frames": len(audio), "duration": len(audio) / SR})
        rows += [{**r, "relative_path": relative} for r in speaker_rows]
    for name, items in (("material.jsonl.gz", rows), ("material-index.jsonl.gz", index)):
        with gzip.open(tmp_path / name, "wt") as stream:
            stream.writelines(json.dumps(item) + "\n" for item in items)
    template = [["p", 0.5, 2.5, 6], ["q", 2.0, 3.5, 6], ["p", 4.0, 5.5, 6]]
    record = AudioRecord("w0", "speaker_attributed_asr",
                         (AudioSlot("mixture", AudioRef("m", "1", "train", "template/c/a", start=0.0, duration=6.0)),),
                         "", language="zh", metadata={"meeting_template": {"turns": template}})
    write_records([record], tmp_path / "windows.jsonl.gz")
    sf.write(tmp_path / "noise.wav", np.random.default_rng(1).normal(0, 0.1, SR).astype(np.float32), SR)
    RecordingSet.from_recordings([Recording.from_file(tmp_path / "noise.wav", recording_id="noise")]).to_file(
        tmp_path / "noise.jsonl.gz")
    specs = [
        DatasetSpec("m", "1", ("zh",), ("speaker_attributed_asr",), (
            ArtifactRef("windows", "audio-records", "data", "windows.jsonl.gz"),
            ArtifactRef("material", "jsonl-metadata", "data", "material.jsonl.gz"),
            ArtifactRef("material-index", "jsonl-metadata", "data", "material-index.jsonl.gz")),
            {"train": Split({"records": ("windows",)}),
             "material": Split({"meeting_material": ("material",), "audio_index": ("material-index",)})}),
        DatasetSpec("noise", "1", ("und",), ("augmentation",),
                    (ArtifactRef("r", "lhotse-recordings", "data", "noise.jsonl.gz"),),
                    {"train": Split({"recordings": ("r",)})}),
    ]
    (tmp_path / "catalog.jsonl").write_text("\n".join(json.dumps(s.to_dict()) for s in specs))
    (tmp_path / "roots.json").write_text(json.dumps({"data": str(tmp_path)}))
    online = {"material": {"dataset_id": "m", "version": "1", "split": "material"},
              "noise_sources": [{"dataset_id": "noise", "version": "1", "split": "train"}],
              "corpus_weights": {"c": 1.0}, "dry_prob": 1.0}
    return {"catalog": str(tmp_path / "catalog.jsonl"), "roots": str(tmp_path / "roots.json"), "seed": 7,
            "sampling_rate": SR, "augmentation": {"speed_prob": 1.0, "speed_factors": [0.9]},
            "train": [{"dataset_id": "m", "version": "1", "split": "train", "online_meeting": online}]}


def test_catalog_dataset_synthesizes_a_new_meeting_each_epoch(meeting_catalog):
    from open_audio_llm.data.catalog_dataset import CatalogSwiftDataset

    dataset = CatalogSwiftDataset(meeting_catalog, message_format="qwen3_asr")
    first, again = dataset[0], dataset[0]
    # same epoch and index, same meeting (the WAV header's PEAK chunk carries a clock, so compare samples)
    assert first["solution"] == again["solution"]
    assert np.array_equal(sf.read(BytesIO(first["audios"][0]))[0], sf.read(BytesIO(again["audios"][0]))[0])
    system, _, answer = first["messages"]
    assert "[S1][0.00-2.50]" in system["content"]  # the timestamped SOT protocol
    lines = answer["content"].split("<asr_text>", 1)[1].splitlines()
    parsed = [LINE.fullmatch(line).groups() for line in lines]
    assert {label for label, *_ in parsed} == {"S1", "S2"} and float(parsed[0][1]) == pytest.approx(0.5)
    audio, rate = sf.read(BytesIO(first["audios"][0]))
    assert rate == SR and len(audio) == 6 * SR  # the window, with no speed change (timestamps)
    dataset.set_epoch(1)
    assert dataset[0]["solution"] != first["solution"]
    with pytest.raises(ValueError, match="training only"):
        CatalogSwiftDataset({**meeting_catalog, "validation": meeting_catalog["train"]}, training=False,
                            message_format="qwen3_asr")
