import json
from collections import Counter

import pytest

from open_audio_llm.integrations.ms_swift.grpo_prompts import GRPOPromptPlan
from test_catalog_sampler import dataset


def recipe():
    sources = [
        {"dataset_id": "a", "version": "v", "split": "train", "weight": 3},
        {"dataset_id": "b", "version": "v", "split": "train", "weight": 1},
        {"dataset_id": "hard", "version": "v", "split": "all", "weight": 1},
    ]
    ds = dataset(sources, [60, 20, 20])
    ds.config["replay"] = {"epoch_samples": 80, "window_samples": 8}
    return ds


def write_plan(path, entries):
    path.write_text("".join(json.dumps(e) + "\n" for e in entries))
    return str(path)


def test_replay_draws_skip_mined_only_sources_and_mined_entries_take_their_share(tmp_path):
    ds = recipe()
    mined = [{"source": "hard@v:all", "index": [20, i, n], "record_id": str(i)} for n, i in enumerate([85, 90])]
    plan = GRPOPromptPlan(ds, prompts=40, epoch=3, mined_only=["hard@v:all"],
                          mined=[{"plan": write_plan(tmp_path / "m.jsonl", mined), "share": 0.25}])
    assert len(plan) == 40
    replay, extra = plan.plan[:30], plan.plan[30:]
    assert all(index < 80 for _, index, _ in replay)
    assert all(epoch == 3 for epoch, _, _ in replay)
    assert Counter(tuple(t) for t in extra).keys() <= {(20, 85, 0), (20, 90, 1)}
    assert len(extra) == 10


def test_without_mined_plans_the_replay_draw_is_unchanged():
    ds = recipe()
    assert len(GRPOPromptPlan(ds, prompts=40, epoch=3).plan) == 40


def test_mined_entries_must_index_the_same_records(tmp_path):
    ds = recipe()
    wrong = [{"source": "hard@v:all", "index": [20, 85, 0], "record_id": "84"}]
    plan = GRPOPromptPlan(ds, prompts=40, epoch=3, mined=[{"plan": write_plan(tmp_path / "m.jsonl", wrong), "share": 0.2}])
    with pytest.raises(ValueError, match="not the mined record 84"):
        plan[len(plan) - 1]
    outside = [{"source": "a@v:train", "index": [20, 85, 0], "record_id": "85"}]
    with pytest.raises(ValueError, match="not in source"):
        GRPOPromptPlan(ds, prompts=40, epoch=3, mined=[{"plan": write_plan(tmp_path / "m.jsonl", outside), "share": 0.2}])


def test_building_a_mined_plan_reads_no_mined_record(tmp_path):
    # A Lance read before the DataLoader forks crashes or hangs its workers.
    ds = recipe()
    mined = [{"source": "hard@v:all", "index": [20, 85, 0], "record_id": "85"}]
    path = write_plan(tmp_path / "m.jsonl", mined)

    class Unread(list):
        def __getitem__(self, index):
            raise AssertionError(f"read record {index}")

    ds.records = Unread(ds.records)
    GRPOPromptPlan(ds, prompts=40, epoch=3, mined_only=["hard@v:all"], mined=[{"plan": path, "share": 0.25}])


class _Rendering:
    """A dataset view that renders a GRPO sample the way CatalogSwiftDataset does (grpo=True)."""

    def __init__(self, ds):
        self.__dict__.update(vars(ds))

    def __getitem__(self, draw):
        _, record, _ = draw
        return {"audios": [b"RIFF"], "task": "ts_asr", "solution": "language None<asr_text>",
                "online_mix_info": {"mixture": {"kind": "absent", "interferers": [
                    {"clip": f"c{record}", "lookalike": True}]}}}


def test_every_prompt_carries_its_origin_and_online_mix_build(tmp_path):
    ds = _Rendering(recipe())
    mined = [{"source": "hard@v:all", "index": [20, 85, 0], "record_id": "85"}]
    path = tmp_path / "task-7" / "mining" / "plan-ts-absent.jsonl"
    path.parent.mkdir(parents=True)
    plan = GRPOPromptPlan(ds, prompts=40, epoch=3, mined_only=["hard@v:all"],
                          mined=[{"plan": write_plan(path, mined), "share": 0.25}])
    replay, mined_prompt = plan[0], plan[len(plan) - 1]
    assert "online_mix_info" not in replay
    meta = json.loads(replay["grpo_meta"])
    assert meta["origin"] == "replay" and meta["source"] in {"a@v:train", "b@v:train"}
    assert meta["draw"][0] == 3 and meta["record_id"] == str(meta["draw"][1])
    assert meta["task"] == "ts_asr" and meta["target_empty"] is True
    assert meta["mixture"]["interferers"] == [{"clip": f"c{meta['draw'][1]}", "lookalike": True}]
    meta = json.loads(mined_prompt["grpo_meta"])
    assert meta["origin"] == "task-7/plan-ts-absent" and meta["source"] == "hard@v:all" and meta["draw"] == [20, 85, 0]
