import json
from collections import Counter

import pytest

from open_audio_llm.integrations.ms_swift.grpo_prompts import GRPOPromptPlan
from test_catalog_sampler import dataset


def recipe():
    sources = [
        {"dataset_id": "a", "version": "v", "split": "train", "weight": 3},
        {"dataset_id": "b", "version": "v", "split": "train", "weight": 1},
        {"dataset_id": "hard", "version": "v", "split": "all", "weight": 1, "grpo_mined_only": True},
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
    plan = GRPOPromptPlan(ds, prompts=40, epoch=3, mined=[{"plan": write_plan(tmp_path / "m.jsonl", mined), "share": 0.25}])
    assert len(plan) == 40
    replay, extra = plan.plan[:30], plan.plan[30:]
    assert all(index < 80 for _, index, _ in replay)
    assert all(epoch == 3 for epoch, _, _ in replay)
    assert Counter(tuple(t) for t in extra).keys() <= {(20, 85, 0), (20, 90, 1)}
    assert len(extra) == 10


def test_without_mined_plans_the_replay_draw_is_unchanged():
    ds = recipe()
    ds.sources[2].pop("grpo_mined_only")
    assert len(GRPOPromptPlan(ds, prompts=40, epoch=3).plan) == 40


def test_mined_entries_must_index_the_same_records(tmp_path):
    ds = recipe()
    wrong = [{"source": "hard@v:all", "index": [20, 85, 0], "record_id": "84"}]
    with pytest.raises(ValueError, match="is not 84"):
        GRPOPromptPlan(ds, prompts=40, epoch=3, mined=[{"plan": write_plan(tmp_path / "m.jsonl", wrong), "share": 0.2}])
    outside = [{"source": "a@v:train", "index": [20, 85, 0], "record_id": "85"}]
    with pytest.raises(ValueError, match="not in source"):
        GRPOPromptPlan(ds, prompts=40, epoch=3, mined=[{"plan": write_plan(tmp_path / "m.jsonl", outside), "share": 0.2}])
