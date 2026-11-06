import random

import numpy as np
import pytest
from audio_data_contract import AudioRecord, AudioRef, AudioSlot

from open_audio_llm.data.hotwords import sample_ts_hotwords
from open_audio_llm.data.online_dataset import OnlineAudioDataset
from open_audio_llm.data.records import ResolvedAudioRecord
from open_audio_llm.tsasr.ts_prompt import TS_SYSTEM, ts_system

POOL = [f"pool{i}" for i in range(200)]


def test_ts_system_without_hotwords_is_the_fixed_prompt():
    assert TS_SYSTEM == "Transcribe only the enrolled speaker. Output nothing if they do not speak."
    assert ts_system() == TS_SYSTEM
    assert ts_system("N/A") == TS_SYSTEM
    assert ts_system("e警保,义警") == TS_SYSTEM + "\nHotwords: e警保,义警"


def test_list_keeps_every_target_word_and_only_real_interferer_words():
    for seed in range(200):
        words = sample_ts_hotwords(["e警保", "接警台"], ["义警", "接警台", "义警"], POOL, max_hotwords=30,
                                   prompt_prob=1.0, rng=random.Random(seed)).split(",")
        assert {"e警保", "接警台"} <= set(words)
        assert len(words) == len(set(words)) <= 30
        assert set(words) <= {"e警保", "接警台", "义警", *POOL}


def test_interferer_words_appear_and_the_order_is_shuffled():
    lists = [sample_ts_hotwords(["目标词"], ["别人的词"], POOL, prompt_prob=1.0, rng=random.Random(s)).split(",")
             for s in range(400)]
    assert 0.3 < sum("别人的词" in w for w in lists) / len(lists) < 0.7  # 0..all hard negatives per sample
    assert len({w.index("目标词") for w in lists}) > 5


def test_prompt_probability_and_cap():
    assert sample_ts_hotwords(["a"], [], POOL, prompt_prob=0.0, rng=random.Random(0)) == "N/A"
    assert sample_ts_hotwords([], [], [], prompt_prob=1.0, rng=random.Random(0)) == "N/A"
    capped = sample_ts_hotwords([f"t{i}" for i in range(40)], ["x"], POOL, max_hotwords=30, prompt_prob=1.0,
                                rng=random.Random(1)).split(",")
    assert len(capped) == 30 and all(w.startswith("t") for w in capped)


class _Dataset(OnlineAudioDataset):
    sampling_rate = 16000

    def _resolve_audio(self, resolved, rng):
        return {"enrollment": np.zeros(16000, np.float32), "mixture": np.zeros(32000, np.float32)}


def _ts_record(target):
    return AudioRecord(
        id="ts", task="ts_asr", language="zh", target=target,
        audio_slots=(AudioSlot("enrollment", AudioRef("p", "1", "zh", "e")),
                     AudioSlot("mixture", AudioRef("p", "1", "zh", "m"))),
        hotwords=("e警保",) if target else (),
        metadata={"interferer_hotwords": ["义警"]},
    )


@pytest.mark.parametrize("target", ["打开e警保。", ""])
def test_ts_sample_carries_the_list_and_keeps_the_transcript(target):
    pytest.importorskip("qwen_asr")
    data = _Dataset([ResolvedAudioRecord(_ts_record(target))], message_format="qwen3_asr",
                    ts_hotword_prompt_prob=1.0)
    samples = [data[(epoch, 0, None)] for epoch in range(40)]
    systems = [s["messages"][0]["content"] for s in samples]
    assert all(s == TS_SYSTEM or s.startswith(TS_SYSTEM + "\nHotwords: ") for s in systems)
    if target:  # a present target always lists its own word
        assert all("e警保" in s for s in systems)
    else:  # an absent target's list holds only other people's and unspoken words; some draws list nothing
        assert not any("e警保" in s for s in systems) and any("义警" in s for s in systems)
    assert all(s["messages"][-1]["content"].endswith("<asr_text>" + target) for s in samples)


def test_ts_hotwords_off_keeps_the_fixed_prompt():
    pytest.importorskip("qwen_asr")
    data = _Dataset([ResolvedAudioRecord(_ts_record("打开e警保。"))], message_format="qwen3_asr")
    assert all(data[(epoch, 0, None)]["messages"][0]["content"] == TS_SYSTEM for epoch in range(10))


def test_confusables_join_plain_and_ts_lists():
    from open_audio_llm.data.hotwords import sample_hotwords

    for seed in range(50):
        plain = sample_hotwords(["e警保"], POOL, prompt_prob=1.0, rng=random.Random(seed), confusables=["义警"],
                                confusable_rng=random.Random(seed)).split(",")
        assert {"e警保", "义警"} <= set(plain) and len(plain) == len(set(plain))
        ts = sample_ts_hotwords(["e警保"], [], POOL, prompt_prob=1.0, rng=random.Random(seed),
                                confusables=["义警", "e警保"]).split(",")
        assert {"e警保", "义警"} <= set(ts) and ts.count("e警保") == 1


def test_confusables_leave_the_rng_stream_unchanged():
    from open_audio_llm.data.hotwords import sample_hotwords

    a, b = random.Random(7), random.Random(7)
    sample_hotwords(["x"], POOL, prompt_prob=1.0, rng=a)
    sample_hotwords(["x"], POOL, prompt_prob=1.0, rng=b, confusables=["y"], confusable_rng=random.Random(1))
    assert a.random() == b.random()
