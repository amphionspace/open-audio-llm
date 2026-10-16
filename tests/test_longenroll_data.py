from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("torchaudio")
transformers = pytest.importorskip("transformers")

from open_audio_llm.longenroll.backbone import feat_extract_output_lengths  # noqa: E402
from open_audio_llm.longenroll.data import (  # noqa: E402
    DUMMY_ENROLL_SECONDS,
    TS_SYSTEM,
    LongEnrollFeatures,
    collate,
)
from open_audio_llm.longenroll.model import BOTH, NONE  # noqa: E402
from open_audio_llm.longenroll.train import score, validation_batches  # noqa: E402

SPECIAL = {"<|im_start|>": 1, "<|im_end|>": 2, "<|audio_start|>": 3, "<|audio_end|>": 4, "<|audio_pad|>": 5}


class Tokenizer:
    """Specials map to fixed ids, every other character to one id."""

    pad_token_id = 0

    def convert_tokens_to_ids(self, token):
        return SPECIAL[token]

    def __call__(self, text, add_special_tokens=False):
        ids = []
        while text:
            special = next((s for s in SPECIAL if text.startswith(s)), None)
            if special:
                ids.append(SPECIAL[special])
                text = text[len(special):]
            else:
                ids.append(100 + ord(text[0]) % 1000)
                text = text[1:]
        return {"input_ids": ids}


@pytest.fixture(scope="module")
def features():
    extractor = transformers.WhisperFeatureExtractor(feature_size=128)
    return LongEnrollFeatures(SimpleNamespace(feature_extractor=extractor, tokenizer=Tokenizer()))


def sample(task, audios, answer, system="", mix=None):
    out = {"task": task, "audios": audios, "dataset_id": "x",
           "messages": [{"role": "system", "content": system}, {"role": "user", "content": "<audio>"},
                        {"role": "assistant", "content": answer}]}
    if mix is not None:
        out["mix_wav"] = mix
    return out


def test_ts_and_plain_examples(features):
    rng = np.random.default_rng(0)
    enroll = rng.standard_normal(16000 * 40).astype(np.float32) * 0.1
    mix = rng.standard_normal(16000 * 7).astype(np.float32) * 0.1
    ts = features(sample("ts_asr", [enroll], "language English<asr_text>hi", system="native concat prompt", mix=mix),
                  np.random.default_rng(1))
    assert ts["mode"] == BOTH and ts["system"] == TS_SYSTEM and ts["task"] == "ts"
    assert ts["enroll_mel"].shape[-1] == 4000 and ts["mix_mel"].shape[-1] == 700
    assert ts["n_audio"] == int(feat_extract_output_lengths(torch.tensor(700)))
    assert ts["input_ids"].count(SPECIAL["<|audio_pad|>"]) == ts["n_audio"]
    answer = [t for t, label in zip(ts["input_ids"], ts["labels"]) if label != -100]
    assert answer == Tokenizer()("language English<asr_text>hi")["input_ids"] + [SPECIAL["<|im_end|>"]]
    assert 5 * 100 - 3 <= ts["ecapa_fbank"].shape[0] <= 10 * 100  # 5-10 s head, 100 fbank frames/s

    short = features(sample("ts_asr", [enroll[: 16000 * 7]], "language English<asr_text>hi", mix=mix),
                     np.random.default_rng(2))
    assert 7 * 100 - 3 <= short["ecapa_fbank"].shape[0] <= 7 * 100  # <= 10 s enrollment: ECAPA gets all of it
    assert features(sample("ts_asr", [enroll], "language English<asr_text>hi", mix=mix), None)[
        "ecapa_fbank"].shape[0] <= 10 * 100  # validation: a long enrollment gives its first 10 s

    absent = features(sample("ts_asr", [enroll], "language None<asr_text>", mix=mix), None)
    assert absent["task"] == "ts_absent"

    hot = features(sample("asr_hotwords", [mix], "language Chinese<asr_text>你好", system="Hotwords: 你好"), None)
    assert hot["mode"] == NONE and hot["system"] == "Hotwords: 你好" and hot["task"] == "hotword"
    assert hot["enroll_mel"].shape[-1] == int(DUMMY_ENROLL_SECONDS * 100) and hot["enroll_sec"] == 0.0

    batch = collate([ts, hot], pad_id=0)
    assert batch["input_ids"].shape == batch["labels"].shape == batch["attention_mask"].shape
    assert batch["mode"].tolist() == [BOTH, NONE] and batch["task"] == ["ts", "hotword"]


def test_validation_batches_cover_each_record_once():
    durations = np.array([5.0, 30.0, 12.0, 1.0, 29.0, 7.0, 3.0])
    dataset = SimpleNamespace(config={"batching": {"max_duration": 40, "max_samples": 3}},
                              records=SimpleNamespace(durations=durations), __len__=None)
    dataset = type("D", (), {"config": dataset.config, "records": dataset.records,
                             "__len__": lambda self: len(durations)})()
    seen = []
    for rank in range(2):
        for batch in validation_batches(dataset, rank, 2):
            assert len(batch) <= 3 and sum(durations[i] for _, i, _ in batch) <= 40
            seen += [i for _, i, _ in batch]
    assert sorted(seen) == list(range(len(durations)))


def test_score_units():
    assert score("language Chinese<asr_text>今天天气很好。", "language Chinese<asr_text>今天天气好", "Chinese") == (1, 6)
    assert score("language English<asr_text>Hello, world!", "language English<asr_text>hello word", "English") == (1, 2)
    assert score("language None<asr_text>", "language English<asr_text>", "None") == (0, 1)
    assert score("language None<asr_text>", "language English<asr_text>hi", "None") == (1, 1)


def test_metrics_send_the_same_keys_on_every_rank():
    from open_audio_llm.longenroll.train import TRAIN_KEYS, Metrics

    seen_ts, seen_asr = Metrics(TRAIN_KEYS), Metrics(TRAIN_KEYS)
    seen_ts.add_tasks("", ["ts"], torch.tensor([2.0]), torch.tensor([4]))
    seen_asr.add_tasks("", ["asr", "hotword"], torch.tensor([1.0, 3.0]), torch.tensor([2, 6]))
    one, two = (m.reduce("cpu", distributed=False) for m in (seen_ts, seen_asr))
    assert list(one) == list(two) == list(TRAIN_KEYS)
    assert one["ce_ts"] == 2.0 and one["ce_asr"] == 0.0 and two["tok_hotword"] == 6.0
    with pytest.raises(KeyError):
        seen_ts.add("unknown", 1.0)
