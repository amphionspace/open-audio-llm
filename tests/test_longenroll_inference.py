from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("torchaudio")
transformers = pytest.importorskip("transformers")

from open_audio_llm.longenroll.ecapa import ECAPATDNN  # noqa: E402
from open_audio_llm.longenroll.features import (  # noqa: E402
    DUMMY_ENROLL_SECONDS,
    TS_SYSTEM,
    LongEnrollFeatures,
    request_system,
)
from open_audio_llm.longenroll.frontend import (  # noqa: E402
    cast_for_inference,
    embed_requests,
    load_waveform,
    parse_answer,
)
from open_audio_llm.longenroll.model import BOTH, NONE  # noqa: E402

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
            ids.append(SPECIAL[special] if special else 100 + ord(text[0]) % 1000)
            text = text[len(special) if special else 1:]
        return {"input_ids": ids}


def sample(task, audios, answer, system="", mix=None):
    out = {"task": task, "audios": audios, "dataset_id": "x",
           "messages": [{"role": "system", "content": system}, {"role": "user", "content": "<audio>"},
                        {"role": "assistant", "content": answer}]}
    if mix is not None:
        out["mix_wav"] = mix
    return out


@pytest.fixture(scope="module")
def features():
    extractor = transformers.WhisperFeatureExtractor(feature_size=128)
    return LongEnrollFeatures(SimpleNamespace(feature_extractor=extractor, tokenizer=Tokenizer()))


def test_request_matches_validation_example(features):
    rng = np.random.default_rng(0)
    enroll = rng.standard_normal(16000 * 40).astype(np.float32) * 0.1
    mix = rng.standard_normal(16000 * 7).astype(np.float32) * 0.1
    answer = "language English<asr_text>hi"
    for task, request, record in (
        ("ts", features.request("ts", mix, enroll), sample("ts_asr", [enroll], answer, mix=mix)),
        ("hotword", features.request("hotword", mix, hotwords="a,b"),
         sample("asr_hotwords", [mix], answer, system="Hotwords: a,b")),
        ("asr", features.request("asr", mix), sample("asr", [mix], answer)),
    ):
        example = features(record, None)  # validation: no rng
        assert request["mode"] == example["mode"] and request["system"] == example["system"], task
        for key in ("mix_mel", "enroll_mel", "ecapa_fbank"):
            assert torch.equal(request[key], example[key]), (task, key)
        prompt = [t for t, label in zip(example["input_ids"], example["labels"]) if label == -100]
        assert request["input_ids"] == prompt and set(request["labels"]) == {-100}, task
    assert features.request("asr", mix)["enroll_mel"].shape[-1] == int(DUMMY_ENROLL_SECONDS * 100)
    assert features.request("asr", mix)["mode"] == NONE and features.request("ts", mix, enroll)["mode"] == BOTH
    with pytest.raises(ValueError):
        features.request("ts", mix)
    with pytest.raises(ValueError):
        features.request("asr", mix, enroll)


def test_request_system():
    assert request_system("ts") == TS_SYSTEM
    assert request_system("asr") == "" and request_system("hotword") == ""
    assert request_system("hotword", "甲,乙") == "Hotwords: 甲,乙"
    with pytest.raises(ValueError):
        request_system("sot")


def test_parse_answer():
    assert parse_answer("language Chinese<asr_text>你好") == ("Chinese", "你好")
    assert parse_answer("language None<asr_text>") == ("None", "")
    assert parse_answer("no tag") == ("", "no tag")


def test_load_waveform_span_and_resample(tmp_path):
    import soundfile as sf

    tone = np.sin(2 * np.pi * 440 * np.arange(8000 * 3) / 8000).astype(np.float32) * 0.5
    sf.write(tmp_path / "a.wav", np.stack([tone, tone], axis=1), 8000, subtype="FLOAT")
    whole = load_waveform(str(tmp_path / "a.wav"))
    assert whole.dtype == np.float32 and whole.shape == (16000 * 3,)
    span = load_waveform({"path": str(tmp_path / "a.wav"), "start": 1.0, "duration": 0.5})
    assert span.shape == (8000,)


class FakeModel:
    """Embeds each item as n_audio rows holding its mode and mixture length; records batches."""

    device = torch.device("cpu")

    def __init__(self):
        self.batches = []

    def encode_enrollment(self, mels, fbanks):
        return None

    def encode_mix(self, mix_mels, enroll, modes):
        self.batches.append([m.shape[-1] for m in mix_mels])
        rows = [torch.full((int(m.shape[-1]) // 100, 2), float(m.shape[-1])) for m in mix_mels]
        for row, mode in zip(rows, modes.tolist()):
            row[:, 1] = mode
        return torch.cat(rows)


def test_embed_requests_returns_request_order(features):
    class Items:
        def request(self, task, mixture, enrollment=None, hotwords=""):
            frames = len(mixture) // 160
            return {"mix_mel": torch.zeros(128, frames), "enroll_mel": None, "ecapa_fbank": None,
                    "mode": BOTH if task == "ts" else NONE, "n_audio": frames // 100}

    seconds = [3, 1, 2, 5, 4]
    requests = [{"task": "ts" if s % 2 else "asr", "mixture": np.zeros(16000 * s)} for s in seconds]
    for workers in (1, 3):  # threads build the features; order and grouping must not change
        model = FakeModel()
        items, embeds, _, _ = embed_requests(model, Items(), requests, batch_size=2, workers=workers)
        assert model.batches == [[100, 200], [300, 400], [500]]  # grouped by length
        assert [i["n_audio"] for i in items] == seconds
        for s, request, embed in zip(seconds, requests, embeds):
            assert embed.shape[0] == s and torch.all(embed[:, 0] == s * 100)
            assert torch.all(embed[:, 1] == (BOTH if request["task"] == "ts" else NONE))


def test_threaded_requests_match_serial(features):
    from concurrent.futures import ThreadPoolExecutor

    rng = np.random.default_rng(3)
    calls = [("ts", rng.standard_normal(16000 * s).astype(np.float32) * 0.1,
              rng.standard_normal(16000 * (s + 9)).astype(np.float32) * 0.1) for s in (2, 5, 3, 7)]
    serial = [features.request(*c) for c in calls]
    with ThreadPoolExecutor(4) as pool:
        threaded = list(pool.map(lambda c: features.request(*c), calls))
    for a, b in zip(serial, threaded):
        assert a["input_ids"] == b["input_ids"]
        for key in ("mix_mel", "enroll_mel", "ecapa_fbank"):
            assert torch.equal(a[key], b[key])


def test_cast_for_inference_keeps_ecapa_fp32():
    model = torch.nn.Module()
    model.ecapa = ECAPATDNN()
    model.speaker = torch.nn.Linear(4, 4)
    stats = torch.full_like(model.ecapa.bn.running_mean, 1.0 + 2 ** -12)  # not representable in bf16
    model.ecapa.bn.running_mean.copy_(stats)
    cast_for_inference(model, "cpu")
    assert model.speaker.weight.dtype == torch.bfloat16
    assert model.ecapa.bn.running_mean.dtype == torch.float32 and torch.equal(model.ecapa.bn.running_mean, stats)
    assert not any(p.requires_grad for p in model.parameters()) and not model.training


def test_vllm_prompt_has_one_audio_pad(features):
    ids = features.prompt.build(1, None, system=TS_SYSTEM)[0]
    assert ids.count(SPECIAL["<|audio_pad|>"]) == 1
    full = features.prompt.build(13, None, system=TS_SYSTEM)[0]
    pad = SPECIAL["<|audio_pad|>"]
    assert [t for t in full if t != pad] == [t for t in ids if t != pad]


def test_optimizer_groups_skip_decay_on_biases_norms_and_gates():
    from open_audio_llm.longenroll.model import LongEnrollTSASR

    model = torch.nn.Module()
    model.asr = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.LayerNorm(4))
    model.enroll_encoder = torch.nn.Module()
    model.enroll_encoder.proj = torch.nn.Linear(4, 4, bias=False)
    model.enroll_encoder.depth_norms = torch.nn.ModuleDict({"12": torch.nn.RMSNorm(4)})
    model.enroll_encoder.query = torch.nn.Parameter(torch.zeros(3, 4))
    model.gca_gates = torch.nn.ParameterDict({"12": torch.nn.Parameter(torch.tensor(0.01))})
    model.speaker = torch.nn.Module()
    model.speaker.gate = torch.nn.Parameter(torch.tensor(0.01))
    model.frozen = torch.nn.Linear(4, 4)
    model.frozen.requires_grad_(False)
    groups = LongEnrollTSASR.optimizer_groups(model, 1e-4, 1e-5, 2.0, 0.1)
    names = {id(p): n for n, p in model.named_parameters()}
    got = {g["name"]: (sorted(names[id(p)] for p in g["params"]), g["lr"], g["weight_decay"]) for g in groups}
    assert got == {
        "new": (["enroll_encoder.proj.weight", "enroll_encoder.query"], 1e-4, 0.1),
        "new_no_decay": (["enroll_encoder.depth_norms.12.weight"], 1e-4, 0.0),
        "gates": (["gca_gates.12", "speaker.gate"], 2e-4, 0.0),
        "backbone_no_decay": (["asr.0.bias", "asr.1.bias", "asr.1.weight"], 1e-5, 0.0),
        "backbone": (["asr.0.weight"], 1e-5, 0.1),
    }
    assert groups[-1]["name"] == "backbone"  # train.py logs the last group's rate as backbone_lr
