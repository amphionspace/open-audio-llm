import gzip
import json
from dataclasses import replace

import pytest
import torch
from audio_data_contract import AudioRecord, AudioRef, AudioSlot

from open_audio_llm.data.think import NO_EXPLANATION, ThinkConfig, ThinkTable, think_answer

KEY = "emilia2_mix@mix-20261013-ctxclean-20261110"
THINK = "zha 这个音，说话人说是「这是山楂的楂」，写作「楂」。"


def record(record_id="train_2spk:a", target="这是山楂的楂。", language="zh", task="asr"):
    return AudioRecord(id=record_id, task=task, language=language, target=target,
                       audio_slots=(AudioSlot("primary", AudioRef("t", "1", "train", "1")),))


@pytest.fixture
def table(tmp_path):
    def build(explained_prob=1.0, ordinary_prob=1.0, rows=None, keys=(KEY,)):
        path = tmp_path / "think.jsonl.gz"
        rows = rows or [
            {"dataset": KEY, "record_id": "train_2spk:a", "think_on": True, "think": THINK},
            {"dataset": KEY, "record_id": "train_2spk:b", "think_on": False, "think": None},
        ]
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            stream.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        return ThinkTable(ThinkConfig((str(path),), explained_prob, ordinary_prob), set(keys))
    return build


def test_listed_records_follow_their_think_and_ordinary_records_get_the_fixed_sentence(table):
    always = table()
    assert always.choose(KEY, record(), "s") == THINK
    assert always.choose(KEY, record("train_2spk:b"), "s") is None
    assert always.choose(KEY, record("train_2spk:c"), "s") == NO_EXPLANATION
    never = table(explained_prob=0.0, ordinary_prob=0.0)
    assert never.choose(KEY, record(), "s") is None
    assert never.choose(KEY, record("train_2spk:c"), "s") is None


def test_only_chinese_asr_with_a_transcript_in_a_source_with_a_think_file_can_think(table):
    always = table(keys=(KEY, "other@1"))
    assert always.choose("other@1", record("x"), "s") is None
    assert always.choose(KEY, record("train_2spk:c", language="en"), "s") is None
    assert always.choose(KEY, record(target=" "), "s") is None
    assert always.choose(KEY, replace(record(), task="speaker_attributed_asr"), "s") is None
    assert always.choose(KEY, record("train_2spk:c", language="Chinese", task="ts_asr"), "s") == NO_EXPLANATION


def test_draws_are_fixed_by_the_example_seed(table):
    half = table(explained_prob=0.5, ordinary_prob=0.5)
    draws = [half.choose(KEY, record(), f"7:0:{i}") for i in range(200)]
    assert draws == [half.choose(KEY, record(), f"7:0:{i}") for i in range(200)]
    assert 60 < draws.count(THINK) < 140


def test_think_files_must_match_the_trained_versions(table):
    with pytest.raises(ValueError, match="no train source reads"):
        table(keys=("emilia2_mix@mix-20261013",))
    with pytest.raises(ValueError, match="listed twice"):
        table(rows=[{"dataset": KEY, "record_id": "a", "think_on": False, "think": None}] * 2)
    with pytest.raises(ValueError, match="probability"):
        ThinkConfig(("x",), 1.5, 0.0)


def test_native_answer_puts_the_think_block_first():
    pytest.importorskip("qwen_asr")
    from open_audio_llm.data.qwen3_asr import native_messages

    answer = native_messages(record(), "N/A", THINK)[-1]["content"]
    assert answer == f"<think>\n{THINK}\n</think>\n\nlanguage Chinese<asr_text>这是山楂的楂。"
    assert answer == think_answer(THINK, native_messages(record(), "N/A")[-1]["content"])


def test_template_leaves_the_opening_think_tag_unsupervised():
    module = pytest.importorskip("open_audio_llm.integrations.ms_swift.register_qwen3_asr")
    template = object.__new__(module.Qwen3ASRTemplate)
    opening = [151667, 198]
    tokenizer = type("Tokenizer", (), {"encode": lambda self, text, add_special_tokens: opening})()
    template.processor = type("Processor", (), {"tokenizer": tokenizer})()
    encoded = {"input_ids": [1, 2, *opening, 7, 151668, 9], "labels": [-100, -100, *opening, 7, 151668, 9],
               "loss_scale": [0, 0, 1, 1, 1, 1, 1]}
    template._prefill_think_open(encoded)
    assert encoded["labels"] == [-100, -100, -100, -100, 7, 151668, 9]
    assert encoded["loss_scale"] == [0, 0, 0, 0, 1, 1, 1]
    plain = {"input_ids": [1, 2, 5, 6], "labels": [-100, -100, 5, 6]}
    template._prefill_think_open(plain)
    assert plain["labels"] == [-100, -100, 5, 6]


def test_think_rows_skip_replay_kl_and_keep_the_header_to_the_language_tag():
    from open_audio_llm.integrations.ms_swift.retention import answer_parts

    asr_text, think_end, newline, pad = 50, 60, 9, -100
    # Labels are aligned to the input; position 0 is never supervised.
    plain = [pad, pad, 1, 2, asr_text, 3, 4, 5]
    thinking = [pad, pad, 7, 8, think_end, newline, 1, 2, asr_text, 3]
    labels = torch.full((2, 10), pad)
    labels[0, :len(plain)] = torch.tensor(plain)
    labels[1] = torch.tensor(thinking)
    think_rows, header = answer_parts(labels, asr_text, think_end)
    assert think_rows.tolist() == [False, True]
    # plain: 1 2 <asr_text> | 3 4 5; thinking: 7 8 </think> | \n 1 2 <asr_text> | 3
    assert header.tolist() == [True, True, True, False, False, False,
                               False, False, False, True, True, True, True, False]


def test_catalog_dataset_reads_think_paths_and_finds_the_source_of_a_record(table, tmp_path):
    from open_audio_llm.data.catalog_dataset import CatalogSwiftDataset, read_data_config

    path = tmp_path / "data.yaml"
    path.write_text("catalog: c\nroots: r.json\ntrain: [{dataset_id: a, version: '1', split: train}]\n"
                    "think: {files: [think/podcast.jsonl.gz], explained_prob: 0.5, ordinary_prob: 0.01}\n")
    assert read_data_config(path)["think"]["files"] == [str(tmp_path / "think/podcast.jsonl.gz")]
    dataset = object.__new__(CatalogSwiftDataset)
    dataset.think = table(keys=(KEY, "other@1"))
    dataset._source_starts, dataset._source_keys = [0, 5], ["other@1", KEY]
    assert dataset._think(7, record(), "s") == THINK
    assert dataset._think(3, record(), "s") is None


def test_serving_chat_template_opens_a_think_block_only_when_asked():
    from pathlib import Path

    import jinja2.sandbox

    import open_audio_llm.integrations.vllm as vllm_integration

    path = Path(vllm_integration.__file__).with_name("qwen3_asr_think.jinja")
    # The environment transformers and vLLM render chat templates with.
    env = jinja2.sandbox.ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    template = env.from_string(path.read_text())
    messages = [{"role": "system", "content": "Hotwords: 楂"},
                {"role": "user", "content": [{"type": "audio", "audio": "a.wav"}]}]
    plain = template.render(messages=messages, add_generation_prompt=True)
    assert plain.endswith("<|im_start|>system\nHotwords: 楂<|im_end|>\n<|im_start|>user\n"
                          "<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n<|im_start|>assistant\n")
    assert template.render(messages=messages, add_generation_prompt=True, enable_thinking=False) == plain
    assert template.render(messages=messages, add_generation_prompt=True, enable_thinking=True) == plain + "<think>\n"
