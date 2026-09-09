from audio_data_contract import (
    AudioRecord,
    AudioRef,
    AudioSlot,
    PromptAudio,
    PromptTemplate,
    PromptText,
    render_example,
)

from open_audio_llm.data.records import to_swift_record


def test_interleaving_derives_audio_order():
    record = AudioRecord(
        id="utt2",
        task="compare",
        audio_slots=(
            AudioSlot("first", AudioRef("a", "1", "train", "cut-a")),
            AudioSlot("second", AudioRef("b", "2", "dev", "cut-b")),
            AudioSlot("third", AudioRef("a", "1", "train", "cut-c")),
        ),
        target="answer",
    )
    template = PromptTemplate(
        "interleaved",
        "1",
        (
            (
                PromptText("A"),
                PromptAudio("second"),
                PromptText("B"),
                PromptAudio("first"),
                PromptAudio("third"),
            ),
        ),
    )
    example = render_example(record, template)
    swift = to_swift_record(
        example,
        {slot.name: f"/{slot.ref.cut_id}.wav" for slot in record.audio_slots},
        solution="answer",
    )
    assert swift["messages"][0]["content"] == "A<audio>B<audio><audio>"
    assert swift["audios"] == ["/cut-b.wav", "/cut-a.wav", "/cut-c.wav"]
    assert swift["audio_slot_count"] == 3


def test_catalog_record_hotwords_populate_distractor_pool():
    from open_audio_llm.data.hotwords import build_hotword_pool
    from open_audio_llm.data.records import ResolvedAudioRecord

    record = AudioRecord(
        id="hotword-record",
        task="asr_hotwords",
        audio_slots=(AudioSlot("primary", AudioRef("demo", "1", "train", "cut")),),
        target="hello world",
        hotwords=("hello", "world"),
    )
    assert build_hotword_pool([ResolvedAudioRecord(record)]) == ["hello", "world"]
