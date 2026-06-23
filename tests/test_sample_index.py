from open_audio_llm.data.sample_index import (
    SampleIndexRecord,
    read_jsonl,
    to_sharegpt_record,
    write_jsonl,
)


def test_sample_index_roundtrip(tmp_path):
    path = tmp_path / "sample_index.jsonl"
    record = SampleIndexRecord(
        id="utt1",
        dataset_id="toy",
        task="ts_asr",
        audio_path="/tmp/mixed.wav",
        enrollment_audio="/tmp/enroll.wav",
        text="hello",
        real_hotwords=["hello"],
    )
    write_jsonl([record], path)
    loaded = list(read_jsonl(path))
    assert loaded[0].task == "ts_asr"
    sharegpt = to_sharegpt_record(loaded[0], "prompt<audio><audio>", "answer")
    assert sharegpt["audios"] == ["/tmp/enroll.wav", "/tmp/mixed.wav"]
