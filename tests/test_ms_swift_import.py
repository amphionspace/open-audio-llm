from open_audio_llm.integrations.ms_swift.template import normalize_audio_sample


def test_ms_swift_template_normalizes_audio_column():
    sample = normalize_audio_sample({"audio": "a.wav", "task": "asr"})
    assert sample.audios == ["a.wav"]
    assert sample.task == "asr"
