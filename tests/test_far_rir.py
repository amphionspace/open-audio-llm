import wave
from pathlib import Path

import numpy as np

from open_audio_llm.data.far_rir import is_real_far, select_sim_far


def _write(path: Path, audio: np.ndarray, sample_rate: int = 16000) -> None:
    pcm = np.clip(audio, -1, 1)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes((pcm * 32767).astype(np.int16).tobytes())


def test_real_far_names_skip_small_rooms():
    assert is_real_far("RVB2014_type1_rir_largeroom1_far_angla.wav")
    assert not is_real_far("RVB2014_type1_rir_smallroom1_far_angla.wav")
    assert not is_real_far("RVB2014_type1_rir_largeroom1_near_angla.wav")


def test_sim_far_keeps_responses_closest_to_zero_drr(tmp_path):
    root = tmp_path / "smallroom"
    room = root / "Room001"
    room.mkdir(parents=True)
    # 7 x 5 x 3 m and absorption 0.238 gives Sabine T60 about 0.5 s.
    (root / "room_info").write_text(
        "small-Room001 7 5 3 1 1 1 0.238\n", encoding="utf-8"
    )
    impulse = np.zeros(1600, dtype=np.float32)
    impulse[0] = 1
    _write(room / "Room001-00001.wav", impulse)
    balanced = np.zeros(1600, dtype=np.float32)
    balanced[100] = 1
    balanced[200:] = 0.05
    _write(room / "Room001-00002.wav", balanced)
    chosen = select_sim_far([type("Rec", (), {"sources": [type("S", (), {"source": str(root / "unused.wav")})()]})()])
    assert chosen == [str(room / "Room001-00002.wav")]
