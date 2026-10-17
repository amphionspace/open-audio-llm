"""Far-field subset of the SLR reverberation catalogs.

Ordinary reverberation draws from the whole RIR source. The far branch draws
from the same files the far-field speech-noise set used: REVERB recordings
whose names contain ``far`` and are not small rooms, and the simulated small
rooms nearest a 7 m × 5 m × 3 m room with Sabine T60 around 0.5 s. Within each
of those rooms, the kept responses are the ones whose direct-to-reverberant
ratio is closest to 0 dB.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SR = 16000


def media_path(item) -> str:
    recording = getattr(item, "recording", item)
    return str(recording.sources[0].source)


def is_real_far(path: str) -> bool:
    name = Path(path).name.lower()
    return "far" in name and "smallroom" not in name


def sabine_t60(length: float, width: float, height: float, alpha: float) -> float:
    volume = length * width * height
    surface = 2.0 * (length * width + length * height + width * height)
    return 0.161 * volume / (alpha * surface)


def drr_db(rir: np.ndarray, sample_rate: int = SR) -> float:
    peak = int(np.argmax(np.abs(rir)))
    direct_end = min(len(rir), peak + int(0.0025 * sample_rate))
    direct = rir[max(0, peak - 8) : direct_end]
    late = rir[direct_end:]
    direct_energy = float(np.sum(direct**2)) + 1e-12
    late_energy = float(np.sum(late**2)) + 1e-12
    return 10.0 * np.log10(direct_energy / late_energy)


def read_channel0(path: str | Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    audio, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
    return np.ascontiguousarray(audio[:, 0]), int(sample_rate)


def sim_room_candidates(root: Path) -> list[dict]:
    ranked = []
    for line in (root / "room_info").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        name, length, width, height, _rx, _ry, _rz, alpha = line.split()
        length, width, height, alpha = map(float, (length, width, height, alpha))
        floor = sorted((length, width), reverse=True)
        size_err = abs(floor[0] - 7.0) + abs(floor[1] - 5.0) + abs(height - 3.0)
        t60 = sabine_t60(length, width, height, alpha)
        room_id = name.split("-", 1)[1]
        room_dir = root / room_id
        if room_dir.is_dir():
            ranked.append(
                {
                    "room": name,
                    "room_dir": room_dir,
                    "t60_sabine": t60,
                    "score": size_err + 4.0 * abs(t60 - 0.5),
                }
            )
    ranked.sort(key=lambda item: item["score"])
    return ranked[:8]


def smallroom_root(recordings) -> Path:
    for item in recordings:
        path = Path(media_path(item))
        for parent in path.parents:
            if parent.name == "smallroom" and (parent / "room_info").is_file():
                return parent
    raise ValueError("simulated RIR source has no smallroom/room_info")


def select_real_far(recordings) -> list:
    return [item for item in recordings if is_real_far(media_path(item))]


def select_sim_far(recordings) -> list[str]:
    root = smallroom_root(recordings)
    rooms = [
        room
        for room in sim_room_candidates(root)
        if 0.4 <= room["t60_sabine"] <= 0.6
    ][:5]
    if not rooms:
        raise ValueError(f"no small room near T60 0.5 under {root}")
    chosen = []
    for room in rooms:
        scored = []
        for wav in sorted(room["room_dir"].glob("*.wav")):
            rir, sample_rate = read_channel0(wav)
            scored.append((abs(drr_db(rir, sample_rate)), wav))
        scored.sort(key=lambda item: item[0])
        shortlist = scored[: max(1, len(scored) // 5)]
        chosen.extend(str(wav) for _gap, wav in shortlist)
    return chosen


def select_drr_far(recordings, keep: float = 0.2) -> list:
    """The ``keep`` fraction of responses whose direct-to-reverberant ratio is closest to 0 dB."""
    scored = []
    for item in recordings:
        rir, sample_rate = read_channel0(media_path(item))
        scored.append((abs(drr_db(rir, sample_rate)), item))
    scored.sort(key=lambda pair: pair[0])
    return [item for _gap, item in scored[: max(1, int(len(scored) * keep))]]


def far_recordings(dataset_id: str, recordings) -> list:
    if "dns5" in dataset_id:
        # DNS5 responses are not grouped into rooms, so the DRR rule runs over the whole set.
        chosen = select_drr_far(recordings)
    elif "real" in dataset_id:
        chosen = select_real_far(recordings)
    elif "sim" in dataset_id:
        chosen = select_sim_far(recordings)
    else:
        raise ValueError(f"{dataset_id} is not a real or simulated RIR set")
    if not chosen:
        raise ValueError(f"{dataset_id} contains no far-field RIRs")
    return chosen
