"""Meetings synthesized on the fly from real meeting timelines and real conversational speech.

A source with ``online_meeting`` holds window records of real meeting timelines (``metadata.meeting_template``:
each turn's speaker, start, end and character count, in seconds from the window start). Each time a record is
drawn, every template speaker is voiced by a speaker of another real conversation (the ``material`` split): that
speaker's consecutive utterances, from the same part of their conversation, fill the template speaker's turns. Who
talks over whom, and how often a turn sits inside another speaker's long turn, stay a real meeting's; the voices and
words are new on every pass. The speakers are then placed at their own positions in one real room, given their own
levels, and real meeting-room noise is added.

Labels are the utterances' annotated boundaries at their new places, in the aligned SOT timestamp format.
"""

from __future__ import annotations

import gzip
import json
import math
import re
from bisect import bisect_left
from dataclasses import dataclass, field

import numpy as np

# Template turns with at most this many characters are back-channels ("嗯", "对"); they are voiced by
# back-channels, and longer turns by longer utterances.
BACKCHANNEL_CHARS = 2
# Characters of a turn: everything but punctuation and spaces (the pool builder's count).
CHARS_PATTERN = re.compile(r"[\W_]+")
_SILENT_RMS = 1e-6
_MAX_PEAK = 0.99
# Noise clips lose their digitally silent 100 ms frames (power below -120 dBFS, under any 16-bit quantization floor)
# before a window is cut: meeting_room_noise@v2-20261014 train has AISHELL-4 clips that are up to 88% zeros, in runs
# of up to 266 s, and a window cut inside one had no noise.
_NOISE_FRAME_SECONDS = 0.1
_DIGITAL_SILENCE = 1e-12


@dataclass(frozen=True)
class OnlineMeetingConfig:
    # Material split ({dataset_id, version, split}) and the acoustic resources, as Catalog sources.
    material: dict
    rir_sources: tuple[dict, ...] = ()
    noise_sources: tuple[dict, ...] = ()
    # Relative weight of each material corpus; corpora not listed are not used.
    corpus_weights: dict = field(default_factory=lambda: {"alimeeting": 1.0, "aishell5": 1.0})
    # How many of a speaker's next utterances may be skipped to find one that fits the turn; beyond
    # ``max_fit_ratio`` (longer or shorter) the closest-fitting utterance of the speaker is taken instead.
    lookahead: int = 8
    max_fit_ratio: float = 1.5
    # A template turn longer than the chosen utterance is filled with the speaker's next utterances, this far apart,
    # until less than ``min_fill_seconds`` of it is left.
    fill_gap_seconds: tuple[float, float] = (0.1, 0.4)
    min_fill_seconds: float = 1.0
    # Silence kept before the same speaker's next turn (labels of one speaker never overlap).
    min_gap_seconds: float = 0.05
    # Material filter: utterances other speakers overlapped by more than this share in their source meeting carry
    # the others' leakage through the headset. Back-channels are always allowed.
    max_overlap: float = 1.0
    teacher_material: bool = True
    # Acoustics. dry_prob: no room at all (close-talk); otherwise one real room per meeting, a distinct measured
    # position per speaker (wrapping when the room has fewer), responses cut to rir_seconds after the direct path.
    dry_prob: float = 0.0
    rir_seconds: float = 1.0
    min_room_positions: int = 4
    # Per-speaker level (dB, normal, clipped to +-2 sigma) relative to the meeting, measured on each one's speech.
    level_db_std: float = 3.0
    snr_db: tuple[float, float] = (9.0, 28.0)
    speech_dbfs: tuple[float, float] = (-30.0, -22.0)

    def __post_init__(self):
        if not isinstance(self.material, dict) or not {"dataset_id", "version", "split"} <= set(self.material):
            raise ValueError("online_meeting.material must name dataset_id, version and split")
        if not self.corpus_weights or any(not math.isfinite(w) or w < 0 for w in self.corpus_weights.values()) \
                or not sum(self.corpus_weights.values()) > 0:
            raise ValueError("corpus_weights must be non-negative with a positive sum")
        if self.lookahead < 1 or self.max_fit_ratio < 1 or self.min_fill_seconds < 0 or self.min_gap_seconds <= 0:
            raise ValueError("lookahead >= 1, max_fit_ratio >= 1, min_fill_seconds >= 0 and min_gap_seconds > 0")
        for name in ("fill_gap_seconds", "snr_db", "speech_dbfs"):
            pair = getattr(self, name)
            if len(pair) != 2 or pair[0] > pair[1]:
                raise ValueError(f"{name} must be an ordered pair")
        if self.speech_dbfs[1] > 0 or self.fill_gap_seconds[0] < 0:
            raise ValueError("speech_dbfs must be <= 0 dBFS and fill gaps non-negative")
        if not 0 <= self.max_overlap <= 1 or not 0 <= self.dry_prob <= 1:
            raise ValueError("max_overlap and dry_prob must be between 0 and 1")
        if self.rir_seconds <= 0 or self.level_db_std < 0 or self.min_room_positions < 1:
            raise ValueError("rir_seconds > 0, level_db_std >= 0 and min_room_positions >= 1")
        if self.dry_prob < 1 and not self.rir_sources:
            raise ValueError("online_meeting needs rir_sources unless dry_prob is 1")
        if not self.noise_sources:
            raise ValueError("online_meeting needs noise_sources (real meeting-room noise)")


class MaterialSpeaker:
    """One speaker's utterances of one conversation, in order, as spans of one track."""

    __slots__ = ("key", "track", "offsets", "durations", "texts", "starts", "backchannel", "eligible", "_sorted")

    def __init__(self, key, track, rows, config):
        self.key, self.track = key, track
        self.offsets = np.array([r["offset"] for r in rows], dtype=np.float64)
        self.durations = np.array([r["duration"] for r in rows], dtype=np.float64)
        self.texts = [r["text"] for r in rows]
        self.starts = np.array([r["source_start"] for r in rows], dtype=np.float64)
        self.backchannel = np.array([r["chars"] <= BACKCHANNEL_CHARS for r in rows])
        self.eligible = np.array([
            (r["chars"] <= BACKCHANNEL_CHARS or r["overlap"] <= config.max_overlap)
            and (config.teacher_material or r.get("origin", "human") == "human") for r in rows])
        # Per kind, eligible utterances by duration, for the closest-fit fallback.
        self._sorted = {}
        for kind in (True, False):
            index = np.flatnonzero(self.eligible & (self.backchannel == kind))
            order = index[np.argsort(self.durations[index], kind="stable")]
            self._sorted[kind] = (self.durations[order], order)

    def __len__(self):
        return len(self.durations)

    @property
    def seconds(self):
        return float(self.durations[self.eligible].sum())

    def closest(self, want, limit, kind, used):
        """The unused eligible utterance of this kind closest to ``want`` seconds and at most ``limit``."""
        durations, order = self._sorted[kind]
        top = bisect_left(durations, limit + 1e-9)
        if not top:
            return None
        middle = min(bisect_left(durations, want), top - 1)
        best, best_gap = None, math.inf
        # Walk outwards from the closest duration until an unused one is found on each side.
        for step in (-1, 1):
            j = middle if step == -1 else middle + 1
            while 0 <= j < top:
                index = int(order[j])
                if index not in used:
                    gap = abs(math.log(durations[j] / want))
                    if gap < best_gap:
                        best, best_gap = index, gap
                    break
                j += step
        return best


def load_material(rows, config, track_path):
    """{corpus: [conversation: [MaterialSpeaker]]} from material rows; ``track_path(row)`` resolves a track."""
    grouped = {}
    for row in rows:
        grouped.setdefault(row["corpus"], {}).setdefault(row["meeting"], {}).setdefault(row["speaker"], []).append(row)
    material = {}
    for corpus, meetings in grouped.items():
        if not config.corpus_weights.get(corpus):
            continue
        conversations = []
        for meeting, speakers in sorted(meetings.items()):
            voices = [MaterialSpeaker((corpus, meeting, speaker), track_path(rows[0]),
                                      sorted(rows, key=lambda r: r["index"]), config)
                      for speaker, rows in sorted(speakers.items())]
            voices = [v for v in voices if v.eligible.any()]
            if voices:
                conversations.append(voices)
        if conversations:
            material[corpus] = conversations
    if not material:
        raise ValueError("online_meeting material holds no utterance of a weighted corpus")
    return material


def read_material_rows(paths):
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)


def choose_voices(rng, material, count, weights):
    """``count`` material speakers of one corpus: one conversation's, more conversations' when it has too few."""
    corpora = sorted(c for c in material if weights.get(c))
    corpus = rng.choices(corpora, weights=[weights[c] for c in corpora])[0]
    conversations = material[corpus]
    order = list(range(len(conversations)))
    rng.shuffle(order)
    voices = []
    for number in order:
        # Which speakers of a larger conversation stay silent is random too.
        voices += rng.sample(conversations[number], len(conversations[number]))
        if len(voices) >= count:
            break
    if len(voices) < count:
        raise ValueError(f"material corpus {corpus} has {len(voices)} speakers; the window needs {count}")
    return corpus, voices


def _start_cursor(voice, moment):
    """The first utterance at or after ``moment`` s into the source conversation."""
    return min(int(np.searchsorted(voice.starts, moment)), len(voice) - 1)


def fill_turns(rng, turns, length, voice, cursor, config):
    """Place ``voice``'s utterances on one template speaker's turns ([(start, end, chars)], sorted).

    Returns [(start, utterance index)]. Every utterance ends before the speaker's next turn (minus
    min_gap_seconds) and inside the window.
    """
    placed, used = [], set()
    for number, (start, end, characters) in enumerate(turns):
        boundary = (turns[number + 1][0] if number + 1 < len(turns) else length) - config.min_gap_seconds
        kind = characters <= BACKCHANNEL_CHARS
        position = start
        while True:
            want, limit = max(end - position, 0.05), boundary - position
            if limit <= 0.05:
                break
            choice, best = None, math.inf
            for step in range(config.lookahead):
                index = (cursor + step) % len(voice)
                duration = voice.durations[index]
                if (index in used or not voice.eligible[index] or voice.backchannel[index] != kind
                        or duration > limit):
                    continue
                gap = abs(math.log(duration / want))
                if gap < best:
                    choice, best = index, gap
            if choice is None or best > math.log(config.max_fit_ratio):
                fallback = voice.closest(want, limit, kind, used)
                if fallback is not None and (choice is None
                                             or abs(math.log(voice.durations[fallback] / want)) < best):
                    choice = fallback
            if choice is None:
                break
            placed.append((position, choice))
            used.add(choice)
            cursor = (choice + 1) % len(voice)
            position += voice.durations[choice] + rng.uniform(*config.fill_gap_seconds)
            # A back-channel turn holds one back-channel; a longer turn is filled until little of it is left.
            if kind or end - position < config.min_fill_seconds:
                break
            kind = False
    return placed


def plan_meeting(rng, template, length, material, config):
    """Voice a template window; returns (corpus, [(start, duration, label speaker, voice, utterance index)])."""
    by_speaker = {}
    for speaker, start, end, characters in template:
        by_speaker.setdefault(str(speaker), []).append((float(start), float(end), int(characters)))
    for turns in by_speaker.values():
        turns.sort()
    corpus, voices = choose_voices(rng, material, len(by_speaker), config.corpus_weights)
    # The speaker who talks most gets the voice with the most speech, so nobody runs out of words.
    needs = sorted(by_speaker, key=lambda s: -sum(e - b for b, e, _ in by_speaker[s]))
    voices = sorted(voices[: len(needs)], key=lambda v: -v.seconds)
    # All voices start at one moment of the source conversation, so they talk about the same things.
    moment = rng.uniform(0.0, max(float(v.starts[-1]) for v in voices))
    placed = []
    for speaker, voice in zip(needs, voices):
        cursor = _start_cursor(voice, moment)
        for start, index in fill_turns(rng, by_speaker[speaker], length, voice, cursor, config):
            placed.append((start, float(voice.durations[index]), speaker, voice, index))
    placed.sort(key=lambda item: (item[0], item[2]))
    return corpus, placed


def meeting_target(placed):
    """SOT timestamp lines ([S1][0.00-2.50] text), speakers numbered by first speech; also the speaker map."""
    mapping, lines, ends = {}, [], {}
    for start, duration, speaker, voice, index in placed:
        end = start + duration
        if round(start, 2) >= round(end, 2):
            continue
        if start < ends.get(speaker, -math.inf):
            raise ValueError("a planned speaker overlaps itself")
        ends[speaker] = end
        label = mapping.setdefault(speaker, f"S{len(mapping) + 1}")
        lines.append(f"[{label}][{start:.2f}-{end:.2f}] {voice.texts[index]}")
    return "\n".join(lines), mapping


def room_of(rir_id):
    """Real-room groups of SLR28 (amphiondata.multispeaker.farfield.room_of); other sets are not rooms."""
    parts = rir_id.split("_")
    if parts[:2] == ["RVB2014", "type1"]:
        return "RVB2014_" + parts[3]
    if len(parts) > 4 and parts[0] == "RWCP" and parts[1] in {"type1", "type2"} and parts[4] != "ane":
        return "RWCP_" + parts[4]
    return None


def group_rooms(items, identify, minimum):
    rooms = {}
    for item in items:
        room = room_of(identify(item))
        if room:
            rooms.setdefault(room, []).append(item)
    rooms = {name: sorted(group, key=identify) for name, group in rooms.items() if len(group) >= minimum}
    if not rooms:
        raise ValueError(f"no real room with {minimum} measured positions among the rir_sources")
    return [rooms[name] for name in sorted(rooms)]


def _tile(rng, clip, length):
    clip = np.asarray(clip, dtype=np.float32)
    if len(clip) < length:
        clip = np.tile(clip, -(-length // len(clip)))
    start = rng.randrange(len(clip) - length + 1)
    return clip[start : start + length]


def _drop_digital_silence(clip, sampling_rate):
    clip = np.asarray(clip, dtype=np.float32)
    frame = round(_NOISE_FRAME_SECONDS * sampling_rate)
    frames = clip[: len(clip) // frame * frame].reshape(-1, frame)
    live = (frames.astype(np.float64) ** 2).mean(axis=1) > _DIGITAL_SILENCE
    if live.all() or not live.any():
        return clip
    return frames[live].reshape(-1)


def render_meeting(rng, tracks, spans, length, config, *, rooms, load_rir, noise_loader, sampling_rate, info=None):
    """Reverberate each dry speaker track at its own position in one room, set levels, add room noise.

    ``tracks`` are dry speaker signals on the window timeline and ``spans`` each one's speech as sample
    spans; ``rooms`` are groups of RIR items, ``load_rir(item)`` one response, ``noise_loader(rng)`` a noise clip.
    """
    # fftconvolve pads to a fast FFT length; oaconvolve's block sizes follow the response length, and a response
    # cut at its direct path (an arbitrary length) made one 120 s convolution take up to 10 s instead of 0.1 s.
    from scipy.signal import fftconvolve

    dry = config.dry_prob and rng.random() < config.dry_prob
    room = None if dry else rooms[rng.randrange(len(rooms))]
    positions = list(range(len(room))) if room else []
    rng.shuffle(positions)
    wet, active = [], np.zeros(length, dtype=bool)
    for number, (track, own) in enumerate(zip(tracks, spans)):
        signal = np.asarray(track, dtype=np.float32)
        if room:
            impulse = np.asarray(load_rir(room[positions[number % len(positions)]]), dtype=np.float32)
            impulse = impulse[int(np.argmax(np.abs(impulse))) :][: round(config.rir_seconds * sampling_rate)]
            norm = float(np.linalg.norm(impulse))
            if norm > 0:
                # The direct path stays at t = 0, so the labels keep their times.
                signal = fftconvolve(signal, impulse / norm)[:length].astype(np.float32)
        mask = np.zeros(length, dtype=bool)
        for begin, end in own:
            mask[begin:end] = True
        active |= mask
        level = float(np.sqrt(np.mean(signal[mask] ** 2))) if mask.any() else 0.0
        if level > _SILENT_RMS:
            sigma = config.level_db_std
            gain_db = min(max(rng.gauss(0.0, sigma), -2 * sigma), 2 * sigma) if sigma else 0.0
            signal = signal * (10 ** (gain_db / 20) / level)
        wet.append(signal)
    speech = np.sum(wet, axis=0) if wet else np.zeros(length, dtype=np.float32)
    power = float(np.mean(speech[active] ** 2)) if active.any() else 0.0
    snr = rng.uniform(*config.snr_db)
    noise = _tile(rng, _drop_digital_silence(noise_loader(rng), sampling_rate), length)
    noise_power = float(np.mean(noise**2))
    if power > 0:
        # A silent noise window leaves the speech without noise; it never replaces the speech.
        scale = math.sqrt(power / (noise_power * 10 ** (snr / 10))) if noise_power > 0 else 0.0
        mixture = speech + noise * scale
        gain = 10 ** (rng.uniform(*config.speech_dbfs) / 20) / math.sqrt(power)
    else:
        mixture = noise
        gain = 10 ** (config.speech_dbfs[0] / 20) / max(math.sqrt(noise_power), _SILENT_RMS)
    peak = float(np.max(np.abs(mixture))) * gain
    if peak > _MAX_PEAK:
        gain *= _MAX_PEAK / peak
    if info is not None:
        info.update(room=None if room is None else len(room), snr_db=snr)
    return (mixture * gain).astype(np.float32)
