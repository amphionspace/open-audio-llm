"""Online enrollment views of complete, timestamped SOT records.

Candidate identities and source spans are facts supplied by data preparation;
random cropping never infers a speaker identity from a filename.
"""

import math
import random
import re
from dataclasses import dataclass, replace

from audio_data_contract import AudioRef, AudioSlot

from .sot import TIMESTAMP_FORMAT

TARGET_FORMAT = "target_speaker_timestamps_v1"
LINE = re.compile(r"\[([ST][1-9]\d*)\]\[(\d+\.\d{2})-(\d+\.\d{2})\] (.+)")


@dataclass(frozen=True)
class EnrollmentConfig:
    # ECAPA2 motivates 1--5 s coverage; uniform sampling is our experiment choice.
    min_seconds: float = 1.0
    max_seconds: float = 5.0
    max_targets: int = 3
    absent_probability: float = 0.2
    mode: str = "random"
    probability: float = 0.5

    def __post_init__(self):
        if not (math.isfinite(self.min_seconds) and math.isfinite(self.max_seconds)
                and 1 <= self.min_seconds <= self.max_seconds <= 5):
            raise ValueError("Invalid enrollment duration range")
        if type(self.max_targets) is not int or not 1 <= self.max_targets <= 3:
            raise ValueError("max_targets must be 1, 2 or 3")
        if self.mode not in {"random", "all", "targets_only"}:
            raise ValueError("Unknown enrollment mode")
        if not 0 <= self.absent_probability <= 1 or not 0 <= self.probability <= 1:
            raise ValueError("Enrollment probabilities must be in [0, 1]")


def target_prompt(count, mode):
    if not 1 <= count <= 3 or mode not in {"all", "targets_only"}:
        raise ValueError("Expected 1--3 enrollments and all/targets_only mode")
    prefix = (f"前 {count} 段音频为参考说话人，依次编号 T1 至 T{count}，最后一段为待转写音频。\n")
    scope = ("仅转写参考说话人的发言，" if mode == "targets_only" else
             "转写所有人的发言，其他说话人按首次发声编号 S1、S2……。\n")
    return prefix + scope + "每次发言一行：[编号][开始秒-结束秒] 文本。"


def parse_target_segments(text, duration, *, count=3, mode="all"):
    """Strict public parser; empty output is valid, malformed output is not."""
    segments, last_start, ends, anonymous = [], -1.0, {}, []
    for line in text.splitlines():
        if not line.strip():
            continue
        match = LINE.fullmatch(line.strip())
        if not match:
            raise ValueError(f"Invalid transcript line: {line!r}")
        label, start, end, body = match.groups()
        start, end = float(start), float(end)
        if (not 0 <= start < end <= duration + .005001 or not body.strip()
                or start < last_start or start < ends.get(label, 0)):
            raise ValueError("Invalid or unordered utterance boundaries")
        if label.startswith("T") and int(label[1:]) > count:
            raise ValueError("Unregistered target ID")
        if label.startswith("S"):
            if mode == "targets_only":
                raise ValueError("Non-target output in targets_only mode")
            if label not in anonymous:
                anonymous.append(label)
        last_start, ends[label] = start, end
        segments.append({"speaker_id": label, "start": start, "end": end, "text": body})
    if anonymous != [f"S{i + 1}" for i in range(len(anonymous))]:
        raise ValueError("Anonymous IDs must follow first speech")
    return segments


def render_target(record, identities, mode):
    turns = parse_target_segments(record.target, record.audio_slots[-1].ref.duration)
    labels = record.metadata["speaker_identities"]
    targets = {identity: f"T{i + 1}" for i, identity in enumerate(identities)}
    anonymous, lines = {}, []
    for turn in turns:
        identity = labels[turn["speaker_id"]]
        label = targets.get(identity)
        if label is None:
            if mode == "targets_only":
                continue
            label = anonymous.setdefault(identity, f"S{len(anonymous) + 1}")
        lines.append(f'[{label}][{turn["start"]:.2f}-{turn["end"]:.2f}] {turn["text"]}')
    return "\n".join(lines)


def _overlap(a, b):
    # source_spans refer to original cuts, also for pre-mixed synthetic audio.
    key = lambda r: (r.dataset_id, r.version, r.split, r.cut_id)
    if key(a) != key(b):
        return False
    if a.duration is None or b.duration is None:
        raise ValueError("Enrollment provenance needs explicit source durations")
    return max(a.start or 0, b.start or 0) < min(
        (a.start or 0) + a.duration, (b.start or 0) + b.duration)


def eligible_candidates(record, config):
    if record.task != "speaker_attributed_asr" or record.metadata.get("sot_output_format") != TIMESTAMP_FORMAT:
        raise ValueError("Enrollment needs complete timestamped SOT")
    turns = parse_target_segments(record.target, record.audio_slots[-1].ref.duration)
    identities = record.metadata.get("speaker_identities", {})
    if set(identities) != {t["speaker_id"] for t in turns} or len(set(identities.values())) != len(identities):
        raise ValueError("speaker_identities must cover the complete mixture exactly")
    if not record.metadata.get("source_spans"):
        raise ValueError("Enrollment needs original mixture source_spans")
    spans = [AudioRef.from_dict(r) for r in record.metadata["source_spans"]]
    spans.extend(slot.ref for slot in record.audio_slots)
    groups = {}
    for item in record.metadata.get("enrollment_candidates", []):
        if item.get("single_speaker") is not True:
            raise ValueError("Enrollment candidate must be confirmed single-speaker")
        if item.get("partition") != record.metadata.get("partition") or not item.get("partition"):
            raise ValueError("Enrollment candidate crosses the train/dev/test partition")
        identity = item["speaker_id"]
        if not isinstance(identity, str) or not identity:
            raise ValueError("Enrollment candidate needs a stable speaker identity")
        ref = AudioRef.from_dict(item["ref"])
        if ref.duration is None:
            raise ValueError("Enrollment candidate needs an explicit duration")
        if ref.duration < config.min_seconds or any(_overlap(ref, span) for span in spans):
            continue
        if record.metadata.get("clean", {}).get("pass") is True and item.get("clean_pass") is not True:
            continue
        groups.setdefault(identity, []).append(ref)
    return groups


def sample_enrollment(record, config, seed, *, fixed=None):
    """Materialize one view. ``fixed`` is the frozen evaluation enrollment manifest."""
    groups = eligible_candidates(record, config)
    rng = random.Random(f"{seed}:targets")
    present = set(record.metadata["speaker_identities"].values())
    if fixed is None:
        if not groups or rng.random() >= config.probability:
            return record
        k = rng.randint(1, min(config.max_targets, len(groups)))
        available, selected = sorted(groups), []
        for _ in range(k):
            absent = rng.random() < config.absent_probability
            pool = [s for s in available if (s not in present) == absent]
            # Only use feasible identities; observed absence rates are reported.
            identity = rng.choice(pool or available)
            selected.append(identity)
            available.remove(identity)
        rng.shuffle(selected)
        mode = rng.choice(["all", "targets_only"]) if config.mode == "random" else config.mode
    else:
        selected = [row["speaker_id"] for row in fixed["enrollments"]]
        mode = fixed["mode"]
        if not 1 <= len(selected) <= config.max_targets or len(set(selected)) != len(selected):
            raise ValueError("Invalid fixed enrollment identities")
    target_prompt(len(selected), mode)
    slots, audit = [], []
    for i, identity in enumerate(selected):
        crop_rng = random.Random(f"{seed}:enrollment:{i}:{identity}")
        if fixed is None:
            source = crop_rng.choice(groups[identity])
            length = crop_rng.uniform(config.min_seconds, min(config.max_seconds, source.duration))
            offset = crop_rng.uniform(0, source.duration - length)
            ref = replace(source, start=(source.start or 0) + offset, duration=length, purpose="enrollment")
        else:
            ref = AudioRef.from_dict(fixed["enrollments"][i]["ref"])
            if ref.duration is None or not config.min_seconds <= ref.duration <= config.max_seconds:
                raise ValueError("Fixed enrollment duration outside configured interval")
            if not any((ref.dataset_id, ref.version, ref.split, ref.cut_id, ref.channel) ==
                       (s.dataset_id, s.version, s.split, s.cut_id, s.channel)
                       and (s.start or 0) <= (ref.start or 0)
                       and (ref.start or 0) + ref.duration <= (s.start or 0) + s.duration + 1e-8
                       for s in groups.get(identity, [])):
                raise ValueError("Fixed enrollment is not inside an eligible reference")
        slots.append(AudioSlot(f"enrollment_{i + 1}", ref, purpose="enrollment"))
        audit.append({"speaker_id": identity, "ref": ref.to_dict(), "present": identity in present})
    return replace(record, audio_slots=(*slots, record.slot("mixture")),
                   target=render_target(record, selected, mode), metadata={
                       **record.metadata, "sot_output_format": TARGET_FORMAT,
                       "enrollment_view": {"mode": mode, "enrollments": audit},
                   })
