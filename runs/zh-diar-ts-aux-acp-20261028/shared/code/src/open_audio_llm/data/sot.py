"""Render aligned source utterances on the existing SOT mixture timeline."""

import math
import re
from dataclasses import replace

TIMESTAMP_FORMAT = "aligned_utterance_timestamps_v1"
TIMESTAMP_SYSTEM = (
    "Transcribe every utterance from every speaker, including overlapping speech. "
    "Output one line per utterance as [S1][0.00-2.50] text, with start and end "
    "times in seconds from the beginning of the audio. Sort lines by start time. "
    "Keep the same speaker ID across turns; assign IDs by first speech. "
    "Use the beginning of the first spoken word and the end of the last spoken "
    "word as the utterance boundaries, excluding leading and trailing silence."
)


def with_sot_timestamps(record, alignments):
    """Join by exact source identity; fail instead of dropping or relabeling audio."""
    if record.task != "speaker_attributed_asr" or len(record.audio_slots) != 1:
        raise ValueError("sot_timestamps requires a single-mixture SOT record")
    duration = record.audio_slots[0].ref.duration
    segments = record.metadata.get("segments")
    if not segments or duration is None:
        raise ValueError("SOT timestamps require segments and mixture duration")
    turns, speakers = [], {}
    for segment in sorted(segments, key=lambda row: (row["start"], row["speaker"])):
        source = segment['source']
        key = tuple(source[field] for field in ('dataset_id', 'version', 'source_id'))
        annotation = alignments[key]
        if annotation['status'] != 'aligned' or annotation['bounds'] is None:
            raise ValueError(f'Unapproved source alignment ({annotation["status"]}): {key}')
        for field in ('audio', 'channel', 'sample_rate', 'start', 'duration',
                      'split', 'source_split', 'recording_id'):
            if annotation[field] != source[field]:
                raise ValueError(f'Alignment source {field} mismatch: {key}')
        label, text = segment['speaker'], segment['text'].strip()
        if annotation['text'].strip() != text:
            raise ValueError(f'Alignment transcript mismatch: {key}')
        # Both producers decode at 16 kHz; allow only floating-point/sample rounding.
        if abs(annotation['audio_duration'] - segment['duration']) > 1 / 16000 + 1e-6:
            raise ValueError(f'Alignment decoded duration mismatch: {key}')
        begin, finish = annotation['bounds']
        start, end = segment['start'] + begin, segment['start'] + finish
        if (not math.isfinite(start) or not math.isfinite(end)
                or not 0 <= begin < finish <= segment['duration'] + 1e-6
                or not 0 <= start < end <= duration + 1e-6
                or not re.fullmatch(r'S[1-9]\d*', label) or not text
                or '\n' in text or '\r' in text
                or round(start, 2) >= round(end, 2)):
            raise ValueError(f'Invalid aligned SOT interval: {record.id}')
        speakers.setdefault(label, []).append(text)
        turns.append((start, end, label, text))
    original = '\n'.join(f'[{label}] {" ".join(speakers[label])}'
                         for label in sorted(speakers, key=lambda value: int(value[1:])))
    if original != record.target:
        raise ValueError(f'SOT segments do not reconstruct the complete transcript: {record.id}')
    # Removing edge silence can change the order of first speech. Keep original
    # provenance labels and explicitly record the mapping used by this target.
    mapping, lines, ends = {}, [], {}
    for start, end, label, text in sorted(turns, key=lambda row: (row[0], int(row[2][1:]))):
        # Adding a source boundary to its offset can round a shared endpoint
        # one representable float above the next turn's start.
        if math.nextafter(start, math.inf) < ends.get(label, 0):
            raise ValueError(f'Overlapping turns from the same speaker: {record.id}')
        ends[label] = end
        mapped = mapping.setdefault(label, f'S{len(mapping) + 1}')
        lines.append(f'[{mapped}][{start:.2f}-{end:.2f}] {text}')
    return replace(record, target='\n'.join(lines), metadata={
        **record.metadata, 'sot_output_format': TIMESTAMP_FORMAT,
        'timestamp_scope': 'first_to_last_aligned_word', 'timestamp_speaker_map': mapping,
    })
