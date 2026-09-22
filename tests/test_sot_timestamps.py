import copy
import gzip
import json
from dataclasses import replace

import numpy as np
import pytest
import soundfile as sf
from audio_data_contract import ArtifactRef, AudioRecord, AudioRef, AudioSlot, DatasetSpec, write_records

from open_audio_llm.data.catalog_dataset import CatalogSwiftDataset
from open_audio_llm.data.qwen3_asr import native_messages
from open_audio_llm.data.sot import TIMESTAMP_FORMAT, TIMESTAMP_SYSTEM, with_sot_timestamps
from open_audio_llm.data.sot_alignment import AlignmentIndex, build_alignment_index, fingerprint
from open_audio_llm.eval.sot_timestamps import summarize_timed_sot
from open_audio_llm.eval.ts_asr import summarize


def source_rows():
    rows = []
    for i, (text, begin, finish) in enumerate([('hello there', .8, 1.), ('good day', .08, .64), ('goodbye', .16, .72)]):
        rows.append(dict(id=str(i), dataset_id='source', version='clean-v1', source_id=str(i),
                         recording_id=str(i), audio={'root_alias': 'data', 'relative_path': f'{i}.wav'},
                         channel=0, sample_rate=16000, start=10., duration=1., text=text,
                         split='train', source_split='train', alignment={
                             'status': 'aligned', 'time_reference': 'source_segment_start',
                             'audio_duration': 1., 'issues': {},
                             'items': [{'text': text, 'start': begin, 'end': finish}]}))
    return rows


def make_index(tmp_path, rows=None):
    root = tmp_path / 'alignments-output'
    for folder in ('alignments', 'summaries'):
        (root / folder).mkdir(parents=True)
    rows = source_rows() if rows is None else rows
    path = root / 'alignments/000000.jsonl.gz'
    with gzip.open(path, 'wt') as stream:
        for row in rows:
            stream.write(json.dumps(row) + '\n')
    (root / 'summaries/000000.json').write_text(json.dumps({'sha256': fingerprint(path), 'records': len(rows)}))
    (root / 'plan.json').write_text(json.dumps({'shards': [{'records': len(rows)}]}))
    (root / 'progress.json').write_text('{"status":"complete"}')
    (root / 'alignment-config.json').write_text(json.dumps({'plan_sha256': fingerprint(root / 'plan.json')}))
    output = tmp_path / 'source.sqlite'
    report = build_alignment_index([root], output)
    return AlignmentIndex(output), report


def record():
    segments = []
    for source, speaker, start in zip(source_rows(), ['S1', 'S2', 'S1'], [0., .2, 2.]):
        source = {key: value for key, value in source.items() if key not in {'alignment', 'id'}}
        segments.append({'speaker': speaker, 'start': start, 'duration': 1.,
                         'text': source['text'], 'source': source})
    return AudioRecord('mixed', 'speaker_attributed_asr', (
        AudioSlot('mixture', AudioRef('sot', 'v1', 'train', 'mixed', duration=4.)),),
        target='[S1] hello there goodbye\n[S2] good day', language='en',
        metadata={'segments': segments})


def test_alignment_join_uses_clip_relative_time_and_first_speech_order(tmp_path):
    index, report = make_index(tmp_path)
    original = record()
    timed = with_sot_timestamps(original, index)
    assert report['counts'] == {'train/source/aligned': 3}
    assert report['sha256'] == fingerprint(index.path)
    with pytest.raises(ValueError, match='checksum'):
        AlignmentIndex(index.path, 'wrong-checksum')
    assert timed.target == '[S1][0.28-0.84] good day\n[S2][0.80-1.00] hello there\n[S2][2.16-2.72] goodbye'
    assert timed.metadata['timestamp_speaker_map'] == {'S2': 'S1', 'S1': 'S2'}
    assert timed.audio_slots == original.audio_slots
    assert timed.metadata['segments'] == original.metadata['segments']
    messages = native_messages(timed, 'N/A')
    assert messages[0]['content'] == TIMESTAMP_SYSTEM
    assert messages[-1]['content'] == 'language English<asr_text>' + timed.target
    # The metadata can be serialized to a spawn worker without an open SQLite handle.
    assert index.__getstate__()['_connection'] is None


@pytest.mark.parametrize('field,value', [('time_reference', 'recording_start'),
    ('items', [{'text': 'hello', 'start': 0., 'end': 2.}]),
    ('items', [{'text': 'hello', 'start': .5, 'end': .5}])])
def test_invalid_alignment_cannot_be_frozen_as_approved(tmp_path, field, value):
    rows = source_rows()
    rows[0]['alignment'][field] = value
    with pytest.raises(ValueError, match='Invalid aligned'):
        make_index(tmp_path, rows)
    assert not (tmp_path / 'source.sqlite').exists()


@pytest.mark.parametrize('field,value,error', [
    ('start', 0., 'source start mismatch'), ('text', 'wrong transcript', 'transcript mismatch'),
    ('version', 'different-version', 'Missing source alignment'),
    ('split', 'dev', 'source split mismatch'),
])
def test_alignment_cannot_be_joined_to_different_audio_or_text(tmp_path, field, value, error):
    index, _ = make_index(tmp_path)
    original = record()
    metadata = copy.deepcopy(original.metadata)
    metadata['segments'][0]['source'][field] = value
    if field == 'text':
        metadata['segments'][0]['text'] = value
    with pytest.raises((ValueError, KeyError), match=error):
        with_sot_timestamps(replace(original, metadata=metadata), index)


def test_reviewed_and_missing_alignment_never_fall_back_to_clip_bounds(tmp_path):
    rows = source_rows()
    rows[0]['alignment']['status'] = 'needs_review'
    index, _ = make_index(tmp_path, rows)
    with pytest.raises(ValueError, match='Unapproved'):
        with_sot_timestamps(record(), index)
    with pytest.raises(KeyError):
        with_sot_timestamps(record(), {})


@pytest.mark.parametrize('next_start,overlapping', [(11.62, False), (11.61, True)])
def test_adjacent_turn_rounding_does_not_hide_real_overlap(tmp_path, next_start, overlapping):
    rows = source_rows()
    rows[0]['duration'] = rows[0]['alignment']['audio_duration'] = 5.04
    rows[0]['alignment']['items'][0].update(start=.08, end=5.04)
    rows[2]['alignment']['items'][0]['start'] = 0.
    index, _ = make_index(tmp_path, rows)
    original = record()
    metadata = copy.deepcopy(original.metadata)
    metadata['segments'][0].update(start=6.58, duration=5.04)
    metadata['segments'][0]['source']['duration'] = 5.04
    metadata['segments'][2]['start'] = next_start
    slots = (replace(original.audio_slots[0], ref=replace(original.audio_slots[0].ref, duration=20.)),)
    original = replace(original, metadata=metadata, audio_slots=slots)
    if overlapping:
        with pytest.raises(ValueError, match='Overlapping turns'):
            with_sot_timestamps(original, index)
    else:
        assert '[11.62-12.34] goodbye' in with_sot_timestamps(original, index).target


@pytest.mark.parametrize('issue', ['tampered', 'incomplete', 'changed_plan'])
def test_index_rejects_incomplete_or_changed_alignment_outputs(tmp_path, issue):
    _, _ = make_index(tmp_path)
    root = tmp_path / 'alignments-output'
    if issue == 'tampered':
        with (root / 'alignments/000000.jsonl.gz').open('ab') as stream:
            stream.write(b'bad')
    elif issue == 'incomplete':
        (root / 'progress.json').write_text('{"status":"running"}')
    else:
        (root / 'plan.json').write_text('{"shards": []}')
    with pytest.raises(ValueError):
        build_alignment_index([root], tmp_path / 'changed.sqlite')
    assert not (tmp_path / 'changed.sqlite').exists()


def evaluation_row(reference, prediction):
    return dict(source='sot', language='en', task='speaker_attributed_asr', duration=4.,
                sot_output_format=TIMESTAMP_FORMAT, reference=reference, prediction=prediction)


def test_timestamp_scoring_keeps_transcription_and_timing_separate(tmp_path):
    index, _ = make_index(tmp_path)
    reference = with_sot_timestamps(record(), index).target
    perfect = summarize([evaluation_row(reference, reference)])['sot']
    assert perfect['error_rate'] == 0
    assert perfect['speaker_attribution']['accuracy'] == 1
    assert perfect['timestamps']['boundary_mae_seconds'] == 0
    assert perfect['timestamps']['f1'] == 1 and perfect['format_valid_rate'] == 1
    shifted = reference.replace('0.28-0.84', '0.88-1.44')
    result = summarize([evaluation_row(reference, shifted)])['sot']
    assert result['error_rate'] == 0
    assert result['timestamps']['f1'] == pytest.approx(2 / 3)
    assert result['timestamps']['boundary_mae_seconds'] == pytest.approx(.2)
    assert result['format_valid_rate'] == 0  # Lines are no longer chronological.
    lexical_error = reference.replace('good day', 'bad day')
    result = summarize_timed_sot([evaluation_row(reference, lexical_error)], False)
    assert result['error_rate'] == .2
    assert result['timestamps']['boundary_mae_seconds'] == 0


def test_missing_extra_malformed_and_permuted_timed_predictions():
    reference = '[S1][0.00-1.00] hello\n[S2][0.20-1.20] world'
    swapped = '[S2][0.00-1.00] hello\n[S1][0.20-1.20] world'
    result = summarize_timed_sot([evaluation_row(reference, swapped)], False)
    assert result['error_rate'] == 0 and result['timestamps']['f1'] == 1
    for prediction, precision, recall in [('', 0, 0), ('[S1][0.00-1.00] hello', 1, .5),
            (reference + '\n[S1][2.00-3.00] again', 2 / 3, 1),
            ('[S1][bad] hello\n[S2][0.20-1.20] world', .5, .5)]:
        result = summarize_timed_sot([evaluation_row(reference, prediction)], False)
        assert result['timestamps']['precision'] == pytest.approx(precision)
        assert result['timestamps']['recall'] == recall
    malformed = summarize_timed_sot([evaluation_row(reference, reference.replace('0.00-1.00', 'bad'))], False)
    assert malformed['error_rate'] == 0 and malformed['format_valid_rate'] == 0


@pytest.mark.parametrize('cached', [False, True])
def test_catalog_timed_audio_stays_in_sync_and_cache_tracks_alignment(tmp_path, cached):
    index, _ = make_index(tmp_path)
    write_records([record()], tmp_path / 'records.jsonl.gz')
    sf.write(tmp_path / 'mixed.wav', np.zeros(64000), 16000)
    (tmp_path / 'audio.jsonl').write_text(json.dumps(dict(cut_id='mixed', root_alias='data',
        relative_path='mixed.wav', duration=4., sample_rate=16000, channels=1, num_frames=64000)) + '\n')
    spec = DatasetSpec(dataset_id='sot', version='v1', languages=('en',), tasks=('speaker_attributed_asr',),
        artifacts=(ArtifactRef('records', 'audio-records', 'data', 'records.jsonl.gz'),
                   ArtifactRef('audio', 'audio-index', 'data', 'audio.jsonl')),
        splits={'train': {'records_artifact': 'records', 'audio_index_artifact': 'audio'}})
    (tmp_path / 'catalog.jsonl').write_text(json.dumps(spec.to_dict()) + '\n')
    (tmp_path / 'roots.json').write_text(json.dumps({'data': str(tmp_path)}))
    source = dict(dataset_id='sot', version='v1', split='train', sot_timestamps=True,
                  sot_alignment_index=str(index.path))
    config = dict(catalog=str(tmp_path / 'catalog.jsonl'), roots=str(tmp_path / 'roots.json'),
                  train=[source], augmentation={'speed_prob': 1., 'speed_factors': [.5]}, seed=42)
    if cached:
        config['metadata_cache'] = str(tmp_path / 'cache')
    dataset = CatalogSwiftDataset(config, message_format='qwen3_asr')
    from open_audio_llm.data.catalog_cache import sampling_cost
    assert sampling_cost(dataset.records[0], dataset)[0] == pytest.approx(4., abs=.001)
    sample = dataset[0]
    assert sample['duration'] == 4.  # Speed augmentation would double this.
    assert '[S1][0.28-0.84]' in sample['messages'][-1]['content']
    if cached:
        from open_audio_llm.data.catalog_cache import source_identity
        before = source_identity(dataset, source)
        import os
        os.utime(index.path, ns=(index.path.stat().st_atime_ns, index.path.stat().st_mtime_ns + 1000))
        assert source_identity(dataset, source) != before
    with pytest.raises(ValueError, match='qwen3_asr'):
        CatalogSwiftDataset(config)
