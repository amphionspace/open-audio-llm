"""Index AmphionData alignment outputs for joining SOT training metadata."""

import argparse
import gzip
import hashlib
import json
import math
import os
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def utterance_bounds(row):
    alignment = row['alignment']
    if alignment['status'] != 'aligned':
        return None
    items = alignment['items']
    if (alignment['time_reference'] != 'source_segment_start' or not items
            or alignment.get('issues')):
        raise ValueError(f'Invalid aligned annotation: {row["id"]}')
    previous = 0.0
    duration = alignment['audio_duration']
    for item in items:
        start, end = item['start'], item['end']
        if (not math.isfinite(start) or not math.isfinite(end)
                or not previous <= start < end <= duration or not item['text'].strip()):
            raise ValueError(f'Invalid aligned boundaries: {row["id"]}')
        previous = end
    return [items[0]['start'], items[-1]['end']]


def build_alignment_index(directories, output):
    """Freeze source identities and utterance bounds, without rewriting any audio."""
    output = Path(output).resolve()
    report_path = output.with_suffix('.json')
    if output.exists() or report_path.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, prefix=output.name, suffix='.tmp', delete=False) as stream:
        temporary = Path(stream.name)
    counts, inputs = Counter(), []
    db = sqlite3.connect(temporary)
    try:
        db.execute('CREATE TABLE alignments (dataset TEXT, version TEXT, source TEXT, '
                   'payload TEXT, PRIMARY KEY (dataset, version, source)) WITHOUT ROWID')
        for directory in directories:
            directory = Path(directory).resolve()
            plan = json.loads((directory / 'plan.json').read_text())
            progress = json.loads((directory / 'progress.json').read_text())
            config = json.loads((directory / 'alignment-config.json').read_text())
            plan_hash = fingerprint(directory / 'plan.json')
            if (progress['status'] not in {'complete', 'complete_with_errors'}
                    or config['plan_sha256'] != plan_hash):
                raise ValueError(f'Alignment output is incomplete or changed: {directory}')
            shards = []
            for number, shard in enumerate(plan['shards']):
                path = directory / 'alignments' / f'{number:06d}.jsonl.gz'
                summary = json.loads((directory / 'summaries' / f'{number:06d}.json').read_text())
                digest = fingerprint(path)
                if digest != summary['sha256']:
                    raise ValueError(f'Alignment shard changed: {path}')
                batch = []
                with gzip.open(path, 'rt', encoding='utf-8') as stream:
                    for line in stream:
                        row = json.loads(line)
                        # A replacement clean source is a different waveform. Never
                        # attach its labels to an already-rendered original mixture.
                        key = (row['dataset_id'], row['version'], row['source_id'])
                        payload = {key: row[key] for key in (
                            'audio', 'channel', 'sample_rate', 'start', 'duration',
                            'text', 'split', 'source_split', 'recording_id')}
                        payload.update(status=row['alignment']['status'],
                                       bounds=utterance_bounds(row),
                                       audio_duration=row['alignment'].get('audio_duration'))
                        batch.append((*key, json.dumps(payload, ensure_ascii=False)))
                        counts[f'{row["split"]}/{row["dataset_id"]}/{payload["status"]}'] += 1
                if len(batch) != summary['records'] or len(batch) != shard['records']:
                    raise ValueError(f'Alignment shard count mismatch: {path}')
                db.executemany('INSERT INTO alignments VALUES (?, ?, ?, ?)', batch)
                db.commit()
                shards.append({'path': str(path), 'sha256': digest, 'records': len(batch)})
            inputs.append({'directory': str(directory), 'plan_sha256': plan_hash,
                           'config': config, 'shards': shards})
    except BaseException:
        db.close()
        temporary.unlink(missing_ok=True)
        raise
    else:
        db.close()
    try:
        os.link(temporary, output)  # Publish once; never overwrite a pinned training view.
    finally:
        temporary.unlink(missing_ok=True)
    report = {'format': 'sot-alignment-index-v1', 'path': str(output),
              'sha256': fingerprint(output), 'counts': dict(counts), 'inputs': inputs}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    return report


class AlignmentIndex:
    def __init__(self, path, expected_sha256=None):
        self.path = Path(path).resolve()
        self.sha256 = fingerprint(self.path) if expected_sha256 else None
        if expected_sha256 and self.sha256 != expected_sha256:
            raise ValueError(f'Alignment index checksum mismatch: {self.path}')
        self._connection, self._pid = None, None

    def __getitem__(self, key):
        if self._pid != os.getpid():
            if self._connection is not None:
                self._connection.close()
            self._connection = sqlite3.connect(self.path.as_uri() + '?mode=ro&immutable=1', uri=True)
            self._pid = os.getpid()
        row = self._connection.execute(
            'SELECT payload FROM alignments WHERE dataset=? AND version=? AND source=?', key,
        ).fetchone()
        if row is None:
            raise KeyError(f'Missing source alignment: {key}')
        return json.loads(row[0])

    def __getstate__(self):
        return {'path': self.path, 'sha256': self.sha256, '_connection': None, '_pid': None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', action='append', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = build_alignment_index(args.input, args.output)
    print(json.dumps({key: report[key] for key in ('path', 'sha256', 'counts')}))


if __name__ == '__main__':
    main()
