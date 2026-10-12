"""Assemble this round's training data configuration without creating new audio.

The previous round's frozen data configuration (zh-first-mix-acp-20261010) supplies
the English meetings, the word-level and earlier synthesis and single-speaker replay
unchanged. Its AISHELL-4, AliMeeting and RAMC sources are replaced by the
teacher-reviewed v2 windows plus the Chinese far-field products (one original
microphone per window, near-to-far simulation); the replay group is scaled down to
make room for Chinese far-field single-speaker replay.

Group weights are sample proportions. Inside a group, sources keep their window
views (or splits) proportional to catalog record counts, as before.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

import yaml


def rows(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as stream:
        yield from (json.loads(line) for line in stream if line.strip())


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    settings = yaml.safe_load(parser.parse_args().config.read_text())['parameters']
    output = Path(settings['output'])
    output.mkdir(parents=True, exist_ok=True)
    weights = settings['weights']
    assert abs(sum(weights.values()) + settings['kept_weight'] - 100) < 1e-9, weights
    data = yaml.safe_load(Path(settings['source_data']).read_text())
    dropped = {(d['dataset_id'], d['version']) for d in settings['dropped_sources']}
    catalog = [e for e in rows(data['catalog']) if (e['dataset_id'], e['version']) not in dropped]
    roots = json.loads(Path(data['roots']).read_text())
    for source in settings['catalogs']:
        new_roots = json.loads(Path(source['roots']).read_text())
        for alias, path in new_roots.items():
            assert roots.get(alias, path) == path, alias
        roots.update(new_roots)
        catalog += list(rows(source['catalog']))
    assert len({(e['dataset_id'], e['version']) for e in catalog}) == len(catalog), 'duplicate catalog entries'
    entries = {(e['dataset_id'], e['version']): e for e in catalog}

    previous = data['train']
    removed = [s for s in previous if (s['dataset_id'], s['version']) in dropped]
    assert {(s['dataset_id'], s['version']) for s in removed} == dropped, 'every dropped source was in use'
    train = [s for s in previous if (s['dataset_id'], s['version']) not in dropped]
    # Kept groups: unchanged weights, except replay, which is rescaled as a whole.
    def is_replay(source):
        # sot_multispeaker_zh_en holds both the 1-speaker replay splits and multi-speaker synthesis.
        return any(source['dataset_id'] == m['dataset_id'] and m.get('split_contains', '') in source['split']
                   for m in settings['replay']['match'])

    replay = [s for s in train if is_replay(s)]
    replay_total = sum(s['weight'] for s in replay)
    assert abs(replay_total - settings['replay']['previous_weight']) < 1e-6, replay_total
    for s in replay:
        s['weight'] *= weights['replay'] / replay_total
    kept_total = sum(s['weight'] for s in train if not is_replay(s))
    assert abs(kept_total - settings['kept_weight']) < 1e-6, kept_total
    report = {'groups': {'replay': weights['replay'], 'kept_unchanged': round(kept_total, 6)}, 'sources': {}}

    for name, group in settings['groups'].items():
        counts = {}
        for spec in group['sources']:
            entry = entries[(spec['dataset_id'], spec['version'])]
            for split in spec['splits']:
                counts[(spec['dataset_id'], spec['version'], split)] = entry['splits'][split]['statistics']['records']
        total = sum(counts.values())
        for spec in group['sources']:
            extra = {k: v for k, v in spec.items() if k not in ('dataset_id', 'version', 'splits')}
            for split in spec['splits']:
                key = (spec['dataset_id'], spec['version'], split)
                train.append({'dataset_id': spec['dataset_id'], 'version': spec['version'], 'split': split,
                              'weight': weights[name] * counts[key] / total, **extra})
        report['groups'][name] = weights[name]
        report['sources'][name] = {'/'.join(k): v for k, v in counts.items()}

    data['train'] = train
    data['augmentation']['noise_exclude_datasets'] = sorted(
        set(data['augmentation']['noise_exclude_datasets']) | set(settings['noise_exclude_datasets']))
    (output / 'catalog.jsonl').write_text(''.join(json.dumps(e, ensure_ascii=False) + '\n' for e in catalog))
    dump(output / 'roots.json', roots)
    data.update(catalog=str(output / 'catalog.jsonl'), roots=str(output / 'roots.json'))
    # A new sampling seed draws different records than earlier rounds that used the same sources.
    data['seed'] = settings['seed']
    (output / 'train-data.yaml').write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))

    assert abs(sum(s['weight'] for s in train) - 100) < 1e-6
    exclude_files = sorted({s['exclude_records'] for s in train if 'exclude_records' in s})
    report.update(passed=True, training_source_count=len(train),
                  removed_sources=sorted({f"{s['dataset_id']}@{s['version']}" for s in removed}),
                  inputs={'source_data': settings['source_data'], 'source_data_sha256': digest(settings['source_data']),
                          'catalogs': {c['catalog']: digest(c['catalog']) for c in settings['catalogs']},
                          'exclude_records': {p: digest(p) for p in exclude_files}})
    dump(output / 'report.json', report)
    print(json.dumps(report['groups']), flush=True)


if __name__ == '__main__':
    main()
