"""Assemble this round's training data configuration without creating new audio.

The previous round's data configuration supplies AISHELL-4, AliMeeting, AMI, the
1-5 speaker synthesis and single-speaker replay unchanged; the 6-12 speaker
whole-sentence synthesis (sot_many_speaker_zh) is dropped. NOTSOFAR, CHiME-6,
RAMC and the word-level 6-12 speaker synthesis come from their own catalogs.

Group weights are sample proportions. Inside a group, the sources of one
dataset keep their window views proportional to record counts, as before.
Two groups need record subsets, expressed as exclude_records ID files:
- the word-level synthesis has one train split; each language and speaker count
  is its own source so every count is weighted equally;
- NOTSOFAR records each meeting once per recording device; sources partitioned
  by the meeting's device count get weight proportional to records / devices,
  so every meeting contributes by its deduplicated windows.
"""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path

import yaml

OLD_GROUPS = {
    'aishell4_meeting_sot': 'aishell4',
    'alimeeting_sdm_meeting_sot': 'alimeeting',
    'ami_sdm_meeting_replay': 'ami',
    # The single matched A/B events source stays with the earlier synthesis.
    'sot_speaker_events_treatment_ab': 'old_synthetic',
}


def rows(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as stream:
        yield from (json.loads(line) for line in stream if line.strip())


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def old_group(source):
    if source['dataset_id'] in OLD_GROUPS:
        return OLD_GROUPS[source['dataset_id']]
    if source['dataset_id'] == 'sot_multispeaker_zh_en' and '1spk' not in source['split']:
        return 'old_synthetic'
    return 'replay'


def split_records(entry, roots, split):
    artifacts = {a['name']: a for a in entry['artifacts']}
    for name in entry['splits'][split]['artifacts']['records']:
        artifact = artifacts[name]
        yield from rows(Path(roots[artifact['root_alias']]) / artifact['relative_path'])


def write_ids(path, ids):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(f'{i}\n' for i in sorted(ids)))
    return str(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    settings = yaml.safe_load(parser.parse_args().config.read_text())['parameters']
    output = Path(settings['output'])
    output.mkdir(parents=True, exist_ok=True)
    weights = settings['weights']
    assert abs(sum(weights.values()) - 100) < 1e-9, weights
    data = yaml.safe_load(Path(settings['source_data']).read_text())
    catalog = [e for e in rows(data['catalog']) if e['dataset_id'] not in settings['dropped_datasets']]
    roots = json.loads(Path(data['roots']).read_text())
    for source in settings['catalogs']:
        new_roots = json.loads(Path(source['roots']).read_text())
        for alias, path in new_roots.items():
            assert roots.get(alias, path) == path, alias
        roots.update(new_roots)
        catalog += list(rows(source['catalog']))
    assert len({(e['dataset_id'], e['version']) for e in catalog}) == len(catalog), 'duplicate catalog entries'
    # Datasets added this round have a single version in the merged catalog.
    entries = {e['dataset_id']: e for e in catalog}

    train = [s for s in data['train'] if s['dataset_id'] not in settings['dropped_datasets']]
    totals = Counter()
    for s in train:
        totals[old_group(s)] += s['weight']
    for s in train:
        s['weight'] *= weights[old_group(s)] / totals[old_group(s)]
    report = {'groups': {}, 'sources': {}}

    # Real meetings and RAMC: official train views, weight proportional to window counts.
    for name, spec in settings['meetings'].items():
        entry = entries[spec['dataset_id']]
        views = {split: entry['splits'][split]['statistics']['records'] for split in spec['splits']}
        if spec.get('deduplicate_devices'):
            devices = defaultdict(set)
            for split in spec['splits']:
                for record in split_records(entry, roots, split):
                    devices[record['metadata']['meeting_id']].add(record['metadata']['recording_id'])
            parts = defaultdict(lambda: defaultdict(set))
            all_ids = {}
            for split in spec['splits']:
                all_ids[split] = set()
                for record in split_records(entry, roots, split):
                    k = len(devices[record['metadata']['meeting_id']])
                    parts[split][k].add(record['id'])
                    all_ids[split].add(record['id'])
            shares = {(split, k): len(ids) / k for split in parts for k, ids in parts[split].items()}
            scale = weights[name] / sum(shares.values())
            for (split, k), share in sorted(shares.items()):
                excluded = all_ids[split] - parts[split][k]
                train.append({'dataset_id': spec['dataset_id'], 'version': entry['version'], 'split': split,
                              'weight': share * scale, 'max_duration': settings['max_duration'],
                              'exclude_records': write_ids(output / 'subsets' / f'{name}-{split}-{k}dev.txt',
                                                           excluded)})
            report['sources'][name] = {
                'meetings': len(devices), 'recordings': sum(len(v) for v in devices.values()),
                'device_count_meetings': dict(Counter(len(v) for v in devices.values())),
                'partitions': {f'{s}/{k}dev': len(parts[s][k]) for s in parts for k in parts[s]}}
        else:
            total = sum(views.values())
            for split, count in views.items():
                train.append({'dataset_id': spec['dataset_id'], 'version': entry['version'], 'split': split,
                              'weight': weights[name] * count / total, 'max_duration': settings['max_duration']})
            report['sources'][name] = {'records': views}

    # Word-level synthesis: one source per language and speaker count.
    spec = settings['word_synthesis']
    entry = entries[spec['dataset_id']]
    groups = defaultdict(set)
    for record in split_records(entry, roots, 'train'):
        groups[(record['language'], record['labels']['speaker_count'])].add(record['id'])
    every = set().union(*groups.values())
    counts = sorted({c for _, c in groups})
    assert counts == spec['speaker_counts'], counts
    for language, group in spec['languages'].items():
        for count in counts:
            ids = groups[(language, count)]
            assert ids, (language, count)
            train.append({'dataset_id': spec['dataset_id'], 'version': entry['version'], 'split': 'train',
                          'weight': weights[group] / len(counts), 'max_duration': settings['max_duration'],
                          'exclude_records': write_ids(output / 'subsets' / f'word-{language}-{count}spk.txt',
                                                       every - ids)})
    report['sources']['word_synthesis'] = {f'{lang}/{count}': len(v) for (lang, count), v in sorted(groups.items())}

    data['train'] = train
    data['augmentation']['noise_exclude_datasets'] = sorted(
        (set(data['augmentation']['noise_exclude_datasets']) - set(settings['dropped_datasets']))
        | set(settings['noise_exclude_datasets']))
    (output / 'catalog.jsonl').write_text(''.join(json.dumps(e, ensure_ascii=False) + '\n' for e in catalog))
    dump(output / 'roots.json', roots)
    data.update(catalog=str(output / 'catalog.jsonl'), roots=str(output / 'roots.json'))
    (output / 'train-data.yaml').write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))

    weight_by_group = Counter()
    for s in train:
        name = next((n for n, m in settings['meetings'].items() if m['dataset_id'] == s['dataset_id']), None)
        if s['dataset_id'] == spec['dataset_id']:
            lang = Path(s['exclude_records']).name.split('-')[1]
            name = spec['languages'][lang]
        weight_by_group[name or old_group(s)] += s['weight']
    report['groups'] = {k: round(v, 6) for k, v in sorted(weight_by_group.items())}
    assert all(abs(report['groups'][k] - weights[k]) < 1e-6 for k in weights), report['groups']
    report.update(passed=True, training_source_count=len(train),
                  inputs={'source_data': settings['source_data'], 'source_data_sha256': digest(settings['source_data']),
                          'catalogs': {c['catalog']: digest(c['catalog']) for c in settings['catalogs']}})
    dump(output / 'report.json', report)
    print(json.dumps(report['groups']), flush=True)


if __name__ == '__main__':
    main()
