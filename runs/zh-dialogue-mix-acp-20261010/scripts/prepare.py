"""Assemble this round's training data configuration without creating new audio.

The previous round's frozen data configuration (zh-farfield-mix-acp-20261010) is
kept; selected groups are rescaled as a whole and new Chinese conversation
groups are added. Root aliases that the shared registry declares are read from
the registry copies (same files, verified by sha256 when the catalog is loaded).

Group weights are sample proportions. Inside a new group, sources keep their
window views (or splits) proportional to catalog record counts, as before.
"""
import argparse
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


def matches(source, rules):
    return any(source['dataset_id'] == r['dataset_id']
               and r.get('split_contains', '') in source['split']
               and r.get('split_excludes', '\0') not in source['split']
               for r in rules)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    settings = yaml.safe_load(parser.parse_args().config.read_text())['parameters']
    output = Path(settings['output'])
    output.mkdir(parents=True, exist_ok=True)
    data = yaml.safe_load(Path(settings['source_data']).read_text())
    catalog = list(rows(data['catalog']))
    roots = json.loads(Path(data['roots']).read_text())
    registry = json.loads(Path(settings['prefer_roots']).read_text())
    moved = {alias: registry[alias] for alias in roots if alias in registry and roots[alias] != registry[alias]}
    roots.update(moved)
    for source in settings['catalogs']:
        new_roots = json.loads(Path(source['roots']).read_text())
        for alias, path in new_roots.items():
            assert roots.get(alias, path) == path, alias
        roots.update(new_roots)
        catalog += list(rows(source['catalog']))
    assert len({(e['dataset_id'], e['version']) for e in catalog}) == len(catalog), 'duplicate catalog entries'
    entries = {(e['dataset_id'], e['version']): e for e in catalog}

    train = data['train']
    report = {'groups': {}, 'sources': {}, 'roots_from_registry': moved}
    claimed = set()
    for name, rule in settings['rescale'].items():
        members = [s for s in train if matches(s, rule['match'])]
        assert not claimed & {id(s) for s in members}, f'{name} overlaps another rescaled group'
        claimed |= {id(s) for s in members}
        total = sum(s['weight'] for s in members)
        assert abs(total - rule['previous_weight']) < 1e-6, (name, total)
        for s in members:
            s['weight'] *= rule['weight'] / total
        report['groups'][name] = rule['weight']
    kept = sum(s['weight'] for s in train if id(s) not in claimed)
    assert abs(kept - settings['kept_weight']) < 1e-6, kept
    report['groups']['kept_unchanged'] = round(kept, 6)

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
                              'weight': group['weight'] * counts[key] / total, **extra})
        report['groups'][name] = group['weight']
        report['sources'][name] = {'/'.join(k): v for k, v in counts.items()}
    assert abs(sum(s['weight'] for s in train) - 100) < 1e-6, sum(s['weight'] for s in train)

    data['train'] = train
    data['augmentation']['noise_exclude_datasets'] = sorted(
        set(data['augmentation']['noise_exclude_datasets']) | set(settings['noise_exclude_datasets']))
    (output / 'catalog.jsonl').write_text(''.join(json.dumps(e, ensure_ascii=False) + '\n' for e in catalog))
    dump(output / 'roots.json', roots)
    data.update(catalog=str(output / 'catalog.jsonl'), roots=str(output / 'roots.json'))
    # A new sampling seed draws different records than earlier rounds that used the same sources.
    data['seed'] = settings['seed']
    (output / 'train-data.yaml').write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))

    exclude_files = sorted({s['exclude_records'] for s in train if 'exclude_records' in s})
    report.update(passed=True, training_source_count=len(train),
                  inputs={'source_data': settings['source_data'], 'source_data_sha256': digest(settings['source_data']),
                          'catalogs': {c['catalog']: digest(c['catalog']) for c in settings['catalogs']},
                          'exclude_records': {p: digest(p) for p in exclude_files}})
    dump(output / 'report.json', report)
    print(json.dumps(report['groups']), flush=True)


if __name__ == '__main__':
    main()
