"""Point the frozen local-training data recipe at this machine's copies, then verify them.

The archived catalog stays byte-identical. Only root aliases move: aliases whose
original tree was split during the AIDC migration get a hard-link mirror built from
declared directory prefixes and the migration path map. Record identity is
proven later by the training sampler signature; this task proves file identity.
"""
import argparse
import gzip
import hashlib
import json
import os
import random
import sqlite3
from pathlib import Path

import soundfile as sf
import yaml


def sha256(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def jsonl(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    settings = yaml.safe_load(parser.parse_args().config.read_text())['parameters']
    output = Path(settings['output'])
    output.mkdir(parents=True, exist_ok=True)
    data = yaml.safe_load(Path(settings['source_data']).read_text())
    catalog_path = Path(settings['catalog'])
    previous_roots = json.loads(Path(settings['previous_roots']).read_text())
    alias_roots = settings['alias_roots']
    mirrored = set(settings['mirrored_aliases'])
    prefixes = settings['prefix_map']
    sample_size = settings['audio_sample_per_index']
    rng = random.Random(settings['seed'])

    catalog = {(r['dataset_id'], r['version']): r for r in jsonl(catalog_path)}
    used, audio_indexes = {}, set()
    for group in ('train', 'validation', 'evaluation'):
        for source in data.get(group) or []:
            entry = catalog[(source['dataset_id'], source['version'])]
            artifacts = {a['name']: a for a in entry['artifacts']}
            for key, value in entry['splits'][source['split']].items():
                for name in [value] if key.endswith('_artifact') else value if key.endswith('_artifacts') else []:
                    used[(entry['dataset_id'], entry['version'], name)] = artifacts[name]
                if key == 'audio_index_artifact':
                    audio_indexes.add((entry['dataset_id'], entry['version'], value))

    def old_path(alias, relative):
        return str(Path(previous_roots[alias]) / relative)

    # Resolve every original file that is not covered by a declared prefix.
    wanted = {old_path(a['root_alias'], a['relative_path'])
              for a in used.values() if a['root_alias'] in mirrored}
    wanted = {p for p in wanted if not any(p == k or p.startswith(k + '/') for k in prefixes)}
    path_map = {}
    with open(settings['path_map'], encoding='utf-8') as stream:
        for line in stream:
            source = line.split('"', 4)[3]
            if source in wanted:
                path_map[source] = json.loads(line)
    missing = sorted(wanted - set(path_map))
    if missing:
        raise SystemExit(f'No migration target for {len(missing)} files, e.g. {missing[:3]}')
    relocated = {item['path']: item for item in
                 json.loads(Path(settings['relocated_manifests']).read_text()).values()}

    mirror = output / 'roots'
    links = {}

    def local_for(path):
        for prefix, target in prefixes.items():
            if path == prefix or path.startswith(prefix + '/'):
                return prefix, target
        return path, path_map[path]['target']

    def link(alias, path):
        # audio-data-contract rejects paths that resolve outside a root, so the
        # mirror holds per-file hard links (same inode, no copy), not symlinks.
        anchor, target = local_for(path)
        target = Path(target) / Path(path).relative_to(anchor) if anchor != path else Path(target)
        relative = Path(path).relative_to(previous_roots[alias])
        destination = mirror / alias / relative
        if destination.exists():
            assert os.path.samefile(destination, target), destination
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(target, destination)
        links[f'{alias}/{relative}'] = str(target)
        return destination

    roots = {alias: str(mirror / alias) for alias in mirrored}
    roots.update(alias_roots)
    report = {'artifacts': [], 'audio_indexes': [], 'links': links}
    for (dataset_id, version, name), artifact in sorted(used.items()):
        alias, relative = artifact['root_alias'], artifact['relative_path']
        if alias in mirrored:
            local = link(alias, old_path(alias, relative))
        else:
            local = Path(roots[alias]) / relative
        if not local.is_file():
            raise SystemExit(f'Missing artifact {dataset_id}@{version}:{name}: {local}')
        source_file = Path(links.get(f'{alias}/{relative}', local))
        row = {'dataset_id': dataset_id, 'version': version, 'name': name, 'local': str(local),
               'source_file': str(source_file)}
        actual = sha256(local)
        row['local_sha256'] = actual
        expected = artifact.get('sha256')
        moved = relocated.get(str(source_file.relative_to(settings['aidc_root']))) \
            if str(source_file).startswith(settings['aidc_root'] + '/') else None
        if expected is None:
            row['integrity'] = 'catalog-has-no-sha256'
            if alias in mirrored:
                source = path_map.get(old_path(alias, relative))
                row['migration_record_sha256'] = source['sha256'] if source else None
        elif actual == expected:
            row['integrity'] = 'sha256-match'
        elif moved and moved['sha256'] == actual and moved['original_sha256'] == expected:
            # The migration rewrote audio paths in this manifest; its original matches.
            row['integrity'] = 'relocated-original-sha256-match'
        else:
            raise SystemExit(f'sha256 mismatch for {dataset_id}@{version}:{name}: {local}')
        report['artifacts'].append(row)

        # Split declarations, not artifact kinds, say which file is an audio index.
        if (dataset_id, version, name) not in audio_indexes:
            continue
        entries = list(jsonl(local))
        # Small meeting/event indexes are checked in full; the 2M synthetic
        # mixtures get a seeded header sample, their identity being in the index.
        full_aliases = mirrored | set(settings['full_audio_check_aliases'])
        full = [e for e in entries if e['root_alias'] in full_aliases]
        rest = [e for e in entries if e['root_alias'] not in full_aliases]
        sample = full + rng.sample(rest, min(sample_size, len(rest)))
        for entry in sample:
            if entry['root_alias'] in mirrored:
                audio = link(entry['root_alias'], old_path(entry['root_alias'], entry['relative_path']))
            else:
                audio = Path(roots[entry['root_alias']]) / entry['relative_path']
            info = sf.info(str(audio))
            if (info.samplerate, info.channels, info.frames) != (
                    entry['sample_rate'], entry['channels'], entry['num_frames']):
                raise SystemExit(f'Audio header mismatch: {audio}')
        report['audio_indexes'].append({'dataset_id': dataset_id, 'version': version, 'name': name,
                                        'entries': len(entries), 'header_checked': len(sample)})

    alignment = Path(settings['sot_alignment_index'])
    expected = {s['sot_alignment_sha256'] for s in data['train'] if s.get('sot_alignment_index')}
    actual = sha256(alignment)
    if expected != {actual}:
        raise SystemExit(f'Alignment sha256 {actual} not in {expected}')
    with sqlite3.connect(f'file:{alignment}?mode=ro', uri=True) as db:
        db.execute('select count(*) from sqlite_master').fetchone()
    report['sot_alignment'] = {'path': str(alignment), 'sha256': actual}

    (output / 'roots.json').write_text(json.dumps(roots, indent=2) + '\n')
    current = {
        'catalog': str(catalog_path.resolve()),
        'roots': str((output / 'roots.json').resolve()),
        'experiment_plan': str(Path(settings['experiment_plan']).resolve()),
        'sot_alignment_index': str(alignment),
    }
    previous_paths = {}
    for key in ('catalog', 'roots', 'experiment_plan'):
        previous_paths[current[key]] = data[key]
        data[key] = current[key]
    for source in data['train'] + (data.get('validation') or []) + (data.get('evaluation') or []):
        if source.get('sot_alignment_index'):
            previous_paths[current['sot_alignment_index']] = source['sot_alignment_index']
            source['sot_alignment_index'] = current['sot_alignment_index']
    data['metadata_cache'] = str(Path(settings['metadata_cache']).resolve())
    (output / 'local-balanced-train.yaml').write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    report['previous_paths'] = previous_paths
    report['status'] = 'passed'
    (output / 'verification.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'links'} | {'links': len(links)},
                     ensure_ascii=False)[:4000])


if __name__ == '__main__':
    main()
