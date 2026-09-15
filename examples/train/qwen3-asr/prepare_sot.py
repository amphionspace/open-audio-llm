#!/usr/bin/env python3
"""Build a local SOT training view and exclude its held-out speakers from replay."""

import argparse
import copy
import itertools
import json
from pathlib import Path

import yaml
from audio_data_contract import DatasetSpec, load_catalog


def prepare(data_root, catalog_path, roots_path, output):
    output.mkdir(parents=True, exist_ok=True)
    synthetic = json.loads((data_root / 'catalog.jsonl').read_text())
    if synthetic['provenance']['status'] != 'complete':
        raise ValueError('SOT synthesis must be complete before freezing a training view')
    catalog = load_catalog(catalog_path)
    heldout = json.loads((data_root / 'heldout-speakers.json').read_text())
    roots = json.loads(roots_path.read_text())
    roots['multispeaker_synthetic'] = str(data_root.resolve())
    root_file = output / 'roots.json'
    root_file.write_text(json.dumps(roots, indent=2) + '\n')
    cells = list(itertools.product(['zh', 'en'], [3, 4, 5], ['staggered', 'dense']))
    artifacts = {item['name']: item for item in synthetic['artifacts']}
    sot = {key: [] for key in ('train', 'dev', 'test')}
    for split in sot:
        original = synthetic['splits'][split]
        for language, speakers, profile in cells:
            prefix = f'{split}/{language}/{speakers}spk/{profile}/'
            names = [name for name in original['records_artifacts']
                     if artifacts[name]['relative_path'].startswith(prefix)]
            if not names:
                raise ValueError(f'Missing synthesis cell: {prefix}')
            alias = f'{split}_{language}_{speakers}spk_{profile}'
            synthetic['splits'][alias] = {
                'group': split, 'records_artifacts': names,
                'audio_index_artifact': original['audio_index_artifact'],
            }
            sot[split].append({'dataset_id': synthetic['dataset_id'], 'version': synthetic['version'],
                               'split': alias, 'max_duration': 28})
    ordinary = [
        {'dataset_id': 'wenetspeech_clean', 'version': 'clean-v3-20260828', 'weight': 20, 'require_clean_pass': True},
        {'dataset_id': 'aishell', 'version': 'icefall-20260908', 'weight': 5},
        {'dataset_id': 'kespeech', 'version': 'icefall-20260908', 'weight': 5},
        {'dataset_id': 'commonvoice_en_clean', 'version': 'clean-v1-20260805', 'weight': 5, 'require_clean_pass': True},
        {'dataset_id': 'librispeech', 'version': 'icefall-20260908', 'weight': 5},
    ]
    for source in ordinary:
        source.update(split='train', task='asr', min_duration=0.5, max_duration=20)
        dataset_id = source['dataset_id']
        if dataset_id in heldout:
            source['exclude_speakers'] = sorted({speaker.removeprefix(dataset_id + ':')
                for split in ('dev', 'test') for speaker in heldout[dataset_id][split]})
    asr_dev = [{'dataset_id': name, 'version': 'icefall-20260908', 'split': 'dev',
                'task': 'asr', 'max_duration': 20}
               for name in ('aishell', 'wenetspeech', 'kespeech', 'librispeech')]
    selected_specs = {(source['dataset_id'], source['version']) for source in ordinary + asr_dev}
    specs = [catalog.get(*key).to_dict() for key in sorted(selected_specs)] + [synthetic]
    catalog_file = output / 'catalog.jsonl'
    for spec in specs:
        DatasetSpec.from_dict(spec)
    catalog_file.write_text('\n'.join(json.dumps(spec, ensure_ascii=False) for spec in specs) + '\n')
    config = {
        'catalog': str(catalog_file.resolve()), 'roots': str(root_file.resolve()),
        'sampling_rate': 16000, 'seed': 42,
        'replay': {'epoch_samples': 20000, 'window_samples': 200},
        'objective': {'sample_mean': True, 'separate_prefix': False, 'replay_kl_weight': 1.0},
        'augmentation': {'speed_prob': 0.25, 'speed_factors': [0.95, 1.0, 1.05],
                         'spec_aug_prob': 0.1, 'frequency_mask_width': 8, 'time_mask_width': 20},
        'batching': {'max_samples': 6, 'max_duration': 90, 'num_buckets': 4, 'drop_last': False},
        'train': [{**source, 'weight': 5} for source in sot['train']] + ordinary,
        'validation': [{**source, 'max_samples': 16} for source in sot['dev'] + asr_dev],
        'evaluation': sot['test'],
    }
    for filename, validation in [('train-data.yaml', config['validation']),
                                  ('sot-dev.yaml', sot['dev']), ('asr-dev.yaml', asr_dev)]:
        selected = copy.deepcopy(config)
        selected['validation'] = validation
        (output / filename).write_text(yaml.safe_dump(selected, allow_unicode=True, sort_keys=False))
    audit = {'sot_train_samples': synthetic['splits']['train']['statistics']['records'],
             'weights': {'sot': 60, 'chinese_asr': 30, 'english_asr': 10},
             'excluded_replay_speakers': {source['dataset_id']: len(source.get('exclude_speakers', [])) for source in ordinary},
             'source': str(data_root.resolve())}
    (output / 'data-audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data-root', 'catalog', 'roots', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    prepare(args.data_root, args.catalog, args.roots, args.output)


if __name__ == '__main__':
    main()
