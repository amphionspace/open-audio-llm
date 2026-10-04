from settings import ROOT, RUNS, OUTPUT, SETTINGS, artifact, child
"""Freeze matched old/new event subsets, preserving the other 65% of training."""
from collections import Counter, defaultdict
import copy
import gzip
import hashlib
import heapq
import json
import os
from pathlib import Path
import shutil
import subprocess
import yaml
from prepare import save

D = ROOT
BASE = Path(SETTINGS['parameters']['reference_experiment'])
OLD_DATA = Path(SETTINGS['parameters']['source_data'])
def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
def rows(path):
    with gzip.open(path, 'rt') as f: yield from map(json.loads, f)
def cell(r):
    return (r['language'], r['labels']['scenario'], int(r['labels']['speaker_count']), float(r['audio_slots'][0]['ref']['duration']))
def main():
    assert json.loads((artifact('release-gate.json')).read_text())['passed']
    training_plan = json.loads((RUNS / 'identity-synthesis-20260927/training-comparison-plan.json').read_text())
    assert sha(BASE / 'train-data.yaml') == training_plan['control']['training_data_sha256']
    initial = BASE / 'initial-checkpoint'; pins = training_plan['initial_weight_pins_from_existing_training_plan']
    for name, pin in pins.items():
        assert (initial / name).stat().st_size == pin['bytes'] and sha(initial / name) == pin['sha256'], name
    treatment = [r for p in sorted((artifact('generated/shards')).glob('*/records.jsonl.gz')) for r in rows(p)]
    desired = Counter(map(cell, treatment)); heaps = defaultdict(list); available = Counter()
    for path in sorted((OLD_DATA / 'shards').glob('*/records.jsonl.gz')):
        for r in rows(path):
            key = cell(r)
            if key not in desired: continue
            available[key] += 1
            rank = int(hashlib.sha256(('matched-control-20260928:' + r['id']).encode()).hexdigest(), 16)
            item = (-rank, r['id'], r); heap = heaps[key]
            if len(heap) < desired[key]: heapq.heappush(heap, item)
            elif item > heap[0]: heapq.heapreplace(heap, item)
    assert all(len(heaps[k]) == n for k, n in desired.items()), {'missing': {str(k): (n, available[k]) for k, n in desired.items() if available[k] < n}}
    control = [r for k in sorted(heaps) for _, _, r in sorted(heaps[k], reverse=True)]
    assert Counter(map(cell, control)) == desired
    summaries = {}
    for arm, data, origin in [('control', control, OLD_DATA), ('treatment', treatment, artifact('generated'))]:
        root = artifact(arm); root.mkdir(exist_ok=False); event = root / 'event-data'; event.mkdir()
        dataset = f'sot_speaker_events_{arm}_ab'; version = 'matched-auto-ab-20260928'
        alias = f'events_{arm}_ab_manifest'; audio_alias = f'events_{arm}_ab_audio'
        roots = json.loads((BASE / 'roots.json').read_text()); roots[alias] = str(event); roots[audio_alias] = str(origin)
        old_index = {r['cut_id']: r for r in rows(origin / 'audio-index.jsonl.gz')}
        with gzip.open(event / 'records.jsonl.gz', 'wt') as f, gzip.open(event / 'audio-index.jsonl.gz', 'wt') as ix:
            for r in sorted(data, key=lambda x: (cell(x), x['id'])):
                for slot in r['audio_slots']: slot['ref'].update(dataset_id=dataset, version=version, split='train')
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
                a = copy.deepcopy(old_index[r['id']]); a['root_alias'] = audio_alias
                assert (origin / a['relative_path']).exists()
                ix.write(json.dumps(a) + '\n')
        hours = sum(r['audio_slots'][0]['ref']['duration'] for r in data) / 3600
        spec = {'schema_version': 'dataset-catalog/1.0', 'dataset_id': dataset, 'version': version,
            'languages': ['zh', 'en'], 'tasks': ['speaker_attributed_asr'],
            'artifacts': [{'name': name, 'kind': kind, 'root_alias': alias, 'relative_path': file, 'sha256': sha(event / file)}
                for name, kind, file in [('records', 'audio-records', 'records.jsonl.gz'), ('index', 'audio-index', 'audio-index.jsonl.gz')]],
            'splits': {'train': {'records_artifact': 'records', 'audio_index_artifact': 'index', 'statistics': {'records': len(data), 'duration_hours': hours}}},
            'provenance': {'synthetic': True, 'source': str(origin), 'experiment_arm': arm,
                'matched_on': ['sample_count', 'language', 'scenario', 'duration', 'speaker_count'], 'human_verified': False}}
        catalog = (BASE / 'catalog.jsonl').read_text().rstrip() + '\n' + json.dumps(spec, ensure_ascii=False) + '\n'
        (root / 'catalog.jsonl').write_text(catalog); (root / 'roots.json').write_text(json.dumps(roots, indent=2) + '\n')
        config = yaml.safe_load((BASE / 'train-data.yaml').read_text())
        config['catalog'] = str(root / 'catalog.jsonl'); config['roots'] = str(root / 'roots.json')
        config['metadata_cache'] = str(RUNS / 'catalog-metadata-cache')
        config['experiment_plan'] = str(root / 'plan.json')
        event_source = next(s for s in config['train'] if s['dataset_id'] == 'sot_speaker_events_zh_en')
        event_source.update(dataset_id=dataset, version=version, require_clean_pass=arm == 'treatment')
        assert event_source['weight'] == 35
        (root / 'train-data.yaml').write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
        plan = json.loads((BASE / 'plan.json').read_text())
        plan.update(name=f'{ROOT.name}-{arm}', status='prepared', max_steps=1000, source_checkpoint=str(initial),
            pilot_training=True, event_dataset=dataset, event_version=version, event_records=len(data),
            arm=arm, scope='Matched old versus automatic-screened new synthetic sources and generator; bounded 1000-step pilot.',
            source_weight_files=pins, event_data={'dataset_id': dataset, 'version': version, 'records': len(data), 'audio_hours': hours})
        (root / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
        for name in ('source', 'dependencies', 'initial-checkpoint'): (root / name).symlink_to(BASE / name, target_is_directory=True)
        from open_audio_llm.run_config import load_config
        recipe = load_config(SETTINGS['parameters']['training_configs'][arm],
                             root.parent.parent)
        recipe['data']['config'] = str(root / 'train-data.yaml')
        recipe['task']['arguments']['--model'] = str(initial)
        recipe['task']['arguments']['--output_dir'] = str(root / 'training')
        (root / 'train.yaml').write_text(yaml.safe_dump(recipe, allow_unicode=True, sort_keys=False))
        audit = (BASE / 'training_audit.py').read_text().replace('args.max_steps == 60000', 'args.max_steps == 1000')
        audit = audit.replace("s['dataset_id'] == 'sot_speaker_events_zh_en'", "s['dataset_id'] == self.phase['event_dataset']")
        audit = audit.replace("source['dataset_id'] == 'sot_speaker_events_zh_en'", "source['dataset_id'] == self.phase['event_dataset']")
        audit = audit.replace("event_sources[0]['version'] == 'events-v1c-20260921'", "event_sources[0]['version'] == self.phase['event_version']")
        audit = audit.replace("== 50000", "== self.phase['event_records']")
        (root / 'training_audit.py').write_text(audit)
        summaries[arm] = {'records': len(data), 'hours': hours, 'dataset_id': dataset,
            'manifest_sha256': sha(event / 'records.jsonl.gz'), 'data_config_sha256': sha(root / 'train-data.yaml')}
    save('matched-data.json', {'arms': summaries, 'equal_cells': True, 'cells': {str(k): n for k, n in desired.items()}, 'initial_weights_verified': pins})
    # Verify the actual training adapter, including real decoded 30/120/300 s
    # examples and the exact 35% sampler quota; no training on a filtered-empty set.
    for arm in ('control', 'treatment'):
        env = {**os.environ, 'PYTHONPATH': SETTINGS['parameters']['verification_pythonpath'],
               'CUDA_VISIBLE_DEVICES': ''}
        subprocess.run(child('verify_data.py', arm), env=env, check=True)
    save('training-ready.json', {'passed': True, 'arms': summaries, 'steps_per_arm': 1000, 'initial_checkpoint': str(initial)})

if __name__ == '__main__': main()
