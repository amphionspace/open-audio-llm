"""Build aligned 6–12 speaker training meetings and a separate selection set."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import importlib.util
import json
import math
import multiprocessing
from pathlib import Path
import random
import sqlite3
import shutil
import tempfile

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve
from transformers import AutoTokenizer
import yaml

RATE = 16000
STATE = {}


def rows(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as stream:
        yield from (json.loads(line) for line in stream if line.strip())


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def write_rows(path, values):
    with Path(path).open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as stream:
        for row in values:
            stream.write((json.dumps(row, ensure_ascii=False) + '\n').encode())


def make_pool(settings, catalog, roots):
    spec = next(e for e in catalog if (e['dataset_id'], e['version']) ==
                (settings['source_dataset'], settings['source_version']))
    artifacts = {a['name']: a for a in spec['artifacts']}
    # SQLite random reads on AFS dominated the first preparation attempt.
    scratch = tempfile.TemporaryDirectory(prefix='many-speaker-alignments-')
    local_db = Path(scratch.name)/'alignments.sqlite'
    shutil.copyfile(settings['alignments'],local_db)
    db = sqlite3.connect(f"file:{local_db}?mode=ro&immutable=1", uri=True)
    pools, counts, hashes = {}, Counter(), {}
    relocated = {}
    for dataset,manifests in settings['source_recordings'].items():
        for manifest in manifests:
            assert digest(manifest['file']) == manifest['sha256']
            hashes[manifest['file']] = manifest['sha256']
            for recording in rows(manifest['file']):
                assert len(recording['sources']) == 1
                relocated[(dataset,recording['id'])] = recording
    for split in ('train',):
        pool, seen = defaultdict(list), set()
        names = spec['splits'][f'{split}_zh_1spk_none']['artifacts']['records']
        for name in names:
            artifact = artifacts[name]
            path = Path(roots[artifact['root_alias']]) / artifact['relative_path']
            hashes[str(path)] = digest(path)
            assert hashes[str(path)] == artifact['sha256'], path
            for record in rows(path):
                for segment in record['metadata']['segments']:
                    source = segment['source']
                    key = tuple(source[k] for k in ('dataset_id', 'version', 'source_id'))
                    if key in seen:
                        continue
                    seen.add(key)
                    assert source['split'] == split and source['source_split'] == 'train'
                    if 'clean' in source['dataset_id']:
                        assert source['upstream_clean_pass'] is True
                    found = db.execute('SELECT payload FROM alignments WHERE dataset=? AND version=? AND source=?', key).fetchone()
                    assert found, key
                    alignment = json.loads(found[0])
                    assert alignment['status'] == 'aligned' and alignment['text'].strip() == source['text'].strip()
                    for field in ('audio', 'channel', 'sample_rate', 'start', 'duration', 'split', 'recording_id'):
                        assert alignment[field] == source[field], (key, field)
                    start, end = alignment['bounds']
                    assert 0 <= start < end <= alignment['audio_duration']
                    if not settings['utterance_seconds'][0] <= end - start <= settings['utterance_seconds'][1]:
                        counts[f'{split}_duration_excluded'] += 1
                        continue
                    recording = relocated[(source['dataset_id'],source['recording_id'])]
                    assert recording['sampling_rate'] == source['sample_rate']
                    resolved = recording['sources'][0]['source']
                    assert Path(resolved).name == Path(source['audio']['relative_path']).name
                    pool[source['speaker']].append({**source, 'bounds': [start, end], 'resolved_audio':resolved,
                                                   'decoded_duration': alignment['audio_duration']})
        pools[split] = dict(pool)
        counts[f'{split}_speakers'] = len(pool)
        counts[f'{split}_utterances'] = sum(map(len, pool.values()))
    db.close(); scratch.cleanup()
    # The aligned catalog contains training splits only. The new selection panel
    # uses AISHELL's official dev voices and original complete-utterance bounds.
    recordings = {r['id']:r for r in rows(settings['dev_recordings'])}
    pool = defaultdict(list)
    for row in rows(settings['dev_supervisions']):
        recording = recordings[row['recording_id']]
        if not settings['utterance_seconds'][0] <= row['duration'] <= settings['utterance_seconds'][1]:
            counts['dev_duration_excluded'] += 1
            continue
        speaker = 'aishell:'+str(row['speaker'])
        path = Path(recording['sources'][0]['source'])
        pool[speaker].append({'dataset_id':'aishell','version':'legacy-20260804','source_id':row['id'],
            'speaker':speaker,'split':'dev','source_split':'dev','recording_id':recording['id'],
            'audio':{'root_alias':'aidc_data','relative_path':str(path.relative_to('/workspace/data'))},
            'channel':0,'start':row['start'],'duration':row['duration'],'sample_rate':recording['sampling_rate'],
            'text':row['text'].strip(),'language':'zh','upstream_clean_pass':False,
            'bounds':[0,row['duration']],'decoded_duration':row['duration']})
    pools['dev'] = dict(pool)
    counts['dev_speakers'] = len(pool); counts['dev_utterances'] = sum(map(len,pool.values()))
    for path in (settings['dev_recordings'],settings['dev_supervisions']): hashes[str(path)] = digest(path)
    assert not (pools['train'].keys() & pools['dev'].keys())
    return pools, {'counts': dict(counts), 'source_manifest_sha256': hashes,
                   'alignments_sha256': digest(settings['alignments'])}


def synthesize(job):
    split, count, number = job
    settings, output, pools, roots = (STATE[k] for k in ('settings', 'output', 'pools', 'roots'))
    rng = random.Random(f"{settings['seed']}:{split}:{count}:{number}")
    pool = pools[split]
    # Sparse CommonVoice identities may have one sentence only; require enough
    # unique speech so a 180 s meeting never pads itself with repeated sentences.
    reserve = settings['target_seconds'] / count * settings['source_speech_reserve_ratio']
    eligible = [s for s in pool if sum(r['bounds'][1]-r['bounds'][0] for r in pool[s]) >= reserve]
    speakers = rng.sample(sorted(eligible), count)
    order, timeline, cursor, previous, used = {}, [], 0.0, None, set()
    while cursor < settings['target_seconds']:
        candidates = [s for s in speakers if s not in order] or [s for s in speakers if s != previous]
        speaker = rng.choice(candidates)
        sources = [s for s in pool[speaker] if (s['dataset_id'], s['source_id']) not in used]
        if not sources:
            candidates = [s for s in speakers if any((r['dataset_id'], r['source_id']) not in used for r in pool[s])]
            if not candidates:
                raise ValueError('Insufficient unique utterances for complete meeting')
            speaker = rng.choice(candidates)
            sources = [s for s in pool[speaker] if (s['dataset_id'], s['source_id']) not in used]
        source = rng.choice(sources)
        used.add((source['dataset_id'], source['source_id']))
        begin, end = source['bounds']
        gap = rng.uniform(*settings['gap_seconds'])
        if timeline and speaker != previous and rng.random() < settings['overlap_probability']:
            gap = -min(rng.uniform(*settings['overlap_seconds']), (timeline[-1]['end'] - timeline[-1]['start']) / 3)
        start = max(0.1, cursor + gap, max((t['end'] + 0.1 for t in timeline if t['identity'] == speaker), default=0))
        order.setdefault(speaker, f'S{len(order) + 1}')
        timeline.append({'speaker': order[speaker], 'identity': speaker, 'start': start,
                         'end': start + end - begin, 'text': source['text'], 'source': source})
        cursor, previous = max(cursor, start + end - begin), speaker
    assert len(order) == count
    duration = cursor + settings['tail_seconds']
    audio = np.zeros(round(duration * RATE), dtype=np.float32)
    rirs = {s: STATE['rirs'][rng.randrange(len(STATE['rirs']))] for s in speakers}
    for turn in timeline:
        source = turn['source']
        wave = STATE['audio_loader'].load_audio(source, roots, RATE)
        assert abs(len(wave) / RATE - source['decoded_duration']) <= 1 / RATE + 1e-6
        a, b = (round(v * RATE) for v in source['bounds'])
        # Include aligned speech completely; fades touch only the context outside it.
        context = round(settings['edge_context_seconds'] * RATE)
        left, right = max(0, a-context), min(len(wave), b+context)
        fragment = wave[left:right].copy()
        level = float(np.sqrt(np.mean(wave[a:b].astype(np.float64) ** 2)))
        assert level > 0 and np.isfinite(fragment).all()
        fragment *= 0.05 * 10 ** (rng.uniform(*settings['gain_db']) / 20) / level
        for n, leading in ((a-left, True), (right-b, False)):
            if n:
                ramp = np.linspace(0, 1, n, dtype=np.float32)
                if leading:
                    fragment[:n] *= ramp
                else:
                    fragment[-n:] *= ramp[::-1]
        rir = rirs[turn['identity']]
        reverbed = fftconvolve(fragment, rir)[:len(fragment) + RATE // 2]
        mixed = reverbed * settings['reverberation_mix']
        mixed[:len(fragment)] += fragment * (1-settings['reverberation_mix'])
        offset = round(turn['start'] * RATE) - (a-left)
        assert offset >= 0
        audio[offset:offset+min(len(mixed),len(audio)-offset)] += mixed[:len(audio)-offset]
    noise = STATE['noises'][rng.randrange(len(STATE['noises']))]
    offset = rng.randrange(len(noise))
    noise = np.resize(np.concatenate((noise[offset:], noise[:offset])), len(audio))
    snr = rng.uniform(*settings['snr_db'])
    audio += noise * np.sqrt(np.mean(audio**2) / (np.mean(noise**2) + 1e-12) / 10**(snr/10))
    assert np.isfinite(audio).all() and np.max(np.abs(audio)) > 0
    audio *= min(1.0, 0.95 / np.max(np.abs(audio)))
    reference = [{k:t[k] for k in ('speaker','start','end','text')} for t in timeline]
    target = '\n'.join(f"[{t['speaker']}][{t['start']:.2f}-{t['end']:.2f}] {t['text']}" for t in reference)
    tokens = math.ceil(duration * 13) + 288 + len(STATE['tokenizer'].encode(target, add_special_tokens=False))
    assert tokens <= settings['max_tokens'], tokens
    clip_id = f'{split}-zh-{count}spk-{number:05d}'
    path = output / 'audio' / split / f'{count}spk' / f'{clip_id}.flac'
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, RATE, subtype='PCM_16')
    record = {'schema_version':'audio-record/1.0', 'id':clip_id, 'task':'speaker_attributed_asr', 'language':'zh',
              'audio_slots':[{'name':'mixture','ref':{'dataset_id':settings['dataset_id'], 'version':settings['version'],
                            'split':f'{split}_{count}spk', 'cut_id':clip_id, 'duration':len(audio)/RATE}}],
              'target':target, 'labels':{'speaker_count':count},
              'metadata':{'sot_output_format':'aligned_utterance_timestamps_v1', 'turns':timeline,
                          'timestamp_source':'upstream_aligned_bounds' if split=='train' else 'human_utterance_bounds',
                          'sequence_length_upper_bound':tokens,
                          'quality':{'all_sources_clean_pass':all(t['source']['upstream_clean_pass'] for t in timeline)},
                          'noise_snr_db':snr, 'human_reviewed':False}}
    index = {'cut_id':clip_id,'root_alias':settings['root_alias'],'relative_path':str(path.relative_to(output)),
             'sample_rate':RATE,'channels':1,'num_frames':len(audio),'duration':len(audio)/RATE,'sha256':digest(path)}
    clip = {'id':clip_id,'dataset':f'synthetic_{count}spk','language':'zh','kind':'synthetic','meeting_id':clip_id,
            'duration':len(audio)/RATE,'reference':reference,'reference_speakers':count,'audio':str(path),
            'audio_sha256':index['sha256']}
    return record,index,clip


def ami_records(settings, output):
    helpers = module('meeting_windows', Path(__file__).parents[1] / 'shared/meeting_windows.py')
    records = {r['id']:r for r in rows(settings['ami_recordings'])}
    heldout = {r['id'] for p in settings['ami_heldout_recordings'] for r in rows(p)}
    assert not (records.keys() & heldout)
    grouped = defaultdict(list)
    for row in rows(settings['ami_supervisions']):
        assert row['text'].strip() and row['speaker'] and row['duration'] > 0
        grouped[row['recording_id']].append(row)
    result,index = [],[]
    for key, recording in records.items():
        source = Path(recording['sources'][0]['source'])
        info = sf.info(source)
        assert info.samplerate == RATE and info.frames == recording['num_samples'] and info.channels == 1
        index.append({'cut_id':key,'root_alias':'aidc_data','relative_path':str(source.relative_to('/workspace/data')),
                      'sample_rate':RATE,'channels':1,'num_frames':info.frames,'duration':info.frames/RATE})
        seen = []
        for n,(a,b) in enumerate(helpers.complete_windows(info.frames, grouped[key], settings['target_seconds'])):
            selected = [r for r in grouped[key] if a <= round(r['start']*RATE) < b]
            assert all(round((r['start']+r['duration'])*RATE) <= b for r in selected)
            seen.extend(r['id'] for r in selected)
            if not selected:
                continue
            target,turns,mapping = helpers.render_target(selected,a/RATE)
            tokens = math.ceil((b-a)/RATE*13)+288+len(STATE['tokenizer'].encode(target,add_special_tokens=False))
            assert tokens <= settings['max_tokens'] and (b-a)/RATE <= 650
            result.append({'schema_version':'audio-record/1.0','id':f'ami-train-{key}-{n:04d}',
                'task':'speaker_attributed_asr','language':'en',
                'audio_slots':[{'name':'mixture','ref':{'dataset_id':settings['ami_dataset_id'],'version':settings['version'],
                    'split':'train','cut_id':key,'start':a/RATE,'duration':(b-a)/RATE}}], 'target':target,
                'labels':{'speaker_count':len(mapping)}, 'metadata':{'sot_output_format':'aligned_utterance_timestamps_v1',
                    'turns':turns,'meeting_id':key,'source_dataset':'ami_sdm','source_version':'icefall-20260908',
                    'source_split':'train','sequence_length_upper_bound':tokens}})
        assert Counter(seen) == Counter(r['id'] for r in grouped[key])
    return result,index


def catalog_entry(settings, output, dataset_id, groups, indexes):
    artifacts,splits = [],{}
    index_path = output / f'{dataset_id}-audio-index.jsonl.gz'
    write_rows(index_path,indexes)
    artifacts.append({'name':'audio-index','kind':'audio-index','root_alias':settings['root_alias'],
                      'relative_path':index_path.name,'sha256':digest(index_path)})
    for split,records in groups.items():
        path = output / f'{dataset_id}-{split}.jsonl.gz'
        write_rows(path,records)
        artifacts.append({'name':split,'kind':'audio-records','root_alias':settings['root_alias'],
                          'relative_path':path.name,'sha256':digest(path),'metadata':{'record_count':len(records)}})
        splits[split] = {'artifacts':{'records':[split],'audio_index':['audio-index']},
                         'statistics':{'records':len(records),'duration_hours':sum(r['audio_slots'][0]['ref']['duration'] for r in records)/3600}}
    return {'schema_version':'dataset-catalog/2.0','dataset_id':dataset_id,'version':settings['version'],
            'languages':['en'] if dataset_id == settings['ami_dataset_id'] else ['zh'],
            'tasks':['speaker_attributed_asr'],'artifacts':artifacts,'splits':splits,
            'provenance':{'preparation_config':settings,'human_reviewed':False}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    settings = yaml.safe_load(parser.parse_args().config.read_text())['parameters']
    output = Path(settings['output']); output.mkdir(parents=True,exist_ok=True)
    data = yaml.safe_load(Path(settings['source_data']).read_text())
    catalog = list(rows(data['catalog'])); roots = json.loads(Path(data['roots']).read_text())
    pools,provenance = make_pool(settings,catalog,roots)
    dump(output/'source-provenance.json',provenance)
    rirs = []
    for r in rows(settings['rir_recordings']):
        wave,rate = sf.read(r['sources'][0]['source'],dtype='float32',always_2d=True)
        assert rate == RATE
        wave = wave[:,0]; wave = wave[np.argmax(np.abs(wave)):]
        rirs.append(wave / (np.linalg.norm(wave)+1e-8))
    noises = []
    for r in rows(settings['noise_recordings']):
        if settings['noise_subset'] in r['sources'][0]['source']:
            wave,rate = sf.read(r['sources'][0]['source'],dtype='float32',always_2d=True)
            assert rate == RATE
            noises.append(wave.mean(1))
    assert noises and rirs
    STATE.update(settings=settings,output=output,pools=pools,roots=roots,rirs=rirs,noises=noises,
                 tokenizer=AutoTokenizer.from_pretrained(settings['tokenizer'],local_files_only=True),
                 audio_loader=module('source_audio',Path(__file__).parents[1]/'shared/source_audio.py'))
    groups,indexes,clips = defaultdict(list),[],[]
    jobs = [(split,count,n) for split in ('dev','train') for count in settings['speaker_counts']
            for n in range(settings[f'{split}_per_count'])]
    with ProcessPoolExecutor(settings['workers'],mp_context=multiprocessing.get_context('fork')) as executor:
        for i,(record,index,clip) in enumerate(executor.map(synthesize,jobs,chunksize=1),1):
            groups[record['audio_slots'][0]['ref']['split']].append(record); indexes.append(index)
            if record['id'].startswith('dev-'): clips.append(clip)
            if i % 100 == 0: print(json.dumps({'generated':i,'total':len(jobs)}),flush=True)
    catalog.append(catalog_entry(settings,output,settings['dataset_id'],groups,indexes))
    ami,ami_index = ami_records(settings,output)
    catalog.append(catalog_entry(settings,output,settings['ami_dataset_id'],{'train':ami},ami_index))
    roots[settings['root_alias']] = str(output)
    dump(output/'roots.json',roots)
    (output/'catalog.jsonl').write_text(''.join(json.dumps(e,ensure_ascii=False)+'\n' for e in catalog))
    data.update(catalog=str(output/'catalog.jsonl'),roots=str(output/'roots.json'))
    def group(s):
        if s['dataset_id'] in ('aishell4_meeting_sot','alimeeting_sdm_meeting_sot','sot_speaker_events_treatment_ab'): return 'meeting'
        if s['dataset_id']=='sot_multispeaker_zh_en' and '1spk' not in s['split']: return 'synthetic'
        return 'replay'
    totals = Counter()
    for s in data['train']: totals[group(s)] += s['weight']
    for s in data['train']: s['weight'] *= settings['weights'][group(s)]/totals[group(s)]
    data['train'] += [{'dataset_id':settings['dataset_id'],'version':settings['version'],'split':f'train_{n}spk',
                       'weight':settings['weights']['many_speaker']/len(settings['speaker_counts']),'max_duration':650}
                      for n in settings['speaker_counts']]
    data['train'].append({'dataset_id':settings['ami_dataset_id'],'version':settings['version'],'split':'train',
                          'weight':settings['weights']['english_meeting'],'max_duration':650})
    data['batching']['max_samples'] = settings['batch_size']
    data['augmentation']['noise_exclude_datasets'] += [settings['dataset_id'],settings['ami_dataset_id']]
    (output/'train-data.yaml').write_text(yaml.safe_dump(data,sort_keys=False,allow_unicode=True))
    manifest = output/'selection-clips.jsonl'
    manifest.write_text(''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in clips))
    plan = json.loads(Path(settings['reference_plan']).read_text())
    plan.update(manifest_sha256=digest(manifest),samples=len(clips),meetings=len(clips),
                audio_hours=sum(c['duration'] for c in clips)/3600,priority_ids=[],
                purpose='6–12 speaker development selection; AISHELL official dev voices, complete-utterance bounds')
    dump(output/'selection-plan.json',plan)
    report = {'passed':True,'synthetic':{s:len(r) for s,r in groups.items()},'ami_records':len(ami),
              'weights':settings['weights'],'pool_counts':provenance['counts'],
              'synthetic_train_hours':sum(i['duration'] for i in indexes if i['cut_id'].startswith('train'))/3600,
              'training_source_count':len(data['train']),'retained_original_source_count':len(data['train'])-8,
              'speaker_disjoint':True,'max_sequence_tokens':max(r['metadata']['sequence_length_upper_bound'] for rs in groups.values() for r in rs)}
    dump(output/'report.json',report); print(json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
