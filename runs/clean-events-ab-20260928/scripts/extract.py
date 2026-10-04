from settings import ROOT, RUNS, OUTPUT, SETTINGS, artifact, child
"""Extract independent variable-length source embeddings with official AntSpeaker."""
import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi

D = ROOT
MODEL = Path(SETTINGS['parameters']['identity_model'])
sys.path.insert(0, str(MODEL / 'upstream'))
import antspeaker.models
from antspeaker.utils.registry import create_model

parser = argparse.ArgumentParser()
parser.add_argument('--worker', type=int, required=True)
parser.add_argument('--pool', default='expanded-clean-pool.jsonl.gz')
parser.add_argument('--prefix', default='')
args = parser.parse_args()
torch.set_num_threads(4)
torch.manual_seed(20260927)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert torch.cuda.is_available()
checkpoint = torch.load(MODEL / 'models/mect_b2_vb2.pt', map_location='cpu', weights_only=True)
model = create_model(copy.deepcopy(checkpoint['config']['model']))
model.load_state_dict(checkpoint['model'], strict=True)
model.eval().cuda()
feat = checkpoint['config']['feature']
roots = json.loads((artifact('roots.json')).read_text())
rows = [json.loads(line) for i, line in enumerate(gzip.open(artifact(args.pool), 'rt')) if i % 2 == args.worker]
entries, vectors = [], []
started = time.monotonic()
for i, row in enumerate(rows):
    reference_only = row.get('identity_reference_only', False)
    speech_start = 0. if reference_only else row['words'][0]['start']
    speech_end = row['duration'] if reference_only else row['words'][-1]['end']
    entry = {'id': row['id'], 'speaker': row['speaker'], 'dataset': row['dataset_id'],
             'text': row['text'], 'speech_seconds': speech_end - speech_start,
             'identity_reference_only': reference_only,
             'duration_kind': 'recording_duration' if reference_only else 'aligned_speech_duration'}
    try:
        path = Path(roots[row['audio']['root_alias']]) / row['audio']['relative_path']
        wave, sr = sf.read(path, dtype='float32', always_2d=True)
        left = max(0, round((row['start'] + speech_start - (.0 if reference_only else .08)) * sr))
        right = min(len(wave), round((row['start'] + speech_end + (.0 if reference_only else .08)) * sr))
        wave = wave[left:right, row['channel']].copy()
        assert len(wave) and np.isfinite(wave).all()
        entry.update(source_path=str(path), source_interval=[left / sr, right / sr],
                     source_channel=row['channel'], input_seconds=len(wave) / sr,
                     pcm_sha256=hashlib.sha256(wave.astype('<f4').tobytes()).hexdigest())
        waveform = torch.from_numpy(wave).unsqueeze(0)
        if sr != feat['sampling_rate']:
            waveform = torchaudio.functional.resample(waveform, sr, feat['sampling_rate'])
        fb = kaldi.fbank(waveform * (1 << 15), num_mel_bins=feat['num_mel_bins'],
            frame_length=feat['frame_length'], frame_shift=feat['frame_shift'], dither=feat['dither'],
            energy_floor=0., window_type='hamming', sample_frequency=feat['sampling_rate'])
        with torch.inference_mode():
            vector = model(fb.unsqueeze(0).transpose(-1, -2).cuda()).float().cpu().numpy()[0]
        assert vector.shape == (192,) and np.isfinite(vector).all() and np.linalg.norm(vector) > 0
        entry.update(status='ok', embedding_index=len(vectors))
        vectors.append(vector / np.linalg.norm(vector))
    except Exception as exc:
        entry.update(status='error', error=f'{type(exc).__name__}: {exc}')
    entries.append(entry)
    if (i + 1) % 100 == 0 or i + 1 == len(rows):
        progress = {'completed': i + 1, 'total': len(rows), 'successful': len(vectors),
                    'elapsed_seconds': time.monotonic() - started}
        target = artifact(f'{args.prefix}progress-{args.worker}.json');temp = target.with_suffix('.tmp')
        temp.write_text(json.dumps(progress));temp.replace(target)
        print(json.dumps(progress), flush=True)
np.savez_compressed(artifact(f'{args.prefix}embeddings-{args.worker}.npz'), embeddings=np.stack(vectors))
(artifact(f'{args.prefix}inputs-{args.worker}.json')).write_text(json.dumps(entries, ensure_ascii=False) + '\n')
