"""Speaker embeddings for (window, speaker) groups of segments with a 3D-Speaker model, as a JSONL worker.

Same rows and CLI as embed_worker.py: each input row names a 16 kHz mono WAV and the [start, end] seconds of
one speaker's segments; the segments are concatenated (up to ``--max-seconds``) and embedded with the
ModelScope speaker-verification pipeline (ERes2NetV2 / CAM++ trained on about 200k Chinese speakers).
Runs in the environment that has ModelScope; uses the first visible GPU when there is one.

usage: embed_worker_3dspeaker.py MODEL_DIR INPUT.jsonl OUTPUT.jsonl [--max-seconds 30]
"""
import argparse
import json

import numpy as np
import soundfile as sf
import torch
from modelscope.pipelines import pipeline

RATE = 16000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('input')
    parser.add_argument('output')
    parser.add_argument('--max-seconds', type=float, default=30.0)
    args = parser.parse_args()
    model = pipeline(task='speaker-verification', model=args.model,
                     device='cuda:0' if torch.cuda.is_available() else 'cpu')
    audio, path = None, None
    with open(args.input) as source, open(args.output, 'w') as sink:
        for line in source:
            row = json.loads(line)
            if row['audio'] != path:
                audio, rate = sf.read(row['audio'], dtype='float32')
                assert rate == RATE and audio.ndim == 1, row['audio']
                path = row['audio']
            pieces, total = [], 0.0
            for start, end in row['segments']:
                take = min(end - start, args.max_seconds - total)
                if take <= 0:
                    break
                pieces.append(audio[int(start * RATE):int((start + take) * RATE)])
                total += take
            embedding = np.asarray(model([np.concatenate(pieces)], output_emb=True)['embs'])[0]
            sink.write(json.dumps({'id': row['id'], 'embedding': embedding.tolist()}) + '\n')


if __name__ == '__main__':
    main()
