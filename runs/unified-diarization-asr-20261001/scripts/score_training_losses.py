"""Score per-sample training loss with a fixed model to find suspicious training targets.

Uses the training template and Catalog encoding without augmentation. Every real
meeting record is scored; other sources are sampled with a fixed seed. High-loss
outliers are candidates for label, alignment or audio problems, not proof of them.
"""
import argparse
import json
import os
from pathlib import Path
import random
import runpy
import subprocess
import sys

import yaml


def worker(settings, rank, world, output):
    import torch

    runpy.run_path(settings['plugin'])
    from swift import get_model_processor, get_template

    from open_audio_llm.data.catalog_dataset import CatalogSwiftDataset, read_data_config

    config = read_data_config(settings['data_config'])
    config['validation'] = [s for s in config['train']
                            if s['dataset_id'] not in settings['skip_sources']]
    model, processor = get_model_processor(settings['model'], model_type=settings['model_type'],
                                           torch_dtype=torch.bfloat16, device_map='cuda:0', attn_impl='sdpa')
    template = get_template(processor, max_length=settings['max_length'])
    template.set_mode('train')  # inference mode encodes no labels
    dataset = CatalogSwiftDataset(config, training=False, encode=template.encode, message_format='qwen3_asr')
    rng = random.Random(settings['seed'])
    selected = [i for i in range(len(dataset))
                if any(k in dataset.records[i].dataset_id for k in settings['full_sources'])
                or rng.random() < settings['sample_rate']]
    model.eval()
    with (output / f'losses-rank{rank}.jsonl').open('w') as stream:
        for i in selected[rank::world]:
            row = dataset.records[i]
            batch = template.data_collator([dataset[i]])
            labels = batch.pop('labels').to('cuda:0')
            batch = {k: (v.to('cuda:0', torch.bfloat16) if v.is_floating_point() else v.to('cuda:0'))
                     for k, v in batch.items() if torch.is_tensor(v) and not k.startswith('catalog')}
            with torch.no_grad():
                logits = model(**batch).logits[:, :-1].float()
            target = labels[:, 1:]
            tokens = int((target != -100).sum())
            ce = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), target.reshape(-1), ignore_index=-100, reduction='sum')
            stream.write(json.dumps({'id': row.record.id, 'dataset_id': row.dataset_id,
                                     'tokens': tokens, 'ce': float(ce) / max(tokens, 1)}) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--rank', type=int)
    args = parser.parse_args()
    raw = yaml.safe_load(args.config.read_text())
    settings = raw['parameters']
    output = Path(settings['output'])
    output.mkdir(parents=True, exist_ok=True)
    gpus = raw['runtime']['gpus']
    if args.rank is not None:
        worker(settings, args.rank, len(gpus), output)
        return
    jobs = [subprocess.Popen([sys.executable, __file__, '--config', str(args.config), '--rank', str(r)],
                             env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(g)},
                             stdout=(output / f'worker{r}.log').open('w'), stderr=subprocess.STDOUT)
            for r, g in enumerate(gpus)]
    codes = [job.wait() for job in jobs]
    if any(codes):
        raise SystemExit(f'workers exited {codes}')


if __name__ == '__main__':
    main()
