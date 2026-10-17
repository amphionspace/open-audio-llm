"""Interpolate the fine-tuned checkpoint toward the original Qwen3-ASR weights (WiSE-FT style).

For each ratio a, every weight present in both models becomes a * base + (1 - a) * fine-tuned, computed in
float32 and stored in the fine-tuned checkpoint's dtype. ``thinker.lm_head.weight`` exists only in the base
files and equals its tied embedding, so it is skipped; ``thinker.audio_tower.sep_token.weight`` exists only in
the fine-tuned model and is kept. Tokenizer, config and every other file are copied from the fine-tuned model.
"""
import argparse
import json
import shutil
from pathlib import Path

import torch
import yaml
from safetensors import safe_open
from safetensors.torch import save_file


def tensors(folder):
    index = json.loads((folder / 'model.safetensors.index.json').read_text())['weight_map']
    handles = {f: safe_open(folder / f, 'pt') for f in set(index.values())}
    return index, handles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    p = yaml.safe_load(parser.parse_args().config.read_text())['parameters']
    base, tuned, out = Path(p['base']), Path(p['fine_tuned']), Path(p['output'])
    base_index, base_files = tensors(base)
    tuned_index, tuned_files = tensors(tuned)
    skipped = sorted(set(base_index) - set(tuned_index))
    kept = sorted(set(tuned_index) - set(base_index))
    assert skipped == ['thinker.lm_head.weight'] and kept == ['thinker.audio_tower.sep_token.weight'], (skipped, kept)
    report = {'skipped_base_only': skipped, 'kept_fine_tuned_only': kept, 'ratios': {}}
    for ratio in p['ratios']:
        target = out / f'base-{int(round(ratio * 100)):02d}'
        target.mkdir(parents=True, exist_ok=True)
        for item in tuned.iterdir():
            if not item.name.endswith('.safetensors') and item.is_file():
                shutil.copy2(item, target / item.name)
        shards = {}
        for name, shard in tuned_index.items():
            weight = tuned_files[shard].get_tensor(name)
            if name in base_index:
                mixed = ratio * base_files[base_index[name]].get_tensor(name).float() + (1 - ratio) * weight.float()
                weight = mixed.to(weight.dtype)
            shards.setdefault(shard, {})[name] = weight.contiguous()
        for shard, weights in shards.items():
            save_file(weights, target / shard, metadata={'format': 'pt'})
        (target / 'complete.json').write_text(json.dumps({'base_ratio': ratio}) + '\n')
        report['ratios'][str(ratio)] = str(target)
        print(f'ratio {ratio}: {target}', flush=True)
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
